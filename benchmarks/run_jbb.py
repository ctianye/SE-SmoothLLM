"""可断点续跑的 JailbreakBench 生成入口。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import httpx
from tqdm import tqdm

from benchmarks.jbb_loader import (
    DEFAULT_CONFIG_PATH,
    BenchmarkSample,
    load_experiment_config,
    load_jbb_samples,
)
from se_smoothllm.backends.base import Backend
from se_smoothllm.backends.openai_compatible import OpenAICompatibleBackend
from se_smoothllm.core import SmoothGuard
from se_smoothllm.judges.prefix import PrefixJudge
from se_smoothllm.models import CopyTrace, DefenseResult
from se_smoothllm.perturbations import RandomSwapPerturbation

Method = Literal["undefended", "smoothllm_fixed", "se_smoothllm"]
SUPPORTED_METHODS: tuple[Method, ...] = (
    "undefended",
    "smoothllm_fixed",
    "se_smoothllm",
)


class BenchmarkRunError(RuntimeError):
    """运行配置、检查点或模型服务不满足实验要求。"""


@dataclass(frozen=True, slots=True)
class RunTask:
    """一条可独立执行和恢复的样本任务。"""

    method: Method
    base_seed: int
    sample: BenchmarkSample

    @property
    def key(self) -> tuple[str, int, str, int]:
        return (self.method, self.base_seed, self.sample.split, self.sample.index)


def config_fingerprint(config: dict[str, Any]) -> str:
    """为完整主配置生成稳定指纹，防止不同配置混写到同一结果文件。"""

    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def effective_sample_seed(base_seed: int, sample: BenchmarkSample) -> int:
    """从主 seed 与样本身份派生独立、可复现的扰动 seed。"""

    identity = f"{base_seed}:{sample.split}:{sample.index}".encode()
    return int.from_bytes(hashlib.sha256(identity).digest()[:8], byteorder="big")


def build_tasks(
    samples: tuple[BenchmarkSample, ...],
    *,
    methods: tuple[Method, ...],
    seeds: tuple[int, ...],
) -> tuple[RunTask, ...]:
    """按照 method、seed、split 和 index 的稳定顺序构造任务。"""

    ordered_samples = sorted(samples, key=lambda item: (item.split, item.index))
    return tuple(
        RunTask(method=method, base_seed=seed, sample=sample)
        for method in methods
        for seed in seeds
        for sample in ordered_samples
    )


def run_task(
    task: RunTask,
    *,
    backend: Backend,
    config: dict[str, Any],
    fingerprint: str,
) -> dict[str, Any]:
    """执行一条任务并返回可直接写入 JSONL 的记录。"""

    judge = PrefixJudge()
    seed = effective_sample_seed(task.base_seed, task.sample)
    if task.method == "undefended":
        result = _run_undefended(task.sample.prompt, backend=backend, judge=judge)
    else:
        defense = _require_mapping(config, "defense")
        perturbation_config = _require_mapping(defense, "perturbation")
        if perturbation_config.get("class_name") != "RandomSwapPerturbation":
            raise BenchmarkRunError("only RandomSwapPerturbation is supported by run_jbb")
        guard = SmoothGuard(
            backend=backend,
            judge=judge,
            perturbation=RandomSwapPerturbation(q=_require_number(perturbation_config, "q_percent")),
            copies=_require_int(defense, "copies"),
            seed=seed,
        )
        result = (
            guard.defend(task.sample.prompt)
            if task.method == "smoothllm_fixed"
            else guard.defend_early(task.sample.prompt)
        )

    return {
        "schema_version": 1,
        "experiment_id": _require_text(config, "experiment_id"),
        "config_sha256": fingerprint,
        "method": task.method,
        "seed": task.base_seed,
        "effective_seed": seed,
        "sample": asdict(task.sample),
        "result": result.model_dump(mode="json"),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


def load_completed_keys(
    path: Path,
    *,
    fingerprint: str,
) -> set[tuple[str, int, str, int]]:
    """读取已有 JSONL 检查点，并拒绝配置混用或损坏记录。"""

    if not path.exists():
        return set()

    completed: set[tuple[str, int, str, int]] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if record["config_sha256"] != fingerprint:
                    raise BenchmarkRunError(
                        f"checkpoint config mismatch at {path}:{line_number}"
                    )
                sample = record["sample"]
                key = (
                    str(record["method"]),
                    int(record["seed"]),
                    str(sample["split"]),
                    int(sample["index"]),
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise BenchmarkRunError(
                    f"invalid checkpoint record at {path}:{line_number}"
                ) from exc
            if key in completed:
                raise BenchmarkRunError(f"duplicate checkpoint key at {path}:{line_number}: {key}")
            completed.add(key)
    return completed


def recover_jsonl_tail(path: Path) -> Literal["unchanged", "completed", "truncated"]:
    """修复断电造成的最后一行；绝不修改文件中间的记录。"""

    if not path.exists():
        return "unchanged"
    payload = path.read_bytes()
    if not payload or payload.endswith(b"\n"):
        return "unchanged"

    last_newline = payload.rfind(b"\n")
    tail_start = last_newline + 1
    tail = payload[tail_start:]
    try:
        json.loads(tail)
    except (UnicodeDecodeError, json.JSONDecodeError):
        with path.open("r+b") as handle:
            handle.truncate(tail_start)
            handle.flush()
            os.fsync(handle.fileno())
        return "truncated"

    descriptor = os.open(path, os.O_APPEND | os.O_WRONLY)
    try:
        os.write(descriptor, b"\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return "completed"


def append_record(path: Path, record: dict[str, Any]) -> None:
    """每完成一个样本立即追加并同步，降低实例中断导致的数据损失。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        remaining = memoryview(encoded)
        while remaining:
            written = os.write(descriptor, remaining)
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def probe_model_service(
    base_url: str, *, timeout: float, api_key: str | None = None
) -> tuple[str, ...]:
    """在正式计费实验前确认 OpenAI 兼容服务可访问且返回模型列表。"""

    endpoint = f"{base_url.rstrip('/')}/models"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    try:
        response = httpx.get(endpoint, headers=headers, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        models = tuple(
            item["id"]
            for item in payload["data"]
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        )
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
        raise BenchmarkRunError(f"model service probe failed: {endpoint}: {exc}") from exc
    if not models:
        raise BenchmarkRunError(f"model service returned no model ids: {endpoint}")
    return models


def main(argv: list[str] | None = None) -> int:
    """解析命令行、恢复检查点并运行选定任务。"""

    parser = _build_parser()
    args = parser.parse_args(argv)
    config_path = Path(args.config).resolve()
    config = load_experiment_config(config_path)
    fingerprint = config_fingerprint(config)

    methods = _selected_methods(args.method, config)
    seeds = _selected_seeds(args.seed, config)
    splits = tuple(args.split or ("harmful", "benign"))
    samples = load_jbb_samples(config_path, splits=splits)
    if args.limit is not None:
        samples = _limit_each_split(samples, args.limit)
    tasks = build_tasks(samples, methods=methods, seeds=seeds)

    output = Path(args.output or _default_output(config)).resolve()
    recovery = recover_jsonl_tail(output)
    if recovery == "truncated":
        print(f"检测到中断写入，已移除检查点最后一个不完整片段：{output}")
    elif recovery == "completed":
        print(f"已为检查点最后一条完整记录补写换行符：{output}")
    completed = load_completed_keys(output, fingerprint=fingerprint)
    pending = tuple(task for task in tasks if task.key not in completed)
    print(
        f"任务总数={len(tasks)}，已完成={len(tasks) - len(pending)}，待运行={len(pending)}，"
        f"输出={output}"
    )
    if args.dry_run or not pending:
        return 0

    generation = _require_mapping(config, "generation")
    model = _require_mapping(config, "model")
    serving = _require_mapping(config, "serving")
    base_url = args.base_url or _require_text(serving, "base_url")
    api_key = os.environ.get(args.api_key_env)
    timeout = float(_require_number(generation, "timeout_seconds"))
    served_models = probe_model_service(
        base_url, timeout=min(timeout, 10.0), api_key=api_key
    )
    print(f"模型服务正常：{', '.join(served_models)}")

    backend = OpenAICompatibleBackend(
        base_url=base_url,
        model=_require_text(model, "id"),
        api_key=api_key,
        system_prompt=_require_text(model, "system_prompt"),
        temperature=_require_number(generation, "temperature"),
        max_tokens=_require_int(generation, "max_tokens"),
        timeout=timeout,
        max_retries=_require_int(generation, "max_retries"),
        retry_backoff=_require_number(generation, "retry_backoff_seconds"),
    )

    failures: list[tuple[RunTask, Exception]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures: dict[Future[dict[str, Any]], RunTask] = {
            executor.submit(
                run_task,
                task,
                backend=backend,
                config=config,
                fingerprint=fingerprint,
            ): task
            for task in pending
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="JBB generation"):
            task = futures[future]
            try:
                append_record(output, future.result())
            except Exception as exc:
                failures.append((task, exc))
                for unfinished in futures:
                    unfinished.cancel()
                break

    if failures:
        task, error = failures[0]
        raise BenchmarkRunError(f"task failed and checkpoint was preserved: {task.key}: {error}") from error
    print(f"本次新增 {len(pending)} 条记录，检查点位于 {output}")
    return 0


def _run_undefended(prompt: str, *, backend: Backend, judge: PrefixJudge) -> DefenseResult:
    generation = backend.generate(prompt)
    judgment = judge.classify(prompt, generation)
    trace = CopyTrace(
        copy_index=1,
        perturbed_prompt=prompt,
        response=generation.text,
        model=generation.model,
        judge_result=judgment,
        latency_ms=generation.latency_ms,
        prompt_tokens=generation.prompt_tokens,
        completion_tokens=generation.completion_tokens,
    )
    votes = {"safe": 0, "jailbroken": 0}
    votes[judgment.label] = 1
    return DefenseResult(
        response=generation.text,
        jailbroken=judgment.jailbroken,
        copies_used=1,
        votes=votes,
        stopped_early=False,
        latency_ms=generation.latency_ms,
        prompt_tokens=generation.prompt_tokens,
        completion_tokens=generation.completion_tokens,
        trace=(trace,),
    )


def _selected_methods(values: list[str] | None, config: dict[str, Any]) -> tuple[Method, ...]:
    configured = values or config.get("methods")
    if not isinstance(configured, list) or not configured:
        raise BenchmarkRunError("methods must be a non-empty list")
    unsupported = [value for value in configured if value not in SUPPORTED_METHODS]
    if unsupported:
        raise BenchmarkRunError(f"unsupported methods: {unsupported}")
    if len(set(configured)) != len(configured):
        raise BenchmarkRunError("methods must not contain duplicates")
    return tuple(configured)  # type: ignore[return-value]


def _selected_seeds(values: list[int] | None, config: dict[str, Any]) -> tuple[int, ...]:
    configured = values or config.get("seeds")
    if (
        not isinstance(configured, list)
        or not configured
        or any(isinstance(value, bool) or not isinstance(value, int) for value in configured)
    ):
        raise BenchmarkRunError("seeds must be a non-empty integer list")
    if len(set(configured)) != len(configured):
        raise BenchmarkRunError("seeds must not contain duplicates")
    return tuple(configured)


def _limit_each_split(
    samples: tuple[BenchmarkSample, ...],
    limit: int,
) -> tuple[BenchmarkSample, ...]:
    selected: list[BenchmarkSample] = []
    for split in ("harmful", "benign"):
        split_samples = [sample for sample in samples if sample.split == split]
        selected.extend(split_samples[:limit])
    return tuple(selected)


def _default_output(config: dict[str, Any]) -> str:
    return f"results/raw/{_require_text(config, 'experiment_id')}.jsonl"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key-env", default="SE_SMOOTHLLM_API_KEY")
    parser.add_argument("--method", action="append", choices=SUPPORTED_METHODS)
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--split", action="append", choices=("harmful", "benign"))
    parser.add_argument("--limit", type=_positive_int)
    parser.add_argument("--workers", type=_positive_int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _require_mapping(mapping: dict[str, Any], name: str) -> dict[str, Any]:
    value = mapping.get(name)
    if not isinstance(value, dict):
        raise BenchmarkRunError(f"{name} must be an object")
    return value


def _require_text(mapping: dict[str, Any], name: str) -> str:
    value = mapping.get(name)
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkRunError(f"{name} must be a non-empty string")
    return value


def _require_int(mapping: dict[str, Any], name: str) -> int:
    value = mapping.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise BenchmarkRunError(f"{name} must be an integer")
    return value


def _require_number(mapping: dict[str, Any], name: str) -> float:
    value = mapping.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkRunError(f"{name} must be a number")
    return float(value)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BenchmarkRunError as error:
        print(f"实验中止：{error}", file=sys.stderr)
        raise SystemExit(1) from error
