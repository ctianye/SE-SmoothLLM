"""对已保存的 Vicuna 最终回答运行可恢复的 JBB Llama-3-8B Refusal Judge。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from tqdm import tqdm

from benchmarks.jbb_loader import load_experiment_config
from benchmarks.jbb_refusal_judge import build_refusal_prompt, parse_refusal_output
from benchmarks.run_jbb import (
    append_record,
    config_fingerprint,
    probe_model_service,
    recover_jsonl_tail,
)
from se_smoothllm.backends.base import Backend
from se_smoothllm.backends.openai_compatible import OpenAICompatibleBackend

DEFAULT_CONFIG_PATH = Path(__file__).parent / "configs" / "llama3_8b_refusal_judge.json"
DEFAULT_INPUT_DIR = Path("results/raw")
DEFAULT_OUTPUT_PATH = Path("results/judged/jbb-llama3-8b-refusal.jsonl")
Split = Literal["harmful", "benign"]


class RefusalJudgeRunError(RuntimeError):
    """输入数据、检查点或 Judge 服务不满足评价约束。"""


@dataclass(frozen=True, slots=True)
class RefusalJudgeTask:
    """一条可独立评价并通过复合键恢复的最终回答。"""

    method: str
    seed: int
    split: Split
    index: int
    goal: str
    response: str
    source_config_sha256: str

    @property
    def key(self) -> tuple[str, int, str, int]:
        return (self.method, self.seed, self.split, self.index)

    @property
    def response_sha256(self) -> str:
        return hashlib.sha256(self.response.encode("utf-8")).hexdigest()


def load_judge_tasks(input_dir: str | Path) -> tuple[RefusalJudgeTask, ...]:
    """读取目录内全部生成 JSONL，并拒绝重复或损坏的复合键。"""

    directory = Path(input_dir)
    paths = sorted(directory.glob("*.jsonl"))
    if not paths:
        raise RefusalJudgeRunError(f"no JSONL generation files found: {directory}")

    tasks: list[RefusalJudgeTask] = []
    seen: set[tuple[str, int, str, int]] = set()
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    sample = record["sample"]
                    result = record["result"]
                    task = RefusalJudgeTask(
                        method=_non_empty_text(record["method"], "method"),
                        seed=_integer(record["seed"], "seed"),
                        split=_split(sample["split"]),
                        index=_integer(sample["index"], "sample.index"),
                        goal=_non_empty_text(sample["goal"], "sample.goal"),
                        response=_text(result["response"], "result.response"),
                        source_config_sha256=_non_empty_text(
                            record["config_sha256"], "config_sha256"
                        ),
                    )
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise RefusalJudgeRunError(
                        f"invalid generation record at {path}:{line_number}"
                    ) from exc
                if task.key in seen:
                    raise RefusalJudgeRunError(
                        f"duplicate generation key at {path}:{line_number}: {task.key}"
                    )
                seen.add(task.key)
                tasks.append(task)

    return tuple(sorted(tasks, key=lambda item: (item.split, item.method, item.seed, item.index)))


def run_judge_task(
    task: RefusalJudgeTask,
    *,
    backend: Backend,
    config: dict[str, Any],
    fingerprint: str,
    request_delay_seconds: float = 0.0,
) -> dict[str, Any]:
    """评价一条最终回答，并保留 Judge 原始输出、格式状态和 token。"""

    judge = _mapping(config, "judge")
    prompt = build_refusal_prompt(
        _non_empty_text(judge.get("prompt_template"), "judge.prompt_template"),
        goal=task.goal,
        response=task.response,
    )
    if request_delay_seconds > 0:
        time.sleep(request_delay_seconds)
    generation = backend.generate(prompt)
    decision = parse_refusal_output(
        generation.text,
        positive_answer=_non_empty_text(judge.get("positive_answer"), "judge.positive_answer"),
        negative_answer=_non_empty_text(judge.get("negative_answer"), "judge.negative_answer"),
    )
    metric_key = _non_empty_text(judge.get("metric_key"), "judge.metric_key")
    judgment = {
        "positive": decision.refused,
        "metric": metric_key,
        "raw_output": generation.text,
        "normalized_output": decision.normalized_output,
        "format_conforming": decision.format_conforming,
        "model": generation.model,
        "latency_ms": generation.latency_ms,
        "prompt_tokens": generation.prompt_tokens,
        "completion_tokens": generation.completion_tokens,
    }
    judgment[metric_key] = decision.refused
    return {
        "schema_version": 1,
        "experiment_id": _non_empty_text(config.get("experiment_id"), "experiment_id"),
        "judge_config_sha256": fingerprint,
        "source": {
            "method": task.method,
            "seed": task.seed,
            "split": task.split,
            "index": task.index,
            "config_sha256": task.source_config_sha256,
            "response_sha256": task.response_sha256,
        },
        "judgment": judgment,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


def load_completed_judgments(
    path: Path,
    *,
    fingerprint: str,
) -> dict[tuple[str, int, str, int], str]:
    """读取 Judge 检查点并返回复合键到源回答哈希的映射。"""

    if not path.exists():
        return {}
    completed: dict[tuple[str, int, str, int], str] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if record["judge_config_sha256"] != fingerprint:
                    raise RefusalJudgeRunError(
                        f"judge checkpoint config mismatch at {path}:{line_number}"
                    )
                source = record["source"]
                key = (
                    str(source["method"]),
                    int(source["seed"]),
                    str(source["split"]),
                    int(source["index"]),
                )
                response_sha256 = _non_empty_text(
                    source["response_sha256"], "source.response_sha256"
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise RefusalJudgeRunError(
                    f"invalid judge checkpoint at {path}:{line_number}"
                ) from exc
            if key in completed:
                raise RefusalJudgeRunError(
                    f"duplicate judge checkpoint key at {path}:{line_number}: {key}"
                )
            completed[key] = response_sha256
    return completed


def main(argv: list[str] | None = None) -> int:
    """加载生成结果、恢复 Judge 检查点并并发评价最终回答。"""

    args = _build_parser().parse_args(argv)
    config = load_experiment_config(Path(args.config).resolve())
    fingerprint = config_fingerprint(config)
    tasks = load_judge_tasks(Path(args.input_dir).resolve())
    input_config = _mapping(config, "input")
    expected_records = _integer(input_config.get("expected_records"), "input.expected_records")
    if len(tasks) != expected_records:
        raise RefusalJudgeRunError(
            f"expected {expected_records} generation records, received {len(tasks)}"
        )
    expected_source_fingerprint = _non_empty_text(
        input_config.get("source_config_sha256"), "input.source_config_sha256"
    )
    source_fingerprints = {task.source_config_sha256 for task in tasks}
    if source_fingerprints != {expected_source_fingerprint}:
        raise RefusalJudgeRunError(
            "generation source config mismatch: "
            f"expected {expected_source_fingerprint}, received {sorted(source_fingerprints)}"
        )

    tasks = _select_tasks(
        tasks,
        splits=tuple(args.split or ("harmful", "benign")),
        methods=tuple(args.method or ()),
        seeds=tuple(args.seed or ()),
        limit=args.limit,
    )
    output = Path(args.output).resolve()
    recovery = recover_jsonl_tail(output)
    if recovery != "unchanged":
        print(f"Judge 检查点尾部恢复状态：{recovery}，文件={output}")
    completed = load_completed_judgments(output, fingerprint=fingerprint)

    pending: list[RefusalJudgeTask] = []
    for task in tasks:
        previous_hash = completed.get(task.key)
        if previous_hash is None:
            pending.append(task)
        elif previous_hash != task.response_sha256:
            raise RefusalJudgeRunError(f"source response changed for checkpoint key: {task.key}")

    print(
        f"Judge任务总数={len(tasks)}，已完成={len(tasks) - len(pending)}，"
        f"待运行={len(pending)}，输出={output}"
    )
    if args.dry_run or not pending:
        return 0

    generation = _mapping(config, "generation")
    serving = _mapping(config, "serving")
    judge = _mapping(config, "judge")
    base_url = args.base_url or _non_empty_text(serving.get("base_url"), "serving.base_url")
    timeout = float(_number(generation.get("timeout_seconds"), "generation.timeout"))
    extra_body_value = generation.get("extra_body", {})
    if not isinstance(extra_body_value, dict):
        raise RefusalJudgeRunError("generation.extra_body must be an object")
    api_key = os.environ.get(args.api_key_env)
    served_models = probe_model_service(
        base_url, timeout=min(timeout, 10.0), api_key=api_key
    )
    print(f"Judge模型服务正常：{', '.join(served_models)}")

    model = args.model or _non_empty_text(judge.get("model_id"), "judge.model_id")
    backend = OpenAICompatibleBackend(
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=_number(generation.get("temperature"), "generation.temperature"),
        max_tokens=_integer(generation.get("max_tokens"), "generation.max_tokens"),
        timeout=timeout,
        max_retries=_integer(generation.get("max_retries"), "generation.max_retries"),
        retry_backoff=_number(
            generation.get("retry_backoff_seconds"), "generation.retry_backoff"
        ),
        extra_body=dict(extra_body_value),
    )

    failures: list[tuple[RefusalJudgeTask, Exception]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures: dict[Future[dict[str, Any]], RefusalJudgeTask] = {
            executor.submit(
                run_judge_task,
                task,
                backend=backend,
                config=config,
                fingerprint=fingerprint,
                request_delay_seconds=args.request_delay_seconds,
            ): task
            for task in pending
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="JBB refusal judge"):
            task = futures[future]
            try:
                append_record(output, future.result())
            except Exception as exc:
                failures.append((task, exc))

    if failures:
        task, error = failures[0]
        print(f"本批有 {len(failures)} 条 Judge 失败；已返回的成功结果均已保存到 {output}")
        raise RefusalJudgeRunError(
            f"judge task failed and checkpoint was preserved: {task.key}: {error}"
        ) from error
    print(f"本次新增 {len(pending)} 条 Judge 记录，检查点位于 {output}")
    return 0


def _select_tasks(
    tasks: tuple[RefusalJudgeTask, ...],
    *,
    splits: tuple[str, ...],
    methods: tuple[str, ...],
    seeds: tuple[int, ...],
    limit: int | None,
) -> tuple[RefusalJudgeTask, ...]:
    selected = tuple(
        task
        for task in tasks
        if task.split in splits
        and (not methods or task.method in methods)
        and (not seeds or task.seed in seeds)
    )
    if limit is None:
        return selected
    limited: list[RefusalJudgeTask] = []
    for split in splits:
        split_tasks = [task for task in selected if task.split == split]
        limited.extend(split_tasks[:limit])
    return tuple(limited)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--api-key-env", default="SE_SMOOTHLLM_API_KEY")
    parser.add_argument("--workers", type=_positive_int, default=8)
    parser.add_argument(
        "--request-delay-seconds",
        type=_non_negative_float,
        default=0.0,
        help="在每次 Judge 请求前等待的秒数；服务商有 RPM 限制时配合 --workers 1 使用",
    )
    parser.add_argument("--split", action="append", choices=("harmful", "benign"))
    parser.add_argument("--method", action="append")
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--limit", type=_positive_int)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _mapping(mapping: dict[str, Any], name: str) -> dict[str, Any]:
    value = mapping.get(name)
    if not isinstance(value, dict):
        raise RefusalJudgeRunError(f"{name} must be an object")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise RefusalJudgeRunError(f"{name} must be text")
    return value


def _non_empty_text(value: object, name: str) -> str:
    text = _text(value, name)
    if not text.strip():
        raise RefusalJudgeRunError(f"{name} must be non-empty")
    return text


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RefusalJudgeRunError(f"{name} must be an integer")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RefusalJudgeRunError(f"{name} must be a number")
    return float(value)


def _split(value: object) -> Split:
    if value not in ("harmful", "benign"):
        raise RefusalJudgeRunError(f"unsupported split: {value}")
    return value  # type: ignore[return-value]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _non_negative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RefusalJudgeRunError as error:
        print(f"Refusal Judge 中止：{error}", file=sys.stderr)
        raise SystemExit(1) from error
