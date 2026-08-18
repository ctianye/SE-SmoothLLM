"""从固定版本的官方来源加载 JailbreakBench 主实验样本。"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx

BenchmarkSplit = Literal["harmful", "benign"]
JsonFetcher = Callable[[str], object]
DatasetLoader = Callable[..., Iterable[Mapping[str, object]]]

DEFAULT_CONFIG_PATH = Path(__file__).parent / "configs" / "vicuna_gcg.json"


class JBBDataError(ValueError):
    """JailbreakBench 配置或数据不满足主实验约束。"""


@dataclass(frozen=True, slots=True)
class BenchmarkSample:
    """一个可追溯到官方来源的统一基准样本。"""

    index: int
    split: BenchmarkSplit
    behavior: str
    category: str
    goal: str
    prompt: str


def load_experiment_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """读取版本控制中的主实验 JSON 配置。"""

    config_path = Path(path)
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise JBBDataError(f"failed to read experiment config: {config_path}") from exc
    if not isinstance(config, dict):
        raise JBBDataError("experiment config must be a JSON object")
    return config


def load_harmful_samples(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    fetch_json: JsonFetcher | None = None,
) -> tuple[BenchmarkSample, ...]:
    """从固定版本的 JBB GCG artifact 中只读取 harmful prompt 及描述字段。"""

    config = load_experiment_config(config_path)
    attack = _require_mapping(config, "attack", context="config")
    url = _require_text(attack, "artifact_url", context="config.attack")
    revision = _require_text(attack, "revision", context="config.attack")
    if revision not in url:
        raise JBBDataError("artifact_url must contain config.attack.revision")

    payload = (fetch_json or _fetch_json)(url)
    if not isinstance(payload, Mapping):
        raise JBBDataError("JBB artifact must be a JSON object")
    records = payload.get("jailbreaks")
    if not isinstance(records, list):
        raise JBBDataError("JBB artifact must contain a jailbreaks list")

    samples = tuple(_harmful_sample(record, position) for position, record in enumerate(records))
    expected_count = _expected_count(config, "harmful")
    return _validate_samples(samples, split="harmful", expected_count=expected_count)


def load_benign_samples(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    dataset_loader: DatasetLoader | None = None,
) -> tuple[BenchmarkSample, ...]:
    """从固定版本的 JBB-Behaviors benign split 加载正常请求。"""

    config = load_experiment_config(config_path)
    dataset = _require_mapping(config, "dataset", context="config")
    dataset_id = _require_text(dataset, "id", context="config.dataset")
    configuration = _require_text(dataset, "configuration", context="config.dataset")
    revision = _require_text(dataset, "revision", context="config.dataset")
    split = _split_config(config, "benign")
    split_name = _require_text(split, "name", context="config.dataset.splits.benign")

    loader = dataset_loader or _load_hugging_face_dataset
    records = loader(
        dataset_id,
        configuration,
        split=split_name,
        revision=revision,
    )
    samples = tuple(_benign_sample(record, position) for position, record in enumerate(records))
    expected_count = _expected_count(config, "benign")
    return _validate_samples(samples, split="benign", expected_count=expected_count)


def load_jbb_samples(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    splits: Sequence[BenchmarkSplit] = ("harmful", "benign"),
    fetch_json: JsonFetcher | None = None,
    dataset_loader: DatasetLoader | None = None,
) -> tuple[BenchmarkSample, ...]:
    """只加载选定 split，并保持 harmful、benign 的稳定顺序。"""

    requested = set(splits)
    unsupported = requested.difference(("harmful", "benign"))
    if unsupported:
        raise JBBDataError(f"unsupported splits: {sorted(unsupported)}")
    if not requested:
        raise JBBDataError("at least one split must be selected")

    harmful = (
        load_harmful_samples(config_path, fetch_json=fetch_json)
        if "harmful" in requested
        else ()
    )
    benign = (
        load_benign_samples(config_path, dataset_loader=dataset_loader)
        if "benign" in requested
        else ()
    )
    samples = harmful + benign
    identities = {(sample.split, sample.index) for sample in samples}
    if len(identities) != len(samples):
        raise JBBDataError("sample identities must be unique by split and index")
    return samples


def _fetch_json(url: str) -> object:
    response = httpx.get(url, follow_redirects=True, timeout=30.0)
    response.raise_for_status()
    return response.json()


def _load_hugging_face_dataset(
    dataset_id: str,
    configuration: str,
    *,
    split: str,
    revision: str,
) -> Iterable[Mapping[str, object]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - 由独立安装验证覆盖
        raise RuntimeError(
            'loading benign samples requires: pip install -e ".[benchmark]"'
        ) from exc
    return load_dataset(dataset_id, configuration, split=split, revision=revision)


def _harmful_sample(record: object, position: int) -> BenchmarkSample:
    context = f"artifact.jailbreaks[{position}]"
    if not isinstance(record, Mapping):
        raise JBBDataError(f"{context} must be an object")
    return BenchmarkSample(
        index=_require_index(record, "index", context=context),
        split="harmful",
        behavior=_require_text(record, "behavior", context=context),
        category=_require_text(record, "category", context=context),
        goal=_require_text(record, "goal", context=context),
        prompt=_require_text(record, "prompt", context=context),
    )


def _benign_sample(record: object, position: int) -> BenchmarkSample:
    context = f"dataset.benign[{position}]"
    if not isinstance(record, Mapping):
        raise JBBDataError(f"{context} must be an object")
    goal = _require_text(record, "Goal", context=context)
    return BenchmarkSample(
        index=_require_index(record, "Index", context=context),
        split="benign",
        behavior=_require_text(record, "Behavior", context=context),
        category=_require_text(record, "Category", context=context),
        goal=goal,
        prompt=goal,
    )


def _validate_samples(
    samples: tuple[BenchmarkSample, ...],
    *,
    split: BenchmarkSplit,
    expected_count: int,
) -> tuple[BenchmarkSample, ...]:
    if len(samples) != expected_count:
        raise JBBDataError(f"expected {expected_count} {split} samples, received {len(samples)}")
    indices = [sample.index for sample in samples]
    if len(set(indices)) != len(indices):
        raise JBBDataError(f"{split} sample indices must be unique")
    return samples


def _expected_count(config: Mapping[str, object], split: BenchmarkSplit) -> int:
    split_config = _split_config(config, split)
    value = split_config.get("expected_count")
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise JBBDataError(f"config.dataset.splits.{split}.expected_count must be positive")
    return value


def _split_config(
    config: Mapping[str, object], split: BenchmarkSplit
) -> Mapping[str, object]:
    dataset = _require_mapping(config, "dataset", context="config")
    splits = _require_mapping(dataset, "splits", context="config.dataset")
    return _require_mapping(splits, split, context="config.dataset.splits")


def _require_mapping(
    record: Mapping[str, object], key: str, *, context: str
) -> Mapping[str, object]:
    value = record.get(key)
    if not isinstance(value, Mapping):
        raise JBBDataError(f"{context}.{key} must be an object")
    return value


def _require_text(record: Mapping[str, object], key: str, *, context: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise JBBDataError(f"{context}.{key} must be a non-empty string")
    return value


def _require_index(record: Mapping[str, object], key: str, *, context: str) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise JBBDataError(f"{context}.{key} must be a non-negative integer")
    return value
