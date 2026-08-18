import json
from pathlib import Path

import httpx
import pytest
from benchmarks.jbb_loader import BenchmarkSample, load_experiment_config
from benchmarks.run_jbb import (
    BenchmarkRunError,
    RunTask,
    append_record,
    build_tasks,
    config_fingerprint,
    effective_sample_seed,
    load_completed_keys,
    probe_model_service,
    recover_jsonl_tail,
    run_task,
)

from se_smoothllm.backends.mock import MockBackend
from se_smoothllm.models import Generation


def _sample(*, split: str = "harmful", index: int = 3) -> BenchmarkSample:
    return BenchmarkSample(
        index=index,
        split=split,  # type: ignore[arg-type]
        behavior="test behavior",
        category="test category",
        goal="test goal",
        prompt="A sufficiently long prompt for a benchmark test.",
    )


def _config(*, copies: int = 10, q: float = 10) -> dict[str, object]:
    config = load_experiment_config()
    config["defense"]["copies"] = copies
    config["defense"]["perturbation"]["q_percent"] = q
    return config


def _generation(text: str, index: int = 1) -> Generation:
    return Generation(
        text=text,
        model="mock-model",
        prompt_tokens=index,
        completion_tokens=index * 2,
        latency_ms=float(index),
    )


def test_effective_sample_seed_is_stable_and_uses_full_sample_identity() -> None:
    harmful = _sample(split="harmful", index=1)
    benign = _sample(split="benign", index=1)

    assert effective_sample_seed(42, harmful) == effective_sample_seed(42, harmful)
    assert effective_sample_seed(42, harmful) != effective_sample_seed(43, harmful)
    assert effective_sample_seed(42, harmful) != effective_sample_seed(42, benign)


def test_build_tasks_has_stable_order_and_unique_keys() -> None:
    samples = (_sample(split="harmful", index=2), _sample(split="benign", index=1))

    tasks = build_tasks(
        samples,
        methods=("undefended", "se_smoothllm"),
        seeds=(42, 43),
    )

    assert len(tasks) == 8
    assert len({task.key for task in tasks}) == 8
    assert tasks[0].key == ("undefended", 42, "benign", 1)
    assert tasks[-1].key == ("se_smoothllm", 43, "harmful", 2)


def test_run_undefended_produces_uniform_trace_record() -> None:
    config = _config()
    backend = MockBackend(_generation("I cannot help.", 7))
    task = RunTask(method="undefended", base_seed=42, sample=_sample())

    record = run_task(
        task,
        backend=backend,
        config=config,  # type: ignore[arg-type]
        fingerprint=config_fingerprint(config),  # type: ignore[arg-type]
    )

    assert backend.call_count == 1
    assert record["method"] == "undefended"
    assert record["result"]["copies_used"] == 1
    assert record["result"]["votes"] == {"safe": 1, "jailbroken": 0}
    assert record["result"]["trace"][0]["model"] == "mock-model"


def test_fixed_and_early_tasks_use_configured_budget() -> None:
    config = _config(copies=10, q=0)
    safe = [_generation("I cannot help.", index) for index in range(1, 11)]
    fixed_backend = MockBackend(safe)
    early_backend = MockBackend(safe)

    fixed = run_task(
        RunTask(method="smoothllm_fixed", base_seed=42, sample=_sample()),
        backend=fixed_backend,
        config=config,  # type: ignore[arg-type]
        fingerprint=config_fingerprint(config),  # type: ignore[arg-type]
    )
    early = run_task(
        RunTask(method="se_smoothllm", base_seed=42, sample=_sample()),
        backend=early_backend,
        config=config,  # type: ignore[arg-type]
        fingerprint=config_fingerprint(config),  # type: ignore[arg-type]
    )

    assert fixed_backend.call_count == fixed["result"]["copies_used"] == 10
    assert early_backend.call_count == early["result"]["copies_used"] == 5
    assert fixed["result"]["jailbroken"] is early["result"]["jailbroken"] is False
    assert fixed["result"]["prompt_tokens"] == 55
    assert early["result"]["prompt_tokens"] == 15


def test_checkpoint_round_trip_and_config_mismatch(tmp_path: Path) -> None:
    config = _config()
    fingerprint = config_fingerprint(config)  # type: ignore[arg-type]
    record = run_task(
        RunTask(method="undefended", base_seed=42, sample=_sample()),
        backend=MockBackend("I cannot help."),
        config=config,  # type: ignore[arg-type]
        fingerprint=fingerprint,
    )
    output = tmp_path / "checkpoint.jsonl"

    append_record(output, record)

    assert load_completed_keys(output, fingerprint=fingerprint) == {
        ("undefended", 42, "harmful", 3)
    }
    with pytest.raises(BenchmarkRunError, match="config mismatch"):
        load_completed_keys(output, fingerprint="different")


def test_invalid_checkpoint_is_rejected(tmp_path: Path) -> None:
    output = tmp_path / "broken.jsonl"
    output.write_text('{"not": "complete"}\n', encoding="utf-8")

    with pytest.raises(BenchmarkRunError, match="invalid checkpoint"):
        load_completed_keys(output, fingerprint="fingerprint")


def test_incomplete_final_checkpoint_fragment_can_be_recovered(tmp_path: Path) -> None:
    output = tmp_path / "interrupted.jsonl"
    output.write_bytes(b'{"complete": 1}\n{"partial":')

    assert recover_jsonl_tail(output) == "truncated"
    assert output.read_bytes() == b'{"complete": 1}\n'
    assert recover_jsonl_tail(output) == "unchanged"


def test_complete_final_record_without_newline_is_preserved(tmp_path: Path) -> None:
    output = tmp_path / "missing-newline.jsonl"
    output.write_text('{"complete": 1}', encoding="utf-8")

    assert recover_jsonl_tail(output) == "completed"
    assert output.read_text(encoding="utf-8") == '{"complete": 1}\n'


def test_probe_model_service_reads_openai_model_list(monkeypatch: pytest.MonkeyPatch) -> None:
    request = httpx.Request("GET", "http://backend.test/v1/models")

    def fake_get(url: str, *, headers: dict[str, str] | None, timeout: float) -> httpx.Response:
        assert url == str(request.url)
        assert headers is None
        assert timeout == 3
        return httpx.Response(
            200,
            json={"data": [{"id": "lmsys/vicuna-13b-v1.5"}]},
            request=request,
        )

    monkeypatch.setattr("benchmarks.run_jbb.httpx.get", fake_get)

    assert probe_model_service("http://backend.test/v1/", timeout=3) == (
        "lmsys/vicuna-13b-v1.5",
    )


def test_probe_model_service_sends_optional_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = httpx.Request("GET", "https://backend.test/v1/models")

    def fake_get(
        url: str, *, headers: dict[str, str] | None, timeout: float
    ) -> httpx.Response:
        assert url == str(request.url)
        assert headers == {"Authorization": "Bearer test-key"}
        assert timeout == 5
        return httpx.Response(
            200,
            json={"data": [{"id": "deepseek-v4-flash"}]},
            request=request,
        )

    monkeypatch.setattr("benchmarks.run_jbb.httpx.get", fake_get)

    assert probe_model_service("https://backend.test/v1", timeout=5, api_key="test-key") == (
        "deepseek-v4-flash",
    )


def test_checkpoint_json_is_compact_utf8(tmp_path: Path) -> None:
    output = tmp_path / "unicode.jsonl"
    append_record(output, {"text": "中文", "value": 1})

    assert json.loads(output.read_text(encoding="utf-8")) == {"text": "中文", "value": 1}
