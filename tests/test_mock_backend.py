import pytest

from se_smoothllm.backends.mock import MockBackend
from se_smoothllm.judges.prefix import PrefixJudge
from se_smoothllm.models import Generation


def test_preset_sequence_records_prompts_and_drives_judge_votes() -> None:
    responses = ["I cannot help with that request."] * 6 + ["Here is the answer."] * 4
    backend = MockBackend(responses)
    judge = PrefixJudge()

    results = [
        judge.classify(
            f"perturbed prompt {index}",
            backend.generate(f"perturbed prompt {index}"),
        )
        for index in range(10)
    ]

    assert sum(not result.jailbroken for result in results) == 6
    assert sum(result.jailbroken for result in results) == 4
    assert backend.call_count == 10
    assert backend.successful_call_count == 10
    assert backend.received_prompts == [f"perturbed prompt {index}" for index in range(10)]
    assert backend.remaining_responses == 0


def test_exhaustion_raises_and_records_the_unexpected_call() -> None:
    backend = MockBackend(["first response"])

    backend.generate("first prompt")
    with pytest.raises(RuntimeError, match="exhausted after 1 configured responses"):
        backend.generate("unexpected prompt")

    assert backend.call_count == 2
    assert backend.successful_call_count == 1
    assert backend.received_prompts[-1] == "unexpected prompt"


def test_cycle_policy_must_be_enabled_explicitly() -> None:
    backend = MockBackend(["first", "second"], exhaustion="cycle")

    texts = [backend.generate("prompt").text for _ in range(3)]

    assert texts == ["first", "second", "first"]


def test_repeat_last_policy_reuses_only_the_final_response() -> None:
    backend = MockBackend(["first", "second"], exhaustion="repeat_last")

    texts = [backend.generate("prompt").text for _ in range(4)]

    assert texts == ["first", "second", "second", "second"]


def test_reset_rewinds_responses_and_clears_trace() -> None:
    configured = Generation(
        text="configured response",
        prompt_tokens=8,
        completion_tokens=2,
        latency_ms=3.5,
    )
    backend = MockBackend(configured)

    assert backend.generate("before reset") is configured
    backend.reset()

    assert backend.call_count == 0
    assert backend.remaining_responses == 1
    assert backend.generate("after reset") is configured


def test_invalid_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one item"):
        MockBackend([])
    with pytest.raises(ValueError, match="unsupported exhaustion policy"):
        MockBackend(["response"], exhaustion="unsupported")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="strings or Generation"):
        MockBackend([object()])  # type: ignore[list-item]
