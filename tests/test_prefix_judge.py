import pytest

from se_smoothllm.backends.mock import MockBackend
from se_smoothllm.core import SmoothGuard
from se_smoothllm.judges.base import Judge
from se_smoothllm.judges.prefix import SMOOTHLLM_REFUSAL_PREFIXES, PrefixJudge
from se_smoothllm.models import Generation, JudgeResult


def _generation(text: str) -> Generation:
    return Generation(text=text, latency_ms=0.0)


@pytest.mark.parametrize("prefix", SMOOTHLLM_REFUSAL_PREFIXES)
def test_official_refusal_phrases_are_classified_safe(prefix: str) -> None:
    result = PrefixJudge().classify("test prompt", _generation(f"Preamble. {prefix} more text."))

    assert result.jailbroken is False
    assert result.label == "safe"
    assert result.reason.startswith("matched refusal phrase:")


@pytest.mark.parametrize(
    "response",
    [
        "Here are detailed instructions.",
        "Certainly, I will comply with the request.",
        "",
        "这是一个没有英文拒答短语的回答。",
    ],
)
def test_response_without_refusal_phrase_is_classified_jailbroken(response: str) -> None:
    result = PrefixJudge().classify("test prompt", _generation(response))

    assert result.jailbroken is True
    assert result.label == "jailbroken"
    assert result.score == 1.0


def test_classification_is_stable_for_the_same_response() -> None:
    judge = PrefixJudge()
    generation = _generation("I apologize, but I cannot help with that request.")

    results = [judge.classify("test prompt", generation) for _ in range(5)]

    assert all(result == results[0] for result in results)


def test_default_matching_is_case_sensitive_but_can_be_overridden() -> None:
    generation = _generation("i'm sorry, but I cannot comply")

    default_result = PrefixJudge(refusal_prefixes=("I'm sorry",)).classify("test prompt", generation)
    insensitive_result = PrefixJudge(
        refusal_prefixes=("I'm sorry",),
        case_sensitive=False,
    ).classify("test prompt", generation)

    assert default_result.label == "jailbroken"
    assert insensitive_result.label == "safe"


class AlwaysSafeJudge(Judge):
    def classify(self, prompt: str, generation: Generation) -> JudgeResult:
        return JudgeResult(
            jailbroken=False,
            score=None,
            reason=f"custom judge accepted prompt={prompt!r}: {generation.text}",
        )


def test_smooth_guard_accepts_a_custom_judge() -> None:
    backend = MockBackend("custom response")
    custom_judge = AlwaysSafeJudge()
    guard = SmoothGuard(backend=backend, judge=custom_judge)

    generation = guard.backend.generate("prompt")
    result = guard.judge.classify("prompt", generation)

    assert guard.judge is custom_judge
    assert result.label == "safe"
