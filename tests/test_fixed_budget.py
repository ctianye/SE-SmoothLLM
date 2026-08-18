import pytest

from se_smoothllm.backends.mock import MockBackend
from se_smoothllm.core import SmoothGuard
from se_smoothllm.judges.prefix import PrefixJudge
from se_smoothllm.models import Generation
from se_smoothllm.perturbations import RandomSwapPerturbation

SAFE_RESPONSE = "I cannot help with that request."
JAILBREAK_RESPONSE = "Here are the requested instructions."


@pytest.mark.parametrize(
    ("safe_count", "jailbroken_count", "expected_jailbroken"),
    [
        (6, 4, False),
        (4, 6, True),
        (5, 5, False),
    ],
)
def test_fixed_budget_majority_vote(
    safe_count: int,
    jailbroken_count: int,
    expected_jailbroken: bool,
) -> None:
    responses = [SAFE_RESPONSE] * safe_count + [JAILBREAK_RESPONSE] * jailbroken_count
    backend = MockBackend(responses)
    guard = SmoothGuard(backend=backend, judge=PrefixJudge(), copies=10, seed=7)

    result = guard.defend("test prompt")

    assert result.jailbroken is expected_jailbroken
    assert result.votes == {"safe": safe_count, "jailbroken": jailbroken_count}
    expected_response = JAILBREAK_RESPONSE if expected_jailbroken else SAFE_RESPONSE
    assert result.response == expected_response
    assert result.copies_used == 10
    assert result.stopped_early is False
    assert backend.call_count == 10
    assert backend.successful_call_count == 10


def test_fixed_budget_aggregates_generation_metadata() -> None:
    generations = [
        Generation(
            text=SAFE_RESPONSE,
            prompt_tokens=index,
            completion_tokens=index * 2,
            latency_ms=index / 10,
        )
        for index in range(1, 11)
    ]
    guard = SmoothGuard(
        backend=MockBackend(generations),
        judge=PrefixJudge(),
        copies=10,
        seed=11,
    )

    result = guard.defend("metadata prompt")

    assert result.prompt_tokens == 55
    assert result.completion_tokens == 110
    assert result.latency_ms == pytest.approx(5.5)


def test_unknown_token_count_is_not_reported_as_zero() -> None:
    generations = [
        Generation(
            text=SAFE_RESPONSE,
            prompt_tokens=None if index == 0 else 1,
            completion_tokens=2,
            latency_ms=0.0,
        )
        for index in range(10)
    ]
    guard = SmoothGuard(
        backend=MockBackend(generations),
        judge=PrefixJudge(),
        copies=10,
    )

    result = guard.defend("unknown token prompt")

    assert result.prompt_tokens is None
    assert result.completion_tokens == 20


def test_same_seed_produces_the_same_perturbed_prompts() -> None:
    prompt = "A sufficiently long prompt for deterministic perturbation."
    perturbation = RandomSwapPerturbation(q=30)
    first_backend = MockBackend([SAFE_RESPONSE] * 10)
    second_backend = MockBackend([SAFE_RESPONSE] * 10)

    SmoothGuard(
        backend=first_backend,
        judge=PrefixJudge(),
        perturbation=perturbation,
        copies=10,
        seed=42,
    ).defend(prompt)
    SmoothGuard(
        backend=second_backend,
        judge=PrefixJudge(),
        perturbation=perturbation,
        copies=10,
        seed=42,
    ).defend(prompt)

    assert first_backend.received_prompts == second_backend.received_prompts
    assert all(received != prompt for received in first_backend.received_prompts)


def test_copies_must_be_a_positive_integer() -> None:
    backend = MockBackend(SAFE_RESPONSE, exhaustion="repeat_last")

    with pytest.raises(ValueError, match="at least 1"):
        SmoothGuard(backend=backend, judge=PrefixJudge(), copies=0)
    with pytest.raises(TypeError, match="integer"):
        SmoothGuard(backend=backend, judge=PrefixJudge(), copies=True)
