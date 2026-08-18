import pytest

from se_smoothllm.backends.mock import MockBackend
from se_smoothllm.core import SmoothGuard
from se_smoothllm.judges.prefix import PrefixJudge
from se_smoothllm.models import Generation, VoteLabel
from se_smoothllm.perturbations import RandomSwapPerturbation


def _generation(label: VoteLabel, index: int) -> Generation:
    response = (
        f"I cannot help with request {index}."
        if label == "safe"
        else f"Here are the requested instructions {index}."
    )
    return Generation(
        text=response,
        prompt_tokens=index,
        completion_tokens=index * 2,
        latency_ms=index / 10,
    )


@pytest.mark.parametrize(
    ("labels", "expected_copies", "expected_jailbroken"),
    [
        (("safe",) * 5 + ("jailbroken",) * 5, 5, False),
        (("jailbroken",) * 6 + ("safe",) * 4, 6, True),
        (("jailbroken", "safe") * 5, 10, False),
    ],
)
def test_fixed_and_early_modes_share_one_execution_semantics(
    labels: tuple[VoteLabel, ...],
    expected_copies: int,
    expected_jailbroken: bool,
) -> None:
    generations = [_generation(label, index) for index, label in enumerate(labels, start=1)]
    backend = MockBackend(generations)
    guard = SmoothGuard(
        backend=backend,
        judge=PrefixJudge(),
        perturbation=RandomSwapPerturbation(q=20),
        copies=10,
        seed=42,
    )

    fixed = guard.defend("A long prompt used to compare deterministic perturbation order.")
    backend.reset()
    early = guard.defend_early("A long prompt used to compare deterministic perturbation order.")

    assert fixed.copies_used == 10
    assert fixed.stopped_early is False
    assert early.copies_used == expected_copies
    assert early.stopped_early is (expected_copies < 10)
    assert backend.call_count == expected_copies
    assert early.jailbroken is fixed.jailbroken is expected_jailbroken
    assert early.trace == fixed.trace[:expected_copies]
    assert early.response == fixed.response

    executed_generations = generations[:expected_copies]
    assert early.latency_ms == pytest.approx(
        sum(generation.latency_ms for generation in executed_generations)
    )
    assert early.prompt_tokens == sum(
        generation.prompt_tokens or 0 for generation in executed_generations
    )
    assert early.completion_tokens == sum(
        generation.completion_tokens or 0 for generation in executed_generations
    )
    assert sum(early.votes.values()) == expected_copies


def test_trace_records_every_copy_field() -> None:
    generation = Generation(
        text="I cannot help with that request.",
        model="test-model",
        prompt_tokens=17,
        completion_tokens=4,
        latency_ms=12.5,
    )
    guard = SmoothGuard(
        backend=MockBackend(generation, exhaustion="repeat_last"),
        judge=PrefixJudge(),
        copies=1,
        seed=7,
    )

    result = guard.defend_early("original prompt")

    assert len(result.trace) == result.copies_used == 1
    item = result.trace[0]
    assert item.copy_index == 1
    assert item.perturbed_prompt == "original prompt"
    assert item.response == generation.text
    assert item.model == "test-model"
    assert item.judge_result.label == "safe"
    assert item.latency_ms == 12.5
    assert item.prompt_tokens == 17
    assert item.completion_tokens == 4
