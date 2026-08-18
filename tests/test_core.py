import random

from se_smoothllm.backends.mock import MockBackend
from se_smoothllm.core import DefenseComponents
from se_smoothllm.judges.base import Judge
from se_smoothllm.judges.prefix import PrefixJudge
from se_smoothllm.models import DefenseResult, Generation, JudgeResult
from se_smoothllm.perturbations import IdentityPerturbation


class RecordingJudge(Judge):
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def classify(self, prompt: str, generation: Generation) -> JudgeResult:
        self.prompts.append(prompt)
        return JudgeResult(jailbroken=False, reason="recorded")


def test_components_accept_replaceable_interfaces() -> None:
    components = DefenseComponents(
        backend=MockBackend("I cannot help with that request."),
        perturbation=IdentityPerturbation(),
        judge=PrefixJudge(),
    )

    assert components.backend.generate("test prompt").text.startswith("I cannot")
    assert components.perturbation.apply("test prompt", rng=random.Random(7)) == "test prompt"


def test_prefix_judge_returns_a_normalized_result() -> None:
    judge = PrefixJudge()

    result = judge.classify("test prompt", Generation(text="I cannot help.", latency_ms=1.0))

    assert result.jailbroken is False
    assert result.score == 0.0


def test_core_passes_the_actual_perturbed_prompt_to_judge() -> None:
    backend = MockBackend("I cannot help.")
    judge = RecordingJudge()
    guard = DefenseComponents(
        backend=backend,
        judge=judge,
        perturbation=IdentityPerturbation(),
        copies=1,
    )

    result = guard.defend("original prompt")

    assert judge.prompts == [result.trace[0].perturbed_prompt]
    assert judge.prompts == ["original prompt"]


def test_defense_result_carries_aggregate_metadata() -> None:
    result = DefenseResult(
        response="I cannot help.",
        jailbroken=False,
        copies_used=3,
        votes={"safe": 3, "jailbroken": 0},
        stopped_early=True,
        latency_ms=12.5,
        prompt_tokens=10,
        completion_tokens=4,
    )

    assert result.votes["safe"] == 3
    assert result.prompt_tokens == 10
