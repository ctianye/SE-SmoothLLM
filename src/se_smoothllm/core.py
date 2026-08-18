"""共享执行器上的固定预算 SmoothLLM 与 SE-SmoothLLM。"""

import random
from collections.abc import Iterable
from dataclasses import dataclass, field

from se_smoothllm.backends.base import Backend
from se_smoothllm.judges.base import Judge
from se_smoothllm.models import CopyTrace, DefenseResult, VoteLabel
from se_smoothllm.perturbations import IdentityPerturbation, Perturbation
from se_smoothllm.stopping import locked_vote


@dataclass(frozen=True, slots=True)
class SmoothGuard:
    """在同一配置和执行器上运行固定预算或精确早停防御。

    两种模式共享扰动、后端、Judge、投票与回答选择逻辑。只有越狱票数
    严格超过安全票数时才判定整体越狱，因此平票归为安全。
    """

    backend: Backend
    judge: Judge
    perturbation: Perturbation = field(default_factory=IdentityPerturbation)
    copies: int = 10
    seed: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.copies, bool) or not isinstance(self.copies, int):
            raise TypeError("copies must be an integer")
        if self.copies < 1:
            raise ValueError("copies must be at least 1")

    def defend(self, prompt: str) -> DefenseResult:
        """运行始终使用 ``copies`` 个副本的固定预算基线。"""

        return self._execute(prompt, check_stopping=False)

    def defend_early(self, prompt: str) -> DefenseResult:
        """运行结论锁定后立即停止的 SE-SmoothLLM。"""

        return self._execute(prompt, check_stopping=True)

    def _execute(self, prompt: str, *, check_stopping: bool) -> DefenseResult:
        """执行共享流程；两种模式只在是否检查停止条件上不同。"""

        perturbation_rng = random.Random(self.seed)
        selection_seed = None if self.seed is None else f"response-selection:{self.seed}"
        selection_rng = random.Random(selection_seed)
        trace: list[CopyTrace] = []
        votes: dict[VoteLabel, int] = {"safe": 0, "jailbroken": 0}

        for copy_index in range(1, self.copies + 1):
            perturbed_prompt = self.perturbation.apply(prompt, rng=perturbation_rng)
            generation = self.backend.generate(perturbed_prompt)
            judgment = self.judge.classify(perturbed_prompt, generation)
            copy_trace = CopyTrace(
                copy_index=copy_index,
                perturbed_prompt=perturbed_prompt,
                response=generation.text,
                model=generation.model,
                judge_result=judgment,
                latency_ms=generation.latency_ms,
                prompt_tokens=generation.prompt_tokens,
                completion_tokens=generation.completion_tokens,
            )
            trace.append(copy_trace)
            votes[judgment.label] += 1

            if check_stopping:
                decision = locked_vote(
                    safe=votes["safe"],
                    jailbroken=votes["jailbroken"],
                    remaining=self.copies - copy_index,
                )
                if decision is not None:
                    break

        verdict = _majority_vote(votes)
        selected_response = _select_response(trace, verdict, rng=selection_rng)
        copies_used = len(trace)

        return DefenseResult(
            response=selected_response,
            jailbroken=verdict == "jailbroken",
            copies_used=copies_used,
            votes=votes,
            stopped_early=copies_used < self.copies,
            latency_ms=sum(item.latency_ms for item in trace),
            prompt_tokens=_sum_optional_tokens(item.prompt_tokens for item in trace),
            completion_tokens=_sum_optional_tokens(item.completion_tokens for item in trace),
            trace=tuple(trace),
        )


def _majority_vote(votes: dict[VoteLabel, int]) -> VoteLabel:
    """应用固定预算和早停模式共享的平票归安全规则。"""

    return "jailbroken" if votes["jailbroken"] > votes["safe"] else "safe"


def _select_response(
    trace: list[CopyTrace],
    verdict: VoteLabel,
    *,
    rng: random.Random,
) -> str:
    """从已观察到的获胜类别回答中随机选择一条。"""

    candidates = [item.response for item in trace if item.judge_result.label == verdict]
    return rng.choice(candidates)


def _sum_optional_tokens(values: Iterable[int | None]) -> int | None:
    """仅在每次调用都报告 token 数时返回总和。"""

    collected = list(values)
    if any(value is None for value in collected):
        return None
    return sum(value for value in collected if value is not None)


DefenseComponents = SmoothGuard
