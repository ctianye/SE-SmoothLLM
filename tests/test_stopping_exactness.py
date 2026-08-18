from itertools import product

from se_smoothllm.backends.mock import MockBackend
from se_smoothllm.core import SmoothGuard
from se_smoothllm.judges.prefix import PrefixJudge
from se_smoothllm.models import VoteLabel
from se_smoothllm.perturbations import RandomSwapPerturbation

N = 10
PROMPT = "A reproducible prompt for exhaustive end-to-end stopping verification."


def _response(label: VoteLabel, index: int) -> str:
    if label == "safe":
        return f"I cannot help with request {index}."
    return f"Here are the requested instructions {index}."


def test_all_n10_vote_sequences_preserve_the_fixed_budget_verdict() -> None:
    sequence_count = 0
    vote_mismatches = 0
    votes_used: list[int] = []

    for sequence in product(("safe", "jailbroken"), repeat=N):
        responses = [_response(label, index) for index, label in enumerate(sequence, start=1)]
        backend = MockBackend(responses)
        guard = SmoothGuard(
            backend=backend,
            judge=PrefixJudge(),
            perturbation=RandomSwapPerturbation(q=10),
            copies=N,
            seed=42,
        )

        fixed = guard.defend(PROMPT)
        backend.reset()
        early = guard.defend_early(PROMPT)

        sequence_count += 1
        vote_mismatches += early.jailbroken != fixed.jailbroken
        votes_used.append(early.copies_used)

        assert fixed.copies_used == N
        assert early.copies_used <= N
        assert backend.call_count == early.copies_used
        assert early.trace == fixed.trace[: early.copies_used]

    assert sequence_count == 2**N == 1024
    assert vote_mismatches == 0
    assert max(votes_used) == N

    # 固定这些统计量，使 README 中的穷举结果不会与实现悄然偏离。
    assert sum(used < N for used in votes_used) == 772
    assert sum(votes_used) == 8492
