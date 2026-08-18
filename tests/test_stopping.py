import pytest

from se_smoothllm.stopping import locked_vote


@pytest.mark.parametrize(
    ("safe", "jailbroken", "remaining", "expected"),
    [
        (3, 6, 2, "jailbroken"),
        (3, 5, 2, None),
        (5, 3, 2, "safe"),
        (4, 3, 2, None),
        (0, 6, 4, "jailbroken"),
        (0, 5, 5, None),
        (5, 0, 5, "safe"),
        (4, 0, 6, None),
    ],
)
def test_locked_vote_uses_exact_early_stopping_conditions(
    safe: int,
    jailbroken: int,
    remaining: int,
    expected: str | None,
) -> None:
    assert locked_vote(safe, jailbroken, remaining) == expected


@pytest.mark.parametrize(
    ("safe", "jailbroken", "expected"),
    [
        (5, 5, "safe"),
        (4, 6, "jailbroken"),
        (6, 4, "safe"),
    ],
)
def test_no_remaining_votes_always_lock_the_final_result(
    safe: int,
    jailbroken: int,
    expected: str,
) -> None:
    assert locked_vote(safe, jailbroken, remaining=0) == expected


def test_locked_vote_matches_all_possible_remaining_vote_assignments() -> None:
    for safe in range(5):
        for jailbroken in range(5):
            for remaining in range(5):
                possible_results = {
                    "jailbroken"
                    if (jailbroken + future_jailbroken) * 2
                    > safe + jailbroken + remaining
                    else "safe"
                    for future_jailbroken in range(remaining + 1)
                }
                expected = possible_results.pop() if len(possible_results) == 1 else None

                assert locked_vote(safe, jailbroken, remaining) == expected


@pytest.mark.parametrize("name", ["safe", "jailbroken", "remaining"])
def test_vote_counts_must_be_non_negative_integers(name: str) -> None:
    values = {"safe": 1, "jailbroken": 1, "remaining": 1}

    values[name] = -1
    with pytest.raises(ValueError, match="non-negative"):
        locked_vote(**values)

    values[name] = True
    with pytest.raises(TypeError, match="integer"):
        locked_vote(**values)
