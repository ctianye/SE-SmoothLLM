import random

import pytest

from se_smoothllm.perturbations import (
    RandomInsertPerturbation,
    RandomPatchPerturbation,
    RandomSwapPerturbation,
)

PERTURBATION_TYPES = (
    RandomSwapPerturbation,
    RandomPatchPerturbation,
    RandomInsertPerturbation,
)


@pytest.mark.parametrize("perturbation_type", PERTURBATION_TYPES)
def test_same_seed_produces_same_result(perturbation_type: type) -> None:
    perturbation = perturbation_type(q=30)
    text = "Reproducible prompt for perturbation testing."

    first = perturbation.apply(text, rng=random.Random(2026))
    second = perturbation.apply(text, rng=random.Random(2026))

    assert first == second


@pytest.mark.parametrize("perturbation_type", PERTURBATION_TYPES)
def test_original_string_is_not_modified(perturbation_type: type) -> None:
    perturbation = perturbation_type(q=50)
    original = "original text"
    snapshot = original[:]

    perturbation.apply(original, rng=random.Random(7))

    assert original == snapshot


@pytest.mark.parametrize("perturbation_type", PERTURBATION_TYPES)
def test_q_zero_returns_text_without_consuming_rng(perturbation_type: type) -> None:
    perturbation = perturbation_type(q=0)
    rng = random.Random(11)
    state_before = rng.getstate()

    result = perturbation.apply("unchanged", rng=rng)

    assert result == "unchanged"
    assert rng.getstate() == state_before


@pytest.mark.parametrize("perturbation_type", PERTURBATION_TYPES)
@pytest.mark.parametrize("text", ["", "a", "中文测试", "🙂", "安全🙂test🚀"])
def test_edge_case_text_does_not_raise(perturbation_type: type, text: str) -> None:
    perturbation = perturbation_type(q=100)

    result = perturbation.apply(text, rng=random.Random(42))

    assert isinstance(result, str)


@pytest.mark.parametrize(
    ("perturbation_type", "measure_changes"),
    [
        (
            RandomSwapPerturbation,
            lambda source, result: sum(
                left != right for left, right in zip(source, result, strict=True)
            ),
        ),
        (
            RandomPatchPerturbation,
            lambda source, result: sum(
                left != right for left, right in zip(source, result, strict=True)
            ),
        ),
        (RandomInsertPerturbation, lambda source, result: len(result) - len(source)),
    ],
)
def test_modified_character_count_matches_q_percent(
    perturbation_type: type,
    measure_changes,
) -> None:
    text = "中" * 200
    q = 25
    expected_changes = int(len(text) * q / 100)
    perturbation = perturbation_type(q=q)

    result = perturbation.apply(text, rng=random.Random(99))

    assert measure_changes(text, result) == expected_changes


@pytest.mark.parametrize("q", [-1, 101, float("nan"), float("inf")])
@pytest.mark.parametrize("perturbation_type", PERTURBATION_TYPES)
def test_invalid_q_is_rejected(perturbation_type: type, q: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 100"):
        perturbation_type(q=q)


@pytest.mark.parametrize("perturbation_type", PERTURBATION_TYPES)
def test_non_numeric_q_and_empty_alphabet_are_rejected(perturbation_type: type) -> None:
    with pytest.raises(TypeError, match="numeric percentage"):
        perturbation_type(q="10")
    with pytest.raises(ValueError, match="alphabet must not be empty"):
        perturbation_type(q=10, alphabet="")


@pytest.mark.parametrize("perturbation_type", PERTURBATION_TYPES)
def test_global_random_module_is_rejected(perturbation_type: type) -> None:
    perturbation = perturbation_type(q=10)

    with pytest.raises(TypeError, match="instance of random.Random"):
        perturbation.apply("long enough text", rng=random)  # type: ignore[arg-type]
