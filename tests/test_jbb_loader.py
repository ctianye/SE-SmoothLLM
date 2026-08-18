from collections.abc import Mapping
from typing import Any

import pytest
from benchmarks.jbb_loader import (
    JBBDataError,
    load_benign_samples,
    load_experiment_config,
    load_harmful_samples,
    load_jbb_samples,
)


def _harmful_records() -> list[dict[str, object]]:
    return [
        {
            "index": index,
            "goal": f"harmful goal {index}",
            "behavior": f"harmful behavior {index}",
            "category": "test category",
            "prompt": f"GCG prompt {index}",
            "response": "ignored artifact response",
            "jailbroken": True,
        }
        for index in range(100)
    ]


def _benign_records() -> list[dict[str, object]]:
    return [
        {
            "Index": index,
            "Goal": f"benign goal {index}",
            "Target": "ignored target",
            "Behavior": f"benign behavior {index}",
            "Category": "test category",
            "Source": "test source",
        }
        for index in range(100)
    ]


def test_load_harmful_samples_uses_only_required_artifact_fields() -> None:
    config = load_experiment_config()
    requested_urls: list[str] = []

    def fetch_json(url: str) -> object:
        requested_urls.append(url)
        return {"parameters": {"attack_success_rate": 1.0}, "jailbreaks": _harmful_records()}

    samples = load_harmful_samples(fetch_json=fetch_json)

    assert len(samples) == 100
    assert len({sample.index for sample in samples}) == 100
    assert all(sample.split == "harmful" and sample.prompt for sample in samples)
    assert not hasattr(samples[0], "response")
    assert not hasattr(samples[0], "jailbroken")
    assert requested_urls == [config["attack"]["artifact_url"]]


def test_load_benign_samples_uses_pinned_hugging_face_split() -> None:
    config = load_experiment_config()
    received: dict[str, object] = {}

    def dataset_loader(
        dataset_id: str,
        configuration: str,
        *,
        split: str,
        revision: str,
    ) -> list[Mapping[str, object]]:
        received.update(
            dataset_id=dataset_id,
            configuration=configuration,
            split=split,
            revision=revision,
        )
        return _benign_records()

    samples = load_benign_samples(dataset_loader=dataset_loader)

    assert len(samples) == 100
    assert len({sample.index for sample in samples}) == 100
    assert all(sample.split == "benign" and sample.prompt == sample.goal for sample in samples)
    assert received == {
        "dataset_id": config["dataset"]["id"],
        "configuration": config["dataset"]["configuration"],
        "split": config["dataset"]["splits"]["benign"]["name"],
        "revision": config["dataset"]["revision"],
    }


def test_load_all_samples_has_unique_split_index_identities() -> None:
    samples = load_jbb_samples(
        fetch_json=lambda _: {"jailbreaks": _harmful_records()},
        dataset_loader=lambda *args, **kwargs: _benign_records(),
    )

    assert len(samples) == 200
    assert len({(sample.split, sample.index) for sample in samples}) == 200


def test_load_selected_benign_split_does_not_fetch_harmful_artifact() -> None:
    def unexpected_fetch(_: str) -> object:
        raise AssertionError("benign-only loading must not fetch the harmful artifact")

    samples = load_jbb_samples(
        splits=("benign",),
        fetch_json=unexpected_fetch,
        dataset_loader=lambda *args, **kwargs: _benign_records(),
    )

    assert len(samples) == 100
    assert all(sample.split == "benign" for sample in samples)


def test_load_selected_harmful_split_does_not_load_benign_dataset() -> None:
    def unexpected_dataset_load(*args: object, **kwargs: object) -> list[Mapping[str, object]]:
        raise AssertionError("harmful-only loading must not load the benign dataset")

    samples = load_jbb_samples(
        splits=("harmful",),
        fetch_json=lambda _: {"jailbreaks": _harmful_records()},
        dataset_loader=unexpected_dataset_load,
    )

    assert len(samples) == 100
    assert all(sample.split == "harmful" for sample in samples)


@pytest.mark.parametrize(
    "field",
    ["goal", "behavior", "category", "prompt"],
)
def test_harmful_samples_reject_empty_required_text(field: str) -> None:
    records = _harmful_records()
    records[0][field] = ""

    with pytest.raises(JBBDataError, match="non-empty string"):
        load_harmful_samples(fetch_json=lambda _: {"jailbreaks": records})


def test_harmful_samples_reject_duplicate_indices() -> None:
    records = _harmful_records()
    records[1]["index"] = records[0]["index"]

    with pytest.raises(JBBDataError, match="indices must be unique"):
        load_harmful_samples(fetch_json=lambda _: {"jailbreaks": records})


def test_source_revisions_are_pinned_in_config() -> None:
    config: dict[str, Any] = load_experiment_config()

    assert config["attack"]["revision"] in config["attack"]["artifact_url"]
    assert config["dataset"]["revision"] == "886acc352a31533ffbcf4ef22c744658688086fc"
    assert config["model"]["revision"] == "c8327bf999adbd2efe2e75f6509fa01436100dc2"
