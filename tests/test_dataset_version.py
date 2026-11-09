"""Tests for the dataset version: a content hash that ignores formatting but not content."""

from pathlib import Path
from typing import Any

import pytest

from trustbench.dataset import dataset_version, load_examples
from trustbench.schema import Example


def make_example(example_id: str, **overrides: Any) -> Example:
    data: dict[str, Any] = {
        "id": example_id,
        "question": "Q one?",
        "reference_answer": "A one.",
        "category": "lookup",
        "difficulty": "easy",
        "evidence": [{"doc_id": "policy", "quote": "Quote one."}],
    }
    data.update(overrides)
    return Example.model_validate(data)


def test_version_ignores_example_order() -> None:
    first, second = make_example("drop-001"), make_example("drop-002")

    assert dataset_version([first, second]) == dataset_version([second, first])


def test_version_ignores_file_formatting(tmp_path: Path) -> None:
    compact = (
        '{"id": "drop-001", "question": "Q one?", "reference_answer": "A one.", '
        '"category": "lookup", "difficulty": "easy", '
        '"evidence": [{"doc_id": "policy", "quote": "Quote one."}]}\n'
    )
    reordered = (
        "\n"
        '{"difficulty":"easy","category":"lookup","reference_answer":"A one.",'
        '"evidence":[{"quote":"Quote one.","doc_id":"policy"}],'
        '"question":"  Q one?  ","id":"drop-001"}\n'
        "\n"
    )
    (tmp_path / "a.jsonl").write_text(compact, encoding="utf-8")
    (tmp_path / "b.jsonl").write_text(reordered, encoding="utf-8")

    version_a = dataset_version(load_examples(tmp_path / "a.jsonl"))
    version_b = dataset_version(load_examples(tmp_path / "b.jsonl"))

    assert version_a == version_b


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("question", "Q changed?"),
        ("reference_answer", "A changed."),
        ("category", "paraphrase"),
        ("difficulty", "hard"),
        ("evidence", [{"doc_id": "policy", "quote": "Quote two."}]),
        ("group", "drop-group"),
    ],
)
def test_version_changes_when_any_field_changes(field: str, value: Any) -> None:
    original = make_example("drop-001")
    changed = make_example("drop-001", **{field: value})

    assert dataset_version([original]) != dataset_version([changed])


def test_version_is_16_hex_characters() -> None:
    version = dataset_version([make_example("drop-001")])

    assert len(version) == 16
    assert all(character in "0123456789abcdef" for character in version)


def test_version_of_a_fixed_dataset_never_changes() -> None:
    # Pinned on purpose: if this fails, the serialization changed (for example, after a
    # Pydantic upgrade), and every dataset version and cached result would shift with it.
    examples = [
        make_example("drop-001"),
        make_example(
            "parking-001",
            question="Q two?",
            reference_answer="A two.",
            category="unanswerable",
            difficulty="medium",
            evidence=[],
        ),
    ]

    assert dataset_version(examples) == "e7e503444fdf515b"
