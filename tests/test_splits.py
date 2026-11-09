"""Tests for the dev/test split and the evidence conflict check."""

from pathlib import Path
from typing import Any

import pytest

from trustbench.schema import Example
from trustbench.splits import (
    Split,
    assign_split,
    find_evidence_conflicts,
    group_key,
    main,
    split_examples,
)


def make_example(
    example_id: str, *, group: str | None = None, quote: str | None = "Quote one."
) -> Example:
    """Build a lookup example quoting `quote`, or an unanswerable one if quote is None."""
    data: dict[str, Any] = {
        "id": example_id,
        "question": "Q?",
        "reference_answer": "A.",
        "category": "lookup" if quote is not None else "unanswerable",
        "difficulty": "easy",
        "evidence": [] if quote is None else [{"doc_id": "policy", "quote": quote}],
        "group": group,
    }
    return Example.model_validate(data)


def test_assign_split_is_deterministic() -> None:
    assert assign_split("drop-001") == assign_split("drop-001")


@pytest.mark.parametrize(
    ("group", "expected"),
    [
        ("drop-001", Split.TEST),
        ("drop-002", Split.DEV),
        ("parking-001", Split.TEST),
        ("census-refund", Split.DEV),
    ],
)
def test_known_groups_keep_their_split(group: str, expected: Split) -> None:
    # Pinned on purpose: if these change, the held-out test split silently changed too.
    assert assign_split(group) == expected


def test_about_forty_percent_of_groups_go_to_test() -> None:
    groups = [f"group-{number}" for number in range(10_000)]
    share = sum(assign_split(group) is Split.TEST for group in groups) / len(groups)

    assert 0.38 < share < 0.42


def test_group_key_is_the_group_or_the_id() -> None:
    assert group_key(make_example("drop-001")) == "drop-001"
    assert group_key(make_example("drop-001", group="census-refund")) == "census-refund"


def test_a_group_never_splits_up() -> None:
    examples = [make_example(f"ex-{number:03d}", group="shared") for number in range(30)]
    splits = split_examples(examples)

    assert sorted(len(members) for members in splits.values()) == [0, 30]


def test_adding_examples_never_moves_existing_ones() -> None:
    originals = [make_example(f"ex-{number:03d}") for number in range(20)]
    extended = originals + [make_example(f"new-{number:03d}") for number in range(50)]

    def split_of(examples: list[Example]) -> dict[str, Split]:
        return {e.id: split for split, members in split_examples(examples).items() for e in members}

    before, after = split_of(originals), split_of(extended)
    assert {example_id: after[example_id] for example_id in before} == before


def test_split_examples_partitions_and_keeps_order() -> None:
    examples = [make_example(f"ex-{number:03d}") for number in range(40)]
    splits = split_examples(examples)

    ids = [example.id for members in splits.values() for example in members]
    assert sorted(ids) == [example.id for example in examples]
    for members in splits.values():
        member_ids = [example.id for example in members]
        assert member_ids == sorted(member_ids)


def test_shared_quote_in_different_groups_is_a_conflict() -> None:
    examples = [make_example("drop-002"), make_example("drop-005")]

    (conflict,) = find_evidence_conflicts(examples)

    assert conflict.example_ids == ("drop-002", "drop-005")
    assert conflict.groups == ("drop-002", "drop-005")


def test_shared_quote_in_the_same_group_is_fine() -> None:
    examples = [
        make_example("drop-002", group="census-refund"),
        make_example("drop-005", group="census-refund"),
    ]

    assert find_evidence_conflicts(examples) == []


def test_quotes_that_differ_only_in_formatting_still_conflict() -> None:
    examples = [
        make_example("advising-001", quote="It\u2019s  open on weekdays."),
        make_example("advising-002", quote="It's open on weekdays."),
    ]

    assert len(find_evidence_conflicts(examples)) == 1


def test_unanswerable_examples_never_conflict() -> None:
    examples = [make_example("parking-001", quote=None), make_example("audit-001", quote=None)]

    assert find_evidence_conflicts(examples) == []


def write_dataset(path: Path, *examples: Example) -> Path:
    path.write_text("".join(example.model_dump_json() + "\n" for example in examples))
    return path


def test_main_prints_the_version_and_split(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = write_dataset(tmp_path / "data.jsonl", make_example("drop-001"))

    assert main([str(dataset)]) == 0

    output = capsys.readouterr().out
    assert "version" in output
    assert "examples  1 (dev 0, test 1)" in output


def test_main_exits_1_on_a_conflict(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dataset = write_dataset(
        tmp_path / "data.jsonl", make_example("drop-002"), make_example("drop-005")
    )

    assert main([str(dataset)]) == 1
    assert "1 conflict(s)" in capsys.readouterr().out


def test_main_exits_2_when_the_dataset_is_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(tmp_path / "missing.jsonl")]) == 2
    assert "No such file or directory" in capsys.readouterr().err
