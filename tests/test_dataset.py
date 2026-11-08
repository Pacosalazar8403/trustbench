"""Tests for the JSONL dataset loader: valid files load, and every problem is reported."""

import json
from pathlib import Path
from typing import Any

import pytest

from trustbench.dataset import DatasetError, DatasetIssue, load_examples
from trustbench.schema import Example


def example_data(example_id: str, **overrides: Any) -> dict[str, Any]:
    """Return raw data for a valid answerable example with the given id."""
    data: dict[str, Any] = {
        "id": example_id,
        "question": "Can I drop a class in week 3 and get a refund?",
        "reference_answer": "Yes. Classes dropped before the census date get a full refund.",
        "category": "lookup",
        "difficulty": "easy",
        "evidence": [
            {
                "doc_id": "registration-policy",
                "quote": "Courses dropped before the census date receive a full refund.",
            }
        ],
    }
    data.update(overrides)
    return data


def as_line(example_id: str, **overrides: Any) -> str:
    """Return one JSONL line holding a valid example, with optional overrides."""
    return json.dumps(example_data(example_id, **overrides))


def write_lines(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def issues_of(path: Path) -> tuple[DatasetIssue, ...]:
    """Load a file that should fail, and return the issues it reported."""
    with pytest.raises(DatasetError) as exc_info:
        load_examples(path)
    return exc_info.value.issues


def test_valid_file_loads_examples_in_order(tmp_path: Path) -> None:
    lines = [as_line("drop-001"), as_line("drop-002")]
    examples = load_examples(write_lines(tmp_path / "data.jsonl", lines))

    assert isinstance(examples, tuple)
    assert all(isinstance(example, Example) for example in examples)
    assert [example.id for example in examples] == ["drop-001", "drop-002"]


def test_blank_lines_are_skipped(tmp_path: Path) -> None:
    lines = ["", as_line("drop-001"), "   ", as_line("drop-002"), ""]
    examples = load_examples(write_lines(tmp_path / "data.jsonl", lines))

    assert len(examples) == 2


def test_windows_line_endings_are_accepted(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    content = as_line("drop-001") + "\r\n" + as_line("drop-002") + "\r\n"
    path.write_bytes(content.encode("utf-8"))

    assert len(load_examples(path)) == 2


def test_invalid_json_is_reported_with_its_line_number(tmp_path: Path) -> None:
    lines = [as_line("drop-001"), "{not json"]
    (issue,) = issues_of(write_lines(tmp_path / "data.jsonl", lines))

    assert issue.line == 2
    assert issue.message.startswith("invalid JSON")


def test_line_that_is_not_an_object_is_reported(tmp_path: Path) -> None:
    lines = ['["not", "an", "object"]']
    (issue,) = issues_of(write_lines(tmp_path / "data.jsonl", lines))

    assert issue == DatasetIssue(line=1, message="expected a JSON object, got list")


def test_missing_field_is_reported_with_the_field_name(tmp_path: Path) -> None:
    data = example_data("drop-001")
    del data["question"]
    (issue,) = issues_of(write_lines(tmp_path / "data.jsonl", [json.dumps(data)]))

    assert issue == DatasetIssue(line=1, message="question: Field required")


def test_rule_across_fields_is_reported(tmp_path: Path) -> None:
    lines = [as_line("drop-001", category="unanswerable")]
    (issue,) = issues_of(write_lines(tmp_path / "data.jsonl", lines))

    expected = "example: Value error, unanswerable examples must not have evidence"
    assert issue == DatasetIssue(line=1, message=expected)


def test_duplicate_ids_are_reported_with_both_line_numbers(tmp_path: Path) -> None:
    lines = [as_line("drop-001"), as_line("drop-002"), as_line("drop-001")]
    (issue,) = issues_of(write_lines(tmp_path / "data.jsonl", lines))

    expected = "duplicate id 'drop-001' (first seen on line 1)"
    assert issue == DatasetIssue(line=3, message=expected)


def test_every_problem_in_the_file_is_reported_together(tmp_path: Path) -> None:
    lines = ["{not json", as_line("drop-001"), as_line("Bad_ID"), as_line("drop-001")]
    issues = issues_of(write_lines(tmp_path / "data.jsonl", lines))

    assert [issue.line for issue in issues] == [1, 3, 4]


def test_error_message_lists_every_issue(tmp_path: Path) -> None:
    path = write_lines(tmp_path / "data.jsonl", ["{not json", as_line("Bad_ID")])

    with pytest.raises(DatasetError) as exc_info:
        load_examples(path)

    message = str(exc_info.value)
    assert message.startswith(f"{path}: 2 problems")
    assert "line 1: invalid JSON" in message
    assert "line 2: id: String should match pattern" in message


def test_empty_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text("", encoding="utf-8")
    (issue,) = issues_of(path)

    assert issue == DatasetIssue(line=None, message="dataset has no examples")


def test_file_that_is_not_utf8_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_bytes(b"\xff\xfe not utf-8")
    (issue,) = issues_of(path)

    assert issue.line is None
    assert issue.message.startswith("file is not valid UTF-8")


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_examples(tmp_path / "missing.jsonl")
