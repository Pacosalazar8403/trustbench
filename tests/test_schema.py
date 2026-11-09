"""Tests for the dataset schema: what a valid example looks like, and what gets rejected."""

from typing import Any

import pytest
from pydantic import ValidationError

from trustbench.schema import Category, Difficulty, Evidence, Example


def make_example(**overrides: Any) -> dict[str, Any]:
    """Return raw data for a valid answerable example, with optional overrides."""
    data: dict[str, Any] = {
        "id": "withdraw-001",
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


def test_valid_example_parses_into_typed_fields() -> None:
    example = Example.model_validate(make_example())

    expected = Evidence(
        doc_id="registration-policy",
        quote="Courses dropped before the census date receive a full refund.",
    )
    assert example.id == "withdraw-001"
    assert example.category is Category.LOOKUP
    assert example.difficulty is Difficulty.EASY
    assert example.evidence == (expected,)
    assert example.answerable


def test_unanswerable_example_has_no_evidence() -> None:
    data = make_example(
        category="unanswerable",
        reference_answer="The policies do not cover this. Contact the Registrar.",
        evidence=[],
    )
    example = Example.model_validate(data)

    assert not example.answerable
    assert example.evidence == ()


def test_answerable_example_without_evidence_is_rejected() -> None:
    with pytest.raises(ValidationError, match="need at least one piece of evidence"):
        Example.model_validate(make_example(evidence=[]))


def test_unanswerable_example_with_evidence_is_rejected() -> None:
    with pytest.raises(ValidationError, match="must not have evidence"):
        Example.model_validate(make_example(category="unanswerable"))


def test_multi_passage_example_with_one_piece_of_evidence_is_rejected() -> None:
    with pytest.raises(ValidationError, match="at least two pieces of evidence"):
        Example.model_validate(make_example(category="multi_passage"))


def test_multi_passage_example_with_two_pieces_of_evidence_is_accepted() -> None:
    evidence = [
        {
            "doc_id": "registration-policy",
            "quote": "Courses dropped before the census date receive a full refund.",
        },
        {
            "doc_id": "tuition-policy",
            "quote": "Refunds are issued to the original form of payment.",
        },
    ]
    example = Example.model_validate(make_example(category="multi_passage", evidence=evidence))

    assert len(example.evidence) == 2


def test_unknown_field_is_rejected() -> None:
    data = make_example()
    data["refrence_answer"] = data.pop("reference_answer")

    with pytest.raises(ValidationError, match="refrence_answer"):
        Example.model_validate(data)


@pytest.mark.parametrize("field", ["question", "reference_answer"])
def test_blank_text_is_rejected(field: str) -> None:
    with pytest.raises(ValidationError, match=rf"\n{field}\n"):
        Example.model_validate(make_example(**{field: "   "}))


def test_text_is_stripped() -> None:
    example = Example.model_validate(make_example(question="  Can I drop a class?  "))

    assert example.question == "Can I drop a class?"


@pytest.mark.parametrize(
    "bad_id",
    ["", "Withdraw-001", "withdraw_001", "withdraw 001", "-withdraw", "withdraw--001"],
)
def test_malformed_ids_are_rejected(bad_id: str) -> None:
    with pytest.raises(ValidationError, match=r"\nid\n"):
        Example.model_validate(make_example(id=bad_id))


def test_unknown_category_is_rejected() -> None:
    with pytest.raises(ValidationError, match=r"\ncategory\n"):
        Example.model_validate(make_example(category="trivia"))


def test_examples_are_immutable() -> None:
    example = Example.model_validate(make_example())

    with pytest.raises(ValidationError, match="frozen"):
        example.question = "Something else?"  # type: ignore[misc]


def test_group_is_optional() -> None:
    assert Example.model_validate(make_example()).group is None
    assert Example.model_validate(make_example(group="census-refund")).group == "census-refund"


def test_malformed_group_is_rejected() -> None:
    with pytest.raises(ValidationError, match=r"\ngroup\n"):
        Example.model_validate(make_example(group="Census Refund"))
