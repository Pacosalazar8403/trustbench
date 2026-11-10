"""Tests for the deterministic scorers, including a check against the real dataset."""

from pathlib import Path

import pytest

from trustbench.dataset import load_examples
from trustbench.schema import Category, Difficulty, Evidence, Example
from trustbench.scoring import abstains, coverage, extract_facts, fact_score, main, score_answer

DATASET = Path(__file__).resolve().parent.parent / "data" / "uccs.jsonl"


def make_example(category: Category, answer: str) -> Example:
    evidence = () if category is Category.UNANSWERABLE else (Evidence(doc_id="doc", quote="q"),)
    return Example(
        id="example-001",
        question="A question?",
        reference_answer=answer,
        category=category,
        difficulty=Difficulty.EASY,
        evidence=evidence,
    )


@pytest.mark.parametrize(
    "answer",
    [
        "The documents I have don't say whether admission can be deferred.",
        "The policies I have don\u2019t cover parking permit refunds.",
        "The advising information I have doesn't explain this.",
        "The documents I have explain how to enroll, but not how to change the card.",
        "I don't know.",
        "I couldn't find that in the UCCS pages.",
    ],
)
def test_abstentions_are_recognized(answer: str) -> None:
    assert abstains(answer)


@pytest.mark.parametrize(
    "answer",
    [
        "Yes. You can defer your admission for up to one year.",
        "No. Your minor doesn't appear on your diploma.",
        "Not attending classes does not cancel your registration.",
    ],
)
def test_real_answers_are_not_abstentions(answer: str) -> None:
    assert not abstains(answer)


def test_equal_facts_are_normalized_to_the_same_form() -> None:
    first = extract_facts("Call (719) 255-3391 or Bursar@UCCS.edu; the fee is $50.00 and 67%.")
    second = extract_facts("Phone 719-255-3391, email bursar@uccs.edu. Pay $50, 67 percent.")
    assert first == second == {"phone:7192553391", "email:bursar@uccs.edu", "usd:50", "pct:67"}


def test_numbers_keep_their_value_not_their_spelling() -> None:
    assert extract_facts("code 004509, a 2.0 GPA") == extract_facts("code 4509, a 2 GPA")


def test_phone_digits_are_not_counted_again_as_numbers() -> None:
    assert extract_facts("Call 719-255-4444.") == {"phone:7192554444"}


def test_all_reference_facts_must_be_present() -> None:
    score = fact_score("Call 719-255-4444; visits are $20.", "Visits cost $20.")
    assert score.passed is False
    assert score.reason == "missing phone:7192554444"


def test_extra_facts_in_the_answer_are_not_penalized() -> None:
    reference = "Support is available 24/7 through the Student Support app."
    answer = "Yes, 24/7 through the Student Support app. You can also call or text 988."
    assert fact_score(reference, answer).passed is True


def test_a_reference_without_facts_is_left_for_a_judge() -> None:
    score = fact_score("Contact your academic advisor.", "Talk to your advisor.")
    assert score.passed is None
    assert not score.applies


def test_unanswerable_examples_are_scored_by_abstention() -> None:
    example = make_example(Category.UNANSWERABLE, "The documents I have don't say.")
    assert score_answer(example, "The documents don't mention a deposit.").passed is True
    assert score_answer(example, "Yes, the deposit is $300.").passed is False


def test_answerable_examples_are_scored_by_facts() -> None:
    example = make_example(Category.LOOKUP, "The school code is 004509.")
    assert score_answer(example, "Use 004509.").passed is True
    assert score_answer(example, "I don't know.").passed is False


def test_coverage_counts_what_each_scorer_can_judge() -> None:
    examples = [
        make_example(Category.UNANSWERABLE, "The documents don't say."),
        make_example(Category.LOOKUP, "It costs $20."),
        make_example(Category.LOOKUP, "Ask your advisor."),
    ]
    assert coverage(examples) == {"abstention": 1, "key facts": 1, "needs a judge": 1}


def test_every_reference_answer_in_the_dataset_passes_its_own_scorer() -> None:
    for example in load_examples(DATASET):
        score = score_answer(example, example.reference_answer)
        assert score.passed is not False, f"{example.id}: {score.reason}"


def test_main_reports_coverage_for_the_dataset(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(DATASET)]) == 0
    output = capsys.readouterr().out
    assert "abstention" in output
    assert "reference answers pass or are left for a judge" in output


def test_main_returns_2_for_a_missing_dataset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(tmp_path / "missing.jsonl")]) == 2
    assert capsys.readouterr().out.startswith("error:")
