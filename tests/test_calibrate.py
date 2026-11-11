"""Tests for calibrating the judge against human labels, with a fake judge."""

import json
from pathlib import Path

import pytest

from trustbench.calibrate import judge_labels, kappa_interval, main, report, verdict
from trustbench.label import Label, append_label
from trustbench.runner import AnswerCache, Generation
from trustbench.schema import Category, Difficulty, Evidence, Example


class FakeJudge:
    """Passes an answer that mentions the advisor; counts its calls."""

    name = "fake-judge format=json"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str) -> Generation:
        self.calls += 1
        graded = prompt.split("Answer to grade: ")[1].split("\n")[0]
        verdict = "correct" if "advisor" in graded else "incorrect"
        return Generation(json.dumps({"reason": "read it", "verdict": verdict}), 9, 0.5)


def make_example(example_id: str, reference: str) -> Example:
    return Example(
        id=example_id,
        question=f"Who do I ask? ({example_id})",
        reference_answer=reference,
        category=Category.LOOKUP,
        difficulty=Difficulty.EASY,
        evidence=(Evidence(doc_id="advising", quote="q"),),
    )


def make_label(example_id: str, answer: str, verdict: str, note: str = "") -> Label:
    return Label.model_validate(
        {
            "answer_key": f"{example_id}:{answer}",
            "example_id": example_id,
            "model": "ollama:m temperature=0.0 seed=0",
            "answer": answer,
            "label": verdict,
            "note": note,
            "dataset_version": "v1",
            "labeled_at": "2026-11-11T20:00:00+00:00",
        }
    )


PROSE = "Contact your academic advisor."
FACTS = "Call 719-255-3260."


def scenario() -> tuple[dict[str, Example], list[Label]]:
    """Six prose answers and two with facts; the judge disagrees with the person once."""
    examples = {f"ex-00{n}": make_example(f"ex-00{n}", PROSE) for n in range(1, 7)}
    examples["ex-007"] = make_example("ex-007", FACTS)
    examples["ex-008"] = make_example("ex-008", FACTS)
    labels = [
        make_label("ex-001", "Ask your advisor.", "correct"),
        make_label("ex-002", "Ask your advisor.", "correct"),
        make_label("ex-003", "Ask the registrar.", "incorrect"),
        make_label("ex-004", "Ask the registrar.", "incorrect"),
        make_label("ex-005", "Ask your advisor, maybe.", "incorrect", "too vague"),
        make_label("ex-006", "No idea.", "skip"),
        make_label("ex-007", "Call 719-255-3260 or your advisor.", "correct"),
        make_label("ex-008", "Call 719-255-0000.", "incorrect"),
    ]
    return examples, labels


def test_kappa_interval_is_reproducible_and_contains_kappa() -> None:
    first = [True] * 20 + [False] * 15 + [True] * 5 + [False] * 10
    second = [True] * 20 + [False] * 15 + [False] * 5 + [True] * 10
    interval = kappa_interval(first, second)
    assert interval is not None
    assert interval == kappa_interval(first, second)
    assert interval[0] < 0.4 < interval[1]


def test_kappa_interval_needs_variation() -> None:
    assert kappa_interval([True] * 5, [True] * 5) is None


def test_skipped_labels_are_left_out_and_judgments_are_cached(tmp_path: Path) -> None:
    examples, labels = scenario()
    judge = FakeJudge()
    items = judge_labels(labels, examples, judge, AnswerCache(tmp_path / "j.jsonl"), "v1")
    judge_labels(labels, examples, judge, AnswerCache(tmp_path / "j.jsonl"), "v1")

    assert len(items) == 7
    assert judge.calls == 7
    assert [item.judge.passed for item in items] == [True, True, False, False, True, True, False]
    assert [item.scorer.applies for item in items] == [False] * 5 + [True, True]


def test_report_compares_all_three_raters_and_lists_disagreements(tmp_path: Path) -> None:
    examples, labels = scenario()
    items = judge_labels(labels, examples, FakeJudge(), AnswerCache(tmp_path / "j.jsonl"), "v1")
    lines = report(items, exclude=set())

    assert lines[0] == "7 labeled answers, 0 without a judge verdict"
    assert lines[1].startswith("person vs judge, all (7 answers): agree on 6 of 7 (86%)")
    assert lines[2].startswith("person vs judge, scorers cannot judge (5 answers): agree on 4")
    assert lines[4].startswith("person vs scorers (2 answers): agree on 2 of 2 (100%)")
    assert "pass rate: person 43%, judge 57%" in lines
    disagreements = [line for line in lines if line.startswith("  disagree")]
    assert disagreements == [
        "  disagree ex-005 (ollama:m): person fail, judge pass: read it | note: too vague"
    ]


def test_a_kappa_just_over_the_bar_on_few_labels_is_inconclusive(tmp_path: Path) -> None:
    examples, labels = scenario()
    items = judge_labels(labels, examples, FakeJudge(), AnswerCache(tmp_path / "j.jsonl"), "v1")
    last = report(items, set())[-1]
    assert last.startswith("verdict: INCONCLUSIVE (kappa 0.62, 95% CI [")
    assert last.endswith("contains 0.6; label more answers)")


def test_the_verdict_uses_the_whole_interval() -> None:
    agree = [True] * 20 + [False] * 20
    assert verdict(agree, agree).startswith("verdict: TRUSTED (kappa 1.00, 95% CI [1.00, 1.00]")
    opposite = [not value for value in agree]
    assert verdict(agree, opposite).startswith("verdict: NOT TRUSTED (kappa -1.00")
    assert verdict([], []).startswith("verdict: INCONCLUSIVE (kappa cannot be estimated")


def test_report_adds_a_row_without_labels_that_were_not_blind(tmp_path: Path) -> None:
    examples, labels = scenario()
    items = judge_labels(labels, examples, FakeJudge(), AnswerCache(tmp_path / "j.jsonl"), "v1")
    lines = report(items, exclude={"ex-005", "ex-001"})
    assert any(
        line.startswith("person vs judge, without 2 seen (5 answers): agree on 5 of 5")
        for line in lines
    )


def write_files(tmp_path: Path) -> tuple[Path, Path]:
    examples, labels = scenario()
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(
        "".join(example.model_dump_json() + "\n" for example in examples.values()),
        encoding="utf-8",
    )
    label_file = tmp_path / "labels.jsonl"
    for label in labels:
        append_label(label_file, label)
    return dataset, label_file


def test_main_reports_and_reuses_judgments(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset, labels = write_files(tmp_path)
    judge = FakeJudge()
    argv = [str(dataset), "--labels", str(labels), "--judgments", str(tmp_path / "j.jsonl")]

    assert main(argv, make_judge=lambda model, url: judge) == 0
    assert main(argv, make_judge=lambda model, url: judge) == 0
    output = capsys.readouterr().out
    assert "judge fake-judge format=json" in output
    assert "verdict: INCONCLUSIVE" in output
    assert judge.calls == 7


def test_main_rejects_labels_for_unknown_examples(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset, labels = write_files(tmp_path)
    append_label(labels, make_label("ex-999", "Ask your advisor.", "correct"))
    argv = [str(dataset), "--labels", str(labels), "--judgments", str(tmp_path / "j.jsonl")]
    assert main(argv, make_judge=lambda model, url: FakeJudge()) == 2
    assert "not in the dataset: ex-999" in capsys.readouterr().out
