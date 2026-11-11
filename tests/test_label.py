"""Tests for blind human labeling, with scripted answers instead of a keyboard."""

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from trustbench.corpus import build_corpus
from trustbench.label import (
    DEFAULT_LABELS,
    Candidate,
    Label,
    LabelError,
    append_label,
    card,
    find_candidates,
    label_session,
    main,
    plan_sample,
    read_labels,
)
from trustbench.retrieval import build_index
from trustbench.runner import Answer, AnswerCache, Generation, OllamaModel, run
from trustbench.schema import Category, Difficulty, Evidence, Example

DOCUMENTS = {"aid": "Call the financial aid office at 719-255-3460 for help with the FAFSA."}


def make_example(example_id: str, reference: str = "Call 719-255-3460.") -> Example:
    return Example(
        id=example_id,
        question=f"How do I reach financial aid? ({example_id})",
        reference_answer=reference,
        category=Category.LOOKUP,
        difficulty=Difficulty.EASY,
        evidence=(Evidence(doc_id="aid", quote="Call the financial aid office"),),
    )


def make_candidate(number: int, prose: bool) -> Candidate:
    reference = "Contact the office." if prose else "Call 719-255-3460."
    example = make_example(f"ex-{number:03d}", reference)
    answer = Answer(
        key=f"key-{number:03d}",
        example_id=example.id,
        model="ollama:secret-model",
        dataset_version="v1",
        answer=f"Answer {number}.",
        tokens=1,
        seconds=0.1,
    )
    return Candidate(example, answer)


def make_label(key: str = "key-001", verdict: str = "correct") -> Label:
    return Label.model_validate(
        {
            "answer_key": key,
            "example_id": "ex-001",
            "model": "m",
            "answer": "a",
            "label": verdict,
            "dataset_version": "v1",
            "labeled_at": "2026-11-11T20:00:00+00:00",
        }
    )


def scripted(*replies: str) -> Callable[[str], str]:
    """An `input` replacement that returns the given replies, then acts like Ctrl+D."""
    queue: Iterator[str] = iter(replies)

    def ask(prompt: str) -> str:
        try:
            return next(queue)
        except StopIteration:
            raise EOFError from None

    return ask


def test_labels_round_trip_and_a_missing_file_has_none(tmp_path: Path) -> None:
    path = tmp_path / "labels" / "dev.jsonl"
    assert read_labels(path) == []
    append_label(path, make_label())
    assert read_labels(path) == [make_label()]


def test_a_damaged_label_line_is_reported_with_its_number(tmp_path: Path) -> None:
    path = tmp_path / "dev.jsonl"
    path.write_text(make_label().model_dump_json() + '\n{"label": "maybe"}\n', encoding="utf-8")
    with pytest.raises(LabelError, match="line 2"):
        read_labels(path)


def test_the_card_shows_the_answer_but_not_the_model() -> None:
    text = card(3, 60, make_candidate(1, prose=True))
    assert text.startswith("---- 3 of 60 ")
    assert "QUESTION   How do I reach financial aid?" in text
    assert "REFERENCE  Contact the office." in text
    assert "EVIDENCE   (aid) Call the financial aid office" in text
    assert "ANSWER     Answer 1." in text
    assert "secret-model" not in text


def test_the_sample_is_fixed_by_the_seed_and_mostly_prose() -> None:
    candidates = [make_candidate(n, prose=n % 2 == 0) for n in range(40)]
    sample = plan_sample(candidates, count=8, seed=0)

    assert sample == plan_sample(list(reversed(candidates)), count=8, seed=0)
    assert sample != plan_sample(candidates, count=8, seed=1)
    assert sum(candidate.prose for candidate in sample) == 6
    assert len({candidate.answer.key for candidate in sample}) == 8


def test_the_sample_fills_from_the_other_group_when_one_runs_short() -> None:
    candidates = [make_candidate(n, prose=n < 2) for n in range(10)]
    sample = plan_sample(candidates, count=6, seed=0)
    assert len(sample) == 6
    assert sum(candidate.prose for candidate in sample) == 2
    assert len(plan_sample(candidates, count=50, seed=0)) == 10


def test_a_session_saves_labels_and_notes_and_stops_on_quit(tmp_path: Path) -> None:
    sample = [make_candidate(n, prose=True) for n in range(1, 5)]
    path = tmp_path / "dev.jsonl"
    shown: list[str] = []

    given = label_session(
        sample, path, "v1", scripted("c", "", "x", "I", "wrong office", "s", "", "q"), shown.append
    )

    labels = read_labels(path)
    assert given == 3
    assert [label.label for label in labels] == ["correct", "incorrect", "skip"]
    assert labels[1].note == "wrong office"
    assert labels[0].model == "ollama:secret-model"
    assert len(shown) == 4


def test_a_new_session_resumes_after_the_labeled_answers(tmp_path: Path) -> None:
    sample = [make_candidate(n, prose=True) for n in range(1, 4)]
    path = tmp_path / "dev.jsonl"
    label_session(sample, path, "v1", scripted("c", ""), lambda text: None)
    shown: list[str] = []

    label_session(sample, path, "v1", scripted("i", "", "c", ""), shown.append)

    assert [label.example_id for label in read_labels(path)] == ["ex-001", "ex-002", "ex-003"]
    assert shown[0].startswith("---- 2 of 3 ")


def test_ctrl_d_ends_a_session_without_saving_a_half_label(tmp_path: Path) -> None:
    path = tmp_path / "dev.jsonl"
    given = label_session([make_candidate(1, prose=True)], path, "v1", scripted("c"), print)
    assert given == 0
    assert read_labels(path) == []


class FakeSystem:
    def __init__(self, model: str) -> None:
        self.name = OllamaModel(model).name

    def generate(self, prompt: str) -> Generation:
        return Generation("Call 719-255-3460.", tokens=1, seconds=0.1)


def write_project(tmp_path: Path, examples: list[Example]) -> tuple[Path, Path]:
    corpus = tmp_path / "corpus"
    (corpus / "raw").mkdir(parents=True)
    (corpus / "sources.txt").write_text("aid https://example.edu/aid\n", encoding="utf-8")
    (corpus / "raw" / "aid.html").write_text(
        f"<main><p>{DOCUMENTS['aid']}</p></main>", encoding="utf-8"
    )
    build_corpus(corpus)
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(
        "".join(example.model_dump_json() + "\n" for example in examples), encoding="utf-8"
    )
    return dataset, corpus


def test_candidates_include_every_cached_answer_from_each_model(tmp_path: Path) -> None:
    examples = [make_example("ex-001"), make_example("ex-002")]
    index = build_index(DOCUMENTS)
    cache = AnswerCache(tmp_path / "a.jsonl")
    run(examples, index, FakeSystem("one"), cache, "v1")
    run(examples[:1], index, FakeSystem("two"), cache, "v1")
    names = [OllamaModel("one").name, OllamaModel("two").name, OllamaModel("three").name]

    found = find_candidates(examples, index, cache, names, k=5)
    assert [(c.example.id, c.answer.model) for c in found] == [
        ("ex-001", names[0]),
        ("ex-001", names[1]),
        ("ex-002", names[0]),
    ]


def test_main_prints_the_rubric_and_labels_the_sample(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    examples = [make_example("ex-001"), make_example("ex-002", "Contact the office.")]
    dataset, corpus = write_project(tmp_path, examples)
    run(examples, build_index(DOCUMENTS), FakeSystem("one"), AnswerCache(tmp_path / "a.jsonl"), "v")
    labels = tmp_path / "labels.jsonl"
    argv = [str(dataset), str(corpus), "--models", "one", "--split", "all"]
    argv += ["--cache", str(tmp_path / "a.jsonl"), "--labels", str(labels)]

    assert main(argv, ask=scripted("c", "", "i", "")) == 0
    output = capsys.readouterr().out
    assert output.startswith("Grade each answer against the reference answer")
    assert "2 answers in this sample (1 that the scorers cannot judge), 0 labels so far" in output
    assert "2 labels saved this session; 0 of 2 left" in output
    assert len(read_labels(labels)) == 2


def test_main_returns_2_without_cached_answers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset, corpus = write_project(tmp_path, [make_example("ex-001")])
    argv = [str(dataset), str(corpus), "--models", "one", "--split", "all"]
    argv += ["--cache", str(tmp_path / "a.jsonl"), "--labels", str(tmp_path / "l.jsonl")]
    assert main(argv, ask=scripted()) == 2
    assert "no cached answers from one on all" in capsys.readouterr().out


def test_the_committed_label_file_is_valid() -> None:
    if not DEFAULT_LABELS.exists():
        pytest.skip("no labels committed yet")
    labels = read_labels(DEFAULT_LABELS)
    assert len({label.answer_key for label in labels}) == len(labels)
