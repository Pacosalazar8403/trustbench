"""Tests for committed run results and exporting them from the answer cache."""

import json
from pathlib import Path

import pytest

from trustbench import runner
from trustbench.corpus import build_corpus
from trustbench.dataset import load_examples
from trustbench.results import (
    RESULTS_DIR,
    Results,
    ResultsError,
    corpus_sha256,
    default_path,
    main,
    prompt_sha256,
    read_results,
    write_results,
)
from trustbench.retrieval import build_index
from trustbench.runner import AnswerCache, Generation, OllamaModel, run
from trustbench.schema import Category, Difficulty, Evidence, Example

DATASET = Path("data/uccs.jsonl")


def make_results(**changes: object) -> Results:
    fields: dict[str, object] = {
        "model": "ollama:m temperature=0.0 seed=0",
        "split": "dev",
        "k": 5,
        "dataset_version": "v1",
        "prompt_sha256": "p",
        "corpus_sha256": "c",
        "answers": {"ex-002": "Two.", "ex-001": "One."},
    }
    return Results.model_validate(fields | changes)


def test_default_path_names_the_split_and_model() -> None:
    assert default_path("llama3.2:3b", "dev") == Path("results/dev-llama3.2-3b.json")


def test_results_round_trip_as_sorted_readable_json(tmp_path: Path) -> None:
    path = tmp_path / "results" / "dev-m.json"
    write_results(make_results(), path)
    text = path.read_text(encoding="utf-8")

    assert read_results(path) == make_results()
    assert text.endswith("}\n")
    assert text.index('"ex-001"') < text.index('"ex-002"')
    assert '\n  "answers": {\n    "ex-001": "One.",' in text


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("not json", "Invalid JSON"),
        (json.dumps({"format": 1}), "model: Field required"),
        (json.dumps(make_results().model_dump() | {"k": 0}), "k: Input should be greater"),
        (json.dumps(make_results().model_dump() | {"extra": 1}), "extra: Extra inputs"),
        (json.dumps(make_results().model_dump() | {"format": 2}), "format: Input should be 1"),
    ],
)
def test_a_bad_results_file_is_reported_with_its_path(
    tmp_path: Path, content: str, message: str
) -> None:
    path = tmp_path / "bad.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ResultsError, match=f"{path.name}.*{message}"):
        read_results(path)


def test_a_missing_results_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ResultsError, match="no such results file"):
        read_results(tmp_path / "missing.json")


def test_the_prompt_hash_follows_the_answer_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    before = prompt_sha256()
    assert before == prompt_sha256()
    monkeypatch.setattr(runner, "PROMPT_TEMPLATE", runner.PROMPT_TEMPLATE + " ")
    assert prompt_sha256() != before


def write_corpus(path: Path, text: str) -> Path:
    (path / "raw").mkdir(parents=True)
    (path / "sources.txt").write_text("aid https://example.edu/aid\n", encoding="utf-8")
    (path / "raw" / "aid.html").write_text(f"<main><p>{text}</p></main>", encoding="utf-8")
    build_corpus(path)
    return path


def test_the_corpus_hash_follows_the_text_but_not_download_times(tmp_path: Path) -> None:
    first = write_corpus(tmp_path / "a", "Call 719-255-3460.")
    again = write_corpus(tmp_path / "b", "Call 719-255-3460.")
    changed = write_corpus(tmp_path / "c", "Call 719-255-0000.")

    assert corpus_sha256(first) == corpus_sha256(again)
    assert corpus_sha256(first) != corpus_sha256(changed)


def make_example(example_id: str) -> Example:
    return Example(
        id=example_id,
        question=f"How do I reach financial aid? ({example_id})",
        reference_answer="Call 719-255-3460.",
        category=Category.LOOKUP,
        difficulty=Difficulty.EASY,
        evidence=(Evidence(doc_id="aid", quote="q"),),
    )


class FakeSystem:
    def __init__(self, model: str) -> None:
        self.name = OllamaModel(model).name

    def generate(self, prompt: str) -> Generation:
        return Generation("Call 719-255-3460.", tokens=1, seconds=0.1)


def write_project(tmp_path: Path) -> tuple[Path, Path, list[Example]]:
    examples = [make_example("ex-001"), make_example("ex-002")]
    corpus = write_corpus(tmp_path / "corpus", "Call the financial aid office at 719-255-3460.")
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(
        "".join(example.model_dump_json() + "\n" for example in examples), encoding="utf-8"
    )
    return dataset, corpus, examples


def test_main_exports_every_cached_answer_with_its_provenance(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset, corpus, examples = write_project(tmp_path)
    index = build_index({"aid": "Call the financial aid office at 719-255-3460."})
    run(examples, index, FakeSystem("sys"), AnswerCache(tmp_path / "a.jsonl"), "v")
    out = tmp_path / "results" / "dev-sys.json"
    argv = [str(dataset), str(corpus), "--model", "sys", "--split", "all"]
    argv += ["--cache", str(tmp_path / "a.jsonl"), "--out", str(out)]

    assert main(argv) == 0
    results = read_results(out)
    assert results.model == "ollama:sys temperature=0.0 seed=0"
    assert results.split == "all"
    assert results.answers == {"ex-001": "Call 719-255-3460.", "ex-002": "Call 719-255-3460."}
    assert results.prompt_sha256 == prompt_sha256()
    assert results.corpus_sha256 == corpus_sha256(corpus)
    assert f"wrote 2 all answers from ollama:sys temperature=0.0 seed=0 to {out}" in (
        capsys.readouterr().out
    )


def test_main_refuses_to_export_a_run_with_missing_answers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset, corpus, examples = write_project(tmp_path)
    index = build_index({"aid": "Call the financial aid office at 719-255-3460."})
    run(examples[:1], index, FakeSystem("sys"), AnswerCache(tmp_path / "a.jsonl"), "v")
    out = tmp_path / "out.json"
    argv = [str(dataset), str(corpus), "--model", "sys", "--split", "all"]
    argv += ["--cache", str(tmp_path / "a.jsonl"), "--out", str(out)]

    assert main(argv) == 2
    assert "1 of 2 all examples have no cached answer from sys (first: ex-002)" in (
        capsys.readouterr().out
    )
    assert not out.exists()


def test_every_committed_results_file_is_valid_and_names_real_examples() -> None:
    files = sorted(RESULTS_DIR.glob("*.json"))
    if not files:
        pytest.skip("no results committed yet")
    ids = {example.id for example in load_examples(DATASET)}
    for path in files:
        results = read_results(path)
        assert results.answers, path
        assert set(results.answers) <= ids, path
