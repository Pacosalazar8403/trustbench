"""Tests for the model runner and its answer cache, with fake models and a fake server."""

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar

import pytest

from trustbench.corpus import build_corpus
from trustbench.retrieval import Chunk, build_index
from trustbench.runner import (
    ABSTAIN,
    Answer,
    AnswerCache,
    Generation,
    OllamaModel,
    RunnerError,
    build_prompt,
    cache_key,
    main,
    run,
)
from trustbench.schema import Category, Difficulty, Evidence, Example
from trustbench.scoring import abstains

DOCUMENTS = {
    "permits": "Parking permits are refunded in full before the census date.",
    "aid": "Call the financial aid office at 719-255-3460 for help with the FAFSA.",
}


class FakeModel:
    """Answers every prompt with a fixed text and counts how often it was called."""

    def __init__(self, reply: str = "An answer.", name: str = "fake") -> None:
        self.reply = reply
        self._name = name
        self.prompts: list[str] = []

    @property
    def name(self) -> str:
        return self._name

    def generate(self, prompt: str) -> Generation:
        self.prompts.append(prompt)
        return Generation(self.reply, tokens=3, seconds=0.5)


def make_example(
    example_id: str,
    question: str,
    reference: str = "An answer.",
    category: Category = Category.LOOKUP,
) -> Example:
    evidence = () if category is Category.UNANSWERABLE else (Evidence(doc_id="aid", quote="q"),)
    return Example(
        id=example_id,
        question=question,
        reference_answer=reference,
        category=category,
        difficulty=Difficulty.EASY,
        evidence=evidence,
    )


def test_the_abstain_sentence_is_recognized_by_the_abstention_scorer() -> None:
    assert abstains(ABSTAIN)


def test_prompt_numbers_chunks_and_names_their_documents() -> None:
    prompt = build_prompt(
        "When are permits refunded?",
        [Chunk("permits", 0, "Refunded before census."), Chunk("aid", 2, "Call us.")],
    )
    assert "[1] (permits) Refunded before census." in prompt
    assert "[2] (aid) Call us." in prompt
    assert ABSTAIN in prompt
    assert prompt.endswith("Question: When are permits refunded?\nAnswer:")


def test_prompt_says_so_when_nothing_was_retrieved() -> None:
    assert "(no documents found)" in build_prompt("Anything?", [])


def test_cache_key_changes_with_the_model_or_the_prompt() -> None:
    key = cache_key("model-a", "prompt")
    assert key == cache_key("model-a", "prompt")
    assert key != cache_key("model-b", "prompt")
    assert key != cache_key("model-a", "prompt!")


def make_answer(key: str = "k1") -> Answer:
    return Answer(
        key=key,
        example_id="ex-001",
        model="fake",
        dataset_version="v1",
        answer="An answer.",
        tokens=3,
        seconds=0.5,
    )


def test_cache_survives_a_restart(tmp_path: Path) -> None:
    path = tmp_path / "runs" / "answers.jsonl"
    AnswerCache(path).put(make_answer())
    reopened = AnswerCache(path)
    assert len(reopened) == 1
    assert reopened.get("k1") == make_answer()
    assert reopened.get("missing") is None


def test_a_damaged_cache_line_is_reported_with_its_line_number(tmp_path: Path) -> None:
    path = tmp_path / "answers.jsonl"
    path.write_text(make_answer().model_dump_json() + '\n{"key": "k2"}\n', encoding="utf-8")
    with pytest.raises(RunnerError, match="line 2"):
        AnswerCache(path)


def test_a_second_run_is_answered_entirely_from_the_cache(tmp_path: Path) -> None:
    index = build_index(DOCUMENTS)
    examples = [make_example("ex-001", "permit refund"), make_example("ex-002", "aid phone")]
    model = FakeModel()

    first = run(examples, index, model, AnswerCache(tmp_path / "a.jsonl"), "v1", k=1)
    second = run(examples, index, model, AnswerCache(tmp_path / "a.jsonl"), "v1", k=1)

    assert [result.cached for result in first] == [False, False]
    assert [result.cached for result in second] == [True, True]
    assert len(model.prompts) == 2
    assert [result.answer for result in second] == [result.answer for result in first]


def test_changing_k_or_the_model_is_a_cache_miss(tmp_path: Path) -> None:
    index = build_index(DOCUMENTS)
    examples = [make_example("ex-001", "permits refunded financial aid")]
    cache = AnswerCache(tmp_path / "a.jsonl")

    run(examples, index, FakeModel(), cache, "v1", k=1)
    assert run(examples, index, FakeModel(), cache, "v1", k=1)[0].cached is True
    assert not run(examples, index, FakeModel(), cache, "v1", k=2)[0].cached
    assert not run(examples, index, FakeModel(name="other"), cache, "v1", k=1)[0].cached


def test_a_larger_k_that_retrieves_the_same_chunks_reuses_the_answer(tmp_path: Path) -> None:
    index = build_index(DOCUMENTS)
    examples = [make_example("ex-001", "financial aid phone")]
    cache = AnswerCache(tmp_path / "a.jsonl")

    run(examples, index, FakeModel(), cache, "v1", k=1)
    assert run(examples, index, FakeModel(), cache, "v1", k=5)[0].cached


def test_the_dataset_version_is_recorded_but_does_not_invalidate_answers(
    tmp_path: Path,
) -> None:
    index = build_index(DOCUMENTS)
    examples = [make_example("ex-001", "permit refund")]
    cache = AnswerCache(tmp_path / "a.jsonl")

    first = run(examples, index, FakeModel(), cache, "v1", k=1)
    second = run(examples, index, FakeModel(), cache, "v2", k=1)

    assert first[0].answer.dataset_version == "v1"
    assert second[0].cached


class FakeOllama(BaseHTTPRequestHandler):
    """Answers /api/generate like Ollama, or with an error for an unknown model."""

    requests: ClassVar[list[dict[str, Any]]] = []

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOllama.requests.append(body)
        if body["model"] == "missing":
            self._send(404, {"error": "model 'missing' not found"})
        else:
            self._send(
                200,
                {
                    "response": "  Call 719-255-3460.\n",
                    "eval_count": 7,
                    "total_duration": 2_500_000_000,
                },
            )

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:
        pass


@pytest.fixture
def ollama_url() -> Iterator[str]:
    FakeOllama.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def test_ollama_model_sends_a_deterministic_request_and_reads_the_reply(
    ollama_url: str,
) -> None:
    generation = OllamaModel("llama3.2:1b", ollama_url).generate("a prompt")

    assert generation == Generation("Call 719-255-3460.", tokens=7, seconds=2.5)
    assert FakeOllama.requests == [
        {
            "model": "llama3.2:1b",
            "prompt": "a prompt",
            "stream": False,
            "options": {"temperature": 0.0, "seed": 0},
        }
    ]


def test_ollama_model_name_includes_its_settings() -> None:
    assert OllamaModel("llama3.2:1b").name == "ollama:llama3.2:1b temperature=0.0 seed=0"


def test_ollama_errors_are_reported_clearly(ollama_url: str) -> None:
    with pytest.raises(RunnerError, match=r"HTTP 404.*not found"):
        OllamaModel("missing", ollama_url).generate("a prompt")


def test_an_unreachable_ollama_is_reported_clearly() -> None:
    with pytest.raises(RunnerError, match="is it running"):
        OllamaModel("llama3.2:1b", "http://127.0.0.1:9", timeout=2).generate("a prompt")


def write_project(tmp_path: Path, examples: list[Example]) -> tuple[Path, Path]:
    corpus = tmp_path / "corpus"
    (corpus / "raw").mkdir(parents=True)
    sources = "".join(f"{doc} https://example.edu/{doc}\n" for doc in sorted(DOCUMENTS))
    (corpus / "sources.txt").write_text(sources, encoding="utf-8")
    for doc, text in DOCUMENTS.items():
        (corpus / "raw" / f"{doc}.html").write_text(f"<main><p>{text}</p></main>", "utf-8")
    build_corpus(corpus)
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(
        "".join(example.model_dump_json() + "\n" for example in examples), encoding="utf-8"
    )
    return dataset, corpus


def test_main_answers_scores_and_then_reuses_the_cache(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    examples = [
        make_example("ex-001", "financial aid phone", "Call 719-255-3460."),
        make_example("ex-002", "who wrote the fight song", category=Category.UNANSWERABLE),
    ]
    dataset, corpus = write_project(tmp_path, examples)
    model = FakeModel("Call 719-255-3460.")
    argv = [str(dataset), str(corpus), "--model", "fake", "--split", "all"]
    argv += ["--cache", str(tmp_path / "answers.jsonl")]

    assert main(argv, make_model=lambda name, url: model) == 0
    output = capsys.readouterr().out
    assert "2 answers: 0 from the cache, 2 new (1.0s, 6 tokens)" in output
    assert "deterministic scorers judged 2 of 2: 1 passed (50%)" in output
    assert "failed ex-002: answered a question the documents do not answer" in output

    assert main(argv, make_model=lambda name, url: model) == 0
    assert "2 answers: 2 from the cache, 0 new (0.0s, 0 tokens)" in capsys.readouterr().out
    assert len(model.prompts) == 2


def test_main_limits_and_counts_abstentions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    examples = [make_example(f"ex-00{number}", "permit refund") for number in (1, 2, 3)]
    dataset, corpus = write_project(tmp_path, examples)
    argv = [str(dataset), str(corpus), "--model", "fake", "--split", "all", "--limit", "2"]
    argv += ["--cache", str(tmp_path / "answers.jsonl")]

    assert main(argv, make_model=lambda name, url: FakeModel(ABSTAIN)) == 0
    output = capsys.readouterr().out
    assert "2 all examples" in output
    assert "abstained on 2 answerable question(s)" in output


def test_main_returns_2_when_the_model_cannot_be_reached(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset, corpus = write_project(tmp_path, [make_example("ex-001", "permit refund")])
    argv = [str(dataset), str(corpus), "--model", "llama3.2:1b", "--split", "all"]
    argv += ["--cache", str(tmp_path / "answers.jsonl"), "--ollama-url", "http://127.0.0.1:9"]

    assert main(argv) == 2
    assert "is it running" in capsys.readouterr().out
