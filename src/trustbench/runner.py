"""Run a language model over a dataset with retrieved context, and cache every answer.

For each example, the runner retrieves the top k chunks with the BM25 baseline, builds a
prompt that asks the model to answer only from those chunks, and sends it to the model.
When the chunks do not contain the answer, the prompt asks for one exact sentence,
`ABSTAIN`, which the abstention scorer recognizes.

Every answer is cached, one JSON object per line, under a key that hashes everything that
decides the model's output: the model's name (including its generation settings) and the
full prompt (which already contains the question and the retrieved chunks). Running the
same configuration again costs nothing, and changing the model, the prompt template, k, or
the chunking changes the key, so a stale answer is never reused. The dataset version is
recorded with each answer for provenance but is not part of the key: editing one reference
answer should not throw away hundreds of answers whose prompts did not change.

Models are anything with a `name` and a `generate(prompt)` method, so tests use a fake
model and need neither Ollama nor a network. `OllamaModel` talks to a local Ollama server.

Run `python -m trustbench.runner data/uccs.jsonl corpus/uccs --model llama3.2:1b --limit 10`
to answer the first 10 dev questions and score them with the deterministic scorers.
"""

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, ValidationError

from trustbench.dataset import DatasetError, dataset_version, load_examples
from trustbench.evidence import CorpusError, load_corpus_text
from trustbench.retrieval import BM25, Chunk, build_index
from trustbench.schema import Category, Example
from trustbench.scoring import abstains, score_answer
from trustbench.splits import Split, assign_split, group_key

ABSTAIN = "The documents don't say."
DEFAULT_K = 5
DEFAULT_CACHE = Path("runs/answers.jsonl")
OLLAMA_URL = "http://localhost:11434"

PROMPT_TEMPLATE = (
    "Answer the question using only the documents below. Keep the answer short, and include "
    "any specific numbers, dates, prices, phone numbers, and email addresses the documents "
    "give for it.\n"
    "If the documents do not contain the answer, reply with exactly this sentence and "
    "nothing else: {abstain}\n"
    "\n"
    "Documents:\n"
    "{context}\n"
    "\n"
    "Question: {question}\n"
    "Answer:"
)


class RunnerError(Exception):
    """A model could not be reached, or the answer cache could not be read."""


@dataclass(frozen=True)
class Generation:
    """One model output, with how many tokens it took and how long, in seconds."""

    text: str
    tokens: int
    seconds: float


class Model(Protocol):
    """Anything that turns a prompt into a generation, with a name that identifies it."""

    @property
    def name(self) -> str: ...

    def generate(self, prompt: str) -> Generation: ...


@dataclass(frozen=True)
class OllamaModel:
    """A model served by Ollama, called with greedy decoding and a fixed seed."""

    model: str
    base_url: str = OLLAMA_URL
    timeout: float = 300.0
    temperature: float = 0.0
    seed: int = 0

    @property
    def name(self) -> str:
        return f"ollama:{self.model} temperature={self.temperature} seed={self.seed}"

    def generate(self, prompt: str) -> Generation:
        body = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature, "seed": self.seed},
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace").strip()
            raise RunnerError(f"Ollama returned HTTP {error.code}: {detail}") from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise RunnerError(
                f"could not reach Ollama at {self.base_url} ({error}); is it running?"
            ) from error
        return Generation(
            text=str(data["response"]).strip(),
            tokens=int(data.get("eval_count", 0)),
            seconds=int(data.get("total_duration", 0)) / 1e9,
        )


class Answer(BaseModel):
    """A cached answer: what was asked, of which model, and what it said."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str
    example_id: str
    model: str
    dataset_version: str
    answer: str
    tokens: int
    seconds: float


class AnswerCache:
    """Answers stored one JSON object per line, appended as soon as each is made.

    Appending right away means an interrupted run keeps every answer it finished.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._answers: dict[str, Answer] = {}
        if path.exists():
            lines = path.read_text(encoding="utf-8").splitlines()
            for number, line in enumerate(lines, start=1):
                if not line.strip():
                    continue
                try:
                    answer = Answer.model_validate_json(line)
                except ValidationError as error:
                    raise RunnerError(
                        f"{path}: line {number} is not a valid cached answer: "
                        f"{error.errors()[0]['msg']}"
                    ) from error
                self._answers[answer.key] = answer

    def __len__(self) -> int:
        return len(self._answers)

    def get(self, key: str) -> Answer | None:
        return self._answers.get(key)

    def put(self, answer: Answer) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(answer.model_dump_json() + "\n")
        self._answers[answer.key] = answer


@dataclass(frozen=True)
class Result:
    """One example's answer, and whether it came from the cache."""

    example: Example
    answer: Answer
    cached: bool


def build_prompt(question: str, chunks: Sequence[Chunk]) -> str:
    """Number the retrieved chunks, name their documents, and ask the question."""
    context = "\n\n".join(
        f"[{number}] ({chunk.doc_id}) {chunk.text}" for number, chunk in enumerate(chunks, 1)
    )
    return PROMPT_TEMPLATE.format(
        abstain=ABSTAIN, context=context or "(no documents found)", question=question
    )


def prompt_for(example: Example, index: BM25, k: int = DEFAULT_K) -> str:
    """Retrieve the top k chunks for an example's question and build its prompt."""
    chunks = [chunk for chunk, _ in index.search(example.question, k)]
    return build_prompt(example.question, chunks)


def select_examples(
    examples: Iterable[Example], split: str, limit: int | None = None
) -> list[Example]:
    """Examples in ID order, from one split (`dev` or `test`) or `all`, up to `limit`."""
    if split not in ("dev", "test", "all"):
        raise ValueError(f"split must be dev, test, or all, got {split!r}")
    selected = [
        example
        for example in sorted(examples, key=lambda example: example.id)
        if split == "all" or assign_split(group_key(example)) is Split(split)
    ]
    return selected[:limit]


def cache_key(model_name: str, prompt: str) -> str:
    """Hash everything that decides a model's output: the model, its settings, the prompt."""
    payload = json.dumps([model_name, prompt], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run(
    examples: Iterable[Example],
    index: BM25,
    model: Model,
    cache: AnswerCache,
    version: str,
    k: int = DEFAULT_K,
    on_result: Callable[[int, Result], None] | None = None,
) -> list[Result]:
    """Answer every example, using the cache where possible, in the order given."""
    results = []
    for position, example in enumerate(examples, start=1):
        prompt = prompt_for(example, index, k)
        key = cache_key(model.name, prompt)
        answer = cache.get(key)
        cached = answer is not None
        if answer is None:
            generation = model.generate(prompt)
            answer = Answer(
                key=key,
                example_id=example.id,
                model=model.name,
                dataset_version=version,
                answer=generation.text,
                tokens=generation.tokens,
                seconds=generation.seconds,
            )
            cache.put(answer)
        result = Result(example, answer, cached)
        results.append(result)
        if on_result is not None:
            on_result(position, result)
    return results


def summarize(results: Sequence[Result]) -> list[str]:
    """Describe a run: cache use, cost, and what the deterministic scorers found."""
    new = [result for result in results if not result.cached]
    lines = [
        f"{len(results)} answers: {len(results) - len(new)} from the cache, {len(new)} new "
        f"({sum(result.answer.seconds for result in new):.1f}s, "
        f"{sum(result.answer.tokens for result in new)} tokens)"
    ]
    scores = [(result, score_answer(result.example, result.answer.answer)) for result in results]
    judged = [(result, score) for result, score in scores if score.applies]
    passed = sum(1 for _, score in judged if score.passed)
    if judged:
        lines.append(
            f"deterministic scorers judged {len(judged)} of {len(results)}: "
            f"{passed} passed ({passed / len(judged):.0%})"
        )
    else:
        lines.append(f"deterministic scorers judged 0 of {len(results)}")
    abstained = sum(
        1
        for result in results
        if result.example.category is not Category.UNANSWERABLE and abstains(result.answer.answer)
    )
    lines.append(f"abstained on {abstained} answerable question(s)")
    lines.extend(
        f"  failed {result.example.id}: {score.reason}"
        for result, score in judged
        if score.passed is False
    )
    return lines


def main(
    argv: Sequence[str] | None = None,
    make_model: Callable[[str, str], Model] = lambda model, url: OllamaModel(model, url),
) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m trustbench.runner",
        description="Answer dataset questions with a model and retrieved context, with caching.",
    )
    parser.add_argument("dataset", type=Path, help="JSONL dataset file")
    parser.add_argument("corpus_dir", type=Path, help="corpus folder with a built manifest")
    parser.add_argument("--model", required=True, help="Ollama model name, e.g. llama3.2:1b")
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    parser.add_argument("--limit", type=int, help="answer only the first N examples")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="chunks retrieved per question")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="answer cache file")
    parser.add_argument("--ollama-url", default=OLLAMA_URL)
    args = parser.parse_args(argv)
    if args.k < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--k and --limit must be at least 1")

    try:
        examples = load_examples(args.dataset)
        documents = load_corpus_text(args.corpus_dir)
        cache = AnswerCache(args.cache)
    except (DatasetError, CorpusError, RunnerError, FileNotFoundError) as error:
        print(f"error: {error}")
        return 2

    selected = select_examples(examples, args.split, args.limit)
    model = make_model(args.model, args.ollama_url)
    print(f"model {model.name}")
    print(f"{len(selected)} {args.split} examples, k={args.k}, cache {args.cache} ({len(cache)})")

    def report(position: int, result: Result) -> None:
        status = "cached" if result.cached else f"{result.answer.seconds:.1f}s"
        print(f"[{position}/{len(selected)}] {result.example.id} {status}", flush=True)

    started = time.perf_counter()
    try:
        results = run(
            selected,
            build_index(documents),
            model,
            cache,
            dataset_version(examples),
            k=args.k,
            on_result=report,
        )
    except RunnerError as error:
        print(f"error: {error}")
        print("answers finished before the error are saved in the cache")
        return 2
    for line in summarize(results):
        print(line)
    print(f"done in {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
