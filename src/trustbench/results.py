"""Frozen run results: one system's answers on one split, in a file committed to git.

Generating answers needs a model, a GPU, and the corpus text, and CI has none of them. So
answers are generated where the hardware is, exported to a results file, and committed;
the regression gate in CI compares committed results and needs only this repository.

A results file records what produced its answers: the model with its settings, the
split, k, the dataset version, and hashes of the answer prompt template and of the
corpus snapshot. Answers are sorted by example ID and the file is pretty-printed, so a
pull request that changes the system shows, in its diff, exactly which answers changed.

Run `python -m trustbench.results data/uccs.jsonl corpus/uccs --model llama3.2:3b` to
export a model's cached dev answers to `results/dev-llama3.2-3b.json`.
"""

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from trustbench import runner
from trustbench.compare import load_answers
from trustbench.corpus import MANIFEST_FILE, read_manifest
from trustbench.dataset import DatasetError, dataset_version, load_examples
from trustbench.evidence import CorpusError, load_corpus_text
from trustbench.retrieval import build_index
from trustbench.runner import DEFAULT_CACHE, DEFAULT_K, AnswerCache, OllamaModel, RunnerError

RESULTS_DIR = Path("results")


class ResultsError(Exception):
    """A results file could not be read, or a run could not be exported."""


class Results(BaseModel):
    """One system's answers on one split, and everything that produced them."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    format: Literal[1] = 1
    model: str
    split: Literal["dev", "test", "all"]
    k: int = Field(ge=1)
    dataset_version: str
    prompt_sha256: str
    corpus_sha256: str
    answers: dict[str, str]


def prompt_sha256() -> str:
    """Hash of the answer prompt template, so a results file records which prompt it used."""
    return hashlib.sha256(runner.PROMPT_TEMPLATE.encode("utf-8")).hexdigest()


def corpus_sha256(corpus_dir: Path) -> str:
    """Hash of the corpus snapshot: every document ID with the hash of its text.

    Download times are left out, so rebuilding the same text gives the same hash.
    """
    entries = sorted(read_manifest(corpus_dir / MANIFEST_FILE), key=lambda entry: entry.doc_id)
    listing = "".join(f"{entry.doc_id} {entry.sha256}\n" for entry in entries)
    return hashlib.sha256(listing.encode("utf-8")).hexdigest()


def default_path(model: str, split: str) -> Path:
    """`results/dev-llama3.2-3b.json` for model `llama3.2:3b` on dev."""
    return RESULTS_DIR / f"{split}-{model.replace(':', '-').replace('/', '-')}.json"


def write_results(results: Results, path: Path) -> None:
    """Write pretty, key-sorted JSON, so diffs between versions are readable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(results.model_dump(mode="json"), indent=2, sort_keys=True, ensure_ascii=False)
    path.write_text(text + "\n", encoding="utf-8")


def read_results(path: Path) -> Results:
    try:
        return Results.model_validate_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ResultsError(f"{path}: no such results file") from error
    except ValidationError as error:
        first = error.errors()[0]
        where = ".".join(str(part) for part in first["loc"]) or "file"
        raise ResultsError(f"{path}: {where}: {first['msg']}") from error


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m trustbench.results",
        description="Export a model's cached answers on one split to a results file.",
    )
    parser.add_argument("dataset", type=Path, help="JSONL dataset file")
    parser.add_argument("corpus_dir", type=Path, help="corpus folder with a built manifest")
    parser.add_argument("--model", required=True, help="Ollama model name, e.g. llama3.2:3b")
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="chunks retrieved per question")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="answer cache file")
    parser.add_argument(
        "--out", type=Path, help="results file (default: results/<split>-<model>.json)"
    )
    args = parser.parse_args(argv)
    if args.k < 1:
        parser.error("--k must be at least 1")

    try:
        dataset = load_examples(args.dataset)
        examples = runner.select_examples(dataset, args.split)
        index = build_index(load_corpus_text(args.corpus_dir))
        cache = AnswerCache(args.cache)
        corpus_hash = corpus_sha256(args.corpus_dir)
    except (DatasetError, CorpusError, RunnerError, FileNotFoundError) as error:
        print(f"error: {error}")
        return 2

    name = OllamaModel(args.model).name
    answers, missing = load_answers(examples, index, cache, name, args.k)
    if missing:
        print(
            f"error: {len(missing)} of {len(examples)} {args.split} examples have no cached "
            f"answer from {args.model} (first: {missing[0]}). Run: python -m "
            f"trustbench.runner {args.dataset} {args.corpus_dir} --model {args.model} "
            f"--split {args.split} --k {args.k}"
        )
        return 2

    results = Results(
        model=name,
        split=args.split,
        k=args.k,
        dataset_version=dataset_version(dataset),
        prompt_sha256=prompt_sha256(),
        corpus_sha256=corpus_hash,
        answers=answers,
    )
    out = args.out or default_path(args.model, args.split)
    write_results(results, out)
    print(f"wrote {len(answers)} {args.split} answers from {name} to {out}")
    print(
        f"dataset {results.dataset_version}, prompt {results.prompt_sha256[:12]}, "
        f"corpus {results.corpus_sha256[:12]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
