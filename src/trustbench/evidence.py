"""Check that every evidence quote in a dataset appears in its source document.

The corpus text lives only on the machine that built it, so this check runs locally,
not in CI. Each document's text is checked against the manifest hash first, so a quote is
only ever verified against exactly the snapshot the manifest describes.
"""

import argparse
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from trustbench.corpus import MANIFEST_FILE, TEXT_DIR, normalize_text, read_manifest, text_sha256
from trustbench.dataset import DatasetError, load_examples
from trustbench.schema import Example

# How many characters of context to show after the point where a quote stops matching.
_CONTEXT = 30


@dataclass(frozen=True)
class EvidenceProblem:
    """One evidence quote that could not be verified."""

    example_id: str
    evidence_index: int
    doc_id: str
    message: str

    def __str__(self) -> str:
        where = f"{self.example_id} evidence[{self.evidence_index}] ({self.doc_id})"
        return f"{where}: {self.message}"


class CorpusError(Exception):
    """Raised when the corpus text on disk does not match its manifest."""


def load_corpus_text(corpus_dir: Path) -> dict[str, str]:
    """Return the normalized text of every document in the manifest.

    Raises:
        CorpusError: if a text file is missing or its hash differs from the manifest.
    """
    documents: dict[str, str] = {}
    for entry in read_manifest(corpus_dir / MANIFEST_FILE):
        path = corpus_dir / TEXT_DIR / f"{entry.doc_id}.txt"
        if not path.exists():
            raise CorpusError(f"no text for {entry.doc_id!r}: expected {path}")
        text = path.read_text(encoding="utf-8")
        if text_sha256(text) != entry.sha256:
            raise CorpusError(f"text for {entry.doc_id!r} does not match the manifest hash")
        documents[entry.doc_id] = normalize_text(text)
    return documents


def check_evidence(
    examples: Sequence[Example], documents: Mapping[str, str]
) -> list[EvidenceProblem]:
    """Return one problem for every evidence quote that is not found in its document."""
    problems: list[EvidenceProblem] = []
    for example in examples:
        for index, evidence in enumerate(example.evidence):
            document = documents.get(evidence.doc_id)
            message: str | None
            if document is None:
                message = "unknown doc_id: not in the corpus manifest"
            else:
                message = _quote_problem(normalize_text(evidence.quote), document)
            if message is not None:
                problems.append(EvidenceProblem(example.id, index, evidence.doc_id, message))
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point: `python -m trustbench.evidence <dataset> <corpus_dir>`."""
    parser = argparse.ArgumentParser(
        prog="python -m trustbench.evidence",
        description="Check that every evidence quote appears in its corpus document.",
    )
    parser.add_argument("dataset", type=Path, help="JSONL dataset file")
    parser.add_argument("corpus_dir", type=Path, help="built corpus folder")
    args = parser.parse_args(argv)

    try:
        examples = load_examples(args.dataset)
        documents = load_corpus_text(args.corpus_dir)
    except (DatasetError, CorpusError) as exc:
        print(exc, file=sys.stderr)
        return 2

    problems = check_evidence(examples, documents)
    quotes = sum(len(example.evidence) for example in examples)
    for problem in problems:
        print(problem)
    if problems:
        print(f"{len(problems)} of {quotes} evidence quote(s) not verified")
        return 1
    print(f"all {quotes} evidence quote(s) in {len(examples)} example(s) verified")
    return 0


def _quote_problem(quote: str, document: str) -> str | None:
    """Describe where a quote stops matching a document, or return None if it matches."""
    if quote in document:
        return None
    matched = _longest_matching_prefix(quote, document)
    if matched == 0:
        return "quote not found: no part of it appears in the document"
    page_start = document.find(quote[:matched]) + matched
    quote_rest = quote[matched : matched + _CONTEXT]
    page_rest = document[page_start : page_start + _CONTEXT]
    return (
        f"quote not found: the first {matched} of {len(quote)} characters match; "
        f"the quote continues {quote_rest!r}, the page continues {page_rest!r}"
    )


def _longest_matching_prefix(quote: str, document: str) -> int:
    """Return the length of the longest prefix of `quote` that appears in `document`.

    Binary search works because matching is monotonic: if a prefix appears in the
    document, every shorter prefix appears too.
    """
    low, high = 0, len(quote)
    while low < high:
        middle = (low + high + 1) // 2
        if quote[:middle] in document:
            low = middle
        else:
            high = middle - 1
    return low


if __name__ == "__main__":
    raise SystemExit(main())
