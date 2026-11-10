"""A BM25 retrieval baseline over the evidence corpus, and its measured recall.

A RAG system can only answer from what it retrieves, so retrieval is measured on its own,
before any language model is involved. Every answerable example already names the
documents that contain its answer (its evidence), which makes those documents the answer
key for retrieval: **recall@k** is the share of examples where a correct document appears
in the top k chunks. No model and no human labels are needed.

Documents are split into overlapping windows of words, and windows are ranked with BM25,
the classic keyword ranking used by search engines: a chunk scores higher when it contains
the query's rare words, contains them more often, and is shorter. Ties are broken by
document ID and chunk position, so the same query always returns the same ranking.

Run `python -m trustbench.retrieval data/uccs.jsonl corpus/uccs` to measure recall@k on the
dev and test splits. It needs the corpus built locally (see the README).
"""

import argparse
import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from trustbench.dataset import DatasetError, load_examples
from trustbench.evidence import CorpusError, load_corpus_text
from trustbench.schema import Category, Example
from trustbench.splits import Split, assign_split, group_key

CHUNK_WORDS = 100
CHUNK_OVERLAP = 25
DEFAULT_KS = (1, 3, 5, 10)

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "my",
        "of",
        "on",
        "or",
        "so",
        "that",
        "the",
        "their",
        "this",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "will",
        "with",
        "you",
        "your",
    ]
)


@dataclass(frozen=True)
class Chunk:
    """A window of words from one document, at a position within it."""

    doc_id: str
    index: int
    text: str


def tokenize(text: str) -> list[str]:
    """Lowercase words and numbers, without common words that say nothing about topic."""
    return [token for token in _TOKEN.findall(text.lower()) if token not in _STOPWORDS]


def chunk_words(
    doc_id: str, text: str, size: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP
) -> list[Chunk]:
    """Split text into windows of `size` words, each sharing `overlap` words with the last.

    The overlap keeps a sentence that crosses a window boundary whole in at least one chunk.
    """
    if size < 1 or not 0 <= overlap < size:
        raise ValueError("size must be at least 1, and overlap between 0 and size - 1")
    words = text.split()
    step = size - overlap
    chunks = []
    for index, start in enumerate(range(0, max(len(words) - overlap, 1), step)):
        window = words[start : start + size]
        if window:
            chunks.append(Chunk(doc_id, index, " ".join(window)))
    return chunks


class BM25:
    """Okapi BM25 ranking over a fixed list of chunks.

    `k1` controls how quickly repeated words stop adding score, and `b` how strongly long
    chunks are penalized. The inverse document frequency uses the form that is never
    negative, so a word found in most chunks adds a little score instead of subtracting.
    """

    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        if not chunks:
            raise ValueError("need at least one chunk to search")
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self._counts = [Counter(tokenize(chunk.text)) for chunk in self.chunks]
        self._lengths = [sum(counts.values()) for counts in self._counts]
        self._average_length = sum(self._lengths) / len(self._lengths) or 1.0
        frequencies = Counter(token for counts in self._counts for token in counts)
        total = len(self.chunks)
        self._idf = {
            token: math.log(1 + (total - count + 0.5) / (count + 0.5))
            for token, count in frequencies.items()
        }

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        """Return up to k chunks with a positive score, best first."""
        if k < 1:
            raise ValueError("k must be at least 1")
        terms = [term for term in set(tokenize(query)) if term in self._idf]
        scored = []
        for position, counts in enumerate(self._counts):
            score = 0.0
            for term in terms:
                frequency = counts.get(term, 0)
                if frequency:
                    norm = 1 - self.b + self.b * self._lengths[position] / self._average_length
                    score += (
                        self._idf[term] * frequency * (self.k1 + 1) / (frequency + self.k1 * norm)
                    )
            if score > 0:
                scored.append((self.chunks[position], score))
        scored.sort(key=lambda item: (-item[1], item[0].doc_id, item[0].index))
        return scored[:k]


def build_index(documents: dict[str, str]) -> BM25:
    """Chunk every document, in doc_id order, and index the chunks."""
    chunks = [
        chunk for doc_id in sorted(documents) for chunk in chunk_words(doc_id, documents[doc_id])
    ]
    return BM25(chunks)


def recall_at_k(
    index: BM25, examples: Iterable[Example], ks: Sequence[int] = DEFAULT_KS
) -> dict[int, tuple[int, int]]:
    """For each k, count answerable examples with a correct document in the top k chunks.

    Returns `{k: (hits, total)}`. Unanswerable examples have no correct document, so they
    are skipped.
    """
    answerable = [example for example in examples if example.category is not Category.UNANSWERABLE]
    counts = {k: 0 for k in ks}
    deepest = max(ks)
    for example in answerable:
        gold = {evidence.doc_id for evidence in example.evidence}
        ranked = [chunk.doc_id for chunk, _ in index.search(example.question, deepest)]
        for k in ks:
            if gold.intersection(ranked[:k]):
                counts[k] += 1
    return {k: (counts[k], len(answerable)) for k in ks}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m trustbench.retrieval",
        description="Measure BM25 recall@k on the dev and test splits.",
    )
    parser.add_argument("dataset", type=Path, help="JSONL dataset file")
    parser.add_argument("corpus_dir", type=Path, help="corpus folder with a built manifest")
    args = parser.parse_args(argv)

    try:
        examples = load_examples(args.dataset)
        documents = load_corpus_text(args.corpus_dir)
    except (DatasetError, CorpusError, FileNotFoundError) as error:
        print(f"error: {error}")
        return 2

    index = build_index(documents)
    print(f"{len(index.chunks)} chunks from {len(documents)} documents")
    print(
        f"{'split':6} {'examples':>8}  " + "  ".join(f"{'recall@' + str(k):>9}" for k in DEFAULT_KS)
    )
    for split in Split:
        subset = [example for example in examples if assign_split(group_key(example)) is split]
        results = recall_at_k(index, subset)
        total = next(iter(results.values()))[1]
        cells = "  ".join(
            f"{hits / total:>9.1%}" if total else f"{'-':>9}" for hits, _ in results.values()
        )
        print(f"{split.value:6} {total:>8}  {cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
