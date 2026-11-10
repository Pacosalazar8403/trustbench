"""Tests for BM25 retrieval and recall@k, on small corpora built for each test."""

from pathlib import Path

import pytest

from trustbench.corpus import build_corpus
from trustbench.retrieval import (
    BM25,
    Chunk,
    build_index,
    chunk_words,
    main,
    recall_at_k,
    tokenize,
)
from trustbench.schema import Category, Difficulty, Evidence, Example


def make_example(
    example_id: str, question: str, docs: tuple[str, ...], quote: str = "q"
) -> Example:
    category = Category.UNANSWERABLE if not docs else Category.LOOKUP
    return Example(
        id=example_id,
        question=question,
        reference_answer="An answer.",
        category=category,
        difficulty=Difficulty.EASY,
        evidence=tuple(Evidence(doc_id=doc, quote=quote) for doc in docs),
    )


def test_tokenize_lowercases_and_drops_common_words() -> None:
    assert tokenize("What is the FAFSA school code for UCCS?") == [
        "fafsa",
        "school",
        "code",
        "uccs",
    ]


def test_chunks_overlap_and_cover_every_word() -> None:
    text = " ".join(f"w{number}" for number in range(10))
    chunks = chunk_words("doc", text, size=4, overlap=1)
    assert [chunk.text for chunk in chunks] == ["w0 w1 w2 w3", "w3 w4 w5 w6", "w6 w7 w8 w9"]
    assert [chunk.index for chunk in chunks] == [0, 1, 2]


def test_a_short_document_is_one_chunk_and_an_empty_one_is_none() -> None:
    assert chunk_words("doc", "only three words", size=4, overlap=1) == [
        Chunk("doc", 0, "only three words")
    ]
    assert chunk_words("doc", "   ", size=4, overlap=1) == []


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (4, 4), (4, -1)])
def test_chunking_rejects_impossible_windows(size: int, overlap: int) -> None:
    with pytest.raises(ValueError, match="overlap"):
        chunk_words("doc", "some words here", size=size, overlap=overlap)


def test_rare_words_outrank_common_words() -> None:
    index = BM25(
        [
            Chunk("tuition", 0, "tuition is due at census tuition payment"),
            Chunk("parking", 0, "parking permits are virtual tuition"),
            Chunk("housing", 0, "housing applications open tuition"),
        ]
    )
    best, _ = index.search("virtual tuition", k=1)[0]
    assert best.doc_id == "parking"


def test_shorter_chunks_win_when_word_counts_tie() -> None:
    index = BM25(
        [
            Chunk("long", 0, "permit " + "filler " * 20),
            Chunk("short", 0, "permit refund"),
            Chunk("other", 0, "unrelated text"),
        ]
    )
    assert [chunk.doc_id for chunk, _ in index.search("permit", k=2)] == ["short", "long"]


def test_ties_are_broken_by_document_and_position() -> None:
    index = BM25(
        [Chunk("b-doc", 0, "census"), Chunk("a-doc", 1, "census"), Chunk("a-doc", 0, "census")]
    )
    ranked = [(chunk.doc_id, chunk.index) for chunk, _ in index.search("census", k=3)]
    assert ranked == [("a-doc", 0), ("a-doc", 1), ("b-doc", 0)]


def test_search_returns_only_chunks_that_match() -> None:
    index = BM25([Chunk("a", 0, "census date"), Chunk("b", 0, "parking permit")])
    assert [chunk.doc_id for chunk, _ in index.search("census", k=10)] == ["a"]
    assert index.search("nothing matches", k=10) == []
    with pytest.raises(ValueError, match="k must be"):
        index.search("census", k=0)


def test_recall_counts_hits_at_each_depth_and_skips_unanswerable() -> None:
    index = build_index(
        {
            "parking": "parking permits are refunded before the census date",
            "housing": "housing applications open in december",
            "tuition": "tuition payment plans and permits for parking garages",
        }
    )
    examples = [
        make_example("ex-001", "can i get a refund on my parking permit", ("parking",)),
        make_example("ex-002", "when do housing applications open", ("housing",)),
        make_example("ex-003", "payment plans for parking", ("parking",)),
        make_example("ex-004", "is tutoring free", ()),
    ]
    assert recall_at_k(index, examples, ks=(1, 3)) == {1: (2, 3), 3: (3, 3)}


def test_quote_recall_needs_the_chunk_that_holds_the_quote() -> None:
    index = BM25(
        [
            Chunk("refunds", 0, "refund policy overview for parking permits"),
            Chunk("refunds", 1, "permits returned before census get a full refund"),
        ]
    )
    example = make_example(
        "ex-001", "parking permits refund policy", ("refunds",), quote="before census get a full"
    )
    assert recall_at_k(index, [example], ks=(1, 2), level="document") == {1: (1, 1), 2: (1, 1)}
    assert recall_at_k(index, [example], ks=(1, 2), level="quote") == {1: (0, 1), 2: (1, 1)}


def test_quote_recall_accepts_the_same_sentence_on_another_page() -> None:
    index = BM25([Chunk("twin-page", 0, "The census date is the last day to drop")])
    example = make_example(
        "ex-001", "last day to drop", ("cited-page",), quote="The census date is the last day"
    )
    assert recall_at_k(index, [example], ks=(1,), level="document") == {1: (0, 1)}
    assert recall_at_k(index, [example], ks=(1,), level="quote") == {1: (1, 1)}


def test_main_reports_recall_by_split(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    corpus = tmp_path / "corpus"
    (corpus / "raw").mkdir(parents=True)
    (corpus / "sources.txt").write_text("permits https://example.edu/permits\n", encoding="utf-8")
    (corpus / "raw" / "permits.html").write_text(
        "<main><p>Parking permits are refunded before the census date.</p></main>", encoding="utf-8"
    )
    build_corpus(corpus)
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(
        make_example("ex-001", "parking permit refund", ("permits",)).model_dump_json() + "\n",
        encoding="utf-8",
    )

    assert main([str(dataset), str(corpus)]) == 0
    output = capsys.readouterr().out
    assert "1 chunks from 1 documents" in output
    assert "document" in output
    assert "quote" in output
    assert "100.0%" in output


def test_main_returns_2_without_a_built_corpus(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(
        make_example("ex-001", "parking", ("permits",)).model_dump_json() + "\n", encoding="utf-8"
    )
    assert main([str(dataset), str(tmp_path / "no-corpus")]) == 2
    assert capsys.readouterr().out.startswith("error:")
