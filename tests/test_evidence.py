"""Tests for checking evidence quotes against the corpus."""

from pathlib import Path
from typing import Any

import pytest

from trustbench.corpus import build_corpus
from trustbench.evidence import (
    CorpusError,
    EvidenceProblem,
    _longest_matching_prefix,
    check_evidence,
    load_corpus_text,
    main,
)
from trustbench.schema import Example

ADVISING_PAGE = (
    "<main><p>In-Person meetings will be in our office in Main Hall&nbsp;208.</p>"
    "<p>It\u2019s open on weekdays.</p></main>"
)


def make_corpus(root: Path, pages: dict[str, str]) -> Path:
    """Create a corpus folder with one saved page per doc id, and build it."""
    (root / "raw").mkdir(parents=True)
    lines = [f"{doc_id} https://example.edu/{doc_id}" for doc_id in pages]
    (root / "sources.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for doc_id, html in pages.items():
        (root / "raw" / f"{doc_id}.html").write_text(html, encoding="utf-8")
    build_corpus(root)
    return root


def make_example(example_id: str, *evidence: tuple[str, str], category: str = "lookup") -> Example:
    """Build an example whose evidence is the given (doc_id, quote) pairs."""
    data: dict[str, Any] = {
        "id": example_id,
        "question": "Where is the advising office?",
        "reference_answer": "Main Hall 208.",
        "category": category,
        "difficulty": "easy",
        "evidence": [{"doc_id": doc_id, "quote": quote} for doc_id, quote in evidence],
    }
    return Example.model_validate(data)


def test_quotes_match_after_normalization(tmp_path: Path) -> None:
    corpus = make_corpus(tmp_path / "corpus", {"advising-faq": ADVISING_PAGE})
    example = make_example(
        "advising-001",
        ("advising-faq", "our office in Main Hall 208."),
        ("advising-faq", "It's open on weekdays."),
    )

    assert check_evidence([example], load_corpus_text(corpus)) == []


def test_missing_quote_shows_where_it_stops_matching(tmp_path: Path) -> None:
    page = "<main><p>through Student Success Connect:</p></main>"
    corpus = make_corpus(tmp_path / "corpus", {"advising-faq": page})
    example = make_example("advising-001", ("advising-faq", "through Student Success Connect."))

    problems = check_evidence([example], load_corpus_text(corpus))

    expected = (
        "quote not found: the first 31 of 32 characters match; "
        "the quote continues '.', the page continues ':'"
    )
    assert problems == [EvidenceProblem("advising-001", 0, "advising-faq", expected)]


def test_quote_with_nothing_in_common_is_reported(tmp_path: Path) -> None:
    corpus = make_corpus(tmp_path / "corpus", {"short-page": "<main><p>abc</p></main>"})
    example = make_example("short-001", ("short-page", "xyz"))

    (problem,) = check_evidence([example], load_corpus_text(corpus))

    assert problem.message == "quote not found: no part of it appears in the document"


def test_unknown_doc_id_is_reported(tmp_path: Path) -> None:
    corpus = make_corpus(tmp_path / "corpus", {"advising-faq": ADVISING_PAGE})
    example = make_example("advising-001", ("missing-page", "Main Hall 208."))

    (problem,) = check_evidence([example], load_corpus_text(corpus))

    assert problem.doc_id == "missing-page"
    assert problem.message == "unknown doc_id: not in the corpus manifest"


def test_problem_points_at_the_failing_quote(tmp_path: Path) -> None:
    corpus = make_corpus(tmp_path / "corpus", {"advising-faq": ADVISING_PAGE})
    example = make_example(
        "advising-001",
        ("advising-faq", "Main Hall 208."),
        ("advising-faq", "It's open on weekends."),
    )

    (problem,) = check_evidence([example], load_corpus_text(corpus))

    assert problem.evidence_index == 1
    assert str(problem).startswith("advising-001 evidence[1] (advising-faq): quote not found")


def test_unanswerable_examples_have_nothing_to_check(tmp_path: Path) -> None:
    corpus = make_corpus(tmp_path / "corpus", {"advising-faq": ADVISING_PAGE})
    example = make_example("parking-001", category="unanswerable")

    assert check_evidence([example], load_corpus_text(corpus)) == []


def test_missing_text_file_is_a_corpus_error(tmp_path: Path) -> None:
    corpus = make_corpus(tmp_path / "corpus", {"advising-faq": ADVISING_PAGE})
    (corpus / "text" / "advising-faq.txt").unlink()

    with pytest.raises(CorpusError, match="no text for 'advising-faq'"):
        load_corpus_text(corpus)


def test_changed_text_file_is_a_corpus_error(tmp_path: Path) -> None:
    corpus = make_corpus(tmp_path / "corpus", {"advising-faq": ADVISING_PAGE})
    with (corpus / "text" / "advising-faq.txt").open("a", encoding="utf-8") as text:
        text.write("A line someone added by hand.\n")

    with pytest.raises(CorpusError, match="does not match the manifest hash"):
        load_corpus_text(corpus)


@pytest.mark.parametrize(
    ("quote", "document", "expected"),
    [("abcdef", "xxabcxx", 3), ("abc", "abc", 3), ("xyz", "abc", 0)],
)
def test_longest_matching_prefix(quote: str, document: str, expected: int) -> None:
    assert _longest_matching_prefix(quote, document) == expected


def write_dataset(path: Path, *examples: Example) -> Path:
    path.write_text("".join(example.model_dump_json() + "\n" for example in examples))
    return path


def test_main_exits_0_when_every_quote_is_verified(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = make_corpus(tmp_path / "corpus", {"advising-faq": ADVISING_PAGE})
    example = make_example("advising-001", ("advising-faq", "Main Hall 208."))
    dataset = write_dataset(tmp_path / "data.jsonl", example)

    assert main([str(dataset), str(corpus)]) == 0
    assert "all 1 evidence quote(s) in 1 example(s) verified" in capsys.readouterr().out


def test_main_exits_1_when_a_quote_is_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = make_corpus(tmp_path / "corpus", {"advising-faq": ADVISING_PAGE})
    example = make_example("advising-001", ("advising-faq", "Main Hall 209."))
    dataset = write_dataset(tmp_path / "data.jsonl", example)

    assert main([str(dataset), str(corpus)]) == 1
    assert "1 of 1 evidence quote(s) not verified" in capsys.readouterr().out


def test_main_exits_2_when_the_dataset_is_invalid(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = make_corpus(tmp_path / "corpus", {"advising-faq": ADVISING_PAGE})
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("{not json\n", encoding="utf-8")

    assert main([str(dataset), str(corpus)]) == 2
    assert "invalid JSON" in capsys.readouterr().err
