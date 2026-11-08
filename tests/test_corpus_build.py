"""Tests for building the corpus: the source list, extracted text files, and the manifest."""

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from trustbench.corpus import (
    ManifestEntry,
    Source,
    build_corpus,
    main,
    read_manifest,
    read_sources,
    text_sha256,
)

FETCHED = datetime(2026, 11, 8, 3, 42, 0, tzinfo=UTC)


def make_corpus(root: Path, pages: dict[str, str]) -> Path:
    """Create a corpus folder with sources.txt and one saved page per doc id."""
    raw = root / "raw"
    raw.mkdir(parents=True)
    lines = [f"{doc_id} https://example.edu/{doc_id}" for doc_id in pages]
    (root / "sources.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for doc_id, html in pages.items():
        page = raw / f"{doc_id}.html"
        page.write_text(html, encoding="utf-8")
        os.utime(page, (FETCHED.timestamp(), FETCHED.timestamp()))
    return root


def write_sources(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "sources.txt"
    path.write_text(content, encoding="utf-8")
    return path


def test_read_sources_skips_blank_lines_and_comments(tmp_path: Path) -> None:
    path = write_sources(tmp_path, "# pages\n\nadvising-faq https://example.edu/advising\n")

    assert read_sources(path) == [Source("advising-faq", "https://example.edu/advising")]


def test_read_sources_rejects_a_line_without_a_url(tmp_path: Path) -> None:
    path = write_sources(tmp_path, "advising-faq https://example.edu/a\nlonely-id\n")

    with pytest.raises(ValueError, match=r"sources\.txt:2: expected 'doc_id url'"):
        read_sources(path)


def test_read_sources_rejects_an_invalid_doc_id(tmp_path: Path) -> None:
    path = write_sources(tmp_path, "Advising_FAQ https://example.edu/a\n")

    with pytest.raises(ValueError, match="invalid doc_id 'Advising_FAQ'"):
        read_sources(path)


def test_read_sources_rejects_a_url_without_http(tmp_path: Path) -> None:
    path = write_sources(tmp_path, "advising-faq ftp://example.edu/a\n")

    with pytest.raises(ValueError, match="URL must start with"):
        read_sources(path)


def test_read_sources_rejects_duplicate_doc_ids(tmp_path: Path) -> None:
    content = "advising-faq https://example.edu/a\nadvising-faq https://example.edu/b\n"
    path = write_sources(tmp_path, content)

    with pytest.raises(ValueError, match=r"duplicate doc_id 'advising-faq' \(first on line 1\)"):
        read_sources(path)


def test_build_writes_text_files_and_the_manifest(tmp_path: Path) -> None:
    pages = {"advising-faq": "<main><p>Main Hall 208.</p></main>"}
    corpus = make_corpus(tmp_path / "corpus", pages)

    entries = build_corpus(corpus)

    text = (corpus / "text" / "advising-faq.txt").read_text(encoding="utf-8")
    expected = ManifestEntry(
        doc_id="advising-faq",
        url="https://example.edu/advising-faq",
        fetched_at=FETCHED,
        sha256=text_sha256(text),
        characters=len(text),
    )
    assert text == "Main Hall 208.\n"
    assert entries == [expected]
    assert read_manifest(corpus / "manifest.jsonl") == entries


def test_rebuilding_the_same_pages_gives_the_same_manifest(tmp_path: Path) -> None:
    pages = {"a-page": "<main><p>One.</p></main>", "b-page": "<main><p>Two.</p></main>"}
    corpus = make_corpus(tmp_path / "corpus", pages)

    first = build_corpus(corpus)
    second = build_corpus(corpus)

    assert first == second


def test_build_fails_when_a_saved_page_is_missing(tmp_path: Path) -> None:
    corpus = make_corpus(tmp_path / "corpus", {"a-page": "<main><p>One.</p></main>"})
    with (corpus / "sources.txt").open("a", encoding="utf-8") as sources:
        sources.write("missing-page https://example.edu/missing\n")

    with pytest.raises(FileNotFoundError, match="no saved page for 'missing-page'"):
        build_corpus(corpus)


def test_build_fails_when_a_page_has_no_text(tmp_path: Path) -> None:
    corpus = make_corpus(tmp_path / "corpus", {"empty-page": "<main></main>"})

    with pytest.raises(ValueError, match="no text extracted"):
        build_corpus(corpus)


def test_main_builds_the_corpus_and_prints_a_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = make_corpus(tmp_path / "corpus", {"advising-faq": "<main><p>Text.</p></main>"})

    assert main(["build", str(corpus)]) == 0

    output = capsys.readouterr().out
    assert "advising-faq" in output
    assert "1 page(s) built" in output
