"""Tests for reading PDF documents into the corpus, using small PDFs built in memory."""

from pathlib import Path

import pytest

from trustbench.corpus import (
    build_corpus,
    extract_document,
    extract_pdf_text,
    is_pdf,
    normalize_text,
    read_manifest,
    text_sha256,
)


def make_pdf(pages: list[list[str]]) -> bytes:
    """Build a minimal valid PDF with one Helvetica text line per string, one page per list.

    Writing the bytes by hand keeps the tests offline and free of extra dependencies: a PDF
    is a header, numbered objects (catalog, page tree, font, and per page a content stream
    and a page), a cross-reference table of byte offsets, and a trailer.
    """
    objects = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    page_ids = []
    for number, lines in enumerate(pages):
        content_id, page_id = 4 + 2 * number, 5 + 2 * number
        operators = ["BT", "/F1 12 Tf", "72 720 Td", "14 TL"]
        for line in lines:
            escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            operators.append(f"({escaped}) Tj T*")
        operators.append("ET")
        stream = "\n".join(operators).encode("latin-1")
        objects[content_id] = b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream)
        objects[page_id] = (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 3 0 R >> >> /Contents %d 0 R >>" % content_id
        )
        page_ids.append(page_id)
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects[2] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode()

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = {}
    for object_id in sorted(objects):
        offsets[object_id] = len(pdf)
        pdf += b"%d 0 obj\n%s\nendobj\n" % (object_id, objects[object_id])
    xref_offset = len(pdf)
    size = max(objects) + 1
    pdf += b"xref\n0 %d\n0000000000 65535 f \n" % size
    for object_id in range(1, size):
        pdf += b"%010d 00000 n \n" % offsets[object_id]
    pdf += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (size, xref_offset)
    return bytes(pdf)


def save_corpus(root: Path, doc_id: str, data: bytes) -> Path:
    """Create a corpus folder with one source, saved the way the download loop saves it."""
    raw = root / "raw"
    raw.mkdir(parents=True)
    (root / "sources.txt").write_text(
        f"{doc_id} https://example.edu/{doc_id}.pdf\n", encoding="utf-8"
    )
    (raw / f"{doc_id}.html").write_bytes(data)
    return root


def test_pdf_text_has_one_normalized_line_per_text_line() -> None:
    pdf = make_pdf([["Credit  Overload   Requests", "  Contact your advisor.  "]])

    assert extract_pdf_text(pdf) == "Credit Overload Requests\nContact your advisor.\n"


def test_pdf_pages_are_read_in_order() -> None:
    pdf = make_pdf([["Page one."], ["Page two (with parentheses)."]])

    assert extract_pdf_text(pdf) == "Page one.\nPage two (with parentheses).\n"


def test_a_sentence_wrapped_across_lines_still_matches_its_quote() -> None:
    pdf = make_pdf([["If a class will exceed your total", "allowed number of credits."]])

    text = normalize_text(extract_pdf_text(pdf))

    assert "will exceed your total allowed number of credits." in text


def test_documents_are_detected_by_content_not_name() -> None:
    assert is_pdf(make_pdf([["Text."]]))
    assert not is_pdf(b"<main><p>%PDF- is mentioned here.</p></main>")
    assert extract_document(b"<main><p>Main Hall 208.</p></main>") == "Main Hall 208.\n"


def test_build_reads_a_pdf_saved_under_an_html_name(tmp_path: Path) -> None:
    corpus = save_corpus(tmp_path / "corpus", "handbook", make_pdf([["Fall 2026 Handbook"]]))

    entries = build_corpus(corpus)

    text = (corpus / "text" / "handbook.txt").read_text(encoding="utf-8")
    assert text == "Fall 2026 Handbook\n"
    assert entries[0].sha256 == text_sha256(text)
    assert read_manifest(corpus / "manifest.jsonl") == entries


def test_build_explains_a_pdf_without_text(tmp_path: Path) -> None:
    corpus = save_corpus(tmp_path / "corpus", "scanned-form", make_pdf([[]]))

    with pytest.raises(ValueError, match=r"\(a scanned PDF has no text layer\)"):
        build_corpus(corpus)


def test_build_reports_a_damaged_pdf_with_its_file(tmp_path: Path) -> None:
    corpus = save_corpus(tmp_path / "corpus", "broken", b"%PDF-1.4\nthis is not a real PDF\n")

    with pytest.raises(ValueError, match=r"broken\.html: could not read PDF"):
        build_corpus(corpus)
