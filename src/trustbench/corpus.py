"""Build the evidence corpus from saved HTML pages and PDF documents.

Evidence quotes must match their source exactly, but raw HTML hides text behind tags
(`C<span>ourse`), entities (`&amp;`, `&nbsp;`), and typographic characters (curly quotes),
and PDFs store text as positioned fragments rather than sentences. This module turns each
saved document into the plain text a reader sees, one block or line per line, and defines
the normalization used to compare quotes against documents.

A corpus folder holds `sources.txt` (one `doc_id url` per line) and `raw/<doc_id>.html`
(the downloaded documents). A saved file is read as a PDF when its content starts with the
PDF signature `%PDF-`, whatever its file name says, because a download script saves every
URL under the same extension. Building the corpus writes `text/<doc_id>.txt` and
`manifest.jsonl`, which records each document's URL, download time, and a SHA-256 hash of
its extracted text. Only `sources.txt` and the manifest are committed; anyone who downloads
the same documents can rebuild the text and compare hashes to see exactly which changed.
"""

import argparse
import hashlib
import io
import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from bs4 import BeautifulSoup, Tag
from pydantic import BaseModel, ConfigDict
from pypdf import PdfReader
from pypdf.errors import PyPdfError

SOURCES_FILE = "sources.txt"
MANIFEST_FILE = "manifest.jsonl"
RAW_DIR = "raw"
TEXT_DIR = "text"
PDF_SIGNATURE = b"%PDF-"

# Characters that look the same to a reader but are different bytes.
_TYPOGRAPHY = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
    }
)
_WHITESPACE = re.compile(r"\s+")
_DOC_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

# Elements that never contain page text a reader would quote.
_REMOVED = ["script", "style", "noscript", "template", "svg", "nav", "footer"]

# Elements that start a new line when a browser renders them.
_BLOCKS = [
    "address",
    "article",
    "aside",
    "blockquote",
    "dd",
    "details",
    "div",
    "dl",
    "dt",
    "figcaption",
    "figure",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "li",
    "ol",
    "p",
    "pre",
    "section",
    "summary",
    "table",
    "ul",
]


@dataclass(frozen=True)
class Source:
    """A document in the corpus: its id and the URL it was downloaded from."""

    doc_id: str
    url: str


class ManifestEntry(BaseModel):
    """One document in the corpus manifest."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    doc_id: str
    url: str
    fetched_at: datetime
    sha256: str
    characters: int


def normalize_text(text: str) -> str:
    """Normalize text so that quotes can be matched exactly against documents.

    Applies Unicode NFKC (which turns non-breaking spaces into spaces and typographic
    ligatures, common in PDFs, into separate letters), replaces curly quotes and long dashes
    with plain ASCII, and collapses every run of whitespace into a single space. Letter case
    and all other punctuation are kept as they are.
    """
    text = unicodedata.normalize("NFKC", text).translate(_TYPOGRAPHY)
    return _WHITESPACE.sub(" ", text).strip()


def extract_text(html: str) -> str:
    """Return the readable text of a page's main content, one block per line.

    Collapsed sections (accordions, hidden answers) are kept, because their text is in the
    HTML even when a browser does not show it. Inline tags never add spaces, so a word that
    markup splits, like `C<span>ourse`, stays one word.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in _tags(soup, _REMOVED):
        tag.decompose()

    main = soup.find("main")
    root = main if isinstance(main, Tag) else (soup.body or soup)

    for line_break in _tags(root, ["br"]):
        line_break.replace_with("\n")
    for block in _tags(root, _BLOCKS):
        block.insert_before("\n")
        block.insert_after("\n")
    for row in _tags(root, ["tr"]):
        cells = (normalize_text(cell.get_text()) for cell in _tags(row, ["td", "th"]))
        row.replace_with("\n" + " | ".join(cell for cell in cells if cell) + "\n")

    return _join_lines(root.get_text().splitlines())


def extract_pdf_text(data: bytes) -> str:
    """Return the text of every page of a PDF, in page order, one text line per line.

    A sentence that wraps onto the next line still matches a quote, because normalization
    turns the line break into a single space. A scanned PDF has no text layer, so it
    yields an empty string. Raises ValueError if the file cannot be read as a PDF.
    """
    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() for page in reader.pages]
    except PyPdfError as error:
        raise ValueError(f"could not read PDF: {error}") from error
    return _join_lines(line for page in pages for line in page.splitlines())


def is_pdf(data: bytes) -> bool:
    """Return True if the bytes are a PDF, judged by the signature at their start."""
    return data.lstrip().startswith(PDF_SIGNATURE)


def extract_document(data: bytes) -> str:
    """Extract text from a saved document, choosing PDF or HTML by its content."""
    if is_pdf(data):
        return extract_pdf_text(data)
    return extract_text(data.decode("utf-8"))


def text_sha256(text: str) -> str:
    """Return the SHA-256 hex digest of text encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_sources(path: Path) -> list[Source]:
    """Read `doc_id url` lines. Blank lines and lines starting with `#` are skipped."""
    sources: list[Source] = []
    first_seen: dict[str, int] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) != 2:
            raise ValueError(f"{path}:{number}: expected 'doc_id url', got {stripped!r}")
        doc_id, url = parts
        if not _DOC_ID.match(doc_id):
            raise ValueError(f"{path}:{number}: invalid doc_id {doc_id!r}")
        if not url.startswith(("https://", "http://")):
            raise ValueError(f"{path}:{number}: URL must start with http:// or https://")
        if doc_id in first_seen:
            first = first_seen[doc_id]
            raise ValueError(
                f"{path}:{number}: duplicate doc_id {doc_id!r} (first on line {first})"
            )
        first_seen[doc_id] = number
        sources.append(Source(doc_id, url))
    return sources


def build_corpus(corpus_dir: Path) -> list[ManifestEntry]:
    """Extract text for every source, write it to `text/`, and write the manifest."""
    sources = read_sources(corpus_dir / SOURCES_FILE)
    text_dir = corpus_dir / TEXT_DIR
    text_dir.mkdir(exist_ok=True)

    entries: list[ManifestEntry] = []
    for source in sources:
        raw_path = corpus_dir / RAW_DIR / f"{source.doc_id}.html"
        if not raw_path.exists():
            message = f"no saved page for {source.doc_id!r}: expected {raw_path}"
            raise FileNotFoundError(message)

        data = raw_path.read_bytes()
        try:
            text = extract_document(data)
        except ValueError as error:
            raise ValueError(f"{raw_path}: {error}") from error
        if not text:
            hint = " (a scanned PDF has no text layer)" if is_pdf(data) else ""
            raise ValueError(f"no text extracted from {raw_path}{hint}")

        text_path = text_dir / f"{source.doc_id}.txt"
        text_path.write_text(text, encoding="utf-8", newline="\n")
        fetched = datetime.fromtimestamp(raw_path.stat().st_mtime, tz=UTC).replace(microsecond=0)
        entries.append(
            ManifestEntry(
                doc_id=source.doc_id,
                url=source.url,
                fetched_at=fetched,
                sha256=text_sha256(text),
                characters=len(text),
            )
        )

    write_manifest(entries, corpus_dir / MANIFEST_FILE)
    return entries


def write_manifest(entries: Iterable[ManifestEntry], path: Path) -> None:
    """Write one JSON line per manifest entry."""
    lines = [entry.model_dump_json() for entry in entries]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def read_manifest(path: Path) -> list[ManifestEntry]:
    """Read a manifest written by `write_manifest`."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ManifestEntry.model_validate_json(line) for line in lines if line.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point: `python -m trustbench.corpus build <corpus_dir>`."""
    parser = argparse.ArgumentParser(
        prog="python -m trustbench.corpus",
        description="Build the evidence corpus from saved HTML pages and PDF documents.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="extract text and write the manifest")
    build.add_argument("corpus_dir", type=Path, help="folder with sources.txt and raw/")
    args = parser.parse_args(argv)

    entries = build_corpus(args.corpus_dir)
    for entry in entries:
        print(f"{entry.doc_id:40} {entry.characters:>7} chars  {entry.sha256[:12]}")
    print(f"{len(entries)} page(s) built into {args.corpus_dir / TEXT_DIR}")
    return 0


def _join_lines(lines: Iterable[str]) -> str:
    """Normalize each line, drop empty ones, and join the rest with a final newline."""
    kept = [normalized for line in lines if (normalized := normalize_text(line))]
    return "\n".join(kept) + "\n" if kept else ""


def _tags(root: Tag, names: Iterable[str]) -> list[Tag]:
    """Return every descendant tag with one of the given names."""
    return [element for element in root.find_all(list(names)) if isinstance(element, Tag)]


if __name__ == "__main__":
    raise SystemExit(main())
