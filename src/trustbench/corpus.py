"""Extract normalized text from saved HTML pages for the evidence corpus.

Evidence quotes must match their source page exactly, but raw HTML hides text behind tags
(`C<span>ourse`), entities (`&amp;`, `&nbsp;`), and typographic characters (curly quotes).
This module turns a page into the plain text a reader sees, one block per line, and
defines the normalization used to compare quotes against documents.
"""

import hashlib
import re
import unicodedata
from collections.abc import Iterable

from bs4 import BeautifulSoup, Tag

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


def normalize_text(text: str) -> str:
    """Normalize text so that quotes can be matched exactly against documents.

    Applies Unicode NFKC (which turns non-breaking spaces into spaces), replaces curly
    quotes and long dashes with plain ASCII, and collapses every run of whitespace into a
    single space. Letter case and all other punctuation are kept as they are.
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

    lines = (normalize_text(line) for line in root.get_text().splitlines())
    kept = [line for line in lines if line]
    return "\n".join(kept) + "\n" if kept else ""


def text_sha256(text: str) -> str:
    """Return the SHA-256 hex digest of text encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tags(root: Tag, names: Iterable[str]) -> list[Tag]:
    """Return every descendant tag with one of the given names."""
    return [element for element in root.find_all(list(names)) if isinstance(element, Tag)]
