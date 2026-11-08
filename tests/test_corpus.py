"""Tests for corpus text extraction and the normalization used to match quotes."""

from trustbench.corpus import extract_text, normalize_text, text_sha256


def test_normalize_turns_non_breaking_spaces_into_spaces() -> None:
    assert normalize_text("Bursar\u00a0approval") == "Bursar approval"


def test_normalize_replaces_curly_quotes_with_straight_quotes() -> None:
    text = "\u2018W\u2019 and \u201cquoted\u201d"

    assert normalize_text(text) == "'W' and \"quoted\""


def test_normalize_replaces_long_dashes_with_hyphens() -> None:
    assert normalize_text("3\u20135 days \u2014 roughly") == "3-5 days - roughly"


def test_normalize_collapses_whitespace() -> None:
    assert normalize_text("  line one\n\n\tline   two  ") == "line one line two"


def test_normalize_keeps_case_and_other_punctuation() -> None:
    assert normalize_text("Main Hall, Room 208.") == "Main Hall, Room 208."


def test_extract_keeps_a_word_split_by_an_inline_tag() -> None:
    html = "<main><p>C<span>ourse adds after this date</span> require approval.</p></main>"

    assert extract_text(html) == "Course adds after this date require approval.\n"


def test_extract_decodes_entities() -> None:
    html = "<main><p>academic dean, &amp; Bursar&nbsp;approval.</p></main>"

    assert extract_text(html) == "academic dean, & Bursar approval.\n"


def test_extract_puts_each_block_on_its_own_line() -> None:
    html = "<main><h2>Census Date</h2><p>First.</p><p>Second.</p></main>"

    assert extract_text(html) == "Census Date\nFirst.\nSecond.\n"


def test_extract_turns_line_breaks_into_new_lines() -> None:
    html = "<main><p>Address<br>Main Hall, Room 208</p></main>"

    assert extract_text(html) == "Address\nMain Hall, Room 208\n"


def test_extract_joins_table_cells_with_pipes() -> None:
    html = (
        "<main><table>"
        "<tr><td>February 3</td><td><strong>Due Date</strong> BEFORE 5pm</td></tr>"
        "<tr><td>February 3</td><td><p>Census</p><p>Last day to add classes.</p></td></tr>"
        "</table></main>"
    )

    assert extract_text(html) == (
        "February 3 | Due Date BEFORE 5pm\nFebruary 3 | Census Last day to add classes.\n"
    )


def test_extract_keeps_collapsed_content() -> None:
    html = (
        "<main><details><summary>How do class waitlists work?</summary>"
        "<div hidden><p>Students on the waitlist are automatically enrolled.</p></div>"
        "</details></main>"
    )

    assert extract_text(html) == (
        "How do class waitlists work?\nStudents on the waitlist are automatically enrolled.\n"
    )


def test_extract_drops_scripts_and_page_chrome() -> None:
    html = (
        "<body><nav>Menu</nav>"
        "<main><p>Real text.</p><script>var x = 1;</script></main>"
        "<footer>Contact us</footer></body>"
    )

    assert extract_text(html) == "Real text.\n"


def test_extract_falls_back_to_body_without_a_main_element() -> None:
    html = "<html><body><p>Only body text.</p></body></html>"

    assert extract_text(html) == "Only body text.\n"


def test_extract_returns_empty_text_for_an_empty_page() -> None:
    assert extract_text("<main></main>") == ""


def test_text_sha256_matches_the_known_digest() -> None:
    expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"

    assert text_sha256("abc") == expected
