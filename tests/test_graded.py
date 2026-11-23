from pathlib import Path

import pytest

from trustbench.dataset import DatasetError
from trustbench.graded import (
    Grade,
    GradedAnswer,
    Language,
    SourceCheck,
    load_graded,
    summarize,
    summary_report,
)

EXAMPLE = Path(__file__).resolve().parent.parent / "data" / "graded_example.csv"
HEADER = "id,language,round,grade,source_check,topic\n"


def write(tmp_path: Path, text: str, encoding: str = "utf-8") -> Path:
    path = tmp_path / "grades.csv"
    path.write_text(text, encoding=encoding)
    return path


def problems(path: Path) -> list[str]:
    with pytest.raises(DatasetError) as caught:
        load_graded(path)
    return [str(issue) for issue in caught.value.issues]


def test_example_file_loads() -> None:
    answers = load_graded(EXAMPLE)
    assert len(answers) == 24
    assert {answer.language for answer in answers} == set(Language)
    assert {answer.round for answer in answers} == {1, 2}


def test_example_file_covers_every_grade() -> None:
    assert {answer.grade for answer in load_graded(EXAMPLE)} == set(Grade)


def test_only_correct_and_correct_abstain_pass() -> None:
    passing = {grade for grade in Grade if grade.passed}
    assert passing == {Grade.CORRECT, Grade.CORRECTLY_ABSTAINED}


def test_reads_one_row(tmp_path: Path) -> None:
    path = write(tmp_path, HEADER + "q01,Spanish,2,Wrong,Linked a wrong page,parking\n")
    (answer,) = load_graded(path)
    assert answer.id == "q01"
    assert answer.language is Language.SPANISH
    assert answer.round == 2
    assert answer.grade is Grade.WRONG
    assert answer.source_check is SourceCheck.WRONG_PAGE
    assert answer.topic == "parking"
    assert not answer.passed


def test_columns_can_come_in_any_order(tmp_path: Path) -> None:
    text = "topic,grade,id,round,source_check,language\nparking,Correct,q01,1,No link,English\n"
    (answer,) = load_graded(write(tmp_path, text))
    assert answer.passed


def test_quoted_label_with_a_comma(tmp_path: Path) -> None:
    row = 'q01,English,1,"Said it doesn\'t know, but should have answered",No link,parking\n'
    (answer,) = load_graded(write(tmp_path, HEADER + row))
    assert answer.grade is Grade.WRONGLY_ABSTAINED


def test_excel_byte_order_mark_is_ignored(tmp_path: Path) -> None:
    path = write(tmp_path, HEADER + "q01,English,1,Correct,No link,parking\n", "utf-8-sig")
    assert len(load_graded(path)) == 1


def test_curly_apostrophe_and_spaces_are_cleaned(tmp_path: Path) -> None:
    curly = "\N{RIGHT SINGLE QUOTATION MARK}"
    row = f"q01, English ,1,Correctly said it doesn{curly}t know,No link,parking\n"
    (answer,) = load_graded(write(tmp_path, HEADER + row))
    assert answer.grade is Grade.CORRECTLY_ABSTAINED
    assert answer.language is Language.ENGLISH


def test_blank_rows_are_skipped(tmp_path: Path) -> None:
    text = HEADER + "\n,,,,,\nq01,English,1,Correct,No link,parking\n\n"
    assert len(load_graded(write(tmp_path, text))) == 1


def test_missing_and_unknown_columns(tmp_path: Path) -> None:
    text = "id,language,round,grade,topic,answer\nq01,English,1,Correct,parking,secret\n"
    assert problems(write(tmp_path, text)) == [
        "line 1: missing columns: source_check",
        "line 1: unknown columns: answer",
    ]


def test_repeated_column(tmp_path: Path) -> None:
    text = HEADER.strip() + ",topic\n"
    assert problems(write(tmp_path, text)) == ["line 1: repeated columns: topic"]


def test_bad_values_are_all_reported_with_line_numbers(tmp_path: Path) -> None:
    text = HEADER + "q01,English,1,Correct,No link,parking\nQ1,english,0,Great,Maybe,\n"
    found = problems(write(tmp_path, text))
    columns = ["id", "language", "round", "grade", "source_check", "topic"]
    assert len(found) == len(columns)
    for issue, column in zip(found, columns, strict=True):
        assert issue.startswith(f"line 3: {column}:")


def test_round_must_be_a_whole_number(tmp_path: Path) -> None:
    text = HEADER + "q01,English,first,Correct,No link,parking\n"
    (issue,) = problems(write(tmp_path, text))
    assert issue.startswith("line 2: round:")


def test_wrong_number_of_fields(tmp_path: Path) -> None:
    text = HEADER + "q01,English,1\n"
    assert problems(write(tmp_path, text)) == ["line 2: expected 6 fields, got 3"]


def test_duplicate_id_language_round(tmp_path: Path) -> None:
    row = "q01,English,1,Correct,No link,parking\n"
    assert problems(write(tmp_path, HEADER + row + row)) == [
        "line 3: duplicate 'q01' in English round 1 (first seen on line 2)"
    ]


def test_same_id_in_another_language_or_round_is_fine(tmp_path: Path) -> None:
    text = (
        HEADER
        + "q01,English,1,Correct,No link,parking\n"
        + "q01,Spanish,1,Correct,No link,parking\n"
        + "q01,English,2,Correct,No link,parking\n"
    )
    assert len(load_graded(write(tmp_path, text))) == 3


def test_topic_must_match_for_one_id(tmp_path: Path) -> None:
    text = (
        HEADER
        + "q01,English,1,Correct,No link,parking\n"
        + "q01,Spanish,1,Correct,No link,tuition\n"
    )
    assert problems(write(tmp_path, text)) == [
        "line 3: topic 'tuition' for 'q01' does not match 'parking' on line 2"
    ]


def test_header_only(tmp_path: Path) -> None:
    assert problems(write(tmp_path, HEADER)) == ["file has no graded answers"]


def test_empty_file(tmp_path: Path) -> None:
    assert problems(write(tmp_path, "")) == ["file is empty"]


def test_not_utf8(tmp_path: Path) -> None:
    path = tmp_path / "grades.csv"
    path.write_bytes(HEADER.encode() + b"q01,Espa\xf1ol,1,Correct,No link,parking\n")
    (issue,) = problems(path)
    assert issue.startswith("file is not valid UTF-8")


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_graded(tmp_path / "nope.csv")


# Summary


def graded(
    counts: dict[Grade, int], language: Language, round_number: int = 1
) -> list[GradedAnswer]:
    """Make answers with the given number of each grade, one question ID per answer."""
    answers: list[GradedAnswer] = []
    for grade, count in counts.items():
        for _ in range(count):
            answers.append(
                GradedAnswer(
                    id=f"q{len(answers) + 1:02d}",
                    language=language,
                    round=round_number,
                    grade=grade,
                    source_check=SourceCheck.NO_LINK,
                    topic="any",
                )
            )
    return answers


def test_example_file_summary() -> None:
    summaries = summarize(load_graded(EXAMPLE))

    assert [(s.round, s.language) for s in summaries] == [
        (1, Language.ENGLISH),
        (1, Language.SPANISH),
        (2, Language.ENGLISH),
        (2, Language.SPANISH),
    ]
    assert [s.pass_rate.successes for s in summaries] == [4, 2, 6, 4]
    assert all(s.total == 6 for s in summaries)


def test_every_grade_and_source_is_counted_even_when_zero() -> None:
    for summary in summarize(load_graded(EXAMPLE)):
        assert list(summary.grades) == list(Grade)
        assert list(summary.sources) == list(SourceCheck)
        assert sum(summary.grades.values()) == summary.total
        assert sum(summary.sources.values()) == summary.total


def test_partly_correct_fails() -> None:
    answers = graded({Grade.CORRECT: 1, Grade.PARTLY_CORRECT: 3}, Language.ENGLISH)
    (summary,) = summarize(answers)
    assert (summary.pass_rate.successes, summary.total) == (1, 4)


def test_round_one_sanity_check() -> None:
    english = graded(
        {
            Grade.CORRECT: 45,
            Grade.PARTLY_CORRECT: 7,
            Grade.CORRECTLY_ABSTAINED: 2,
            Grade.WRONGLY_ABSTAINED: 6,
        },
        Language.ENGLISH,
    )
    spanish = graded(
        {
            Grade.CORRECT: 35,
            Grade.PARTLY_CORRECT: 9,
            Grade.WRONG: 2,
            Grade.CORRECTLY_ABSTAINED: 3,
            Grade.WRONGLY_ABSTAINED: 11,
        },
        Language.SPANISH,
    )

    first, second = summarize(spanish + english)

    assert first.language is Language.ENGLISH
    assert str(first.pass_rate) == "78.3% [66.4%, 86.9%] (95% CI)"
    assert first.grades[Grade.MADE_UP] == 0
    assert str(second.pass_rate) == "63.3% [50.7%, 74.4%] (95% CI)"
    assert second.grades[Grade.WRONGLY_ABSTAINED] == 11


def test_groups_come_out_in_order_whatever_the_input_order() -> None:
    answers = load_graded(EXAMPLE)
    assert summarize(reversed(answers)) == summarize(answers)


def test_confidence_is_passed_to_the_interval() -> None:
    (summary,) = summarize(graded({Grade.CORRECT: 5}, Language.SPANISH), confidence=0.9)
    assert summary.pass_rate.confidence == 0.9


def test_summary_needs_answers() -> None:
    with pytest.raises(ValueError, match="at least one graded answer"):
        summarize([])


def test_summary_report() -> None:
    lines = summary_report(summarize(load_graded(EXAMPLE)))

    assert lines[:3] == [
        "round 1, English: 6 answers",
        "  pass rate  66.7% [30.0%, 90.3%] (95% CI)",
        "  grades",
    ]
    assert lines[3].split() == ["Correct", "3", "50.0%"]
    labels = [line.strip().rsplit(maxsplit=2)[0] for line in lines if line.startswith("    ")]
    assert labels == [*Grade, *SourceCheck] * 4
    assert lines[-1] == (
        "pass = Correct or Correctly said it doesn't know; Partly correct counts as a fail"
    )
