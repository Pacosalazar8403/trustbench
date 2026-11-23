"""Load human grades for a system TrustBench cannot call, from a CSV file.

Some systems, like a hosted chatbot, can only be tested by hand: a person asks each
question, reads the answer, and grades it in a spreadsheet. This module reads that
spreadsheet, exported as CSV, into validated records, so the rest of TrustBench can
summarize and compare human grades with the same statistics it uses for its own runs.

One row is one graded answer: the question ID, the language it was asked in, the test
round, the grade, how the answer's link checked out, and the question's topic. Problems
are collected for the whole file and reported together, each with its line number, the
same way the JSONL dataset loader does it.
"""

import csv
import io
from enum import StrEnum
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic_core import ErrorDetails

from trustbench.dataset import DatasetError, DatasetIssue
from trustbench.schema import ExampleId, NonEmptyText

COLUMNS = ("id", "language", "round", "grade", "source_check", "topic")


class Language(StrEnum):
    """The language a question was asked in."""

    ENGLISH = "English"
    SPANISH = "Spanish"


class Grade(StrEnum):
    """What a human grader decided about one answer."""

    CORRECT = "Correct"
    PARTLY_CORRECT = "Partly correct"
    WRONG = "Wrong"
    MADE_UP = "Made up an answer"
    CORRECTLY_ABSTAINED = "Correctly said it doesn't know"
    WRONGLY_ABSTAINED = "Said it doesn't know, but should have answered"

    @property
    def passed(self) -> bool:
        """Strict pass/fail for the significance tests.

        Only a fully correct answer, or a correct "I don't know", passes. A partly correct
        answer fails, because a student who acts on half the right answer can still get
        hurt. Reports still show all six grades, so nothing is hidden by this choice.
        """
        return self in (Grade.CORRECT, Grade.CORRECTLY_ABSTAINED)


class SourceCheck(StrEnum):
    """Whether the link the answer gave pointed at the right source."""

    RIGHT_PAGE = "Linked the right page"
    OTHER_UCCS_PAGE = "Linked another UCCS page"
    WRONG_PAGE = "Linked a wrong page"
    NO_LINK = "No link"


class GradedAnswer(BaseModel):
    """One answer from the system under test, as graded by a human."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: ExampleId
    language: Language
    round: Annotated[int, Field(ge=1)]
    grade: Grade
    source_check: SourceCheck
    topic: NonEmptyText

    @property
    def passed(self) -> bool:
        return self.grade.passed


def load_graded(path: Path) -> tuple[GradedAnswer, ...]:
    """Load and validate every graded answer in a CSV file.

    The first row must name exactly the columns in COLUMNS, in any order. Blank rows are
    skipped. Each (id, language, round) may appear only once, and every row for one
    question ID must give it the same topic.

    Raises:
        FileNotFoundError: if the file does not exist.
        DatasetError: if anything in the file is invalid, listing every problem found.
    """
    try:
        # utf-8-sig drops the byte order mark that Excel puts at the start of CSV exports.
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        issue = DatasetIssue(None, f"file is not valid UTF-8: {exc}")
        raise DatasetError(path, [issue]) from exc

    reader = csv.reader(io.StringIO(text))
    header = next(reader, None)
    if header is None:
        raise DatasetError(path, [DatasetIssue(None, "file is empty")])
    names = [name.strip() for name in header]
    header_issues = _check_header(names)
    if header_issues:
        raise DatasetError(path, header_issues)

    answers: list[GradedAnswer] = []
    issues: list[DatasetIssue] = []
    first_seen: dict[tuple[str, Language, int], int] = {}
    topics: dict[str, tuple[str, int]] = {}

    for row in reader:
        line = reader.line_num
        if not any(cell.strip() for cell in row):
            continue
        if len(row) != len(names):
            issues.append(DatasetIssue(line, f"expected {len(names)} fields, got {len(row)}"))
            continue

        data = {name: _clean(cell) for name, cell in zip(names, row, strict=True)}
        try:
            answer = GradedAnswer.model_validate(data)
        except ValidationError as exc:
            for error in exc.errors():
                issues.append(DatasetIssue(line, _describe(error)))
            continue

        key = (answer.id, answer.language, answer.round)
        first = first_seen.get(key)
        if first is not None:
            message = (
                f"duplicate {answer.id!r} in {answer.language} round {answer.round} "
                f"(first seen on line {first})"
            )
            issues.append(DatasetIssue(line, message))
            continue

        known = topics.get(answer.id)
        if known is not None and known[0] != answer.topic:
            message = (
                f"topic {answer.topic!r} for {answer.id!r} does not match "
                f"{known[0]!r} on line {known[1]}"
            )
            issues.append(DatasetIssue(line, message))
            continue

        first_seen[key] = line
        topics.setdefault(answer.id, (answer.topic, line))
        answers.append(answer)

    if not answers and not issues:
        issues.append(DatasetIssue(None, "file has no graded answers"))
    if issues:
        raise DatasetError(path, issues)
    return tuple(answers)


def _check_header(names: list[str]) -> list[DatasetIssue]:
    """Report missing, unknown, and repeated columns in the header row."""
    issues = []
    missing = [column for column in COLUMNS if column not in names]
    unknown = [name for name in names if name not in COLUMNS]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if missing:
        issues.append(DatasetIssue(1, f"missing columns: {', '.join(missing)}"))
    if unknown:
        issues.append(DatasetIssue(1, f"unknown columns: {', '.join(unknown)}"))
    if repeated:
        issues.append(DatasetIssue(1, f"repeated columns: {', '.join(repeated)}"))
    return issues


def _clean(cell: str) -> str:
    """Trim a cell and undo the curly apostrophe that spreadsheets swap in for "'"."""
    return cell.strip().replace("\N{RIGHT SINGLE QUOTATION MARK}", "'")


def _describe(error: ErrorDetails) -> str:
    """Turn one Pydantic error into 'column: message'."""
    location = ".".join(str(part) for part in error["loc"]) or "row"
    return f"{location}: {error['msg']}"
