"""Load dataset examples from a JSONL file, and identify a dataset by its content.

Each non-blank line holds one JSON object that must validate as an Example. Problems are
collected for the whole file and reported together, each with its line number, so a
dataset author can fix every mistake in one pass.
"""

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError
from pydantic_core import ErrorDetails

from trustbench.schema import Example


@dataclass(frozen=True)
class DatasetIssue:
    """One problem found in a dataset file. `line` is None for whole-file problems."""

    line: int | None
    message: str

    def __str__(self) -> str:
        if self.line is None:
            return self.message
        return f"line {self.line}: {self.message}"


class DatasetError(Exception):
    """Raised when a dataset file has one or more problems."""

    def __init__(self, path: Path, issues: list[DatasetIssue]) -> None:
        self.path = path
        self.issues = tuple(issues)
        count = len(self.issues)
        noun = "problem" if count == 1 else "problems"
        details = "\n".join(f"  {issue}" for issue in self.issues)
        super().__init__(f"{path}: {count} {noun}\n{details}")


def load_examples(path: Path) -> tuple[Example, ...]:
    """Load and validate every example in a JSONL dataset file.

    Raises:
        FileNotFoundError: if the file does not exist.
        DatasetError: if anything in the file is invalid, listing every problem found.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        issue = DatasetIssue(None, f"file is not valid UTF-8: {exc}")
        raise DatasetError(path, [issue]) from exc

    examples: list[Example] = []
    issues: list[DatasetIssue] = []
    first_seen: dict[str, int] = {}

    for line_number, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            issues.append(DatasetIssue(line_number, f"invalid JSON: {exc.msg}"))
            continue

        if not isinstance(data, dict):
            kind = type(data).__name__
            issues.append(DatasetIssue(line_number, f"expected a JSON object, got {kind}"))
            continue

        try:
            example = Example.model_validate(data)
        except ValidationError as exc:
            for error in exc.errors():
                issues.append(DatasetIssue(line_number, _describe(error)))
            continue

        first = first_seen.get(example.id)
        if first is not None:
            message = f"duplicate id {example.id!r} (first seen on line {first})"
            issues.append(DatasetIssue(line_number, message))
            continue

        first_seen[example.id] = line_number
        examples.append(example)

    if not examples and not issues:
        issues.append(DatasetIssue(None, "dataset has no examples"))
    if issues:
        raise DatasetError(path, issues)
    return tuple(examples)


def dataset_version(examples: Iterable[Example]) -> str:
    """Return a short content hash that identifies exactly this set of examples.

    Examples are sorted by id and serialized canonically, so the version depends only on
    the examples themselves, not on line order, key order, or whitespace in the file.
    The first 16 hex characters (64 bits) of the SHA-256 digest are plenty to tell
    versions apart while staying readable in logs and reports.
    """
    ordered = sorted(examples, key=lambda example: example.id)
    canonical = "\n".join(example.model_dump_json() for example in ordered)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _describe(error: ErrorDetails) -> str:
    """Turn one Pydantic error into 'field.path: message'."""
    location = ".".join(str(part) for part in error["loc"]) or "example"
    return f"{location}: {error['msg']}"
