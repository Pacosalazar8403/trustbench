"""Deterministic scoring of a system's answer against an example's reference answer.

These scorers are cheap, exact, and reproducible, so they run first. Each one either gives
a verdict or says it does not apply; nothing is guessed:

- **Abstention** (unanswerable examples): the answer passes if it says the documents do not
  contain the answer instead of inventing one.
- **Key facts** (answerable examples): every number, money amount, percentage, phone
  number, and email address in the reference answer must appear in the system's answer.
  Extra facts in the answer are never penalized, so an answer that adds a crisis line or a
  second contact still passes. That also means a made-up extra fact is not caught here.

Examples where neither scorer applies (an answerable reference with no extractable facts)
are left for a later judge rather than being forced into a verdict.

Run `python -m trustbench.scoring data/uccs.jsonl` to see how many examples each scorer can
judge, and to check that every reference answer passes its own scorer.
"""

import argparse
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from trustbench.dataset import DatasetError, load_examples
from trustbench.schema import Category, Example

_APOSTROPHES = str.maketrans({"\u2018": "'", "\u2019": "'"})

_ABSTENTION = re.compile(
    r"\b(?:documents?|polic(?:y|ies)|information|pages?|sources?)\b[^.]{0,80}?"
    r"(?:\b(?:don't|do not|doesn't|does not)\s+"
    r"(?:say|mention|cover|explain|describe|include|specify|state|address|contain)\b"
    r"|\bbut not (?:how|whether|what|where|when|why)\b)"
    r"|\bi (?:don't|do not) know\b"
    r"|\b(?:couldn't|could not|can't|cannot) find\b"
    r"|\bno information (?:about|on)\b",
    re.IGNORECASE,
)

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE = re.compile(r"\(?\b\d{3}\)?[\s.-]?\d{3}[\s.-]\d{4}\b")
_MONEY = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)")
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s?(?:%|percent\b)", re.IGNORECASE)
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


@dataclass(frozen=True)
class Score:
    """A scorer's verdict: passed, failed, or not applicable (`passed is None`)."""

    passed: bool | None
    reason: str

    @property
    def applies(self) -> bool:
        return self.passed is not None


def abstains(answer: str) -> bool:
    """Return True if the answer says the documents do not contain the answer."""
    return _ABSTENTION.search(answer.translate(_APOSTROPHES)) is not None


def extract_facts(text: str) -> frozenset[str]:
    """Return the checkable facts in text, each normalized so equal facts compare equal.

    Emails are lowercased, phone numbers reduced to their digits, money and percentages
    tagged with their unit, and plain numbers written in one canonical form (`2.0` and `2`
    are the same number, and so are `004509` and `4509`). Each kind is removed from the
    text once found, so the digits of a phone number are not counted again as numbers.
    """
    facts: set[str] = set()
    remaining = text

    def take(pattern: re.Pattern[str], make: Callable[[re.Match[str]], str]) -> None:
        nonlocal remaining
        facts.update(make(match) for match in pattern.finditer(remaining))
        remaining = pattern.sub(" ", remaining)

    take(_EMAIL, lambda match: "email:" + match.group(0).lower().rstrip("."))
    take(_PHONE, lambda match: "phone:" + re.sub(r"\D", "", match.group(0)))
    take(_MONEY, lambda match: _number("usd", match.group(1)))
    take(_PERCENT, lambda match: _number("pct", match.group(1)))
    take(_NUMBER, lambda match: _number("n", match.group(0)))
    return frozenset(facts)


def fact_score(reference: str, answer: str) -> Score:
    """Pass if every fact in the reference answer also appears in the system's answer."""
    expected = extract_facts(reference)
    if not expected:
        return Score(None, "no checkable facts in the reference answer")
    missing = expected - extract_facts(answer)
    if missing:
        return Score(False, "missing " + ", ".join(sorted(missing)))
    return Score(True, f"all {len(expected)} fact(s) present")


def score_answer(example: Example, answer: str) -> Score:
    """Score one answer: abstention for unanswerable examples, key facts otherwise."""
    if example.category is Category.UNANSWERABLE:
        if abstains(answer):
            return Score(True, "abstained on an unanswerable question")
        return Score(False, "answered a question the documents do not answer")
    return fact_score(example.reference_answer, answer)


def coverage(examples: Iterable[Example]) -> dict[str, int]:
    """Count how many examples each deterministic scorer can judge."""
    counts = {"abstention": 0, "key facts": 0, "needs a judge": 0}
    for example in examples:
        if example.category is Category.UNANSWERABLE:
            counts["abstention"] += 1
        elif extract_facts(example.reference_answer):
            counts["key facts"] += 1
        else:
            counts["needs a judge"] += 1
    return counts


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m trustbench.scoring",
        description="Show what the deterministic scorers can judge in a dataset.",
    )
    parser.add_argument("dataset", type=Path, help="JSONL dataset file")
    args = parser.parse_args(argv)

    try:
        examples = load_examples(args.dataset)
    except (DatasetError, FileNotFoundError) as error:
        print(f"error: {error}")
        return 2

    counts = coverage(examples)
    for name, count in counts.items():
        print(f"{name:14} {count:>4}  ({count / len(examples):.0%})")
    failures = [
        (example.id, score.reason)
        for example in examples
        if (score := score_answer(example, example.reference_answer)).passed is False
    ]
    for example_id, reason in failures:
        print(f"reference answer fails its own scorer: {example_id}: {reason}")
    if failures:
        return 1
    print(f"all {len(examples)} reference answers pass or are left for a judge")
    return 0


def _number(unit: str, digits: str) -> str:
    value = Decimal(digits.replace(",", ""))
    return f"{unit}:{format(value.normalize(), 'f')}"


if __name__ == "__main__":
    raise SystemExit(main())
