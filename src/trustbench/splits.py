"""Assign dataset examples to a fixed dev split and a held-out test split.

Each group (an example's `group`, or its own id if it has none) is hashed together with a
fixed salt, and the hash decides the split. That makes the split:

- Deterministic: the same group lands in the same split on every machine, every run.
- Stable: adding or removing other examples never moves an existing example.
- Group-aware: all examples in a group land together, so near-copies cannot leak from
  test into dev.

Changing the salt or the test fraction would reshuffle everything, so both are fixed
here, and a test pins known assignments.
"""

import argparse
import hashlib
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from trustbench.corpus import normalize_text
from trustbench.dataset import DatasetError, dataset_version, load_examples
from trustbench.schema import Category, Example

TEST_FRACTION = 0.4
SPLIT_SALT = "trustbench-split-v1"


class Split(StrEnum):
    """Dev is for building and tuning systems; test is held out for final claims."""

    DEV = "dev"
    TEST = "test"


@dataclass(frozen=True)
class EvidenceConflict:
    """One quote that examples in different groups share."""

    doc_id: str
    quote: str
    example_ids: tuple[str, ...]
    groups: tuple[str, ...]

    def __str__(self) -> str:
        examples = ", ".join(self.example_ids)
        groups = ", ".join(self.groups)
        quoted = f"{self.doc_id}: {self.quote[:60]!r}"
        return f"{quoted} is quoted by {examples} in different groups ({groups})"


def group_key(example: Example) -> str:
    """Return the example's group, or its own id if it has no group."""
    return example.group or example.id


def assign_split(group: str) -> Split:
    """Decide a group's split from a salted SHA-256 hash of its name."""
    digest = hashlib.sha256(f"{SPLIT_SALT}:{group}".encode()).digest()
    position = int.from_bytes(digest[:8], "big") / 2**64
    return Split.TEST if position < TEST_FRACTION else Split.DEV


def split_examples(examples: Iterable[Example]) -> dict[Split, tuple[Example, ...]]:
    """Return each split's examples, keeping their original order."""
    buckets: dict[Split, list[Example]] = {split: [] for split in Split}
    for example in examples:
        buckets[assign_split(group_key(example))].append(example)
    return {split: tuple(members) for split, members in buckets.items()}


def find_evidence_conflicts(examples: Iterable[Example]) -> list[EvidenceConflict]:
    """Return every quote that is shared by examples in different groups."""
    quoted_by: dict[tuple[str, str], list[Example]] = defaultdict(list)
    for example in examples:
        for evidence in example.evidence:
            quoted_by[(evidence.doc_id, normalize_text(evidence.quote))].append(example)

    conflicts: list[EvidenceConflict] = []
    for (doc_id, quote), sharing in quoted_by.items():
        groups = sorted({group_key(example) for example in sharing})
        if len(groups) > 1:
            example_ids = tuple(sorted({example.id for example in sharing}))
            conflicts.append(EvidenceConflict(doc_id, quote, example_ids, tuple(groups)))
    return sorted(conflicts, key=lambda conflict: (conflict.doc_id, conflict.quote))


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point: `python -m trustbench.splits <dataset>`.

    Prints the dataset version and its split, then any evidence conflicts. Exit codes: 0 if
    there are no conflicts, 1 if there are, and 2 if the dataset could not be loaded.
    """
    parser = argparse.ArgumentParser(
        prog="python -m trustbench.splits",
        description="Show the dataset version, its dev/test split, and evidence conflicts.",
    )
    parser.add_argument("dataset", type=Path, help="JSONL dataset file")
    args = parser.parse_args(argv)

    try:
        examples = load_examples(args.dataset)
    except (DatasetError, FileNotFoundError) as exc:
        print(exc, file=sys.stderr)
        return 2

    splits = split_examples(examples)
    dev, test = splits[Split.DEV], splits[Split.TEST]
    print(f"dataset   {args.dataset}")
    print(f"version   {dataset_version(examples)}")
    print(f"examples  {len(examples)} (dev {len(dev)}, test {len(test)})")
    print()
    print(f"{'category':<16}{'dev':>5}{'test':>6}")
    counts = Counter(
        (example.category, split) for split, members in splits.items() for example in members
    )
    for category in Category:
        in_dev, in_test = counts[(category, Split.DEV)], counts[(category, Split.TEST)]
        print(f"{category.value:<16}{in_dev:>5}{in_test:>6}")

    conflicts = find_evidence_conflicts(examples)
    if conflicts:
        print()
        for conflict in conflicts:
            print(conflict)
        print(f"{len(conflicts)} conflict(s): examples that share evidence must share a group")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
