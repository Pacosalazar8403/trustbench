"""Agreement between two raters' pass/fail verdicts on the same items, and Cohen's kappa.

Raw agreement overstates how well two raters agree when one verdict is common: if both
raters call 70% of answers correct at random, they still agree on 58% of them. Cohen's
kappa subtracts that chance agreement:

    kappa = (observed - expected) / (1 - expected)

where `expected` is the agreement two raters with the same pass rates would reach by
chance. Kappa is 1 for perfect agreement, 0 for agreement no better than chance, and
negative for agreement worse than chance.

This is how an LLM judge is checked before it is trusted: its verdicts are compared with
a human's, or with a deterministic scorer's, on the same answers.
"""

from collections.abc import Sequence
from dataclasses import dataclass

# Landis and Koch (1977): the usual, if arbitrary, names for ranges of kappa.
_KAPPA_LABELS = (
    (0.0, "poor"),
    (0.2, "slight"),
    (0.4, "fair"),
    (0.6, "moderate"),
    (0.8, "substantial"),
    (1.0, "almost perfect"),
)


@dataclass(frozen=True)
class Agreement:
    """A 2x2 confusion table of two raters' verdicts, with agreement statistics.

    `only_first` counts items the first rater passed and the second failed.
    """

    both_pass: int
    only_first: int
    only_second: int
    both_fail: int

    @property
    def total(self) -> int:
        return self.both_pass + self.only_first + self.only_second + self.both_fail

    @property
    def observed(self) -> float:
        """The share of items both raters gave the same verdict."""
        return (self.both_pass + self.both_fail) / self.total

    @property
    def expected(self) -> float:
        """The agreement expected by chance from the two raters' pass rates."""
        first = (self.both_pass + self.only_first) / self.total
        second = (self.both_pass + self.only_second) / self.total
        return first * second + (1 - first) * (1 - second)

    @property
    def kappa(self) -> float | None:
        """Cohen's kappa, or None when both raters gave every item the same verdict."""
        if self.expected == 1:
            return None
        return (self.observed - self.expected) / (1 - self.expected)

    def __str__(self) -> str:
        kappa = self.kappa
        strength = "undefined" if kappa is None else f"{kappa:.2f} ({describe_kappa(kappa)})"
        return (
            f"agree on {self.both_pass + self.both_fail} of {self.total} "
            f"({self.observed:.0%}), kappa {strength}"
        )


def agreement(first: Sequence[bool], second: Sequence[bool]) -> Agreement:
    """Count how two raters' verdicts on the same items agree and disagree."""
    if len(first) != len(second):
        raise ValueError(
            f"first rater has {len(first)} verdicts and second has {len(second)}; "
            "verdicts must be paired by item"
        )
    if not first:
        raise ValueError("need at least one item")
    pairs = list(zip(first, second, strict=True))
    return Agreement(
        both_pass=sum(1 for a, b in pairs if a and b),
        only_first=sum(1 for a, b in pairs if a and not b),
        only_second=sum(1 for a, b in pairs if b and not a),
        both_fail=sum(1 for a, b in pairs if not a and not b),
    )


def describe_kappa(kappa: float) -> str:
    """Name a kappa value on the Landis and Koch scale."""
    for upper, label in _KAPPA_LABELS:
        if kappa <= upper:
            return label
    return _KAPPA_LABELS[-1][1]
