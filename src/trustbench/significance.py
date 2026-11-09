"""Significance tests for paired pass/fail results, and correction for testing many slices.

McNemar's exact test asks whether a candidate system changed the pass rate compared with a
baseline, using only the examples where the two systems disagree. Examples that both systems
pass, or both fail, say nothing about which system is better, so they are ignored.

When the same comparison is repeated on many slices of a dataset (one per category, say),
some slices will look significant by luck alone. Holm's method adjusts the p-values so the
chance of any false alarm across all slices stays at the chosen level.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class McNemarResult:
    """The outcome of McNemar's exact test.

    `wins` counts examples only the candidate passes, `losses` counts examples only the
    baseline passes, and `p_value` is the two-sided exact p-value.
    """

    wins: int
    losses: int
    p_value: float

    @property
    def disagreements(self) -> int:
        return self.wins + self.losses

    def __str__(self) -> str:
        return f"wins {self.wins}, losses {self.losses}, p = {self.p_value:.4f}"


def mcnemar_exact(baseline: Sequence[bool], candidate: Sequence[bool]) -> McNemarResult:
    """Run McNemar's exact two-sided test on paired pass/fail results.

    `baseline[i]` and `candidate[i]` must be the two systems' results on the same example.
    If the systems were equally good, each disagreement would be a win or a loss with equal
    chance, so the number of wins follows a Binomial(disagreements, 0.5) distribution. The
    p-value is the chance of a split at least as lopsided as the one observed.

    The arithmetic is exact (whole-number binomial coefficients), so the result is valid for
    any sample size, including very small ones.
    """
    if len(baseline) != len(candidate):
        raise ValueError(
            f"baseline has {len(baseline)} results and candidate has {len(candidate)}; "
            "results must be paired by example"
        )
    if not baseline:
        raise ValueError("need at least one paired example")
    for result in (*baseline, *candidate):
        if not isinstance(result, bool):
            raise ValueError("pass/fail results must be True or False")

    wins = sum(1 for old, new in zip(baseline, candidate, strict=True) if new and not old)
    losses = sum(1 for old, new in zip(baseline, candidate, strict=True) if old and not new)
    return McNemarResult(wins=wins, losses=losses, p_value=_two_sided_p(wins, losses))


def _two_sided_p(wins: int, losses: int) -> float:
    """Exact two-sided binomial p-value for a wins/losses split under a fair coin."""
    count = wins + losses
    if count == 0:
        return 1.0
    smaller = min(wins, losses)
    tail = sum(math.comb(count, k) for k in range(smaller + 1))
    return min(1.0, 2 * tail / (1 << count))


def holm_adjust(p_values: Sequence[float]) -> tuple[float, ...]:
    """Holm-adjust p-values for testing several hypotheses at once.

    Returns adjusted p-values in the same order as the input. A hypothesis is rejected at
    level alpha when its adjusted p-value is at most alpha, and the chance of rejecting any
    true hypothesis is then at most alpha, whatever the dependence between the tests.

    The smallest p-value is multiplied by the number of tests, the next by one fewer, and so
    on. A running maximum keeps the adjusted values in the same order as the raw ones, so a
    larger raw p-value never ends up with a smaller adjusted one.
    """
    for p in p_values:
        if not (math.isfinite(p) and 0.0 <= p <= 1.0):
            raise ValueError(f"p-values must be between 0 and 1, got {p}")

    count = len(p_values)
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [0.0] * count
    running_max = 0.0
    for rank, index in enumerate(order):
        scaled = min(1.0, (count - rank) * p_values[index])
        running_max = max(running_max, scaled)
        adjusted[index] = running_max
    return tuple(adjusted)
