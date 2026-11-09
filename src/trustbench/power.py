"""Power analysis for McNemar's exact test: what a dataset of a given size can detect.

A test that finds no significant difference has only proved something if it could have
found one. Power is the chance that the test reports a significant difference when a real
one of a given size exists. The minimum detectable drop is the smallest real fall in pass
rate that the test catches with the chosen power (80% by convention).

Power depends on how often the two systems disagree, not only on how many examples there
are, because McNemar's test looks only at disagreements. The disagreement rate is an
assumption here; real systems' rates must be measured.

Run `python -m trustbench.power` to print the minimum detectable drop for several dataset
sizes.
"""

import argparse
import math
from collections.abc import Sequence
from functools import cache

import numpy as np
import numpy.typing as npt

DEFAULT_DISAGREEMENT_RATE = 0.2
DEFAULT_POWER = 0.8
STUDY_SIZES = (13, 40, 100, 250, 500, 1000)
_STEP = 0.001

FloatArray = npt.NDArray[np.float64]


@cache
def critical_count(disagreements: int, alpha: float = 0.05) -> int:
    """Return the largest smaller-side count that McNemar's exact test calls significant.

    With `disagreements` split into wins and losses, the test rejects exactly when
    `min(wins, losses)` is at most this value. Returns -1 when no split can be significant.
    """
    _check_alpha(alpha)
    if disagreements < 0:
        raise ValueError("disagreements cannot be negative")
    total = 1 << disagreements
    term = 1
    tail = 0
    critical = -1
    for k in range(disagreements // 2 + 1):
        tail += term
        if 2 * tail / total > alpha:
            break
        critical = k
        term = term * (disagreements - k) // (k + 1)
    return critical


def mcnemar_power(examples: int, loss_rate: float, win_rate: float, alpha: float = 0.05) -> float:
    """Return the exact chance that McNemar's test reports a significant difference.

    Each example is independently a loss (only the baseline passes) with probability
    `loss_rate`, a win (only the candidate passes) with probability `win_rate`, and an
    agreement otherwise. The sum runs over every possible number of disagreements and every
    split of them into losses and wins, so no simulation is involved.
    """
    _check_alpha(alpha)
    if examples < 1:
        raise ValueError("need at least one example")
    if loss_rate < 0 or win_rate < 0 or loss_rate + win_rate > 1:
        raise ValueError("loss and win rates must be non-negative and sum to at most 1")
    disagree = loss_rate + win_rate
    if disagree == 0:
        return 0.0
    log_choose, reject = _tables(examples, alpha)
    counts = np.arange(examples + 1, dtype=np.float64)
    # Rows are the number of disagreements, columns the number of losses among them.
    losses = counts[np.newaxis, :]
    others = counts[:, np.newaxis] - losses
    share = loss_rate / disagree
    log_split = log_choose + _log_power(losses, share) + _log_power(others, 1 - share)
    split = np.where(reject, np.exp(log_split), 0.0)
    log_count = log_choose[examples] + _log_power(counts, disagree)
    log_count = log_count + _log_power(examples - counts, 1 - disagree)
    return min(1.0, float(np.exp(log_count) @ split.sum(axis=1)))


def minimum_detectable_drop(
    examples: int,
    disagreement_rate: float = DEFAULT_DISAGREEMENT_RATE,
    power: float = DEFAULT_POWER,
    alpha: float = 0.05,
) -> float | None:
    """Return the smallest pass-rate drop detected with at least `power`, or None.

    The drop is `loss_rate - win_rate` with their sum fixed at `disagreement_rate`. The
    search steps up from zero in increments of 0.001 rather than bisecting, because exact
    tests have small dips in power and a bisection could skip past the first drop that
    works. None means that no drop, up to every disagreement being a loss, reaches the
    power.
    """
    if not 0 < disagreement_rate <= 1:
        raise ValueError("disagreement rate must be above 0 and at most 1")
    if not 0 < power < 1:
        raise ValueError("power must be between 0 and 1")
    steps = round(disagreement_rate / _STEP)
    for step in range(steps + 1):
        drop = min(step * _STEP, disagreement_rate)
        loss_rate = (disagreement_rate + drop) / 2
        win_rate = disagreement_rate - loss_rate
        if mcnemar_power(examples, loss_rate, win_rate, alpha) >= power:
            return drop
    return None


@cache
def _tables(examples: int, alpha: float) -> tuple[FloatArray, npt.NDArray[np.bool_]]:
    """Log binomial coefficients and the rejection region, both indexed [count, losses]."""
    log_factorial = np.array([math.lgamma(n + 1) for n in range(examples + 1)])
    count = np.arange(examples + 1)[:, np.newaxis]
    losses = np.arange(examples + 1)[np.newaxis, :]
    valid = losses <= count
    others = np.where(valid, count - losses, 0)
    log_choose = np.where(
        valid, log_factorial[count] - log_factorial[losses] - log_factorial[others], -np.inf
    )
    critical = np.array([critical_count(n, alpha) for n in range(examples + 1)])
    reject = valid & (np.minimum(losses, others) <= critical[:, np.newaxis])
    return log_choose, reject


def _log_power(exponent: FloatArray, base: float) -> FloatArray:
    """exponent * log(base), with 0 * log(0) taken as 0 so certain events work."""
    if base > 0:
        return exponent * math.log(base)
    return np.where(exponent == 0, 0.0, -np.inf)


def _check_alpha(alpha: float) -> None:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Minimum detectable drop for McNemar's test.")
    parser.add_argument(
        "--disagreement-rate",
        type=float,
        default=DEFAULT_DISAGREEMENT_RATE,
        help="assumed share of examples where the two systems disagree (default 0.2)",
    )
    args = parser.parse_args(argv)
    rate: float = args.disagreement_rate
    print(f"disagreement rate {rate:.0%}, power {DEFAULT_POWER:.0%}, alpha 5% (two-sided)")
    print(f"{'examples':>8}  minimum detectable drop")
    for size in STUDY_SIZES:
        drop = minimum_detectable_drop(size, rate)
        shown = "not detectable" if drop is None else f"{drop * 100:.1f} points"
        print(f"{size:>8}  {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
