"""Confidence intervals for pass rates and for differences between two systems.

Both systems answer the same examples, so their scores come in pairs. Working with the
per-example differences (candidate minus baseline) cancels out how hard each example is,
so what remains is how the two systems actually differ.

A single system's pass rate gets a Wilson score interval instead. It is exact arithmetic,
needs no resampling, and stays sensible on small samples and at rates near 0% or 100%,
where the textbook "rate plus or minus 1.96 standard errors" interval breaks down.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import numpy.typing as npt

DEFAULT_RESAMPLES = 10_000
# Resamples are drawn in chunks, so memory stays bounded however many are requested.
_CHUNK = 1_000


@dataclass(frozen=True)
class BootstrapResult:
    """A paired bootstrap confidence interval for the mean of candidate minus baseline."""

    difference: float
    low: float
    high: float
    confidence: float
    resamples: int

    @property
    def excludes_zero(self) -> bool:
        """Whether the whole interval lies on one side of zero."""
        return self.low > 0 or self.high < 0

    def __str__(self) -> str:
        level = f"{self.confidence:.0%}"
        return f"{self.difference:+.3f} [{self.low:+.3f}, {self.high:+.3f}] ({level} CI)"


def paired_bootstrap(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Return a percentile bootstrap interval for the mean of candidate minus baseline.

    `baseline[i]` and `candidate[i]` must be the two systems' scores on the same example.
    Each resample draws whole examples with replacement, keeping both scores together, so
    the pairing survives resampling. The seed makes the result reproducible.

    Measured coverage is close to the stated level from about 100 examples up, and falls
    far short of it on small samples (see `trustbench.coverage`).
    """
    differences = _paired_differences(baseline, candidate)
    if resamples < 100:
        raise ValueError(f"resamples must be at least 100, got {resamples}")
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be between 0 and 1, got {confidence}")

    rng = np.random.default_rng(seed)
    count = len(differences)
    means = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, _CHUNK):
        stop = min(start + _CHUNK, resamples)
        indices = rng.integers(0, count, size=(stop - start, count))
        means[start:stop] = differences[indices].mean(axis=1)

    tail = (1 - confidence) / 2
    low, high = np.quantile(means, [tail, 1 - tail])
    return BootstrapResult(
        difference=float(differences.mean()),
        low=float(low),
        high=float(high),
        confidence=confidence,
        resamples=resamples,
    )


def _paired_differences(
    baseline: Sequence[float], candidate: Sequence[float]
) -> npt.NDArray[np.float64]:
    """Validate two paired score lists and return candidate minus baseline."""
    if len(baseline) != len(candidate):
        raise ValueError(
            f"baseline has {len(baseline)} scores but candidate has {len(candidate)}; "
            "scores must be paired by example"
        )
    if len(baseline) == 0:
        raise ValueError("need at least one paired example")
    base = np.asarray(baseline, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    if not (np.isfinite(base).all() and np.isfinite(cand).all()):
        raise ValueError("scores must be finite numbers")
    return cand - base


@dataclass(frozen=True)
class WilsonResult:
    """A Wilson score confidence interval for a pass rate."""

    successes: int
    total: int
    low: float
    high: float
    confidence: float

    @property
    def rate(self) -> float:
        return self.successes / self.total

    def __str__(self) -> str:
        level = f"{self.confidence:.0%}"
        return f"{self.rate:.1%} [{self.low:.1%}, {self.high:.1%}] ({level} CI)"


def wilson_interval(successes: int, total: int, *, confidence: float = 0.95) -> WilsonResult:
    """Return the Wilson score interval for `successes` passes out of `total` tries.

    The interval is the set of true pass rates that a normal-approximation test would not
    reject given what was observed. Unlike "rate plus or minus 1.96 standard errors", it
    never leaves [0, 1], it is not zero-width at 0% or 100%, and its coverage stays close
    to the stated level even with a few dozen tries.
    """
    if total < 1:
        raise ValueError(f"total must be at least 1, got {total}")
    if not 0 <= successes <= total:
        raise ValueError(f"successes must be between 0 and {total}, got {successes}")
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be between 0 and 1, got {confidence}")

    z = NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    rate = successes / total
    shrink = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / shrink
    half_width = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / shrink
    # In exact arithmetic 0 of n starts at 0 and n of n ends at 1, but floating point can
    # land a hair off (1e-17 instead of 0), so those two ends are set exactly.
    low = 0.0 if successes == 0 else max(0.0, center - half_width)
    high = 1.0 if successes == total else min(1.0, center + half_width)
    return WilsonResult(successes=successes, total=total, low=low, high=high, confidence=confidence)
