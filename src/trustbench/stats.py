"""Paired statistics for comparing a baseline system with a candidate on the same examples.

Both systems answer the same examples, so their scores come in pairs. Working with the
per-example differences (candidate minus baseline) cancels out how hard each example is,
so what remains is how the two systems actually differ.
"""

from collections.abc import Sequence
from dataclasses import dataclass

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
