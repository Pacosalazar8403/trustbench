"""Measure how often the paired bootstrap's interval contains the true difference.

A "95% interval" is only honest if, across many repeated experiments, it contains the true
value about 95% of the time. This study simulates many baseline/candidate pairs from a
model where the true difference is known, and counts how often the interval covers it.

The model: the baseline passes each example with probability 0.6. Where the baseline
fails, the candidate passes with probability 0.15; where the baseline passes, the
candidate fails with probability 0.05. So the candidate is truly better by
0.4 * 0.15 - 0.6 * 0.05 = 0.03, and the two systems agree on most examples, as real
systems do.
"""

import argparse
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from trustbench.stats import paired_bootstrap

BASELINE_PASS = 0.6
CANDIDATE_GAIN = 0.15
CANDIDATE_LOSS = 0.05
TRUE_DIFFERENCE = (1 - BASELINE_PASS) * CANDIDATE_GAIN - BASELINE_PASS * CANDIDATE_LOSS
STUDY_SIZES = (13, 40, 100, 250)

Scores = npt.NDArray[np.float64]


def simulate_pair(count: int, rng: np.random.Generator) -> tuple[Scores, Scores]:
    """Simulate pass/fail scores for a baseline and a candidate on `count` examples."""
    baseline = (rng.random(count) < BASELINE_PASS).astype(np.float64)
    gains = (baseline == 0) & (rng.random(count) < CANDIDATE_GAIN)
    losses = (baseline == 1) & (rng.random(count) < CANDIDATE_LOSS)
    candidate = baseline.copy()
    candidate[gains] = 1.0
    candidate[losses] = 0.0
    return baseline, candidate


def bootstrap_coverage(
    count: int, trials: int, *, resamples: int = 2_000, seed: int = 12_345
) -> float:
    """Return the share of simulated experiments whose 95% interval covers the truth."""
    rng = np.random.default_rng(seed)
    covered = 0
    for trial in range(trials):
        baseline, candidate = simulate_pair(count, rng)
        result = paired_bootstrap(
            baseline.tolist(), candidate.tolist(), resamples=resamples, seed=trial
        )
        covered += result.low <= TRUE_DIFFERENCE <= result.high
    return covered / trials


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point: `python -m trustbench.coverage [--trials N]`."""
    parser = argparse.ArgumentParser(
        prog="python -m trustbench.coverage",
        description="Measure the paired bootstrap's coverage on simulated experiments.",
    )
    parser.add_argument("--trials", type=int, default=1_000, help="experiments per size")
    args = parser.parse_args(argv)

    print(f"true difference {TRUE_DIFFERENCE:+.3f}, nominal coverage 95%")
    print(f"{'examples':>8}  {'coverage':>8}")
    for count in STUDY_SIZES:
        print(f"{count:>8}  {bootstrap_coverage(count, args.trials):>8.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
