"""Tests for the coverage study: the bootstrap's honesty, measured on simulated data."""

import numpy as np
import pytest

from trustbench.coverage import TRUE_DIFFERENCE, bootstrap_coverage, main, simulate_pair


def test_true_difference_follows_from_the_model() -> None:
    assert pytest.approx(0.03) == TRUE_DIFFERENCE


def test_simulated_systems_differ_by_about_the_true_difference() -> None:
    baseline, candidate = simulate_pair(100_000, np.random.default_rng(0))

    assert candidate.mean() - baseline.mean() == pytest.approx(TRUE_DIFFERENCE, abs=0.005)


def test_coverage_is_close_to_95_percent_on_a_large_sample() -> None:
    coverage = bootstrap_coverage(250, trials=300)

    assert 0.92 <= coverage <= 0.98


def test_coverage_falls_far_short_on_a_small_sample() -> None:
    # A known limitation, pinned on purpose: with 13 examples, many experiments contain no
    # example where the systems disagree, so the interval collapses to zero width.
    coverage = bootstrap_coverage(13, trials=300)

    assert coverage < 0.75


def test_main_prints_one_row_per_size(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--trials", "10"]) == 0

    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "true difference +0.030, nominal coverage 95%"
    assert len(lines) == 2 + 4
