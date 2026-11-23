"""Tests for the paired bootstrap and the Wilson interval, checked against known answers."""

import math

import pytest

from trustbench.stats import BootstrapResult, paired_bootstrap, wilson_interval


def test_identical_systems_give_a_zero_width_interval_at_zero() -> None:
    scores = [1.0, 0.0, 1.0, 1.0, 0.0]

    result = paired_bootstrap(scores, scores)

    assert (result.difference, result.low, result.high) == (0.0, 0.0, 0.0)
    assert not result.excludes_zero


def test_a_constant_improvement_gives_a_zero_width_interval_at_that_value() -> None:
    baseline = [0.2, 0.5, 0.7, 0.1]
    candidate = [score + 0.1 for score in baseline]

    result = paired_bootstrap(baseline, candidate)

    assert result.difference == pytest.approx(0.1)
    assert result.low == pytest.approx(0.1)
    assert result.high == pytest.approx(0.1)


def test_interval_matches_the_normal_approximation_on_a_large_sample() -> None:
    # 400 differences, half 1 and half 0: mean 0.5, standard deviation 0.5, so the
    # standard error is 0.5 / sqrt(400) = 0.025 and a 95% interval is about 0.5 +/- 0.049.
    baseline = [0.0] * 400
    candidate = [1.0] * 200 + [0.0] * 200

    result = paired_bootstrap(baseline, candidate)

    half_width = 1.96 * 0.5 / math.sqrt(400)
    assert result.difference == pytest.approx(0.5)
    assert result.low == pytest.approx(0.5 - half_width, abs=0.005)
    assert result.high == pytest.approx(0.5 + half_width, abs=0.005)


def test_the_same_seed_gives_the_same_interval() -> None:
    baseline = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]
    candidate = [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0]

    first = paired_bootstrap(baseline, candidate, seed=7)
    second = paired_bootstrap(baseline, candidate, seed=7)

    assert first == second


def test_the_interval_contains_the_observed_difference() -> None:
    baseline = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]
    candidate = [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0]

    result = paired_bootstrap(baseline, candidate)

    assert result.low <= result.difference <= result.high


def test_higher_confidence_gives_a_wider_interval() -> None:
    baseline = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]
    candidate = [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]

    narrow = paired_bootstrap(baseline, candidate, confidence=0.80)
    wide = paired_bootstrap(baseline, candidate, confidence=0.99)

    assert wide.high - wide.low > narrow.high - narrow.low


def test_a_clear_improvement_excludes_zero() -> None:
    result = paired_bootstrap([0.0] * 40, [1.0] * 40)

    assert result.excludes_zero


def test_result_prints_as_difference_and_interval() -> None:
    result = BootstrapResult(
        difference=0.05, low=-0.01, high=0.11, confidence=0.95, resamples=10_000
    )

    assert str(result) == "+0.050 [-0.010, +0.110] (95% CI)"


@pytest.mark.parametrize(
    ("baseline", "candidate", "options", "message"),
    [
        ([1.0, 0.0], [1.0], {}, "must be paired"),
        ([], [], {}, "at least one paired example"),
        ([1.0], [0.0], {"resamples": 50}, "resamples must be at least 100"),
        ([1.0], [0.0], {"confidence": 1.0}, "confidence must be between 0 and 1"),
        ([1.0], [float("nan")], {}, "finite"),
    ],
)
def test_invalid_input_is_rejected(
    baseline: list[float], candidate: list[float], options: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        paired_bootstrap(baseline, candidate, **options)  # type: ignore[arg-type]


# Wilson score interval


@pytest.mark.parametrize(
    ("successes", "total", "low", "high"),
    [
        # Published values for the 95% Wilson interval.
        (5, 10, 0.2366, 0.7634),
        (0, 10, 0.0, 0.2775),
        (10, 10, 0.7225, 1.0),
        (47, 60, 0.6638, 0.8688),
    ],
)
def test_wilson_matches_known_values(successes: int, total: int, low: float, high: float) -> None:
    result = wilson_interval(successes, total)

    assert result.low == pytest.approx(low, abs=1e-4)
    assert result.high == pytest.approx(high, abs=1e-4)


@pytest.mark.parametrize("total", [1, 12, 20, 60])
def test_wilson_is_exact_at_the_edges(total: int) -> None:
    assert wilson_interval(0, total).low == 0.0
    assert wilson_interval(total, total).high == 1.0


@pytest.mark.parametrize("successes", range(21))
def test_wilson_contains_the_rate_and_stays_in_bounds(successes: int) -> None:
    result = wilson_interval(successes, 20)

    assert 0.0 <= result.low <= result.rate <= result.high <= 1.0
    assert result.high > result.low


def test_wilson_is_symmetric_between_passes_and_fails() -> None:
    passes = wilson_interval(7, 30)
    fails = wilson_interval(23, 30)

    assert passes.low == pytest.approx(1 - fails.high)
    assert passes.high == pytest.approx(1 - fails.low)


def test_wilson_narrows_with_more_tries_at_the_same_rate() -> None:
    small = wilson_interval(5, 10)
    large = wilson_interval(30, 60)

    assert large.high - large.low < small.high - small.low


def test_wilson_higher_confidence_gives_a_wider_interval() -> None:
    narrow = wilson_interval(45, 60, confidence=0.80)
    wide = wilson_interval(45, 60, confidence=0.99)

    assert wide.low < narrow.low
    assert wide.high > narrow.high


def test_wilson_result_prints_as_rate_and_interval() -> None:
    assert str(wilson_interval(47, 60)) == "78.3% [66.4%, 86.9%] (95% CI)"


@pytest.mark.parametrize(
    ("successes", "total", "confidence", "message"),
    [
        (0, 0, 0.95, "total must be at least 1"),
        (-1, 10, 0.95, "successes must be between 0 and 10"),
        (11, 10, 0.95, "successes must be between 0 and 10"),
        (5, 10, 1.0, "confidence must be between 0 and 1"),
    ],
)
def test_wilson_invalid_input_is_rejected(
    successes: int, total: int, confidence: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        wilson_interval(successes, total, confidence=confidence)
