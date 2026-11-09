"""Tests for the power analysis, checked against the significance test and simulation."""

import math

import numpy as np
import pytest

from trustbench.power import critical_count, main, mcnemar_power, minimum_detectable_drop
from trustbench.significance import mcnemar_exact


@pytest.mark.parametrize(
    ("disagreements", "expected"),
    [(0, -1), (5, -1), (6, 0), (10, 1)],
)
def test_critical_count_matches_known_values(disagreements: int, expected: int) -> None:
    assert critical_count(disagreements) == expected


def test_critical_count_agrees_with_the_significance_test() -> None:
    for disagreements in range(1, 31):
        critical = critical_count(disagreements)
        for wins in range(disagreements + 1):
            losses = disagreements - wins
            baseline = [False] * wins + [True] * losses
            candidate = [True] * wins + [False] * losses
            significant = mcnemar_exact(baseline, candidate).p_value <= 0.05
            assert significant == (min(wins, losses) <= critical)


def test_power_with_no_real_difference_is_at_most_alpha() -> None:
    assert mcnemar_power(100, loss_rate=0.1, win_rate=0.1) <= 0.05


def test_power_is_zero_when_the_systems_never_disagree() -> None:
    assert mcnemar_power(50, loss_rate=0.0, win_rate=0.0) == 0.0


def test_power_when_every_disagreement_is_a_loss_matches_a_hand_calculation() -> None:
    # Every disagreement is a loss, so the test fires exactly when there are 6 or more.
    expected = sum(math.comb(13, k) * 0.2**k * 0.8 ** (13 - k) for k in range(6, 14))
    assert mcnemar_power(13, loss_rate=0.2, win_rate=0.0) == pytest.approx(expected)


def test_power_matches_simulation() -> None:
    rng = np.random.default_rng(99)
    trials = 4_000
    detected = 0
    for _ in range(trials):
        draw = rng.random(40)
        losses = int((draw < 0.15).sum())
        wins = int(((draw >= 0.15) & (draw < 0.20)).sum())
        baseline = [True] * losses + [False] * wins
        candidate = [False] * losses + [True] * wins
        if baseline:
            detected += mcnemar_exact(baseline, candidate).p_value <= 0.05
    exact = mcnemar_power(40, loss_rate=0.15, win_rate=0.05)
    assert detected / trials == pytest.approx(exact, abs=0.03)


def test_power_grows_with_more_examples() -> None:
    powers = [mcnemar_power(n, loss_rate=0.13, win_rate=0.07) for n in (40, 100, 250)]
    assert powers == sorted(powers)
    assert powers[0] < powers[-1]


@pytest.mark.parametrize(
    ("examples", "expected"),
    [(13, None), (40, 0.196), (100, 0.128), (250, 0.082)],
)
def test_minimum_detectable_drop_at_a_20_percent_disagreement_rate(
    examples: int, expected: float | None
) -> None:
    drop = minimum_detectable_drop(examples, disagreement_rate=0.2)
    if expected is None:
        assert drop is None
    else:
        assert drop == pytest.approx(expected)


@pytest.mark.parametrize(
    ("examples", "loss_rate", "win_rate", "alpha", "message"),
    [
        (0, 0.1, 0.1, 0.05, "at least one example"),
        (10, -0.1, 0.1, 0.05, "non-negative"),
        (10, 0.7, 0.4, 0.05, "sum to at most 1"),
        (10, 0.1, 0.1, 0.0, "alpha"),
    ],
)
def test_mcnemar_power_rejects_bad_input(
    examples: int, loss_rate: float, win_rate: float, alpha: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        mcnemar_power(examples, loss_rate, win_rate, alpha)


@pytest.mark.parametrize(
    ("rate", "power", "message"),
    [(0.0, 0.8, "disagreement rate"), (0.2, 1.0, "power")],
)
def test_minimum_detectable_drop_rejects_bad_input(rate: float, power: float, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        minimum_detectable_drop(40, disagreement_rate=rate, power=power)


def test_main_prints_the_study_table(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "disagreement rate 20%, power 80%, alpha 5% (two-sided)"
    assert len(lines) == 8
    assert lines[2].split() == ["13", "not", "detectable"]
    assert lines[4].split() == ["100", "12.8", "points"]
