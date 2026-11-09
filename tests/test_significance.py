"""Tests for McNemar's exact test and Holm correction, checked against known answers."""

import math

import numpy as np
import pytest

from trustbench.significance import holm_adjust, mcnemar_exact


def results(
    wins: int, losses: int, both_pass: int = 0, both_fail: int = 0
) -> tuple[list[bool], list[bool]]:
    """Build paired baseline/candidate results with the given counts of each outcome."""
    baseline = [False] * wins + [True] * losses + [True] * both_pass + [False] * both_fail
    candidate = [True] * wins + [False] * losses + [True] * both_pass + [False] * both_fail
    return baseline, candidate


def test_mcnemar_matches_the_textbook_value() -> None:
    # 9 wins and 1 loss: 2 * (C(10,0) + C(10,1)) / 2**10 = 22 / 1024 exactly.
    result = mcnemar_exact(*results(wins=9, losses=1))
    assert (result.wins, result.losses) == (9, 1)
    assert result.p_value == 22 / 1024


def test_mcnemar_is_symmetric() -> None:
    baseline, candidate = results(wins=7, losses=2)
    forward = mcnemar_exact(baseline, candidate)
    backward = mcnemar_exact(candidate, baseline)
    assert (backward.wins, backward.losses) == (forward.losses, forward.wins)
    assert backward.p_value == forward.p_value


def test_identical_systems_give_p_of_one() -> None:
    result = mcnemar_exact(*results(wins=0, losses=0, both_pass=5, both_fail=3))
    assert result.disagreements == 0
    assert result.p_value == 1.0


def test_a_balanced_split_gives_p_of_one() -> None:
    assert mcnemar_exact(*results(wins=3, losses=3)).p_value == 1.0


def test_agreements_do_not_change_the_p_value() -> None:
    small = mcnemar_exact(*results(wins=6, losses=1))
    padded = mcnemar_exact(*results(wins=6, losses=1, both_pass=100, both_fail=50))
    assert padded.p_value == small.p_value


@pytest.mark.parametrize(
    ("wins", "expected", "significant"),
    [(5, 0.0625, False), (6, 0.03125, True)],
)
def test_six_one_way_disagreements_are_the_minimum_for_significance(
    wins: int, expected: float, significant: bool
) -> None:
    p_value = mcnemar_exact(*results(wins=wins, losses=0)).p_value
    assert p_value == expected
    assert (p_value <= 0.05) is significant


def test_large_samples_stay_exact_and_finite() -> None:
    p_value = mcnemar_exact(*results(wins=600, losses=400)).p_value
    assert 0.0 < p_value < 1e-9


@pytest.mark.parametrize(
    ("baseline", "candidate", "message"),
    [
        ([True, False], [True], "paired by example"),
        ([], [], "at least one paired example"),
        ([True, False], [1, 0.0], "True or False"),
    ],
)
def test_mcnemar_rejects_bad_input(
    baseline: list[bool], candidate: list[bool], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        mcnemar_exact(baseline, candidate)


def test_false_alarm_rate_is_at_most_five_percent_when_systems_are_equal() -> None:
    # Equal systems: each example is a disagreement 30% of the time, and a disagreement is
    # a win or a loss with equal chance. A valid 5% test must flag at most 5% of trials.
    rng = np.random.default_rng(2024)
    trials = 2_000
    false_alarms = 0
    for _ in range(trials):
        both = rng.random(40) < 0.6
        disagree = rng.random(40) < 0.3
        baseline_only = rng.random(40) < 0.5
        baseline = np.where(disagree, baseline_only, both)
        candidate = np.where(disagree, ~baseline_only, both)
        result = mcnemar_exact([bool(x) for x in baseline], [bool(x) for x in candidate])
        false_alarms += result.p_value <= 0.05
    assert false_alarms / trials <= 0.05


def test_mcnemar_result_prints_counts_and_p_value() -> None:
    result = mcnemar_exact(*results(wins=9, losses=1))
    assert str(result) == "wins 9, losses 1, p = 0.0215"


def test_holm_matches_the_textbook_example() -> None:
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx((0.03, 0.06, 0.06))


def test_holm_leaves_a_single_p_value_unchanged() -> None:
    assert holm_adjust([0.02]) == (0.02,)


def test_holm_of_nothing_is_nothing() -> None:
    assert holm_adjust([]) == ()


def test_holm_caps_adjusted_values_at_one() -> None:
    assert holm_adjust([0.5, 0.6, 0.7]) == (1.0, 1.0, 1.0)


def test_holm_never_lowers_a_p_value_and_keeps_the_order() -> None:
    rng = np.random.default_rng(7)
    raw = [float(p) for p in rng.random(25)]
    adjusted = holm_adjust(raw)
    assert all(new >= old for old, new in zip(raw, adjusted, strict=True))
    by_raw = sorted(range(len(raw)), key=lambda index: raw[index])
    ordered = [adjusted[index] for index in by_raw]
    assert ordered == sorted(ordered)


@pytest.mark.parametrize("bad", [-0.1, 1.5, math.nan])
def test_holm_rejects_values_that_are_not_probabilities(bad: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        holm_adjust([0.01, bad])
