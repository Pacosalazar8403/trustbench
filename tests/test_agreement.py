"""Tests for two-rater agreement and Cohen's kappa."""

import pytest

from trustbench.agreement import Agreement, agreement, describe_kappa


def verdicts(
    both_pass: int, only_first: int, only_second: int, both_fail: int
) -> tuple[list[bool], list[bool]]:
    first = [True] * both_pass + [True] * only_first + [False] * only_second
    second = [True] * both_pass + [False] * only_first + [True] * only_second
    return first + [False] * both_fail, second + [False] * both_fail


def test_kappa_matches_the_textbook_example() -> None:
    # 50 items: 20 both yes, 5 first only, 10 second only, 15 both no.
    table = agreement(*verdicts(20, 5, 10, 15))
    assert table == Agreement(both_pass=20, only_first=5, only_second=10, both_fail=15)
    assert table.observed == pytest.approx(0.7)
    assert table.expected == pytest.approx(0.5)
    assert table.kappa == pytest.approx(0.4)


def test_chance_agreement_is_subtracted() -> None:
    # Both raters pass 70% of items independently: high raw agreement, kappa near 0.
    table = agreement(*verdicts(49, 21, 21, 9))
    assert table.observed == pytest.approx(0.58)
    assert table.kappa == pytest.approx(0.0)


def test_perfect_and_opposite_raters() -> None:
    assert agreement(*verdicts(6, 0, 0, 4)).kappa == pytest.approx(1.0)
    assert agreement(*verdicts(0, 5, 5, 0)).kappa == pytest.approx(-1.0)


def test_kappa_is_undefined_when_every_verdict_is_the_same() -> None:
    table = agreement([True] * 5, [True] * 5)
    assert table.observed == 1
    assert table.kappa is None
    assert "kappa undefined" in str(table)


def test_agreement_prints_counts_and_strength() -> None:
    assert str(agreement(*verdicts(20, 5, 10, 15))) == "agree on 35 of 50 (70%), kappa 0.40 (fair)"


@pytest.mark.parametrize(
    ("kappa", "label"),
    [
        (-0.1, "poor"),
        (0.0, "poor"),
        (0.15, "slight"),
        (0.5, "moderate"),
        (0.7, "substantial"),
        (0.9, "almost perfect"),
        (1.0, "almost perfect"),
    ],
)
def test_kappa_names_follow_landis_and_koch(kappa: float, label: str) -> None:
    assert describe_kappa(kappa) == label


def test_unpaired_or_empty_verdicts_are_rejected() -> None:
    with pytest.raises(ValueError, match="paired"):
        agreement([True], [True, False])
    with pytest.raises(ValueError, match="at least one"):
        agreement([], [])
