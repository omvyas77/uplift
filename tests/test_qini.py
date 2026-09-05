"""The three properties that catch essentially every Qini implementation bug:
random scores ~ 0, oracle beats random, and reversing the ranking flips the sign.
"""

import numpy as np
import pytest

from uplift.evaluation.qini import (
    bootstrap_qini_ci,
    paired_bootstrap_difference,
    qini_coefficient,
    qini_curve,
)


def _synthetic(n=100_000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    w = rng.binomial(1, 0.85, n)
    tau = 0.05 * (x > 0)
    y = rng.binomial(1, 0.05 + w * tau)
    return x, w, y, tau


def test_random_scores_score_near_zero():
    _, w, y, _ = _synthetic()
    rng = np.random.default_rng(99)
    assert abs(qini_coefficient(y, w, rng.normal(size=len(y)))) < 0.02


def test_oracle_beats_random():
    _, w, y, tau = _synthetic()
    rng = np.random.default_rng(99)
    assert qini_coefficient(y, w, tau) > qini_coefficient(y, w, rng.normal(size=len(y))) + 0.05


def test_reversed_ranking_flips_the_sign():
    """Guards the tie-aware integration.

    `tau = 0.05 * (x > 0)` has exactly two distinct values, so the whole sample
    is two enormous tied blocks. Integrating per-unit rather than per-block made
    this asymmetric by 0.025 - see the docstring of `qini_curve`.
    """
    _, w, y, tau = _synthetic()
    assert np.isclose(qini_coefficient(y, w, tau), -qini_coefficient(y, w, -tau), atol=0.01)


def test_tie_aware_curve_collapses_tied_blocks():
    _, w, y, tau = _synthetic(n=10_000)
    x_tie, _ = qini_curve(y, w, tau, tie_aware=True)
    x_all, _ = qini_curve(y, w, tau, tie_aware=False)
    assert len(x_tie) == 3  # origin + one point per distinct score
    assert len(x_all) == len(y) + 1


def test_curve_starts_at_the_origin():
    _, w, y, tau = _synthetic(n=10_000)
    x, q = qini_curve(y, w, tau)
    assert x[0] == 0.0 and q[0] == 0.0


def test_constant_score_gives_zero():
    """Every unit tied means no ranking at all, so no lift over random."""
    _, w, y, _ = _synthetic(n=20_000)
    assert abs(qini_coefficient(y, w, np.ones(len(y)))) < 1e-9


@pytest.mark.stat
def test_bootstrap_ci_covers_the_point_estimate():
    from uplift.evaluation.qini import sklift_qini

    _, w, y, tau = _synthetic(n=50_000)
    point = sklift_qini(y, tau, w)
    mean, lo, hi = bootstrap_qini_ci(y, w, tau, n_boot=60, seed=1)
    assert lo < point < hi
    assert lo < mean < hi


@pytest.mark.stat
def test_paired_bootstrap_resolves_oracle_versus_random():
    _, w, y, tau = _synthetic(n=50_000)
    rng = np.random.default_rng(7)
    diff, _lo, _hi, resolved = paired_bootstrap_difference(
        y, w, tau, rng.normal(size=len(y)), n_boot=60, seed=2
    )
    assert diff > 0
    assert resolved, "oracle versus random must be resolvable"


@pytest.mark.stat
def test_paired_bootstrap_does_not_resolve_a_model_against_itself():
    """The null case. If this resolves, the pairing is broken."""
    _, w, y, tau = _synthetic(n=50_000)
    diff, _lo, _hi, resolved = paired_bootstrap_difference(y, w, tau, tau, n_boot=60, seed=3)
    assert abs(diff) < 1e-9
    assert not resolved
