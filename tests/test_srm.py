import numpy as np
import pytest

from uplift.experiment.srm import check_srm, detectable_deviation, simulate_srm_calibration


def test_exact_split_is_not_srm():
    r = check_srm(850_000, 150_000, expected_ratio=0.85)
    assert not r.is_srm
    assert r.p_value > 0.9


def test_clear_mismatch_is_detected():
    r = check_srm(800_000, 200_000, expected_ratio=0.85)
    assert r.is_srm
    assert r.p_value < 1e-10


def test_observed_ratio_and_deviation_are_consistent():
    r = check_srm(850_500, 149_500, expected_ratio=0.85)
    assert np.isclose(r.observed_ratio, 0.8505)
    assert np.isclose(r.abs_deviation, 0.0005)
    assert r.ratio_ci[0] < r.observed_ratio < r.ratio_ci[1]


def test_criteo_allocation_shows_no_srm():
    """The delivered file's actual arm sizes.

    Regression-guards the finding in docs/findings.md: this allocation is exact
    enough that a test with power to resolve 4e-4 has nothing to reject.
    """
    r = check_srm(11_882_655, 2_096_937, expected_ratio=0.85, alpha=0.001)
    assert not r.is_srm
    assert r.abs_deviation < 1e-5


def test_detectable_deviation_shrinks_with_n():
    assert detectable_deviation(10_000) > detectable_deviation(1_000_000)
    assert detectable_deviation(14_000_000) < 1e-3


@pytest.mark.stat
def test_false_positive_rate_matches_alpha():
    out = simulate_srm_calibration(n=200_000, n_sims=2000, seed=3)
    assert out["rejection_rate_at_0.05"] < 0.08  # ~0.05 plus Monte-Carlo slack
    assert out["rejection_rate_at_0.001"] < 0.01


@pytest.mark.stat
def test_power_is_high_against_a_detectable_deviation():
    """Pick the alternative from the module's own power calculation, rather than
    a magic number.

    The build guide suggests asserting near-certain rejection at ratio = 0.8505.
    That is wrong at any n this simulation can afford: at n = 5M a deviation of
    5e-4 is 3.1 standard errors, and against a critical value of 3.29 at
    alpha = 0.001 the power is 44%, which is what the simulation returns. Using
    `detectable_deviation` keeps the test honest and self-consistent.
    """
    n = 5_000_000
    delta = detectable_deviation(n, 0.85, alpha=0.001, power=0.80)
    out = simulate_srm_calibration(n=n, n_sims=300, true_ratio=0.85 + 1.5 * delta, seed=4)
    assert out["rejection_rate_at_0.001"] > 0.9


@pytest.mark.stat
def test_power_is_low_against_an_undetectable_deviation():
    """The complement: a deviation well below the detection threshold should
    mostly go unflagged. Together these two bracket the test's resolution."""
    out = simulate_srm_calibration(n=5_000_000, n_sims=300, true_ratio=0.85002, seed=5)
    assert out["rejection_rate_at_0.001"] < 0.2
