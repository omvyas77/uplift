"""Covariate balance and CATE calibration — two more modules that were at 0%.

`balance.py` produced the finding that all 12 Criteo covariates are imbalanced,
and `calibration.py` produces the slope quoted for every model in the README.
Both were entirely untested.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from uplift.evaluation.calibration import calibration_summary, cate_calibration, response_auc
from uplift.experiment.balance import balance_from_arrays, standardized_mean_differences


# ------------------------------------------------------------------ balance
def test_identical_arms_have_zero_smd():
    """The null case. If this is not ~0, every balance number is suspect."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40_000, 5))
    w = rng.binomial(1, 0.5, 40_000)  # independent of X
    out = balance_from_arrays(X, w)
    assert out["smd"].abs().max() < 0.05
    assert out["balanced"].all()


def test_a_planted_shift_is_detected_with_the_right_magnitude():
    """SMD is a mean difference in pooled-SD units, so a shift of 0.5 SD must
    come back as ~0.5 - not merely 'nonzero'."""
    rng = np.random.default_rng(1)
    n = 60_000
    w = rng.binomial(1, 0.5, n)
    X = rng.normal(size=(n, 2))
    X[w == 1, 0] += 0.5  # half a standard deviation
    out = balance_from_arrays(X, w).set_index("feature")
    assert 0.45 < out.loc["f0", "smd"] < 0.55
    assert abs(out.loc["f1", "smd"]) < 0.05


def test_smd_is_scale_free():
    """Rescaling a covariate must not change its SMD; that is the whole point of
    standardising."""
    rng = np.random.default_rng(2)
    n = 40_000
    w = rng.binomial(1, 0.5, n)
    X = rng.normal(size=(n, 1))
    X[w == 1, 0] += 0.3
    a = balance_from_arrays(X, w)["smd"].iloc[0]
    b = balance_from_arrays(X * 1000.0, w)["smd"].iloc[0]
    assert np.isclose(a, b, atol=1e-9)


def test_the_variance_ratio_catches_equal_means_but_different_spreads():
    """Two arms can share a mean and differ wildly in spread. A mean-only table
    would call that balanced, which is why var_ratio is reported."""
    rng = np.random.default_rng(3)
    n = 60_000
    w = rng.binomial(1, 0.5, n)
    X = np.where(w == 1, rng.normal(0, 3, n), rng.normal(0, 1, n)).reshape(-1, 1)
    out = balance_from_arrays(X, w)
    assert abs(out["smd"].iloc[0]) < 0.05, "means should match"
    assert out["var_ratio"].iloc[0] > 5, "spread difference must be visible"


def test_output_is_sorted_by_absolute_smd():
    """The readout shows the worst offenders first."""
    rng = np.random.default_rng(4)
    n = 30_000
    w = rng.binomial(1, 0.5, n)
    X = rng.normal(size=(n, 3))
    X[w == 1, 2] += 0.4
    X[w == 1, 0] += 0.1
    out = balance_from_arrays(X, w)
    assert out["feature"].iloc[0] == "f2"
    assert list(out["smd"].abs()) == sorted(out["smd"].abs(), reverse=True)


def test_named_columns_are_respected():
    df = pd.DataFrame({"a": [1.0, 2, 3, 4], "b": [4.0, 3, 2, 1], "treatment": [1, 1, 0, 0]})
    out = standardized_mean_differences(df, ["a", "b"])
    assert set(out["feature"]) == {"a", "b"}


# ------------------------------------------------------------------ calibration
def _calibratable(n=200_000, seed=0):
    """Data where the TRUE uplift is known per unit, so a perfectly calibrated
    model is available as a reference."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    w = rng.binomial(1, 0.5, n)
    # CONTINUOUS uplift. A two-valued tau makes pd.qcut collapse every quantile
    # edge onto the same number and return a single bin, which is a property of
    # the fixture rather than of the calibration code.
    tau = 0.02 + 0.03 / (1.0 + np.exp(-x))
    y = rng.binomial(1, 0.05 + w * tau)
    return x, w, y, tau


@pytest.mark.stat
def test_a_perfect_model_has_slope_near_one():
    _, w, y, tau = _calibratable()
    summary = calibration_summary(cate_calibration(y, w, tau, n_bins=5))
    assert 0.7 < summary["calibration_slope"] < 1.3, summary
    assert summary["rank_correlation"] > 0.8


@pytest.mark.stat
def test_a_random_model_has_slope_near_zero():
    """The control. A model that ranks nothing must not look calibrated."""
    _, w, y, _ = _calibratable()
    rng = np.random.default_rng(99)
    summary = calibration_summary(cate_calibration(y, w, rng.normal(size=len(y)), n_bins=5))
    assert abs(summary["calibration_slope"]) < 0.5, summary


def test_observed_uplift_tracks_the_planted_effect():
    """Each bin's OBSERVED uplift must land near the mean of the true tau in
    that bin - the actual definition of calibration, not merely monotonicity."""
    _, w, y, tau = _calibratable(seed=1)
    cal = cate_calibration(y, w, tau, n_bins=4)
    assert len(cal) == 4
    assert cal["observed_uplift"].iloc[0] < cal["observed_uplift"].iloc[-1]
    for _, row in cal.iterrows():
        assert abs(row["observed_uplift"] - row["predicted_uplift"]) < 0.012, row.to_dict()


def test_thin_bins_are_dropped_rather_than_reported_as_noise():
    """A decile with almost no control units yields a meaningless difference;
    `min_per_arm` must drop it instead of publishing it."""
    rng = np.random.default_rng(5)
    n = 2_000
    w = rng.binomial(1, 0.99, n)  # almost no controls
    y = rng.binomial(1, 0.05, n)
    cal = cate_calibration(y, w, rng.normal(size=n), n_bins=10, min_per_arm=30)
    assert len(cal) == 0


def test_calibration_summary_degrades_gracefully_on_too_few_bins():
    empty = pd.DataFrame(columns=["predicted_uplift", "observed_uplift"])
    out = calibration_summary(empty)
    assert np.isnan(out["calibration_slope"])
    assert out["n_deciles"] == 0


def test_confidence_intervals_bracket_the_observed_uplift():
    _, w, y, tau = _calibratable(n=80_000, seed=2)
    cal = cate_calibration(y, w, tau, n_bins=4)
    assert (cal["ci_low"] <= cal["observed_uplift"]).all()
    assert (cal["observed_uplift"] <= cal["ci_high"]).all()


def test_response_auc_measures_response_not_uplift():
    """The point of reporting it is to label it as NOT the objective: a score
    that ranks response perfectly can still be useless for uplift."""
    rng = np.random.default_rng(6)
    n = 20_000
    y = rng.binomial(1, 0.3, n)
    perfect_response = y + rng.normal(0, 0.01, n)
    assert response_auc(y, perfect_response) > 0.95
    assert abs(response_auc(y, rng.normal(size=n)) - 0.5) < 0.05
