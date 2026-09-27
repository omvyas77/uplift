"""The ATE estimators — the module that produces the headline number.

`experiment/ate.py` had ZERO test coverage while every number quoted in the
README and the dashboard came out of it: +20.2% adjusted, +27.1% unadjusted, and
the 7.1-standard-error gap between them that the whole "quote the adjusted one"
argument rests on.

The properties pinned here are the ones that would silently corrupt that
headline: unbiasedness against a known effect, the delta-method relative CI
(which must NOT be the absolute CI divided by the baseline), HC1 robust standard
errors, and Lin's estimator being no worse than the simple difference when the
covariates are balanced.
"""

from __future__ import annotations

import numpy as np
import pytest

from uplift.experiment.ate import (
    ate_difference_in_means,
    ate_from_adjusted,
    ate_lin_regression,
)


def _binary_rct(n=200_000, p0=0.05, tau=0.01, share=0.85, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    w = rng.binomial(1, share, n)
    y = rng.binomial(1, np.clip(p0 + 0.02 * np.tanh(X[:, 0]) + w * tau, 1e-6, 1 - 1e-6))
    return X, w, y


# ---------------------------------------------------------------- difference in means
@pytest.mark.stat
def test_difference_in_means_is_unbiased():
    _, w, y = _binary_rct(tau=0.01, seed=1)
    r = ate_difference_in_means(y, w)
    assert abs(r.ate - 0.01) < 4 * r.se


def test_it_is_literally_the_difference_of_the_two_means():
    """No cleverness: this estimator must equal mean(treated) - mean(control)."""
    _, w, y = _binary_rct(n=20_000, seed=2)
    r = ate_difference_in_means(y, w)
    assert np.isclose(r.ate, y[w == 1].mean() - y[w == 0].mean())
    assert np.isclose(r.baseline_rate, y[w == 0].mean())


def test_the_confidence_interval_is_symmetric_about_the_estimate():
    _, w, y = _binary_rct(n=50_000, seed=3)
    r = ate_difference_in_means(y, w)
    assert np.isclose(r.ci_high - r.ate, r.ate - r.ci_low)
    assert r.ci_low < r.ate < r.ci_high


@pytest.mark.stat
def test_a_null_effect_is_not_declared_significant():
    """The control that makes the other tests mean something."""
    _, w, y = _binary_rct(n=200_000, tau=0.0, seed=4)
    r = ate_difference_in_means(y, w)
    assert r.p_value > 0.01
    assert r.ci_low < 0 < r.ci_high


def test_standard_error_shrinks_with_sample_size():
    """SE should fall roughly as 1/sqrt(n): 25x the data, ~5x tighter."""
    _, w_s, y_s = _binary_rct(n=20_000, seed=5)
    _, w_l, y_l = _binary_rct(n=500_000, seed=5)
    se_small = ate_difference_in_means(y_s, w_s).se
    se_large = ate_difference_in_means(y_l, w_l).se
    assert se_large < se_small
    assert 3.5 < se_small / se_large < 7.0, f"ratio {se_small / se_large:.2f} is not ~5"


# ---------------------------------------------------------------- relative effect
def test_relative_lift_is_the_ratio_minus_one():
    _, w, y = _binary_rct(n=50_000, seed=6)
    r = ate_difference_in_means(y, w)
    expected = y[w == 1].mean() / y[w == 0].mean() - 1.0
    assert np.isclose(r.relative_lift, expected)


def test_relative_ci_is_not_the_absolute_ci_divided_by_the_baseline():
    """The trap this function exists to avoid.

    Dividing the absolute CI by the baseline ignores that the baseline is itself
    estimated, so it understates the width. The delta-method interval on
    log(m1/m0) must therefore be strictly wider than the naive one.
    """
    _, w, y = _binary_rct(n=50_000, seed=7)
    r = ate_difference_in_means(y, w)

    naive_width = (r.ci_high - r.ci_low) / r.baseline_rate
    delta_width = r.relative_ci[1] - r.relative_ci[0]
    assert delta_width > naive_width, (
        f"delta-method width {delta_width:.6f} should exceed the naive "
        f"{naive_width:.6f}; the baseline's own variance is being ignored"
    )


def test_relative_ci_brackets_the_point_estimate():
    _, w, y = _binary_rct(n=50_000, seed=8)
    r = ate_difference_in_means(y, w)
    assert r.relative_ci[0] < r.relative_lift < r.relative_ci[1]


# ---------------------------------------------------------------- Lin
@pytest.mark.stat
def test_lin_recovers_the_effect_on_balanced_data():
    X, w, y = _binary_rct(n=150_000, tau=0.01, seed=9)
    r = ate_lin_regression(y, w, X)
    assert abs(r.ate - 0.01) < 4 * r.se


@pytest.mark.stat
def test_lin_agrees_with_difference_in_means_under_balance():
    """In a clean RCT adjustment tightens the interval and leaves the point
    estimate alone.

    This is the property whose FAILURE on the real Criteo file is finding 3 in
    docs/findings.md - there the two disagree by 7.1 SEs because the covariates
    are imbalanced. On synthetic balanced data they must agree, or that argument
    would have been about a bug in this function instead.
    """
    X, w, y = _binary_rct(n=150_000, tau=0.01, seed=10)
    dim = ate_difference_in_means(y, w)
    lin = ate_lin_regression(y, w, X)
    assert abs(lin.ate - dim.ate) < 3 * dim.se


def test_lin_uses_robust_standard_errors():
    """With a binary outcome the error variance differs by arm, so homoskedastic
    OLS errors are wrong. Confirm HC1 is actually in use by comparing against a
    deliberately non-robust fit."""
    import statsmodels.api as sm

    X, w, y = _binary_rct(n=30_000, seed=11)
    lin = ate_lin_regression(y, w, X)

    Xc = X - X.mean(axis=0, keepdims=True)
    design = np.column_stack([np.ones(len(y)), w, Xc, w[:, None] * Xc])
    plain = sm.OLS(y.astype(float), design).fit()  # homoskedastic
    assert not np.isclose(lin.se, plain.bse[1]), "SE matches the non-robust fit"


def test_lin_is_not_fooled_by_a_constant_covariate():
    """A zero-variance column makes the design matrix rank-deficient. The
    estimate must still come out finite rather than silently NaN."""
    X, w, y = _binary_rct(n=20_000, seed=12)
    X = np.column_stack([X, np.ones(len(X))])
    r = ate_lin_regression(y, w, X)
    assert np.isfinite(r.ate) and np.isfinite(r.se)


# ---------------------------------------------------------------- adjusted outcome
def test_adjusted_estimator_uses_the_passed_baseline_not_the_adjusted_mean():
    """A CUPED-adjusted outcome is no longer on the probability scale, so its own
    control mean is not a rate. The unadjusted baseline must be used for the
    relative figure, or the reported percentage is meaningless."""
    rng = np.random.default_rng(13)
    n = 40_000
    w = rng.binomial(1, 0.85, n)
    y_adj = rng.normal(size=n) + 0.01 * w
    r = ate_from_adjusted(y_adj, w, baseline_rate=0.05)
    assert np.isclose(r.baseline_rate, 0.05)
    assert np.isclose(r.relative_lift, r.ate / 0.05)


def test_summary_strings_render():
    """These go straight into the readout; a formatting error there is a broken
    deliverable, not a cosmetic issue."""
    X, w, y = _binary_rct(n=20_000, seed=14)
    for r in (ate_difference_in_means(y, w), ate_lin_regression(y, w, X)):
        s = r.summary()
        assert r.method in s
        assert "ATE=" in s and "SE=" in s
        assert isinstance(r.to_dict(), dict)
