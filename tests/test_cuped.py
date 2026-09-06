import numpy as np
import pytest

from uplift.data.synthetic import make_rct
from uplift.experiment.cuped import build_cupac_covariate, cuped_adjust, evaluate_cuped


def test_variance_reduction_equals_rho_squared():
    """The identity the whole method rests on: var reduction == rho^2."""
    rng = np.random.default_rng(0)
    n, target_rho = 200_000, 0.6
    x = rng.normal(size=n)
    y = target_rho * x + rng.normal(size=n) * np.sqrt(1 - target_rho**2)

    y_adj, theta = cuped_adjust(y, x)
    rho = np.corrcoef(y, x)[0, 1]

    observed = 1 - y_adj.var(ddof=1) / y.var(ddof=1)
    assert abs(observed - rho**2) < 1e-3
    assert abs(theta - target_rho) < 0.02


def test_cuped_preserves_the_mean():
    rng = np.random.default_rng(1)
    x = rng.normal(size=50_000)
    y = 0.4 * x + rng.normal(size=50_000)
    y_adj, _ = cuped_adjust(y, x)
    assert abs(y_adj.mean() - y.mean()) < 1e-9


def test_an_uninformative_covariate_reduces_nothing():
    rng = np.random.default_rng(2)
    n = 100_000
    y = rng.normal(size=n)
    noise = rng.normal(size=n)
    res = evaluate_cuped(y, rng.binomial(1, 0.85, n), noise)
    assert abs(res.variance_reduction) < 0.01
    assert res.effective_sample_multiplier < 1.02


@pytest.mark.stat
def test_cupac_does_not_move_the_ate():
    """CUPAC must shrink the SE without shifting the point estimate.

    This holds when the covariates are balanced across arms, which is true by
    construction in the synthetic RCT. It notably does NOT hold on the real
    Criteo file, where the arms are imbalanced - see docs/findings.md.
    """
    X, w, y, true_ate = make_rct(n=150_000, treatment_share=0.85, seed=11)
    y = y.astype(float)
    cov = build_cupac_covariate(X, y, w, seed=11, n_estimators=100)
    res = evaluate_cuped(y, w, cov)

    naive = y[w == 1].mean() - y[w == 0].mean()
    y_adj, _ = cuped_adjust(y, cov)
    adjusted = y_adj[w == 1].mean() - y_adj[w == 0].mean()

    assert res.se_after <= res.se_before
    assert abs(adjusted - naive) < 3 * res.se_before
    assert abs(adjusted - true_ate) < 4 * res.se_after


def test_feature_groups_keeps_duplicates_together():
    """Guards the leakage fix: every copy of a feature vector gets one group id."""
    from uplift.experiment.cuped import feature_groups

    X = np.array([[1.0, 2.0], [3.0, 4.0], [1.0, 2.0], [5.0, 6.0], [3.0, 4.0]])
    g = feature_groups(X)
    assert g[0] == g[2]  # the duplicated pair
    assert g[1] == g[4]
    assert len({int(v) for v in g}) == 3


@pytest.mark.stat
def test_grouped_cv_removes_memorisation_of_duplicates():
    """Construct a dataset a plain KFold MUST memorise, and show grouping stops it.

    Every row is duplicated and the outcome is pure noise, so any apparent
    correlation is memorisation rather than signal.
    """
    from uplift.experiment.cuped import build_cupac_covariate

    rng = np.random.default_rng(5)
    n = 4_000
    X_base = rng.normal(size=(n, 4))
    y_base = rng.binomial(1, 0.3, n).astype(float)  # unpredictable from X
    X = np.repeat(X_base, 2, axis=0)
    y = np.repeat(y_base, 2)
    w = rng.binomial(1, 0.5, len(y))

    leaky = np.corrcoef(y, build_cupac_covariate(X, y, w, seed=0, grouped=False))[0, 1]
    honest = np.corrcoef(y, build_cupac_covariate(X, y, w, seed=0, grouped=True))[0, 1]

    assert leaky > honest + 0.10, f"expected memorisation; leaky={leaky:.3f} honest={honest:.3f}"
    assert abs(honest) < 0.15, f"grouped CV should find no signal, got rho={honest:.3f}"
