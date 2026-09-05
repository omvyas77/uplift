"""CATE calibration - "how do you know your uplift model is any good?"

Very few portfolio projects do this, and it is the direct answer to that
question. Bin by PREDICTED uplift, then measure OBSERVED uplift in each bin. A
calibrated model produces a monotone increasing observed column that tracks the
predicted column along the diagonal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def cate_calibration(y, w, tau_hat, n_bins: int = 10, min_per_arm: int = 30) -> pd.DataFrame:
    """Observed versus predicted uplift by decile of the predicted effect."""
    y, w, tau_hat = np.asarray(y, dtype=float), np.asarray(w), np.asarray(tau_hat)
    bins = pd.qcut(tau_hat, n_bins, labels=False, duplicates="drop")
    rows = []
    for b in range(int(np.nanmax(bins)) + 1):
        m = bins == b
        yt, yc = y[m & (w == 1)], y[m & (w == 0)]
        if len(yt) < min_per_arm or len(yc) < min_per_arm:
            continue
        obs = yt.mean() - yc.mean()
        se = np.sqrt(yt.var(ddof=1) / len(yt) + yc.var(ddof=1) / len(yc))
        rows.append(
            {
                "decile": b + 1,
                "n_treatment": len(yt),
                "n_control": len(yc),
                "predicted_uplift": float(tau_hat[m].mean()),
                "observed_uplift": float(obs),
                "se": float(se),
                "ci_low": float(obs - 1.96 * se),
                "ci_high": float(obs + 1.96 * se),
            }
        )
    return pd.DataFrame(rows)


def calibration_summary(cal: pd.DataFrame) -> dict[str, float]:
    """Two numbers to report from the calibration table.

    * `calibration_slope` - regress observed on predicted. Near 1 means
      calibrated; near 0 means the model ranks nothing.
    * `rank_correlation` - Spearman between predicted and observed across
      deciles. Monotonicity, without assuming the scale is right.
    """
    if len(cal) < 3:
        return {
            "calibration_slope": float("nan"),
            "rank_correlation": float("nan"),
            "n_deciles": len(cal),
        }
    p, o = cal["predicted_uplift"].to_numpy(), cal["observed_uplift"].to_numpy()
    slope, intercept, r, _, stderr = stats.linregress(p, o)
    rho, _ = stats.spearmanr(p, o)
    return {
        "calibration_slope": float(slope),
        "calibration_intercept": float(intercept),
        "calibration_slope_se": float(stderr),
        "calibration_r2": float(r**2),
        "rank_correlation": float(rho),
        "n_deciles": len(cal),
        "observed_spread": float(o.max() - o.min()),
    }


def response_auc(y, tau_hat) -> float:
    """Response-model AUC.

    Reported ONLY so it can be labelled as not the objective. AUC measures how
    well you predict WHO RESPONDS, not who responds BECAUSE OF the treatment; a
    model that perfectly ranks response probability can have zero uplift value.
    """
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(np.asarray(y), np.asarray(tau_hat)))
