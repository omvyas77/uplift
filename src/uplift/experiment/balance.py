"""Covariate balance between arms."""

from __future__ import annotations

import numpy as np
import pandas as pd


def standardized_mean_differences(
    df: pd.DataFrame, features: list[str], treatment_col: str = "treatment"
) -> pd.DataFrame:
    """SMD per feature.

    |SMD| < 0.1 is the usual "balanced" rule of thumb from the matching
    literature; for a true RCT at n = 14M expect |SMD| < 0.01.

    The variance ratio is reported alongside deliberately: two groups can have
    identical means and very different spreads, and a mean-only balance table
    would call that balanced.
    """
    t = df[treatment_col].to_numpy().astype(bool)
    rows = []
    for f in features:
        x = df[f].to_numpy(dtype=float)
        m1, m0 = x[t].mean(), x[~t].mean()
        v1, v0 = x[t].var(ddof=1), x[~t].var(ddof=1)
        pooled = np.sqrt((v1 + v0) / 2.0)
        rows.append(
            {
                "feature": f,
                "mean_treatment": m1,
                "mean_control": m0,
                "smd": (m1 - m0) / pooled if pooled > 0 else 0.0,
                "var_ratio": v1 / v0 if v0 > 0 else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    out["balanced"] = out["smd"].abs() < 0.1
    return out.sort_values("smd", key=np.abs, ascending=False).reset_index(drop=True)


def balance_from_arrays(
    X: np.ndarray, w: np.ndarray, names: list[str] | None = None
) -> pd.DataFrame:
    names = names or [f"f{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=names)
    df["treatment"] = w
    return standardized_mean_differences(df, names)
