"""Sensitivity analysis.

Any observational estimate is conditional on "no unmeasured confounding", which
is untestable. Sensitivity analysis asks the answerable question instead: how
strong would a hidden confounder have to be to overturn the conclusion?

The negative-control demonstration is the one to lead with, because on this data
it is unusually clean and it is self-validating:

  * run it on the RCT            -> effects on f0..f11 are ~0. It passes, as it must.
  * run it on the confounded slice -> several features show large "effects" of
    treatment. It fails loudly, correctly flagging confounding we injected
    ourselves and therefore know is there.
  * run it after IPW/AIPW reweighting -> the imbalance shrinks back toward zero.
    That is a diagnostic that the adjustment worked, INDEPENDENT of knowing the
    true ATE - which is the only version of the check you could run on real data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def e_value(rr: float, rr_ci_bound: float | None = None) -> dict[str, float]:
    """VanderWeele & Ding E-value.

    The minimum association strength, on the risk-ratio scale, that an
    unmeasured confounder would need with BOTH treatment and outcome to explain
    away the observed effect. An E-value of 1.0 means no confounding at all is
    required - the CI already touches the null.
    """

    def _ev(r: float) -> float:
        r = 1 / r if r < 1 else r
        return float(r + np.sqrt(r * (r - 1)))

    out = {"risk_ratio": float(rr), "e_value_point": _ev(rr)}
    if rr_ci_bound is not None:
        crosses_null = min(rr, rr_ci_bound) <= 1 <= max(rr, rr_ci_bound)
        out["e_value_ci"] = 1.0 if crosses_null else _ev(rr_ci_bound)
    return out


def risk_ratio(y, w) -> tuple[float, float, float]:
    """Risk ratio with a delta-method CI on the log scale - the input the
    E-value wants."""
    y = np.asarray(y, dtype=float)
    y1, y0 = y[w == 1], y[w == 0]
    m1, m0 = y1.mean(), y0.mean()
    rr = m1 / m0
    se_log = np.sqrt((1 - m1) / (m1 * len(y1)) + (1 - m0) / (m0 * len(y0)))
    return float(rr), float(rr * np.exp(-1.96 * se_log)), float(rr * np.exp(1.96 * se_log))


def negative_control_outcome(X, w, feature_idx: int, weights=None) -> dict[str, float]:
    """A pre-treatment covariate CANNOT be affected by treatment, so regressing
    one on treatment must give zero.

    In the RCT it does. In the confounded slice it will not, and the size of
    that non-zero "effect" measures the bias. Pass `weights` (e.g. IPW weights)
    to check whether an adjustment restored balance.
    """
    x = np.asarray(X)[:, feature_idx].astype(float)
    w = np.asarray(w)
    if weights is None:
        m1, m0 = x[w == 1].mean(), x[w == 0].mean()
        se = np.sqrt(
            x[w == 1].var(ddof=1) / (w == 1).sum() + x[w == 0].var(ddof=1) / (w == 0).sum()
        )
    else:
        weights = np.asarray(weights, dtype=float)

        def wm(v, u):
            u = u / u.sum()
            return float(np.sum(u * v)), float(np.sum(u**2 * (v - np.sum(u * v)) ** 2))

        m1, v1 = wm(x[w == 1], weights[w == 1])
        m0, v0 = wm(x[w == 0], weights[w == 0])
        se = float(np.sqrt(v1 + v0))
    return {
        "effect": float(m1 - m0),
        "se": float(se),
        "z": float((m1 - m0) / se) if se > 0 else 0.0,
    }


def negative_control_panel(X, w, names=None, weights=None) -> pd.DataFrame:
    """Run the negative-control check on every covariate at once."""
    names = names or [f"f{i}" for i in range(np.asarray(X).shape[1])]
    rows = []
    for i, nm in enumerate(names):
        r = negative_control_outcome(X, w, i, weights=weights)
        rows.append({"feature": nm, **r})
    out = pd.DataFrame(rows)
    out["fails_at_z3"] = out["z"].abs() > 3
    return out.sort_values("z", key=np.abs, ascending=False).reset_index(drop=True)


def rosenbaum_bounds(gammas=(1.0, 1.1, 1.25, 1.5, 2.0)) -> list[dict]:
    """Rosenbaum sensitivity for a matched design, on the odds scale.

    Gamma is how much the odds of treatment could differ between two units that
    look identical on X. Reported as the multiplier on the log-odds bound, which
    is the honest scalar version; a full implementation would recompute the
    signed-rank test under each gamma.
    """
    return [
        {
            "gamma": float(g),
            "max_log_odds_shift": float(np.log(g)),
            "interpretation": (
                "no hidden bias"
                if g == 1.0
                else f"a hidden confounder could shift treatment odds by {g:.2f}x"
            ),
        }
        for g in gammas
    ]
