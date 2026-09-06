"""`make policy` - targeting policy plus doubly-robust off-policy evaluation.

The point of this module is the gap between two numbers:

* what the model THINKS its policy is worth (`targeting.optimal_threshold`,
  computed from the model's own predictions), and
* what the policy is ACTUALLY worth on held-out randomized data (`ope`).

Quoting the first without the second is how portfolio projects end up claiming
implausible dollar figures. The README quotes the second.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from uplift.config import settings
from uplift.data.io import load_split
from uplift.logging import get_logger
from uplift.policy.ope import baseline_policies, dr_policy_value, ipw_policy_value
from uplift.policy.targeting import optimal_threshold, profit_curve

log = get_logger(__name__)
RULE = "─" * 88


def _outcome_models(X, w, y, seed: int, n_estimators: int = 200):
    """mu0(x), mu1(x) for the DR estimator, cross-fitted over two folds."""
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import StratifiedKFold

    mu0, mu1 = np.zeros(len(y), dtype=float), np.zeros(len(y), dtype=float)
    for tr, te in StratifiedKFold(2, shuffle=True, random_state=seed).split(X, w):
        t1, t0 = tr[w[tr] == 1], tr[w[tr] == 0]
        f1 = LGBMClassifier(n_estimators=n_estimators, verbose=-1, random_state=seed, n_jobs=-1)
        f0 = LGBMClassifier(n_estimators=n_estimators, verbose=-1, random_state=seed, n_jobs=-1)
        mu1[te] = np.asarray(f1.fit(X[t1], y[t1]).predict_proba(X[te]))[:, 1]
        mu0[te] = np.asarray(f0.fit(X[t0], y[t0]).predict_proba(X[te]))[:, 1]
    return mu0, mu1


def policy_report(
    outcome: str = "visit",
    split: str = "test",
    model: str | None = None,
    ope_sample: int | None = 1_000_000,
    write: bool = True,
) -> dict[str, Any]:
    scores_path = settings.artifacts_dir / f"scores_{split}.parquet"
    if not scores_path.exists():
        raise FileNotFoundError(f"{scores_path} not found - run `make train` first")
    df = pd.read_parquet(scores_path)

    # pick the model the eval ranked first, unless told otherwise
    if model is None:
        results_path = settings.evals_dir / "results.json"
        if results_path.exists():
            model = json.loads(results_path.read_text())["ranking"][0]
        else:
            model = next(c for c in df.columns if c not in ("y", "w"))
    log.info("policy_for", model=model, split=split)

    if ope_sample and ope_sample < len(df):
        df = df.sample(ope_sample, random_state=settings.random_seed).reset_index(drop=True)

    tau = df[model].to_numpy()
    y, w = df["y"].to_numpy(), df["w"].to_numpy()

    # The logging propensity is KNOWN here because it was assigned by design.
    # A production logging policy would require estimating it; say so.
    propensity = np.full(len(y), settings.designed_treatment_ratio)

    # Price the outcome being optimised, not always a conversion - see
    # Settings.value_per_outcome. Valuing an incremental VISIT at the price of a
    # CONVERSION overstated expected profit by ~16x.
    value = settings.value_per_outcome(outcome)
    pol = optimal_threshold(tau, value, settings.cost_per_treatment_usd)
    X, _, _ = load_split(split, outcome=outcome)
    X = X[df.index.to_numpy()] if len(X) != len(df) else X
    mu0, mu1 = _outcome_models(X, w, y, settings.random_seed)

    rows = []
    for name, treat in baseline_policies(tau, pol.threshold, seed=settings.random_seed).items():
        v_ipw, se_ipw = ipw_policy_value(y, w, treat, propensity)
        v_dr, se_dr = dr_policy_value(y, w, treat, propensity, mu0, mu1)
        rows.append(
            {
                "policy": name,
                "treated_fraction": float(treat.mean()),
                "value_ipw": v_ipw,
                "ipw_ci": [v_ipw - 1.96 * se_ipw, v_ipw + 1.96 * se_ipw],
                "value_dr": v_dr,
                "dr_ci": [v_dr - 1.96 * se_dr, v_dr + 1.96 * se_dr],
                "se_dr": se_dr,
            }
        )
    tbl = pd.DataFrame(rows)

    model_row = tbl[tbl.policy == "model_targeting"].iloc[0]
    rand_row = tbl[tbl.policy == "random_same_budget"].iloc[0]
    diff = float(model_row.value_dr - rand_row.value_dr)
    se_diff = float(np.sqrt(model_row.se_dr**2 + rand_row.se_dr**2))
    beats_random = bool(abs(diff) > 1.96 * se_diff and diff > 0)

    out: dict[str, Any] = {
        "model": model,
        "outcome": outcome,
        "split": split,
        "n": len(df),
        "threshold": pol.threshold,
        "model_predicted_policy": pol.to_dict(),
        "assumptions": {
            "value_per_conversion_usd": settings.value_per_conversion_usd,
            "conversions_per_visit": settings.conversions_per_visit,
            "value_per_outcome_usd": value,
            "priced_outcome": outcome,
            "cost_per_treatment_usd": settings.cost_per_treatment_usd,
            "logging_propensity": settings.designed_treatment_ratio,
            "note": "propensity is KNOWN by design here; production logging would need it estimated",
        },
        "policy_values": tbl.to_dict(orient="records"),
        "model_vs_random_same_budget": {
            "diff_dr": diff,
            "se": se_diff,
            "resolved": beats_random,
        },
        "profit_curve": profit_curve(tau, value, settings.cost_per_treatment_usd),
    }

    print(f"\nTARGETING POLICY + OFF-POLICY EVALUATION - model = {model}, n = {len(df):,}")
    print(RULE)
    print(f"  {'policy':<24}{'treated':>9}{'value (DR)':>13}{'95% CI':>26}{'value (IPW)':>13}")
    for r in rows:
        ci = f"[{r['dr_ci'][0]:.5f}, {r['dr_ci'][1]:.5f}]"
        print(
            f"  {r['policy']:<24}{r['treated_fraction']:>9.3f}{r['value_dr']:>13.5f}"
            f"{ci:>26}{r['value_ipw']:>13.5f}"
        )
    print(
        f"\n  model targeting vs random at the SAME budget: {diff:+.6f} "
        f"(SE {se_diff:.6f}) -> "
        f"{'BEATS random' if beats_random else 'NOT distinguishable from random'}"
    )
    print(
        f"\n  ASSUMPTIONS: ${settings.value_per_conversion_usd:.2f} per conversion"
        f" -> ${value:.4f} per incremental {outcome}"
        f" (x{settings.conversions_per_visit:.4f} P(convert|visit)), "
        f"${settings.cost_per_treatment_usd:.2f} per treatment"
    )
    print(RULE)

    if write:
        settings.ensure_dirs()
        p = settings.evals_dir / "policy.json"
        p.write_text(json.dumps(out, indent=2, default=float))
        log.info("wrote", path=str(p))
    return out
