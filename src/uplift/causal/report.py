"""`make causal` - the bias table, the strength curve, IV/CACE, and sensitivity.

The bias table is the single best artifact in this project. Everything else in
this module supports it.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from uplift.causal.confounding import inject_confounding
from uplift.causal.estimators import (
    aipw_ate,
    check_overlap,
    fit_propensity,
    ipw_ate,
    matching_ate,
    naive_ate,
)
from uplift.causal.iv import estimand_table, itt_and_cace
from uplift.causal.sensitivity import (
    e_value,
    negative_control_panel,
    risk_ratio,
)
from uplift.config import settings
from uplift.data.io import load_iv_split
from uplift.logging import get_logger

log = get_logger(__name__)
RULE = "─" * 78
STRENGTHS = (0.0, 0.5, 1.0, 2.0, 4.0)


def bias_table(
    X, w, y, strength: float = 1.0, seed: int = 0, misspecified: bool = False
) -> dict[str, Any]:
    """Run the observational toolkit on a confounded slice and score every
    estimator against the RCT-derived ground truth for that slice."""
    cs = inject_confounding(X, w, y, strength=strength, seed=seed)
    Xo, wo, yo = X[cs.idx], w[cs.idx], y[cs.idx]
    truth = cs.ground_truth_ate

    ps = fit_propensity(Xo, wo, seed=seed, misspecified=misspecified)
    overlap = check_overlap(ps, wo)

    est: dict[str, tuple[float, float | None]] = {}
    est["naive difference-in-means"] = (naive_ate(yo, wo), None)
    est["IPW (clipped, stabilized)"] = (ipw_ate(yo, wo, ps), None)
    est["propensity matching (1:1)"] = (matching_ate(yo, wo, ps), None)
    dr, dr_se = aipw_ate(yo, wo, Xo, ps, seed=seed)
    est["AIPW / doubly robust"] = (dr, dr_se)

    rows = [
        {
            "estimator": "ground truth (from the RCT)",
            "estimate": truth,
            "se": cs.ground_truth_se,
            "bias": 0.0,
            "relative_error": 0.0,
        }
    ]
    for name, (v, se) in est.items():
        rows.append(
            {
                "estimator": name,
                "estimate": float(v),
                "se": se,
                "bias": float(v - truth),
                "relative_error": float((v - truth) / truth) if truth != 0 else float("nan"),
            }
        )
    return {
        "strength": strength,
        "misspecified_propensity": misspecified,
        "n_kept": cs.n_kept,
        "n_original": cs.n_original,
        "ground_truth": truth,
        "ground_truth_se": cs.ground_truth_se,
        "overlap": overlap,
        "rows": rows,
        "_arrays": (Xo, wo, yo, ps),  # for the negative-control panel; stripped before writing
    }


def causal_report(
    sample: int = 2_000_000, outcome: str = "visit", seed: int | None = None, write: bool = True
) -> dict[str, Any]:
    seed = settings.random_seed if seed is None else seed
    X, z, d, y = load_iv_split(split=None, outcome=outcome, sample=sample, seed=seed)
    log.info("loaded", rows=len(y), outcome=outcome)
    out: dict[str, Any] = {"outcome": outcome, "n": len(y), "sample": sample}

    # ---------------- 1. ITT vs CACE ----------------
    iv = itt_and_cace(y, z, d)
    out["iv"] = iv.to_dict()
    out["estimand_table"] = estimand_table(iv)

    # ---------------- 2. the bias table ----------------
    main = bias_table(X, z, y, strength=1.0, seed=seed)
    Xo, wo, yo, ps = main.pop("_arrays")
    out["bias_table"] = main

    # ---------------- 3. bias vs confounding strength ----------------
    curve = []
    for s in STRENGTHS:
        r = bias_table(X, z, y, strength=s, seed=seed)
        r.pop("_arrays")
        curve.append(
            {
                "strength": s,
                "n_kept": r["n_kept"],
                "ground_truth": r["ground_truth"],
                **{row["estimator"]: row["bias"] for row in r["rows"] if row["bias"] != 0.0},
                "overlap_frac_below_01": r["overlap"]["frac_below"],
                "overlap_ps_min": r["overlap"]["ps_min"],
                "frac_treated_outside_control_support": r["overlap"][
                    "frac_treated_outside_control_support"
                ],
            }
        )
        log.info("strength_done", strength=s, n_kept=r["n_kept"])
    out["strength_curve"] = curve

    # ---------------- 4. doubly robust under a misspecified propensity ----------------
    mis = bias_table(X, z, y, strength=2.0, seed=seed, misspecified=True)
    mis.pop("_arrays")
    out["misspecified"] = mis

    # ---------------- 5. negative controls ----------------
    nc_rct = negative_control_panel(X, z, list(settings.feature_cols))
    nc_conf = negative_control_panel(Xo, wo, list(settings.feature_cols))
    ipw_w = np.where(wo == 1, 1.0 / np.clip(ps, 0.02, 0.98), 1.0 / (1 - np.clip(ps, 0.02, 0.98)))
    nc_adj = negative_control_panel(Xo, wo, list(settings.feature_cols), weights=ipw_w)
    out["negative_controls"] = {
        "rct": nc_rct.to_dict(orient="records"),
        "confounded": nc_conf.to_dict(orient="records"),
        "after_ipw": nc_adj.to_dict(orient="records"),
        "max_abs_z": {
            "rct": float(nc_rct["z"].abs().max()),
            "confounded": float(nc_conf["z"].abs().max()),
            "after_ipw": float(nc_adj["z"].abs().max()),
        },
    }

    # ---------------- 6. E-value ----------------
    rr, rr_lo, _ = risk_ratio(yo, wo)
    dr_row = next(r for r in main["rows"] if r["estimator"] == "AIPW / doubly robust")
    out["e_value"] = e_value(rr, rr_lo) | {"applies_to": "the confounded slice's adjusted estimate"}

    # ---------------- print ----------------
    print(f"\nCAUSAL VALIDATION REPORT - outcome = {outcome}, n = {len(y):,}")
    print(RULE)
    print("ITT vs CACE (the `exposure` trap)")
    print("  " + iv.summary().replace("\n", "\n  "))
    print()
    print(f"BIAS TABLE - confounding injected at strength 1.0, {main['n_kept']:,} rows kept")
    print(f"  {'estimator':<30}{'estimate':>12}{'bias':>12}{'rel error':>12}")
    for r in main["rows"]:
        rel = "-" if r["estimator"].startswith("ground") else f"{100 * r['relative_error']:+.1f}%"
        print(f"  {r['estimator']:<30}{r['estimate']:>12.6f}{r['bias']:>12.6f}{rel:>12}")
    print(
        f"  overlap: ps in [{main['overlap']['ps_min']:.4f}, {main['overlap']['ps_max']:.4f}], "
        f"{100 * main['overlap']['frac_below']:.2f}% below 0.01"
    )
    print()
    print("BIAS vs CONFOUNDING STRENGTH")
    cdf = pd.DataFrame(curve)
    print("  " + cdf.to_string(index=False).replace("\n", "\n  "))
    print()
    print(
        "NEGATIVE CONTROLS  (max |z| over f0-f11; a pre-treatment covariate "
        "cannot be affected by treatment)"
    )
    m = out["negative_controls"]["max_abs_z"]
    print(f"  on the RCT              max|z| = {m['rct']:8.2f}   <- should pass")
    print(f"  on the confounded slice max|z| = {m['confounded']:8.2f}   <- should fail loudly")
    print(f"  after IPW reweighting   max|z| = {m['after_ipw']:8.2f}   <- should recover")
    print()
    print(
        f"E-VALUE  RR={out['e_value']['risk_ratio']:.4f}  "
        f"point={out['e_value']['e_value_point']:.3f}  "
        f"ci={out['e_value'].get('e_value_ci', float('nan')):.3f}"
    )
    print(
        f"  DR estimate on the slice: {dr_row['estimate']:.6f} (truth {main['ground_truth']:.6f})"
    )
    print(RULE)

    if write:
        settings.ensure_dirs()
        path = settings.evals_dir / "causal.json"
        path.write_text(json.dumps(out, indent=2, default=float))
        log.info("wrote", path=str(path))
    return out
