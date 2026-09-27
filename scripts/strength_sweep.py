"""Bias against confounding strength, with overlap diagnostics alongside.

A bias table at ONE injection strength says "my estimators failed". A sweep says
where they stop working and what predicts it, which is a result rather than an
embarrassment.

Every estimator is scored against ITS OWN estimand: IPW and AIPW against the
s(X)-weighted ATE, matching and ATT-weighted IPW against the s_t(X)-weighted
ATT. Scoring matching against the ATE is what made it look like the best
estimator in the table - see `ground_truths` in confounding.py.

    uv run python scripts/strength_sweep.py --sample 400000 --seeds 0 1
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from uplift.causal.confounding import inject_confounding, true_propensity_in_slice
from uplift.causal.diagnostics import overlap_report, overlap_verdict
from uplift.causal.estimators import (
    aipw_ate,
    fit_propensity,
    ipw_ate,
    ipw_att,
    matching_att,
    naive_ate,
)
from uplift.config import settings
from uplift.data.io import load_iv_split
from uplift.logging import get_logger

log = get_logger(__name__)

STRENGTHS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)


def run(sample: int, seeds: list[int], outcome: str = "visit") -> pd.DataFrame:
    X, z, _, y = load_iv_split(
        split=None, outcome=outcome, sample=sample, seed=settings.random_seed
    )
    log.info("loaded", rows=len(y))

    rows = []
    for strength in STRENGTHS:
        for seed in seeds:
            cs = inject_confounding(X, z, y, strength=strength, seed=seed)
            Xo, wo, yo = X[cs.idx], z[cs.idx], y[cs.idx]
            ps = fit_propensity(Xo, wo, seed=seed)
            # The ORACLE propensity, known in closed form because we built the
            # selection. Running both separates "the estimator failed" from
            # "the propensity estimate failed".
            ps_true = true_propensity_in_slice(cs.selection_score[cs.idx], float(z.mean()))
            ov = overlap_report(ps, wo)
            verdict = overlap_verdict(ov)

            dr, dr_se = aipw_ate(yo, wo, Xo, ps, seed=seed)
            dr_o, _ = aipw_ate(yo, wo, Xo, ps_true, seed=seed)
            ate, att = cs.ground_truth_ate, cs.ground_truth_att

            est = {
                "naive": (naive_ate(yo, wo), ate),
                "ipw": (ipw_ate(yo, wo, ps), ate),
                "aipw": (dr, ate),
                "matching": (matching_att(yo, wo, ps), att),
                "ipw_att": (ipw_att(yo, wo, ps), att),
                "ipw_oracle": (ipw_ate(yo, wo, ps_true), ate),
                "aipw_oracle": (dr_o, ate),
                "matching_oracle": (matching_att(yo, wo, ps_true), att),
            }
            row = {
                "strength": strength,
                "seed": seed,
                "n_kept": cs.n_kept,
                "truth_ate": ate,
                "truth_ate_se": cs.ground_truth_se,
                "truth_att": att,
                "truth_att_se": cs.ground_truth_att_se,
                "aipw_se": dr_se,
                "ess_frac_control": ov["ess_frac_control"],
                "ess_frac_treated": ov["ess_frac_treated"],
                "frac_clipped_any": ov["frac_clipped_any"],
                "common_support_gap": ov["common_support_gap"],
                "top1pct_weight_share_control": ov["top1pct_weight_share_control"],
                "overlap_verdict": verdict["overall"],
            }
            for name, (value, target) in est.items():
                row[name] = value
                # relative bias against the estimator's OWN estimand
                row[f"bias_{name}"] = (value - target) / target if target else np.nan
            rows.append(row)
            log.info(
                "done",
                strength=strength,
                seed=seed,
                overlap=verdict["overall"],
                bias_aipw=round(row["bias_aipw"], 3),
            )
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Mean over seeds, which is what gets published."""
    cols = [c for c in df.columns if c.startswith("bias_")] + [
        "ess_frac_control",
        "frac_clipped_any",
        "truth_ate",
        "truth_att",
        "n_kept",
    ]
    return df.groupby("strength")[cols].mean().reset_index()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, default=400_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--outcome", default="visit")
    args = ap.parse_args()

    df = run(args.sample, args.seeds, args.outcome)
    settings.ensure_dirs()
    raw = settings.evals_dir / "strength_sweep.csv"
    df.to_csv(raw, index=False)

    summary = summarise(df)
    pd.set_option("display.width", 220)
    print("\nBIAS vs CONFOUNDING STRENGTH (mean over seeds; each vs its OWN estimand)")
    show = summary[
        [
            "strength",
            "bias_naive",
            "bias_ipw",
            "bias_aipw",
            "bias_matching",
            "bias_ipw_oracle",
            "bias_aipw_oracle",
            "ess_frac_control",
            "frac_clipped_any",
        ]
    ].copy()
    for c in show.columns:
        if c.startswith("bias_"):
            show[c] = (100 * show[c]).round(1)
    print(show.to_string(index=False))

    # The headline strength: largest strength where AIPW is within 10% of truth.
    ok = summary[summary["bias_aipw"].abs() < 0.10]
    headline = float(ok["strength"].max()) if len(ok) else None
    print(
        f"\nLargest strength with |AIPW bias| < 10%: {headline}"
        if headline is not None
        else "\nNo strength has |AIPW bias| < 10% - report that, it is the finding."
    )

    (settings.evals_dir / "strength_sweep.json").write_text(
        json.dumps(
            {
                "sample": args.sample,
                "seeds": args.seeds,
                "headline_strength": headline,
                "summary": summary.to_dict(orient="records"),
            },
            indent=2,
            default=float,
        )
        + "\n"
    )
    print(f"wrote {raw} and strength_sweep.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
