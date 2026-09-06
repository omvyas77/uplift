"""`make eval` - Qini with bootstrap CIs, pairwise comparisons, calibration.

Reads the score Parquet written by `uplift train`; never refits a model. That
keeps evaluation cheap enough to afford the bootstrap, and means the test split
is scored exactly once.

The reporting rule this module enforces: never publish a leaderboard without
intervals. On this data most models are inside each other's noise band, and a
ranked table with no CIs would manufacture a winner out of it.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any

import pandas as pd

from uplift.config import settings
from uplift.evaluation.calibration import calibration_summary, cate_calibration, response_auc
from uplift.evaluation.qini import (
    bootstrap_qini_ci,
    paired_bootstrap_difference,
    sklift_auuc,
    sklift_qini,
    uplift_at_k,
)
from uplift.logging import get_logger

log = get_logger(__name__)
RULE = "─" * 92


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def evaluate_and_write(
    outcome: str = "visit",
    n_boot: int = 200,
    eval_sample: int | None = None,
    split: str = "test",
    reference: str = "t_learner",
    write: bool = True,
) -> dict[str, Any]:
    path = settings.artifacts_dir / f"scores_{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `make train` first")
    df = pd.read_parquet(path)
    if eval_sample and eval_sample < len(df):
        df = df.sample(eval_sample, random_state=settings.random_seed).reset_index(drop=True)

    y = df["y"].to_numpy()
    w = df["w"].to_numpy()
    model_names = [c for c in df.columns if c not in ("y", "w")]
    log.info("evaluating", split=split, rows=len(df), models=len(model_names), n_boot=n_boot)

    results: dict[str, Any] = {}
    for name in model_names:
        tau = df[name].to_numpy()
        mean, lo, hi = bootstrap_qini_ci(y, w, tau, n_boot=n_boot, seed=settings.random_seed)
        cal = cate_calibration(y, w, tau)
        results[name] = {
            "qini": sklift_qini(y, tau, w),
            "qini_boot_mean": mean,
            "qini_ci": [lo, hi],
            "auuc": sklift_auuc(y, tau, w),
            "uplift_at_30pct": uplift_at_k(y, tau, w, k=0.3),
            # Labelled, not hidden: AUC measures who RESPONDS, not who responds
            # BECAUSE OF treatment. It is not the objective.
            "response_auc_not_the_objective": response_auc(y, tau),
            "score_mean": float(tau.mean()),
            "score_sd": float(tau.std()),
            **calibration_summary(cal),
            "calibration_table": cal.to_dict(orient="records"),
        }
        log.info("scored", model=name, qini=round(results[name]["qini"], 5))

    # ---- pairwise, paired bootstrap against the reference ----
    ref = reference if reference in model_names else model_names[0]
    for name in model_names:
        if name == ref:
            results[name]["vs_reference"] = "(reference)"
            continue
        d, lo, hi, resolved = paired_bootstrap_difference(
            y,
            w,
            df[name].to_numpy(),
            df[ref].to_numpy(),
            n_boot=n_boot,
            seed=settings.random_seed,
        )
        results[name]["vs_reference"] = {
            "reference": ref,
            "diff": d,
            "ci": [lo, hi],
            "resolved": resolved,
        }

    order = sorted(model_names, key=lambda m: -results[m]["qini"])
    n_resolved = sum(
        1
        for m in model_names
        if isinstance(results[m]["vs_reference"], dict) and results[m]["vs_reference"]["resolved"]
    )

    out = {
        "run_id": pd.Timestamp.utcnow().isoformat(),
        "git_sha": _git_sha(),
        "dataset": {
            "name": "criteo-uplift-v2.1",
            "rows": settings.expected_rows,
            "sha256": settings.criteo_sha256,
            "split": split,
            "n_eval": len(df),
        },
        "outcome": outcome,
        "n_boot": n_boot,
        "reference_model": ref,
        "ranking": order,
        "n_models_resolved_vs_reference": n_resolved,
        "models": results,
    }

    # ---------------- print ----------------
    print(f"\nUPLIFT MODEL EVALUATION - outcome = {outcome}, split = {split}, n = {len(df):,}")
    print(RULE)
    print(
        f"  {'model':<24}{'Qini':>9}{'95% CI':>22}{'AUUC':>9}{'u@30%':>9}"
        f"{'calib slope':>13}{'AUC*':>7}"
    )
    for m in order:
        r = results[m]
        ci = f"[{r['qini_ci'][0]:+.4f}, {r['qini_ci'][1]:+.4f}]"
        print(
            f"  {m:<24}{r['qini']:>9.4f}{ci:>22}{r['auuc']:>9.4f}"
            f"{r['uplift_at_30pct']:>9.4f}{r['calibration_slope']:>13.3f}"
            f"{r['response_auc_not_the_objective']:>7.3f}"
        )
    print(
        "\n  * response AUC - measures who responds, NOT who responds because of "
        "treatment. Not the objective."
    )
    print(f"\n  paired bootstrap vs {ref}:")
    for m in order:
        v = results[m]["vs_reference"]
        if not isinstance(v, dict):
            print(f"    {m:<24} (reference)")
            continue
        verdict = "RESOLVED" if v["resolved"] else "overlapping - indistinguishable"
        print(f"    {m:<24} {v['diff']:+.5f}  [{v['ci'][0]:+.5f}, {v['ci'][1]:+.5f}]  {verdict}")
    print(
        f"\n  {n_resolved} of {len(model_names) - 1} models are distinguishable from the "
        f"reference at n = {len(df):,}."
    )
    print(RULE)

    if write:
        settings.ensure_dirs()
        p = settings.evals_dir / "results.json"
        p.write_text(json.dumps(out, indent=2, default=float))
        log.info("wrote", path=str(p))
    return out
