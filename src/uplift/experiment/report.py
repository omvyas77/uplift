"""The `make experiment` readout: SRM, balance, ATE, variance reduction, power."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import numpy as np

from uplift.config import settings
from uplift.data.io import load_split
from uplift.experiment.ate import ate_difference_in_means, ate_from_adjusted, ate_lin_regression
from uplift.experiment.balance import balance_from_arrays
from uplift.experiment.cuped import build_cupac_covariate, evaluate_cuped
from uplift.experiment.power import allocation_table, mde
from uplift.experiment.srm import check_srm, detectable_deviation, simulate_srm_calibration
from uplift.logging import get_logger

log = get_logger(__name__)
RULE = "─" * 78


def experiment_report(
    outcome: str = "visit",
    sample: int | None = None,
    lin_sample: int = 2_000_000,
    cupac_sample: int = 2_000_000,
    run_calibration: bool = True,
    write: bool = True,
) -> dict[str, Any]:
    """Run the full experiment analysis and print the readout.

    `lin_sample` / `cupac_sample` bound the two estimators that build dense
    design matrices; on 14M x 12 the Lin design matrix alone is 14M x 26
    float64 = 2.9 GB, which does not fit alongside everything else on an 8GB
    box. The difference-in-means headline always uses every row.
    """
    X, w, y = load_split(split=None, outcome=outcome, sample=sample)
    n = len(y)
    n_t, n_c = int(w.sum()), int((1 - w).sum())
    log.info("loaded", rows=n, outcome=outcome)

    out: dict[str, Any] = {"outcome": outcome, "n": n, "n_treatment": n_t, "n_control": n_c}

    # ---------------- SRM ----------------
    srm = check_srm(n_t, n_c, settings.designed_treatment_ratio, settings.srm_alpha)
    out["srm"] = srm.to_dict()
    out["srm"]["detectable_deviation_at_80pct_power"] = detectable_deviation(
        n, settings.designed_treatment_ratio, settings.srm_alpha
    )
    if run_calibration:
        out["srm_calibration_null"] = simulate_srm_calibration(n=200_000, n_sims=1000, seed=3)
        out["srm_calibration_alt"] = simulate_srm_calibration(
            n=200_000, n_sims=1000, true_ratio=0.8505, seed=3
        )

    # ---------------- balance ----------------
    bal = balance_from_arrays(X, w, list(settings.feature_cols))
    out["balance"] = bal.to_dict(orient="records")
    out["max_abs_smd"] = float(bal["smd"].abs().max())

    # ---------------- ATE ----------------
    dim = ate_difference_in_means(y, w)
    results = [dim]
    out["ate"] = {"difference_in_means": dim.to_dict()}

    rng = np.random.default_rng(settings.random_seed)
    idx = rng.choice(n, lin_sample, replace=False) if n > lin_sample else np.arange(n)
    # Difference-in-means on the SAME subsample, so that the comparison against
    # Lin is apples-to-apples. Without this row the reader cannot tell whether a
    # gap between the estimators is adjustment or subsampling.
    if len(idx) < n:
        dim_sub = ate_difference_in_means(y[idx], w[idx])
        dim_sub = replace(dim_sub, method="difference-in-means (sub)")
        results.append(dim_sub)
        out["ate"]["difference_in_means_subsample"] = dim_sub.to_dict() | {"n_used": len(idx)}
    lin = ate_lin_regression(y[idx], w[idx], X[idx])
    lin_note = f" (adjusted estimators on {len(idx):,} sampled rows)" if len(idx) < n else ""
    results.append(lin)
    out["ate"]["lin"] = lin.to_dict() | {"n_used": len(idx)}

    # ---------------- CUPAC ----------------
    cidx = rng.choice(n, cupac_sample, replace=False) if n > cupac_sample else np.arange(n)
    cov = build_cupac_covariate(X[cidx], y[cidx].astype(float), w[cidx], seed=settings.random_seed)
    cuped = evaluate_cuped(y[cidx].astype(float), w[cidx], cov)
    y_adj = y[cidx].astype(float) - cuped.theta * (cov - cov.mean())
    cupac_ate = ate_from_adjusted(y_adj, w[cidx], baseline_rate=float(y[cidx][w[cidx] == 0].mean()))
    results.append(cupac_ate)
    out["cuped"] = cuped.to_dict() | {"n_used": len(cidx)}
    out["ate"]["cupac"] = cupac_ate.to_dict() | {"n_used": len(cidx)}

    # ---------------- power ----------------
    p0 = float(y[w == 0].mean())
    design = mde(n, p0, settings.designed_treatment_ratio)
    out["power"] = design.to_dict()
    out["allocation_table"] = allocation_table(n, p0)

    # ---------------- print ----------------
    print(f"\nEXPERIMENT READOUT - Criteo Uplift v2.1, primary outcome = {outcome}")
    print(RULE)
    print(f"Assignment    n_treatment={n_t:,}  n_control={n_c:,}  ratio={srm.observed_ratio:.7f}")
    print(f"SRM           {srm.summary()}")
    print(
        f"              this test detects deviations >= "
        f"{out['srm']['detectable_deviation_at_80pct_power']:.2e} at 80% power"
    )
    if run_calibration:
        print(
            f"              calibration: null rejects at "
            f"{out['srm_calibration_null']['rejection_rate_at_0.001']:.4f} (alpha=0.001) / "
            f"{out['srm_calibration_null']['rejection_rate_at_0.05']:.4f} (alpha=0.05); "
            f"ratio=0.8505 rejects at "
            f"{out['srm_calibration_alt']['rejection_rate_at_0.001']:.3f}"
        )
    print(
        f"Balance       max |SMD| = {out['max_abs_smd']:.5f} across f0-f11  "
        f"(threshold {settings.smd_threshold})"
    )
    print(f"\nATE ({outcome}){lin_note}")
    for r in results:
        print("  " + r.summary())
    print(f"\nVariance red. CUPAC {cuped.summary()}")
    print(f"Design        {design.summary()}")
    eff = design.allocation_efficiency
    print(
        f"              -> the {settings.designed_treatment_ratio:.0%}/"
        f"{1 - settings.designed_treatment_ratio:.0%} split is {eff:.0%} as efficient as 50/50"
    )
    print(RULE)

    if write:
        settings.ensure_dirs()
        # Per-outcome filename. A single experiment.json meant that running the
        # secondary outcome silently destroyed the primary one's readout.
        path = settings.evals_dir / f"experiment_{outcome}.json"
        path.write_text(json.dumps(out, indent=2, default=float))
        log.info("wrote", path=str(path))
    return out
