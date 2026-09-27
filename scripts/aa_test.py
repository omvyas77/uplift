"""A/A test: split the control arm in half and label one half "treated".

The single best end-to-end validation available, and almost nobody runs it.
There is no effect to find, so ANY non-null result is a bug - in the pipeline,
the split, the metric, or the estimator. Unit tests on synthetic data cannot
catch what this catches, because this exercises the real pipeline on the real
file with a known answer.

It is also immune to the answer-key problem that invalidated the bias table
(findings section 16). Under a true null the effect is zero for the ADJUSTED and
the UNADJUSTED estimand alike, so there is no question of which target to score
against. That makes it the cleanest way to calibrate the noise floor and to
settle whether IPW's -91% is real.

Repeated `--reps` times, it gives the sampling distribution of every estimator
under a true null.

Note the fake assignment mimics the real 85/15 ratio, not 50/50, so the same
unbalanced-allocation code paths are exercised.

    uv run python scripts/aa_test.py --reps 20 --sample 400000
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from uplift.causal.estimators import aipw_ate, fit_propensity, ipw_ate, naive_ate
from uplift.config import settings
from uplift.data.ingest import connect
from uplift.data.io import FEATURES
from uplift.evaluation.qini import sklift_qini
from uplift.experiment.ate import ate_difference_in_means, ate_lin_regression
from uplift.experiment.balance import balance_from_arrays
from uplift.experiment.srm import check_srm
from uplift.logging import get_logger
from uplift.models.learners import ClassTransformation

log = get_logger(__name__)


def load_control_arm(sample: int, outcome: str = "visit") -> tuple[np.ndarray, np.ndarray]:
    """Only units that were never treated. There is no effect among them."""
    con = connect(read_only=True)
    try:
        # Subquery FIRST, sample second. DuckDB otherwise samples the whole
        # table and then applies the filter, which returned 15% of the
        # requested rows - the bug this script surfaced in uplift.data.io.
        df = con.execute(
            f"SELECT * FROM (SELECT {', '.join(FEATURES)}, {outcome} AS y "
            f"FROM units WHERE treatment = 0) "
            f"USING SAMPLE {sample} ROWS (reservoir, {settings.random_seed})"
        ).df()
    finally:
        con.close()
    return df[FEATURES].to_numpy(np.float32), df["y"].to_numpy(np.int8)


def one_rep(X: np.ndarray, y: np.ndarray, seed: int, share: float, train_frac: float) -> dict:
    rng = np.random.default_rng(seed)
    w = rng.binomial(1, share, len(y)).astype(np.int8)  # FAKE assignment

    row: dict[str, float] = {"seed": seed, "n": len(y)}

    # --- design checks: must pass, there is nothing wrong with this split ---
    srm = check_srm(int(w.sum()), int((1 - w).sum()), expected_ratio=share)
    row["srm_p"] = srm.p_value
    row["srm_flagged"] = float(srm.is_srm)
    row["max_abs_smd"] = float(balance_from_arrays(X, w)["smd"].abs().max())

    # --- effect estimators: every one must come back at zero ---
    dim = ate_difference_in_means(y, w)
    row["ate_naive"] = dim.ate
    row["ate_naive_ci_covers_0"] = float(dim.ci_low <= 0 <= dim.ci_high)
    row["ate_naive_p"] = dim.p_value

    idx = rng.choice(len(y), min(200_000, len(y)), replace=False)
    lin = ate_lin_regression(y[idx], w[idx], X[idx])
    row["ate_lin"] = lin.ate
    row["ate_lin_ci_covers_0"] = float(lin.ci_low <= 0 <= lin.ci_high)

    ps_true = np.full(len(w), share)  # known exactly: we assigned it
    ps_hat = fit_propensity(X, w, seed=seed)
    row["ps_hat_sd"] = float(ps_hat.std())
    row["ipw_oracle"] = ipw_ate(y, w, ps_true)
    row["ipw_estimated"] = ipw_ate(y, w, ps_hat)
    row["aipw_oracle"] = aipw_ate(y, w, X, ps_true, seed=seed)[0]
    row["aipw_estimated"] = aipw_ate(y, w, X, ps_hat, seed=seed)[0]
    row["naive_check"] = naive_ate(y, w)

    # --- uplift: Qini must be zero. Anything else is pipeline artifact. ---
    cut = int(train_frac * len(y))
    tr, te = slice(0, cut), slice(cut, None)
    model = ClassTransformation(seed=seed).fit(X[tr], w[tr], y[tr])
    tau = model.predict(X[te])
    row["qini"] = sklift_qini(y[te], tau, w[te])
    row["tau_sd"] = float(tau.std())
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--sample", type=int, default=400_000)
    ap.add_argument("--share", type=float, default=0.85, help="mimic the real allocation")
    ap.add_argument("--train-frac", type=float, default=0.6)
    args = ap.parse_args()

    X, y = load_control_arm(args.sample)
    log.info("control_arm_loaded", rows=len(y), outcome_rate=round(float(y.mean()), 6))

    rows = []
    for seed in range(args.reps):
        rows.append(one_rep(X, y, seed, args.share, args.train_frac))
        log.info("rep", seed=seed, qini=round(rows[-1]["qini"], 5))

    df = pd.DataFrame(rows)
    settings.ensure_dirs()
    df.to_csv(settings.evals_dir / "aa_test.csv", index=False)

    print(
        f"\nA/A TEST - {args.reps} reps, {len(y):,} control-arm units, fake {args.share:.0%} split"
    )
    print("Everything below must be NULL. The true effect is zero by construction.\n")

    print(f"  {'quantity':<20}{'mean':>12}{'SD':>12}{'|mean|/SE':>11}  verdict")
    checks = [
        ("ate_naive", 0.0),
        ("ate_lin", 0.0),
        ("ipw_oracle", 0.0),
        ("ipw_estimated", 0.0),
        ("aipw_oracle", 0.0),
        ("aipw_estimated", 0.0),
        ("qini", 0.0),
    ]
    summary = {}
    for name, target in checks:
        v = df[name]
        se = v.std(ddof=1) / np.sqrt(len(v))
        t = abs(v.mean() - target) / se if se > 0 else np.inf
        verdict = "NULL ok" if t < 3 else "*** NOT NULL ***"
        print(f"  {name:<20}{v.mean():>12.6f}{v.std(ddof=1):>12.6f}{t:>11.1f}  {verdict}")
        summary[name] = {
            "mean": float(v.mean()),
            "sd": float(v.std(ddof=1)),
            "t_vs_null": float(t),
            "is_null": bool(t < 3),
        }

    print(
        f"\n  SRM flagged in           {int(df['srm_flagged'].sum())}/{len(df)} reps "
        f"(expect ~0 at alpha={settings.srm_alpha})"
    )
    print(
        f"  naive ATE CI covered 0   {int(df['ate_naive_ci_covers_0'].sum())}/{len(df)} reps "
        f"(expect ~95%)"
    )
    print(
        f"  max |SMD|                mean {df['max_abs_smd'].mean():.5f}, "
        f"max {df['max_abs_smd'].max():.5f}"
    )
    print(
        f"  estimated ps sd          {df['ps_hat_sd'].mean():.5f} "
        f"(true ps is constant at {args.share})"
    )

    (settings.evals_dir / "aa_test.json").write_text(
        json.dumps(
            {
                "reps": args.reps,
                "sample": args.sample,
                "share": args.share,
                "srm_flagged": int(df["srm_flagged"].sum()),
                "naive_ci_coverage": float(df["ate_naive_ci_covers_0"].mean()),
                "max_abs_smd_mean": float(df["max_abs_smd"].mean()),
                "summary": summary,
            },
            indent=2,
        )
        + "\n"
    )
    failed = [k for k, v in summary.items() if not v["is_null"]]
    print(f"\n  {'ALL NULL' if not failed else 'NOT NULL: ' + ', '.join(failed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
