"""Is the -88% real bias, or is it noise?

Three things, in the order they have to be answered:

1. VERIFY THE IMPLEMENTATION. In-sample propensities or unstabilized
   Horvitz-Thompson weights would each produce the observed signature on their
   own, and both are one-line bugs. Confirm cross-fitting is genuinely
   out-of-fold and that the stabilized path is the Hajek estimator, before
   drawing any conclusion about propensity estimation as such.

2. TWENTY SEEDS AT STRENGTH ZERO. One seed cannot separate bias from variance.
   Mean +/- SD for IPW with the oracle propensity against IPW with an estimated
   one settles it.

3. THE NOISE FLOOR. At strength zero there is no confounding, so any spread is
   the estimator's own sampling distribution.

    uv run python scripts/null_calibration.py --sample 400000 --seeds 20
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from uplift.causal.confounding import inject_confounding, true_propensity_in_slice
from uplift.causal.estimators import aipw_ate, fit_propensity, ipw_ate, naive_ate
from uplift.config import settings
from uplift.data.io import load_iv_split
from uplift.logging import get_logger

log = get_logger(__name__)


def verify_implementation(X, w, seed: int = 0) -> dict:
    """Prove the two implementation details rather than trusting the docstring."""
    ps_oof = fit_propensity(X, w, seed=seed, cross_fit=True)
    ps_in = fit_propensity(X, w, seed=seed, cross_fit=False)

    # An in-sample fit memorises, so its scores are more extreme and correlate
    # with w far more than an out-of-fold one. If these were identical,
    # cross-fitting would not be happening.
    from sklearn.metrics import roc_auc_score

    auc_oof = float(roc_auc_score(w, ps_oof))
    auc_in = float(roc_auc_score(w, ps_in))

    # Hajek check: the stabilized estimator must equal the self-normalized form
    # computed by hand, and must NOT equal the Horvitz-Thompson form.
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.05, len(w)).astype(float)
    ps = np.clip(ps_oof, 0.02, 0.98)
    hajek = float(
        np.sum((w / ps) * y) / np.sum(w / ps)
        - np.sum(((1 - w) / (1 - ps)) * y) / np.sum((1 - w) / (1 - ps))
    )
    ht = float(np.mean(w * y / ps) - np.mean((1 - w) * y / (1 - ps)))
    got_stab = ipw_ate(y, w, ps_oof, stabilized=True)
    got_ht = ipw_ate(y, w, ps_oof, stabilized=False)

    return {
        "auc_out_of_fold": auc_oof,
        "auc_in_sample": auc_in,
        "cross_fitting_is_real": bool(auc_in - auc_oof > 0.005),
        "ps_oof_sd": float(ps_oof.std()),
        "ps_in_sample_sd": float(ps_in.std()),
        "stabilized_matches_hajek": bool(abs(got_stab - hajek) < 1e-9),
        "stabilized_differs_from_ht": bool(abs(got_stab - ht) > 1e-12),
        "ht_matches_horvitz_thompson": bool(abs(got_ht - ht) < 1e-9),
    }


def null_runs(X, z, y, n_seeds: int) -> pd.DataFrame:
    """Strength 0 repeated: no confounding, so spread is pure sampling noise."""
    rows = []
    p_w = float(z.mean())
    for seed in range(n_seeds):
        cs = inject_confounding(X, z, y, strength=0.0, seed=seed)
        Xo, wo, yo = X[cs.idx], z[cs.idx], y[cs.idx]
        ps_hat = fit_propensity(Xo, wo, seed=seed)
        ps_true = true_propensity_in_slice(cs.selection_score[cs.idx], p_w)
        truth = cs.ground_truth_ate

        dr_hat, _ = aipw_ate(yo, wo, Xo, ps_hat, seed=seed)
        dr_true, _ = aipw_ate(yo, wo, Xo, ps_true, seed=seed)
        rows.append(
            {
                "seed": seed,
                "truth": truth,
                "naive": naive_ate(yo, wo),
                "ipw_oracle": ipw_ate(yo, wo, ps_true),
                "ipw_estimated": ipw_ate(yo, wo, ps_hat),
                "aipw_oracle": dr_true,
                "aipw_estimated": dr_hat,
                "ps_hat_sd": float(ps_hat.std()),
            }
        )
        log.info("null_run", seed=seed)
    df = pd.DataFrame(rows)
    for c in ("naive", "ipw_oracle", "ipw_estimated", "aipw_oracle", "aipw_estimated"):
        df[f"bias_{c}"] = df[c] / df["truth"] - 1.0
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, default=400_000)
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args()

    X, z, _, y = load_iv_split(
        split=None, outcome="visit", sample=args.sample, seed=settings.random_seed
    )
    log.info("loaded", rows=len(y))

    print("=== 1. IMPLEMENTATION VERIFICATION ===")
    sub = slice(0, min(150_000, len(y)))
    ver = verify_implementation(X[sub], z[sub])
    for k, v in ver.items():
        print(f"  {k:<32}{v}")

    print(f"\n=== 2. {args.seeds} SEEDS AT STRENGTH 0 (no confounding) ===")
    df = null_runs(X, z, y, args.seeds)
    settings.ensure_dirs()
    df.to_csv(settings.evals_dir / "null_calibration.csv", index=False)

    print(f"\n  truth (mean over seeds): {df['truth'].mean():.6f}\n")
    print(f"  {'estimator':<18}{'mean':>11}{'SD':>11}{'mean bias':>12}{'SD of bias':>12}{'|t|':>7}")
    summary = {}
    for c in ("naive", "ipw_oracle", "ipw_estimated", "aipw_oracle", "aipw_estimated"):
        b = df[f"bias_{c}"]
        # t against the hypothesis that the mean bias is zero
        t = abs(b.mean()) / (b.std(ddof=1) / np.sqrt(len(b))) if b.std(ddof=1) > 0 else np.inf
        print(
            f"  {c:<18}{df[c].mean():>11.6f}{df[c].std(ddof=1):>11.6f}"
            f"{100 * b.mean():>11.1f}%{100 * b.std(ddof=1):>11.1f}%{t:>7.1f}"
        )
        summary[c] = {
            "mean": float(df[c].mean()),
            "sd": float(df[c].std(ddof=1)),
            "mean_bias": float(b.mean()),
            "sd_bias": float(b.std(ddof=1)),
            "t_vs_zero_bias": float(t),
        }

    (settings.evals_dir / "null_calibration.json").write_text(
        json.dumps(
            {"sample": args.sample, "seeds": args.seeds, "verification": ver, "summary": summary},
            indent=2,
        )
        + "\n"
    )
    print("\n  |t| >> 2 means the bias is real, not sampling noise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
