"""ITT versus CACE, using `exposure` as the compliance indicator.

This module exists because of the `exposure` column, and it is where the repo
shows it did not fall into the trap.

The structure is  treatment (random assignment, Z) -> exposure (an ad was
actually shown, D) -> visit (outcome, Y). Not every assigned user gets shown an
ad, so there is one-sided non-compliance: P(D = 1 | Z = 0) = 0 by design, which
is verified empirically in docs/findings.md.

On this data the first stage is only about 0.036, so CACE is roughly 28x the
ITT. That makes the distinction quantitatively dramatic rather than academic -
and it makes conditioning on `exposure` catastrophic rather than merely sloppy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class IVResult:
    itt: float
    itt_se: float
    itt_ci: tuple[float, float]
    first_stage: float
    cace: float
    cace_se: float
    cace_ci: tuple[float, float]
    naive_exposed_vs_control: float
    naive_bias_vs_cace: float
    naive_bias_ratio: float
    n: int

    def summary(self) -> str:
        return (
            f"ITT   = {self.itt:+.6f} [{self.itt_ci[0]:+.6f}, {self.itt_ci[1]:+.6f}]\n"
            f"  first stage P(exposed|assigned) = {self.first_stage:.6f}\n"
            f"  CACE  = {self.cace:+.6f} [{self.cace_ci[0]:+.6f}, {self.cace_ci[1]:+.6f}]"
            f"  ({self.cace / self.itt:.1f}x the ITT)\n"
            f"  naive exposed-vs-control = {self.naive_exposed_vs_control:+.6f}  "
            f"-> {self.naive_bias_ratio:.2f}x the CACE  [INVALID: conditions on a "
            f"post-treatment variable]"
        )

    def to_dict(self) -> dict:
        return asdict(self)


def itt_and_cace(y: np.ndarray, z: np.ndarray, d: np.ndarray, alpha: float = 0.05) -> IVResult:
    """z = randomized assignment, d = actual exposure, y = outcome.

    ITT  = E[Y|Z=1] - E[Y|Z=0]           <- always valid, randomization only
    CACE = ITT / (E[D|Z=1] - E[D|Z=0])   <- the Wald estimator

    Under one-sided non-compliance E[D|Z=0] = 0, so CACE = ITT / P(D=1|Z=1).

    Assumptions for CACE: relevance (nonzero first stage - satisfied, but only
    just, at 0.036), exclusion (assignment affects Y only through exposure),
    monotonicity (no defiers), and independence (holds by randomization).

    Exclusion is the one that is NOT guaranteed and cannot be tested: being
    assigned to the campaign could affect a user through channels other than
    seeing this ad - retargeting bid changes, frequency capping elsewhere.
    Naming an untestable assumption beats pretending it holds automatically.
    """
    y = np.asarray(y, dtype=float)
    y1, y0 = y[z == 1], y[z == 0]
    itt = y1.mean() - y0.mean()
    itt_se = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))

    first_stage = d[z == 1].mean() - d[z == 0].mean()
    if first_stage <= 0:
        raise ValueError(f"first stage is {first_stage:.6f}; the instrument is irrelevant")
    cace = itt / first_stage
    # Delta method. The first stage is estimated from 14M rows so its own
    # variance is negligible relative to the ITT's; at a weaker n this term
    # would need the full two-term delta expansion.
    cace_se = itt_se / abs(first_stage)

    zcrit = stats.norm.ppf(1 - alpha / 2)
    naive = y[d == 1].mean() - y0.mean()

    return IVResult(
        itt=float(itt),
        itt_se=float(itt_se),
        itt_ci=(float(itt - zcrit * itt_se), float(itt + zcrit * itt_se)),
        first_stage=float(first_stage),
        cace=float(cace),
        cace_se=float(cace_se),
        cace_ci=(float(cace - zcrit * cace_se), float(cace + zcrit * cace_se)),
        naive_exposed_vs_control=float(naive),
        naive_bias_vs_cace=float(naive - cace),
        naive_bias_ratio=float(naive / cace) if cace != 0 else float("nan"),
        n=len(y),
    )


def estimand_table(r: IVResult) -> list[dict]:
    """The three numbers side by side, with what each one actually answers."""
    return [
        {
            "estimand": "ITT",
            "estimate": r.itt,
            "answers": "What happens if we launch the campaign?",
            "valid": "Always - pure randomization",
        },
        {
            "estimand": "CACE",
            "estimate": r.cace,
            "answers": "What is the effect on users who actually see an ad?",
            "valid": "Under exclusion + monotonicity (exclusion is untestable)",
        },
        {
            "estimand": "exposed-vs-control",
            "estimate": r.naive_exposed_vs_control,
            "answers": "Nothing",
            "valid": "INVALID - conditions on a post-treatment variable",
        },
    ]


def heterogeneous_cace(X, y, z, d, seed: int = 0, cv: int = 3, max_n: int = 300_000):
    """Heterogeneous complier effects via EconML's intent-to-treat DRIV.

    Subsampled hard: DRIV cross-fits three nuisance models, and the first stage
    here is 0.036, so the estimator is working with very little signal.
    """
    from econml.iv.dr import LinearIntentToTreatDRIV
    from lightgbm import LGBMClassifier, LGBMRegressor

    X, y, z, d = map(np.asarray, (X, y, z, d))
    if len(X) > max_n:
        idx = np.random.default_rng(seed).choice(len(X), max_n, replace=False)
        X, y, z, d = X[idx], y[idx], z[idx], d[idx]

    driv = LinearIntentToTreatDRIV(
        model_y_xw=LGBMRegressor(n_estimators=200, verbose=-1, random_state=seed),
        model_t_xwz=LGBMClassifier(n_estimators=200, verbose=-1, random_state=seed),
        flexible_model_effect=LGBMRegressor(n_estimators=200, verbose=-1, random_state=seed),
        cv=cv,
        random_state=seed,
    )
    driv.fit(Y=y.astype(float), T=d, Z=z, X=X)
    return driv
