"""Sample ratio mismatch.

The interesting part of this module is not the chi-square test - it is knowing
what the test does at n = 14M, and validating it by simulation rather than
trusting it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class SRMResult:
    n_treatment: int
    n_control: int
    observed_ratio: float
    expected_ratio: float
    chi2: float
    p_value: float
    is_srm: bool
    abs_deviation: float
    ratio_ci: tuple[float, float]

    def summary(self) -> str:
        verdict = "SRM DETECTED" if self.is_srm else "no SRM"
        return (
            f"{verdict}: observed {self.observed_ratio:.7f} "
            f"(95% CI {self.ratio_ci[0]:.6f}-{self.ratio_ci[1]:.6f}) "
            f"vs designed {self.expected_ratio:.6f}; "
            f"chi2={self.chi2:.3f}, p={self.p_value:.4f}"
        )

    def to_dict(self) -> dict:
        return asdict(self)


def check_srm(
    n_treatment: int,
    n_control: int,
    expected_ratio: float = 0.85,
    alpha: float = 0.001,
) -> SRMResult:
    """Chi-square goodness-of-fit against the DESIGNED allocation.

    alpha defaults to 0.001, not 0.05. SRM checks run on every experiment and
    every metric; at 0.05 you would raise a false alarm on 1 experiment in 20.
    """
    n = n_treatment + n_control
    observed = np.array([n_treatment, n_control], dtype=float)
    expected = np.array([n * expected_ratio, n * (1.0 - expected_ratio)])

    chi2 = float(np.sum((observed - expected) ** 2 / expected))
    p = float(stats.chi2.sf(chi2, df=1))

    ratio = n_treatment / n
    lo, hi = stats.binomtest(n_treatment, n).proportion_ci(confidence_level=0.95)

    return SRMResult(
        n_treatment=n_treatment,
        n_control=n_control,
        observed_ratio=ratio,
        expected_ratio=expected_ratio,
        chi2=chi2,
        p_value=p,
        is_srm=p < alpha,
        abs_deviation=abs(ratio - expected_ratio),
        ratio_ci=(float(lo), float(hi)),
    )


def simulate_srm_calibration(
    n: int = 1_000_000,
    ratio: float = 0.85,
    n_sims: int = 1000,
    true_ratio: float | None = None,
    seed: int = 0,
) -> dict[str, float]:
    """Under H0 the rejection rate should equal alpha; under H1 it should be high.

    Calling this is what turns "I wrote an SRM check" into "I validated my SRM
    check". Run it once with true_ratio=None (expect ~0.001 and ~0.05) and once
    with true_ratio=0.8505 (expect near 1.0).
    """
    rng = np.random.default_rng(seed)
    actual = ratio if true_ratio is None else true_ratio
    draws = rng.binomial(n, actual, size=n_sims)
    results = [check_srm(int(t), n - int(t), expected_ratio=ratio) for t in draws]
    return {
        "n_per_sim": float(n),
        "true_ratio": float(actual),
        "designed_ratio": float(ratio),
        "rejection_rate_at_0.001": float(np.mean([r.is_srm for r in results])),
        "rejection_rate_at_0.05": float(np.mean([r.p_value < 0.05 for r in results])),
    }


def detectable_deviation(
    n: int, ratio: float = 0.85, alpha: float = 0.001, power: float = 0.80
) -> float:
    """The smallest deviation from `ratio` this test detects at n, with `power`.

    This is the number that makes the large-sample discussion concrete: if it
    comes out below the precision to which the design ratio was published, then
    a rejection tells you nothing about data quality.
    """
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    se = np.sqrt(ratio * (1 - ratio) / n)
    return float((z_a + z_b) * se)
