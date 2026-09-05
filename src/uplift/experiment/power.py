"""Power, MDE, and allocation efficiency.

This module contains the most quotable finding in the project: an 85/15 split
is 4*r*(1-r) = 51% as statistically efficient as 50/50. Half the sample is
effectively wasted from a pure-precision standpoint - and the design is still
correct, because a business does not want to withhold ads from half its users
just to tighten a confidence interval. The 85/15 allocation buys revenue with
precision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class PowerResult:
    mde_absolute: float
    mde_relative: float
    n_total: int
    treatment_share: float
    baseline_rate: float
    power: float
    alpha: float
    allocation_efficiency: float
    n_equivalent_balanced: int

    def summary(self) -> str:
        return (
            f"n={self.n_total:,}  share={self.treatment_share:.2f}  "
            f"p0={self.baseline_rate:.6f}  "
            f"MDE={self.mde_absolute:.6f} abs / {100 * self.mde_relative:.3f}% rel  "
            f"efficiency={self.allocation_efficiency:.3f} "
            f"(= {self.n_equivalent_balanced:,} balanced units)"
        )

    def to_dict(self) -> dict:
        return asdict(self)


def mde(
    n_total: int,
    baseline_rate: float,
    treatment_share: float = 0.5,
    power: float = 0.80,
    alpha: float = 0.05,
) -> PowerResult:
    """Minimum detectable effect for a two-proportion test with unequal allocation."""
    n_t = n_total * treatment_share
    n_c = n_total * (1 - treatment_share)
    p = baseline_rate
    var = p * (1 - p) / n_t + p * (1 - p) / n_c

    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    mde_abs = (z_a + z_b) * np.sqrt(var)

    # Efficiency of an r/(1-r) split relative to 50/50 is 4*r*(1-r).
    eff = 4 * treatment_share * (1 - treatment_share)

    return PowerResult(
        mde_absolute=float(mde_abs),
        mde_relative=float(mde_abs / p),
        n_total=int(n_total),
        treatment_share=float(treatment_share),
        baseline_rate=float(p),
        power=float(power),
        alpha=float(alpha),
        allocation_efficiency=float(eff),
        n_equivalent_balanced=int(n_total * eff),
    )


def required_n(
    baseline_rate: float,
    mde_relative: float,
    treatment_share: float = 0.5,
    power: float = 0.80,
    alpha: float = 0.05,
) -> int:
    p = baseline_rate
    delta = p * mde_relative
    z_a, z_b = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    per_unit_var = p * (1 - p) * (1 / treatment_share + 1 / (1 - treatment_share))
    return int(np.ceil(((z_a + z_b) ** 2 * per_unit_var) / delta**2))


def allocation_table(
    n_total: int, baseline_rate: float, shares=(0.5, 0.85), power: float = 0.80, alpha: float = 0.05
) -> list[dict]:
    """The README table: what the 85/15 design costs in precision."""
    rows = []
    ref = mde(n_total, baseline_rate, 0.5, power, alpha)
    for s in shares:
        r = mde(n_total, baseline_rate, s, power, alpha)
        rows.append(
            {
                "treatment_share": s,
                "allocation_efficiency": r.allocation_efficiency,
                "variance_multiple_vs_balanced": (ref.mde_absolute / r.mde_absolute) ** -2,
                "mde_relative_pct": 100 * r.mde_relative,
                "n_multiple_for_same_mde": 1.0 / r.allocation_efficiency,
            }
        )
    return rows
