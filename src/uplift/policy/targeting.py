"""From model scores to a decision.

A Qini curve is not a decision. A policy is.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class TargetingPolicy:
    threshold: float
    treated_fraction: float
    expected_incremental_outcomes: float
    expected_profit: float
    value_per_outcome: float
    cost_per_treatment: float

    def to_dict(self) -> dict:
        return asdict(self)


def optimal_threshold(
    tau_hat: np.ndarray,
    value_per_outcome: float,
    cost_per_treatment: float,
    grid: int = 200,
) -> TargetingPolicy:
    """Treat unit i when tau_hat_i * value > cost.

    The break-even threshold is cost/value; the grid sweep exists so the
    dashboard can show the profit curve and so the decision is visibly a
    business decision rather than a statistical one.

    NOTE the trap this walks into on purpose: `expected_profit` is computed from
    the model's OWN predictions. If the model is overconfident, so is this
    number. That is exactly why `uplift.policy.ope` exists and why the README
    quotes the off-policy value, not this one.
    """
    tau_hat = np.asarray(tau_hat, dtype=float)
    candidates = np.quantile(tau_hat, np.linspace(0, 1, grid))
    best: TargetingPolicy | None = None
    for c in candidates:
        treat = tau_hat >= c
        if treat.sum() == 0:
            continue
        inc = float(tau_hat[treat].sum())
        profit = inc * value_per_outcome - treat.sum() * cost_per_treatment
        if best is None or profit > best.expected_profit:
            best = TargetingPolicy(
                threshold=float(c),
                treated_fraction=float(treat.mean()),
                expected_incremental_outcomes=inc,
                expected_profit=float(profit),
                value_per_outcome=float(value_per_outcome),
                cost_per_treatment=float(cost_per_treatment),
            )
    assert best is not None
    return best


def profit_curve(
    tau_hat, value_per_outcome: float, cost_per_treatment: float, grid: int = 50
) -> list[dict]:
    """Model-predicted profit across targeting depths, for the dashboard."""
    tau_hat = np.asarray(tau_hat, dtype=float)
    order = np.argsort(-tau_hat)
    sorted_tau = tau_hat[order]
    rows = []
    for frac in np.linspace(0.02, 1.0, grid):
        k = max(1, int(frac * len(tau_hat)))
        inc = float(sorted_tau[:k].sum())
        rows.append(
            {
                "treated_fraction": float(frac),
                "n_treated": k,
                "threshold": float(sorted_tau[k - 1]),
                "predicted_incremental_outcomes": inc,
                "predicted_profit": inc * value_per_outcome - k * cost_per_treatment,
            }
        )
    return rows
