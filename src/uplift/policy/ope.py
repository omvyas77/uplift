"""Off-policy evaluation on held-out RANDOMIZED data.

This is the honest number. The propensity is known exactly because it was
assigned by design - which is the rare, comfortable case, and it is worth saying
out loud that a production logging policy would require estimating it.
"""

from __future__ import annotations

import numpy as np


def ipw_policy_value(y, w, policy_treat, propensity):
    """Value of a deterministic policy under randomized logging.

    Only units whose ACTUAL assignment matches what the policy would do are
    informative; each is reweighted by the probability of that match.
    """
    y = np.asarray(y, dtype=float)
    policy_treat = np.asarray(policy_treat).astype(int)
    p = np.where(policy_treat == 1, propensity, 1 - propensity)
    match = np.asarray(w).astype(int) == policy_treat
    vals = np.where(match, y / p, 0.0)
    return float(vals.mean()), float(vals.std(ddof=1) / np.sqrt(len(vals)))


def dr_policy_value(y, w, policy_treat, propensity, mu0, mu1):
    """Doubly robust policy value.

    Lower variance than IPW, and unbiased if either the outcome model or the
    propensity is right - and here the propensity IS right, by design.
    """
    y = np.asarray(y, dtype=float)
    policy_treat = np.asarray(policy_treat).astype(int)
    mu_pi = np.where(policy_treat == 1, mu1, mu0)
    p = np.where(policy_treat == 1, propensity, 1 - propensity)
    match = np.asarray(w).astype(int) == policy_treat
    scores = mu_pi + match * (y - mu_pi) / p
    return float(scores.mean()), float(scores.std(ddof=1) / np.sqrt(len(scores)))


def baseline_policies(tau_hat, threshold: float, seed: int = 0) -> dict[str, np.ndarray]:
    """The four policies to compare, including the one most projects omit.

    `random_same_budget` is the row that matters: if the model's policy does not
    beat random targeting at the SAME budget, the model adds nothing. Finding
    that out is a legitimate result, not a failure.
    """
    tau_hat = np.asarray(tau_hat)
    model = tau_hat >= threshold
    frac = float(model.mean())
    rng = np.random.default_rng(seed)
    return {
        "treat_nobody": np.zeros(len(tau_hat), dtype=bool),
        "treat_everybody": np.ones(len(tau_hat), dtype=bool),
        "model_targeting": model,
        "random_same_budget": rng.random(len(tau_hat)) < frac,
    }
