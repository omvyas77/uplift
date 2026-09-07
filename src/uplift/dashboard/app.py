"""Uplift dashboard.

Five tabs, in the order an analyst would actually reason: what happened in the
experiment, what the models found, what decision follows, whether the causal
machinery can be trusted, and what the caveats are.

It reads `evals/*.json` and never retrains anything. The business assumptions are
sliders rather than constants, so the sensitivity of every dollar figure is
visible instead of hidden behind one hardcoded number.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from uplift.config import settings

st.set_page_config(page_title="Uplift", page_icon="📈", layout="wide")


@st.cache_data(show_spinner=False)
def load(name: str) -> dict[str, Any] | None:
    p: Path = settings.evals_dir / name
    return json.loads(p.read_text()) if p.exists() else None


def missing(name: str, how: str) -> None:
    st.warning(f"`evals/{name}` not found. Generate it with `{how}`.")


def pct(x: float) -> str:
    return f"{100 * x:+.2f}%"


experiment = load("experiment_visit.json")
conversion = load("experiment_conversion.json")
results = load("results.json")
causal = load("causal.json")
policy = load("policy.json")

st.title("Uplift — causal experimentation on 14M randomized records")
st.caption(
    "Criteo Uplift v2.1 · intention-to-treat on `treatment` · primary outcome `visit` · "
    "every number below is reproducible from a clean clone"
)

tabs = st.tabs(
    ["Experiment", "Uplift models", "Targeting policy", "Causal validation", "Method notes"]
)

# ─────────────────────────── 1. experiment ───────────────────────────
with tabs[0]:
    if not experiment:
        missing("experiment_visit.json", "make experiment")
    else:
        srm = experiment["srm"]
        c = st.columns(4)
        c[0].metric("Units", f"{experiment['n']:,}")
        c[1].metric("Treated share", f"{srm['observed_ratio']:.6f}")
        c[2].metric("SRM", "none" if not srm["is_srm"] else "DETECTED", f"p = {srm['p_value']:.3f}")
        c[3].metric(
            "Max |SMD|",
            f"{experiment['max_abs_smd']:.4f}",
            "above the 0.02 clean-RCT line",
            delta_color="inverse",
        )

        st.subheader("Average treatment effect")
        ate = pd.DataFrame(
            [
                {
                    "estimator": v.get("method", k),
                    "ATE": v["ate"],
                    "95% CI": f"[{v['ci_low']:+.6f}, {v['ci_high']:+.6f}]",
                    "relative": pct(v["relative_lift"]),
                    "SE": v["se"],
                    "n": v.get("n_used", v["n"]),
                }
                for k, v in experiment["ate"].items()
            ]
        )
        st.dataframe(ate, hide_index=True, width="stretch")
        st.info(
            "**Quote the adjusted number.** Under clean randomization adjustment only "
            "tightens the interval; here it moves the point estimate by 7.1 SEs, because "
            "the covariates are both imbalanced and strongly predictive. See Causal "
            "validation and `docs/findings.md`."
        )

        if conversion:
            st.subheader("Secondary outcome — conversion")
            cc = st.columns(3)
            cc[0].metric(
                "Unadjusted", pct(conversion["ate"]["difference_in_means"]["relative_lift"])
            )
            cc[1].metric("Adjusted (Lin)", pct(conversion["ate"]["lin"]["relative_lift"]))
            cc[2].metric("Design MDE", f"{100 * conversion['power']['mde_relative']:.2f}%")
            st.caption(
                "0.29% base rate. The adjustment moves the same direction as `visit` but is "
                "only 1.5 SEs — consistent in sign with a better-resolved result, not resolved."
            )

        st.subheader("Covariate balance")
        bal = pd.DataFrame(experiment["balance"])
        st.dataframe(bal, hide_index=True, width="stretch")

        st.subheader("Design efficiency")
        st.dataframe(pd.DataFrame(experiment["allocation_table"]), hide_index=True, width="stretch")
        st.caption(
            "The 85/15 split is 51% as efficient as 50/50 — and is still the right business "
            "call. It buys revenue with precision."
        )

# ─────────────────────────── 2. models ───────────────────────────
with tabs[1]:
    if not results:
        missing("results.json", "make train && make eval")
    else:
        st.subheader(f"Qini on the held-out test split (n = {results['dataset']['n_eval']:,})")
        rows = []
        for name in results["ranking"]:
            m = results["models"][name]
            ref = m.get("vs_reference")
            if isinstance(ref, dict):
                verdict = "resolved" if ref["resolved"] else "overlapping — indistinguishable"
                delta = f"{ref['diff']:+.5f}"
            else:
                verdict, delta = "(reference)", "—"
            rows.append(
                {
                    "model": name,
                    "Qini": round(m["qini"], 5),
                    "95% CI": f"[{m['qini_ci'][0]:+.4f}, {m['qini_ci'][1]:+.4f}]",
                    "vs T-learner": delta,
                    "verdict": verdict,
                    "calib. slope": round(m["calibration_slope"], 3),
                    "response AUC*": round(m["response_auc_not_the_objective"], 3),
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(
            "\\* Response AUC measures who RESPONDS, not who responds BECAUSE OF treatment. "
            "Shown only so it can be labelled as not the objective."
        )
        st.info(
            f"**{results['n_models_resolved_vs_reference']} of 5** models are distinguishable "
            "from the reference. The DR-learner is not — publishing this leaderboard without "
            "intervals would have declared a winner over a gap the data cannot resolve."
        )

        pick = st.selectbox("CATE calibration", results["ranking"])
        cal = pd.DataFrame(results["models"][pick]["calibration_table"])
        if not cal.empty:
            st.line_chart(cal.set_index("decile")[["predicted_uplift", "observed_uplift"]])
            st.dataframe(cal, hide_index=True, width="stretch")

# ─────────────────────────── 3. policy ───────────────────────────
with tabs[2]:
    if not policy:
        missing("policy.json", "make policy")
    else:
        st.subheader("Business assumptions")
        st.caption("Every dollar figure below is a function of these two numbers.")
        c1, c2 = st.columns(2)
        value_conv = c1.slider(
            "Value per conversion ($)",
            1.0,
            100.0,
            float(policy["assumptions"]["value_per_conversion_usd"]),
            1.0,
        )
        cost = c2.slider(
            "Cost per treatment ($)",
            0.001,
            0.10,
            float(policy["assumptions"]["cost_per_treatment_usd"]),
            0.001,
            format="%.3f",
        )
        per_visit = value_conv * settings.conversions_per_visit
        st.caption(
            f"A visit converts with probability {settings.conversions_per_visit:.4f}, so an "
            f"incremental **visit** is worth **${per_visit:.4f}** — not ${value_conv:.2f}. "
            "Pricing a visit at the conversion price overstated profit 20x before this was fixed."
        )

        st.subheader("Off-policy value on held-out randomized data")
        pv_rows = [
            {
                "policy": v["policy"],
                "treated": f"{v['treated_fraction']:.1%}",
                "value (DR)": round(v["value_dr"], 5),
                "95% CI": f"[{v['dr_ci'][0]:.5f}, {v['dr_ci'][1]:.5f}]",
                "value (IPW)": round(v["value_ipw"], 5),
            }
            for v in policy["policy_values"]
        ]
        st.dataframe(pd.DataFrame(pv_rows), hide_index=True, width="stretch")
        mvr = policy["model_vs_random_same_budget"]
        st.success(
            f"Model targeting beats random at the SAME budget by "
            f"{mvr['diff_dr']:+.5f} (SE {mvr['se']:.5f}) — "
            f"{'resolved' if mvr['resolved'] else 'NOT resolved'}. "
            "This is the row most projects omit."
        )
        st.warning(
            "But **treat-everybody has the highest raw value.** With a positive effect nearly "
            "everywhere and a very cheap treatment, blanket treatment is hard to beat on outcome "
            "alone. The targeting case here is efficiency, not raw lift."
        )
        if policy.get("profit_curve"):
            pc = pd.DataFrame(policy["profit_curve"])
            pc["profit_at_slider"] = (
                pc["predicted_incremental_outcomes"] * per_visit - pc["n_treated"] * cost
            )
            st.line_chart(pc.set_index("treated_fraction")[["profit_at_slider"]])

# ─────────────────────────── 4. causal ───────────────────────────
with tabs[3]:
    if not causal:
        missing("causal.json", "make causal")
    else:
        st.subheader("The `exposure` trap — ITT vs CACE vs the invalid comparison")
        st.dataframe(pd.DataFrame(causal["estimand_table"]), hide_index=True, width="stretch")
        iv = causal["iv"]
        st.caption(
            f"First stage P(exposed | assigned) = {iv['first_stage']:.4f}. Only 3.6% of "
            f"targeted users were ever shown an ad, so CACE is {iv['cace'] / iv['itt']:.1f}x the "
            "ITT. Do not quote CACE to stakeholders — it describes 3.6% of the population and "
            "leans on an untestable exclusion restriction."
        )

        st.subheader("Bias table — observational estimators vs RCT ground truth")
        bt = pd.DataFrame(causal["bias_table"]["rows"])
        st.dataframe(bt, hide_index=True, width="stretch")
        st.error(
            "The naive comparison is not merely biased but **sign-flipped**: it reports a "
            "negative effect where the truth is positive."
        )

        st.subheader("Bias vs confounding strength")
        sc = pd.DataFrame(causal["strength_curve"])
        bias_cols = [c for c in sc.columns if c.startswith("bias[")]
        st.line_chart(sc.set_index("strength")[bias_cols])
        st.line_chart(sc.set_index("strength")[["frac_treated_outside_control_support"]])
        st.caption(
            "As overlap collapses the adjusted estimators start to fail too. IPW degrades "
            "fastest — it divides by propensities approaching zero. That is the concrete answer "
            "to *when should I not trust a propensity-score analysis*."
        )

        st.subheader("Negative controls")
        nz = causal["negative_controls"]["max_abs_z"]
        n1, n2, n3 = st.columns(3)
        n1.metric("on the raw RCT", f"{nz['rct']:.2f}")
        n2.metric("confounded slice", f"{nz['confounded']:.2f}")
        n3.metric("after IPW", f"{nz['after_ipw']:.2f}")
        st.caption(
            "The RCT baseline is **not** ~0, because this file carries real covariate "
            "imbalance — independent confirmation of that finding by a different method than "
            "the SMD analysis that first found it. Judge adjustment against 26.18, not zero."
        )

# ─────────────────────────── 5. notes ───────────────────────────
with tabs[4]:
    st.subheader("Honest limitations")
    st.markdown(
        """
- **The uplift signal is weak.** The DR-learner is statistically indistinguishable from the
  T-learner at 2.8M held-out rows. Bootstrap CIs are reported so you can see that rather
  than trust a leaderboard.
- **All 12 covariates are imbalanced** in the delivered file (Welch |t| 6.9 to 67.2), yet 11 of
  12 have identical medians — the imbalance is in the tails. Adjustment therefore *moves*
  the ATE by 25%, and the adjusted number is the defensible one.
- **Features are anonymized and randomly projected.** There is no interpretability and no
  story about *who* responds. Do not write "f3 is likely user age".
- **No time dimension**, so no genuine sequential monitoring and no classic
  difference-in-differences. The peeking problem is demonstrated by simulation and labelled
  as such.
- **CACE relies on the exclusion restriction**, which is untestable and plausibly false here:
  being reachable by an ad correlates with browsing, which correlates with visiting.
- **The confounding in the causal module is injected by me**, so it is a controlled
  demonstration of estimator behaviour, not evidence about real-world confounding.
- **Every dollar figure rests on two assumed constants**, both exposed as sliders above.
        """
    )
    st.subheader("Provenance")
    if results:
        st.json(
            {
                "run_id": results.get("run_id"),
                "git_sha": results.get("git_sha"),
                "dataset": results.get("dataset"),
                "bootstrap": results.get("boot_note"),
            }
        )
