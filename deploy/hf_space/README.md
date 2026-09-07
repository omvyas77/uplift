---
title: Uplift — Causal Experimentation Platform
emoji: 📈
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Uplift — Causal Experimentation & Uplift-Modeling Platform

Live dashboard for a causal platform built on the **Criteo Uplift v2.1**
incrementality dataset — 13,979,592 randomized records, an 85/15 design.

**Source code:** https://github.com/omvyas77/uplift

## What this shows

Of the 576,824 visits in the treated arm, roughly **92,000 were actually caused**
by the campaign. Six out of seven would have happened anyway.

- **Experiment** — SRM, covariate balance, ATE (adjusted and unadjusted), CUPAC
  variance reduction, design efficiency
- **Uplift models** — six learners ranked by Qini **with bootstrap confidence
  intervals**, plus CATE calibration. The DR-learner is *not* distinguishable
  from the T-learner even at 2.8M held-out rows.
- **Targeting policy** — doubly-robust off-policy evaluation against four
  baselines, with the business assumptions as sliders
- **Causal validation** — the bias table: observational estimators scored against
  randomized ground truth. The naive comparison is off by −705% and
  **sign-flipped**.
- **Method notes** — the honest limitations

This dashboard reads pre-computed results from `evals/*.json`. It never trains
anything.
