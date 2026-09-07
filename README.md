# Uplift — Causal Experimentation & Uplift-Modeling Platform

> Which users should we target, and what is the *incremental* effect?

**[▶ Live dashboard](https://huggingface.co/spaces/omvyas77/uplift)** · [source](https://github.com/omvyas77/uplift)

A causal platform built on the Criteo incrementality dataset — **13,979,592
randomized records**, an 85/15 design. The models are the easy part. What this
repo is actually about: **I had a randomized experiment, so I used it as an
answer key to test which causal methods recover the truth when you take the
randomization away — and then built the targeting system on what survived.**

## Headline results

| | |
|---|---|
| Data | Criteo Uplift v2.1 — 13,979,592 units, 85/15, SHA256-verified |
| ATE on `visit` | **+20.2% relative** (adjusted) · +27.1% unadjusted — [why the adjusted one](#the-number-to-quote-is-202-not-271) |
| Best uplift model | S-learner, Qini **0.0893** [95% CI +0.0779, +0.1005] |
| Models statistically indistinguishable | **DR-learner vs T-learner** (+0.0040, CI crosses 0) |
| Naive observational bias | **−705%, and sign-flipped**; matching recovers to −13.6% |
| Design efficiency | 85/15 is **51%** as efficient as 50/50 |
| Variance reduction (CUPAC) | **30.9%** (= ρ², ρ = 0.556) ≈ 1.45× sample |
| Delivery rate | **3.6%** of targeted users were ever shown an ad |

## What a marketing dashboard would have said, and what is true

Of the **576,824 visits** in the treated arm, roughly **92,000 were actually
caused by the campaign**. Six out of seven would have happened anyway.

| | Unadjusted | Adjusted (defensible) |
|---|---|---|
| Incremental visits caused | 122,895 | **92,214** |
| Incremental conversions | 13,687 | **11,641** |
| Incremental revenue | $342,183 | **$291,015** |
| ROAS | 2.88× | **2.45×** |
| Cost per incremental visit | $0.97 | **$1.29** |

*Assumptions: $25.00 per conversion, $0.01 per treatment. Both live in
`src/uplift/config.py` and are dashboard sliders. Never quote a dollar figure
from this repo without them.*

**The biggest finding is operational, not statistical.** Only 3.6% of targeted
users were ever shown an ad — 11.9M "targeted", ~428,000 reached. The entire
measured lift comes from that 3.6%. Delivery, not targeting, is where the
leverage is.

## Model results (`visit`, held-out test, n = 2,796,413)

| Model | Qini | 95% CI | vs T-learner | Calib. slope |
|---|---|---|---|---|
| S-learner | **0.0893** | [+0.0779, +0.1005] | +0.0182, resolved | 0.954 |
| Causal forest | 0.0877 | [+0.0762, +0.0980] | +0.0165, resolved | 1.086 |
| Class transformation | 0.0828 | [+0.0715, +0.0942] | +0.0114, resolved | 0.496 |
| X-learner | 0.0778 | [+0.0680, +0.0890] | +0.0066, resolved (barely) | 0.665 |
| DR-learner | 0.0749 | [+0.0651, +0.0860] | +0.0040, **overlapping** | 0.747 |
| T-learner | 0.0712 | [+0.0620, +0.0825] | (reference) | 0.541 |

> The DR-learner is **not distinguishable** from the T-learner even at 2.8M
> held-out rows. Publishing this leaderboard without intervals would have
> declared a winner over a gap the data cannot resolve.

## The bias table — the artifact this repo exists for

Take the RCT. Deliberately destroy the randomization by keeping units as a
function of their covariates. Run the observational toolkit on the wreckage.
Compare to the answer you already had.

| Estimator | Estimate | Bias | Relative error |
|---|---|---|---|
| Ground truth (from the RCT) | 0.005314 | — | — |
| **Naive difference-in-means** | −0.032148 | −0.037462 | **−705%** |
| IPW (clipped, stabilized) | 0.001404 | −0.003910 | −73.6% |
| **Propensity matching (1:1)** | 0.004590 | −0.000724 | **−13.6%** |
| AIPW / doubly robust | 0.002641 | −0.002673 | −50.3% |

The naive comparison is not merely mis-scaled — it is **sign-flipped**, reporting
a negative effect where the truth is positive.

Two honest departures from the textbook: **matching beats AIPW here** (AIPW is
nonetheless the only estimator whose bias does not grow as overlap collapses,
and is best at strength 4.0), and relative errors are inflated by a small
denominator because the s(X)-weighting halves the target.

At confounding strength 0 the naive bias is **+0.000040** — the apparatus
validating itself. If that row were not ~0, the table would be measuring its own
bug.

## The `exposure` trap

`exposure` records whether the user was actually *shown* an ad. It is
**post-treatment**. Conditioning on it — as a filter or a feature — breaks
randomization.

| Estimand | Value | What it answers | Valid? |
|---|---|---|---|
| **ITT** | +0.010473 | "What happens if we launch?" | Always — pure randomization |
| CACE | +0.291005 | "Effect on users who see an ad?" | Under exclusion + monotonicity |
| Exposed-vs-control | +0.378360 | *nothing* | **INVALID** |

With a 3.6% first stage, CACE is 27.8× the ITT, and the invalid comparison
overstates the CACE by 30% and the ITT by 36×. `src/uplift/data/io.py` returns
`exposure` from exactly one function, used only by the IV module.

## Findings that contradict the plan I was working from

Full detail in **[docs/findings.md](docs/findings.md)**.

1. **There is no SRM.** The allocation is exact to 1.3×10⁻⁷ while the test
   resolves 4×10⁻⁴. The expected large-sample false alarm never fires.
2. **All 12 covariates are imbalanced** (Welch |t| 6.9 to 67.2, all past
   Bonferroni) — yet 11 of 12 have *identical medians*. The imbalance is in the
   tails. Clustering halves it (0.0476 → 0.0224) and
   corr(cluster visit rate, cluster treated share) = **+0.50**: high-propensity
   strata are over-represented in the treated arm.
3. **CUPAC gives 30.9% variance reduction**, not the predicted few percent, and
   survives `GroupKFold` on the exact feature vector (ρ 0.5557 → 0.5561), so it
   is not duplicate leakage. Visit rates span **0.11% to 61%** across strata.
4. **The X-learner was anti-ranked** (Qini −0.0016, decile rank correlation
   −0.44) because EconML reuses the propensity as the blending weight — on 85/15
   that puts 85% of the weight on the model imputed from the 15% arm. `blend=0.5`
   moves it to +0.0778.

### The number to quote is 20.2%, not 27.1%

Under clean randomization, covariate adjustment tightens the interval and leaves
the point estimate alone. Here it moves it by **7.1 standard errors** — and
subsampling is ruled out: difference-in-means on the *same* 2M rows gives
+0.010204 against Lin's +0.007760. That is finding 2 showing up in the headline
number, and it is a 25% haircut on the impact claim.

## Quickstart

```bash
uv sync --extra causal --extra dbt --extra serve --extra orchestrate --extra dev
make download && make ingest      # SHA256-verified; ~90s to ingest
make dbt-build                    # 6 models, 46 data tests
make experiment                   # SRM, balance, ATE, CUPAC, power
```

Then the modelling path:

```bash
make train      # 6 uplift models, ~9.5 min
make eval       # Qini + bootstrap CIs -> evals/results.json
make causal     # the bias table, IV/CACE, negative controls
make policy     # targeting policy + DR off-policy evaluation
make api        # http://localhost:8000/docs
make dashboard  # http://localhost:8501
```

The dashboard is deployed to a [Hugging Face Space](https://huggingface.co/spaces/omvyas77/uplift)
by `python scripts/deploy_space.py`, which assembles a self-contained payload
from this repo. It runs the *same* dashboard module as `make dashboard` via a
thin entrypoint, so the live page and the local one cannot drift.

`make test` runs 68 tests on synthetic data with known ground truth — CI never
downloads the 311 MB source file.

## What this demonstrates

- **Experiment analysis** — SRM with a calibration simulation, covariate balance,
  power/MDE, CUPED/CUPAC, Lin's estimator with HC1 errors, a peeking simulation
- **Heterogeneous effects** — S/T/X/DR-learners, class transformation, causal
  forest, evaluated by Qini/AUUC **with bootstrap CIs** and CATE calibration
- **Causal identification validated against randomized ground truth**
- **ITT vs CACE via instrumental variables**, and why `exposure` breaks a naive
  analysis
- **Analytics engineering** — dbt staging→marts, 46 data tests, a metrics layer
- **Ops** — Dagster assets with blocking SRM/balance checks, Docker Compose, a CI
  regression gate that fails on a real regression but tolerates noise

## Honest limitations

- **The uplift signal is weak.** The DR-learner is indistinguishable from the
  T-learner at 2.8M rows. CIs are reported so you can see that.
- **Features are anonymized and randomly projected.** No interpretability, no
  story about *who* responds.
- **All 12 covariates are imbalanced** in the delivered file, which is why the
  adjusted and unadjusted ATEs differ by 25%. Both are always reported.
- **No time dimension**, so no genuine sequential monitoring and no classic
  difference-in-differences. The peeking problem is *simulated* and labelled as
  such. DiD is deliberately not built — synthesizing a panel would be worse than
  omitting one.
- **CACE relies on the exclusion restriction**, untestable and plausibly false
  here: being reachable correlates with browsing, which correlates with visiting.
  Do not quote the +28.7pp figure to stakeholders.
- **The confounding is injected by me** — a controlled demonstration of estimator
  behaviour, not evidence about real-world confounding.
- **One test is a strict xfail.** AIPW does not beat IPW under a misspecified
  propensity here (0/6 seeds). Left failing loudly rather than deleted; the open
  hypothesis is in the xfail reason.
- **Bootstrap CIs draw 1M of 2.8M rows** per replicate, so they are conservative
  (wider) by ~√(n/b). Point estimates always use the full split.

## Why not Kubernetes?

This is a batch analytics pipeline with a single-user serving surface. Docker
Compose plus a weekly scheduled run is the correct amount of infrastructure.
Running a cluster for a portfolio demo would be paying for complexity that buys
nothing.

## Stack

Python 3.12 · uv · DuckDB · dbt · Polars/pandas · LightGBM · EconML 0.17 ·
scikit-uplift · statsmodels · FastAPI · Streamlit · Dagster · Docker · GitHub
Actions · ruff/mypy/pytest

## License

MIT
