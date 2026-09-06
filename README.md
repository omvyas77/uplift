# Uplift — Causal Experimentation & Uplift-Modeling Platform

> Which users should we target, and what is the *incremental* effect?

A causal platform built on the Criteo incrementality data (13,979,592 randomized
units). The framing is not "an uplift modeling project" — it is:

> *I had a randomized experiment, so I used it as an answer key to test which
> causal methods actually recover the truth when you take the randomization
> away — and then I built the targeting system on top of what survived.*

## Headline results

| | |
|---|---|
| Data | Criteo Uplift v2.1 — 13,979,592 randomized units, 85/15 allocation, SHA256 verified |
| **ATE (`visit`), adjusted** | **+20.2% relative** [95% CI +18.5%, +22.0%] |
| ATE (`visit`), unadjusted | +27.1% [+26.2%, +28.0%] — *see "why the adjusted number is the headline"* |
| ATE (`conversion`), adjusted | +49.3% [+40.4%, +58.2%] |
| Variance reduction (CUPAC) | 30.9% (= ρ², ρ = 0.556), ≈1.45× effective sample |
| Compliance | P(exposed \| treated) = **0.036** — the campaign reached 3.6% of who it targeted |
| Design efficiency | the 85/15 split is **51%** as efficient as 50/50 |
| SRM | none — allocation exact to 1.3×10⁻⁷ |

## The finding this project is actually about

Adjusting for covariates moved the treatment effect by **7.1 standard errors**.

In a clean RCT that must not happen: adjustment tightens the interval and leaves
the point estimate alone. Here it did not, and chasing why produced the result
that makes the rest of the repo worth reading.

Eight of the twelve anonymized features are effectively categorical — `f1` takes
**60 distinct values across 14M rows**, and 98.77% of rows share a single one.
Clustering on the covariates and recomputing balance *within* cluster halves the
imbalance (pooled max\|SMD\| 0.0476 → 0.0224), and the treated share tracks
outcome propensity:

```
corr(cluster visit rate, cluster treatment ratio) = +0.50 Pearson (p=0.026)
                                                    +0.60 Spearman (p=0.005)
```

Users more likely to visit anyway are over-represented in the treated arm. That
biases the naive difference upward, which is why **+20.2% is the number quoted
and +27.1% is shown only as the unadjusted comparison** — a 25% haircut on the
headline impact claim.

Full write-up, including one retracted claim of my own, in
**[docs/findings.md](docs/findings.md)**.

## What the numbers mean commercially

Assumptions, stated because every figure depends on them: **$25.00 per
conversion**, **$0.01 per treatment**. Both live in `src/uplift/config.py` and
are dashboard sliders.

| | Unadjusted | Adjusted (defensible) |
|---|---|---|
| Incremental visits caused | 122,895 | **92,214** |
| — of the 576,824 visits in the treated arm | 21.3% | **16.0%** |
| Incremental conversions caused | 13,687 | **11,641** |
| Incremental revenue | $342,183 | **$291,015** |
| Net of $118,827 campaign cost | $223,356 | **$172,188** |
| ROAS | 2.88× | **2.45×** |

**The reframe:** of the 576,824 visits a marketing dashboard would credit to this
campaign, roughly **92,000 were actually caused by it**. Six out of seven would
have happened without spending a cent.

**The operational finding is bigger than the modelling one.** Only 3.6% of
targeted users were ever shown an ad — 11.9M "targeted", ~428,000 reached. The
entire measured lift comes from that 3.6%. Delivery, not targeting, is where the
leverage is.

## What this demonstrates

- **Experiment analysis** — SRM with a *validated* calibration simulation,
  covariate balance, power/MDE, allocation efficiency, CUPED/CUPAC, Lin's
  estimator with HC1 robust SEs
- **Heterogeneous effects** — S/T/X/DR-learners, class transformation, causal
  forest; Qini/AUUC **with bootstrap CIs** and paired-bootstrap comparisons
- **Causal identification validated against randomized ground truth** — inject
  selection bias into the RCT, then check which estimators recover the answer
- **ITT vs CACE** via instrumental variables, and why `exposure` breaks a naive
  analysis
- **Analytics engineering** — dbt staging→marts, 46 data tests, a metrics layer
- **Ops** — Dagster assets with blocking data-quality checks, Docker Compose, CI

## Honest limitations

- **Uplift signal in this data is weak.** Models are reported with CIs precisely
  so the reader can see which are indistinguishable.
- **Features are anonymized and randomly projected.** There is no interpretability
  and no domain story about *who* responds. `f3` is not "user age".
- **No time dimension**, so no genuine sequential monitoring and no classic DiD.
  The peeking problem is demonstrated by *simulation* and labelled as such
  (naive 18.25% false-positive rate at a nominal 5%; O'Brien–Fleming restores
  6.25%).
- **CACE relies on the exclusion restriction**, which is untestable and shaky
  here — being *reachable* correlates with actively browsing, which correlates
  with visiting. CACE stays a secondary estimand. It is **not** a stakeholder
  number.
- **The confounding in the observational module is injected by me**, so it is a
  controlled demonstration of estimator behaviour, not evidence about real-world
  confounding.
- **The `conversion` adjustment is not resolved** — 1.5 SEs against `visit`'s
  7.1, because a 0.29% base rate carries far less information.
- **Within-cluster imbalance is still 0.0224**, above the 0.02 a clean RCT should
  satisfy. Stratification explains about half the imbalance; the rest is an open
  question.

## Quickstart

```bash
make install
make download      # 311 MB, SHA256-verified across three mirrors
make ingest        # csv.gz -> Parquet -> DuckDB, with structural assertions
make dbt-build     # 6 models, 46 tests
make experiment    # SRM, balance, ATE, CUPAC, power
```

## Why not Kubernetes?

This is a batch analytics pipeline with a single-user serving surface. Docker
Compose plus a scheduled run is the correct amount of infrastructure. Running a
cluster for a portfolio demo would be paying for complexity that buys nothing.

## Performance notes

Measured on an 8-core / 8 GB machine, because the usual advice is backwards here:

- **`n_jobs=-1` is correct.** Pinning threads (`n_jobs=1`) is **6× slower** on
  prediction (48.5s vs 7.9s per 500k rows) and 1.6× slower on fit.
- **Cost is dominated by scoring, not fitting.** A 400-tree LightGBM predicts
  200k rows in ~5s, so scoring 2.8M costs ~70s *per call* — and an S- or
  T-learner needs two per split. `train` therefore scores only `test` by default.
- EconML is **not** the bottleneck: DR-learner's nine nuisance fits cost 15.2s on
  200k rows, and the causal forest predicts fastest of the six.

## License

MIT
