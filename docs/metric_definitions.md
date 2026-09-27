# Metric definitions

Every metric this project reports, defined once. The computation lives in
[`dbt/models/marts/mart_metrics.sql`](../dbt/models/marts/mart_metrics.sql) and
is tested in `dbt/models/marts/schema.yml`; this file is the prose that goes
with it.

If a number in the README, the dashboard or a notebook disagrees with this page,
this page and the model are right and the other thing is stale.

---

## Why this is a model, not a dbt semantic layer

The first version of this layer used `semantic_models` plus MetricFlow metrics,
which is the more fashionable answer. dbt refused to parse it:

```
The semantic layer requires a time spine model with granularity DAY or smaller
in the project, but none was found.
```

MetricFlow is built around time-series aggregation. **This dataset has no time
dimension** — no dates, no user ids, no sessions (see
[findings](findings.md)). Satisfying the parser would have meant adding a
calendar table that no metric uses, in a project whose stated finding is that
there is no time column.

So the metrics layer is a model plus this document. What is lost is the `mf`
CLI. What is kept is the substance: the definitions are single-sourced,
version-controlled, tested, and queryable by anything that can read a table.

There is a second reason to prefer this here. dbt's documentation lists
MetricFlow support for Snowflake, BigQuery, Databricks, Postgres and Redshift —
**DuckDB is not on that list**. Building a headline deliverable on an
unsupported adapter combination would have been a liability, not a feature.

---

## The metrics

### `visit_rate` — NORTH STAR

```
visits / units          = P(visit)
```

Observed: **0.046992** overall.

The primary decision metric. Chosen over `conversion_rate` because a 4.7% base
rate gives usable precision — this is the Criteo paper's own recommendation.
Concretely, at this design the experiment can resolve a **1.05% relative**
change in visit rate at 80% power, against **4.76%** for conversion.

Reported as both an absolute difference and a relative lift. The relative
confidence interval uses the delta method on `log(m1/m0)` — **not** the absolute
interval divided by the baseline, because the baseline is itself estimated and
ignoring its variance understates the width.

### `conversion_rate` — SECONDARY / GUARDRAIL

```
conversions / units     = P(conversion)
```

Observed: **0.002917** overall.

Revenue lives here, which is why it is reported at all. At ~0.29% it is far too
rare to rank on, so it is a guardrail rather than an optimisation target. Uplift
models are trained and evaluated on `visit`; conversion appears in the
experiment readout with its precision caveat attached.

### `compliance_rate` — IV FIRST STAGE

```
exposures among treated / treated units  = P(exposed | treated)
```

Observed: **0.036037**.

This is the instrumental-variables first stage, and

```
CACE = ITT / compliance_rate
```

At 3.6% it is very low, which is why the complier effect comes out roughly
**28×** the intention-to-treat effect. It is also the single most actionable
operational number in the project: 11.9M users were "targeted" and about
428,000 were reached.

The control arm's compliance is **exactly zero** — verified, 0 violations — which
is what makes this clean one-sided non-compliance and licenses the Wald
estimator above.

### `exposure_rate` — diagnostic only

```
exposures / units
```

Observed: **0.030631**.

Included for completeness. It mixes both arms and so is not an estimand; use
`compliance_rate` for anything that matters.

---

## Estimands, and which one answers which question

`exposure` is a **post-treatment** variable. Conditioning on it — as a filter or
as a feature — breaks randomization.

| Estimand | Value | Answers | Valid? |
|---|---|---|---|
| **ITT** | +0.010473 | "What happens if we launch the campaign?" | Always — pure randomization |
| CACE | +0.291005 | "What is the effect on users who actually see an ad?" | Under exclusion + monotonicity |
| exposed-vs-control | +0.378360 | *nothing* | **INVALID** |

The ITT is the launch/kill number. CACE is a secondary estimand that describes
3.6% of the population and leans on an exclusion restriction that is untestable
and plausibly false here — being *reachable* by an ad correlates with browsing,
which correlates with visiting. It should not be quoted to stakeholders.

---

## Business assumptions

Every monetary figure is a function of two constants, which live in
`src/uplift/config.py` and are exposed as dashboard sliders:

| Assumption | Value |
|---|---|
| Value per conversion | $25.00 |
| Cost per treatment | $0.01 |
| P(conversion \| visit), measured | 0.062068 |

An incremental **visit** is therefore worth `$25.00 × 0.062068 ≈ $1.55`, not
$25.00. `Settings.value_per_outcome(outcome)` enforces this. Pricing a visit at
the conversion price overstated expected profit by roughly 20× and moved the
optimal targeting fraction from 18.1% to 52.8% — because the break-even
threshold is `cost / value`.
