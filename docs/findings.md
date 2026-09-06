# Findings

Empirical observations from the delivered Criteo Uplift v2.1 file. Each one was
found by running a check that was written to *discover* something rather than to
confirm it, and each is reported here rather than being tuned away.

Everything below is reproducible with `make ingest && make experiment`.

---

## 1. The structural checks both pass

| Check | Result |
|---|---|
| `exposure = 1 ⟹ treatment = 1` | **0 violations** in 13,979,592 rows |
| `conversion = 1 ⟹ visit = 1` | **0 violations** in 13,979,592 rows |
| Control-arm exposure rate | **exactly 0.000000** |

The build guide says to verify funnel monotonicity empirically rather than
assume it. It holds. The control arm is never exposed, which is what makes this
clean **one-sided non-compliance** and licenses `CACE = ITT / P(exposed | treated)`.

## 2. There is no SRM — and the interesting part is why the test could have found one

```
n_treatment = 11,882,655   n_control = 2,096,937   ratio = 0.8500001
chi2 = 0.000   p = 0.9989   ->  no SRM
```

The observed allocation deviates from the designed 0.85 by **1.3 × 10⁻⁷**. At
n = 14M this chi-square test detects deviations of 3.95 × 10⁻⁴ at 80% power —
roughly 3,000× larger than what is actually there.

This is worth stating precisely because the *opposite* is the usual outcome. The
standard large-sample failure mode is that a design ratio published to two
decimals ("0.85") gets rejected at n = 14M because the test resolves
thousandths. Here the delivered file is allocated to seven decimal places, so
the test has nothing to reject. The lesson is unchanged: **report the effect
size, not only the p-value** — it is just that here the effect size is what
exonerates the data rather than what indicts the test.

The SRM check is *validated*, not merely written: `simulate_srm_calibration`
rejects at 0.0020 (α = 0.001) and 0.0630 (α = 0.05) under the null, matching
the nominal rates to Monte-Carlo slack.

## 3. All 12 covariates are imbalanced — and the imbalance is entirely in the tails

This is the finding that changed how the rest of the repo is built.

| feature | SMD | Welch \|t\| | median difference | skew |
|---|---|---|---|---|
| f3 | −0.0488 | 67.2 | 0.0000 | −3.19 |
| f6 | −0.0404 | 54.7 | 0.0000 | −1.14 |
| f5 | −0.0306 | 42.3 | 0.0000 | −7.09 |
| f1 | +0.0240 | 33.7 | 0.0000 | +14.00 |
| f9 | +0.0240 | 32.5 | 0.0000 | +2.82 |
| … | … | … | … | … |
| f11 | −0.0052 | 6.9 | 0.0000 | −14.81 |

**All 12 of 12** features exceed a Bonferroni-corrected threshold of |t| > 3.93.
Yet **11 of 12 have exactly identical medians across arms**, and the features are
extremely skewed (|skew| up to 14.8).

So the arms differ in the *mean* of every feature while agreeing on the
*median* of almost all of them. The imbalance lives in the tails.

Candidate explanations, tested:

- **Duplicate rows? No — and my first reading of this was wrong.** The file has
  1,259,545 exact duplicate rows (9.0%), and the rate differs sharply by arm:
  10.3% of treated rows versus 2.3% of control. That asymmetry looked alarming
  until it was decomposed. Duplication is a function of how many rows land in
  each distinct feature cell, and the treated arm has 5.7× more rows in the same
  feature space:

  | arm | rows | distinct cells | rows per cell | duplicate fraction |
  |---|---|---|---|---|
  | control | 2,096,937 | 2,047,781 | 1.0240 | 2.3% |
  | treated | 11,882,655 | 10,661,560 | 1.1145 | 10.3% |

  Duplicate fraction tracks rows-per-cell exactly. The asymmetry is **arm size,
  not data quality** — a pigeonhole consequence of 8 of the 12 features being
  near-discrete (see below).

  This also **retracts** an earlier claim in this document. Deduplicating raises
  max |SMD| from 0.0488 to 0.0849, which was first read as "the duplicates were
  masking the imbalance". That inference is invalid: deduplication removes
  proportionally more treated rows, shifting the arm ratio from 0.8500 to
  0.8389. It is a *biased* operation on this file and cannot be used as a
  balance diagnostic at all.

- **Heavy tails?** Yes, and the structure is now explicit. Eight of the twelve
  features are effectively categorical — `f1` takes only **60 distinct values
  across 14M rows**, `f5` 132, `f11` 136, `f4` 260, `f3` 552 — and 98.77% of all
  rows share a single `f1` value. The remaining 1.2% form a long tail in which
  the visit rate climbs monotonically from 4.4% to 48%.

- **Strata with unequal treated shares? Yes — this is the mechanism.** Clustering
  the covariates (k-means, k=20, 2M rows) and recomputing balance *within*
  cluster halves the imbalance:

  ```
  pooled            max |SMD| = 0.0476
  within-cluster    max |SMD| = 0.0224   (size-weighted)
  ```

  And the treated share varies systematically across clusters in a specific
  direction — **corr(cluster visit rate, cluster treatment ratio) = +0.50
  Pearson (p = 0.026), +0.60 Spearman (p = 0.005)**. Cluster visit rates span
  0.11% to 61%, a 555× range, and the high-propensity strata carry treatment
  ratios up to 0.880 against 0.844 in the low-propensity ones.

  So users more likely to visit anyway are over-represented in the treated arm.
  That biases an unadjusted difference-in-means **upward**, which is exactly the
  direction and roughly the magnitude of the adjustment gap below.

### What this changes

The design is randomized by construction, so `treatment` remains a valid
instrument and the intention-to-treat contrast remains the headline. But the
delivered file is not the pristine RCT the modelling literature assumes, and
that shows up directly in the estimates:

```
difference-in-means      ATE = +0.010342   (unadjusted)
Lin regression-adjusted  ATE = +0.007760   (−25%)
CUPAC-adjusted           ATE = +0.007919   (−23%)
```

**The adjusted number is the one to quote: +20.2%, not +27.1%.** The two
covariate-adjusted estimators agree with each other and disagree with the raw
difference by about a quarter. Under clean randomization they should all
agree, and adjustment should only tighten the interval. They diverge here
because the covariates are both **imbalanced** (this section) and **strongly
predictive of the outcome** (section 4) — which is exactly the condition under
which adjustment moves a point estimate rather than merely sharpening it.

Subsampling is ruled out as the explanation: difference-in-means on the *same*
2M rows the adjusted estimators used gives +0.010204, essentially the full-data
+0.010342. Lin therefore sits **7.1 standard errors** below the unadjusted
estimate on identical data.

Consequences carried through the repo:

- The dbt balance test is split in two. A **WARN** at |SMD| < 0.02 (what a clean
  RCT should satisfy — this fails, deliberately, and stays failing) and an
  **ERROR** at |SMD| < 0.10 (the matching-literature rule of thumb) which is
  what actually blocks the pipeline and the Dagster asset check. Widening the
  tolerance to make the dashboard green would have destroyed the finding.
- Both the adjusted and unadjusted ATE are reported everywhere, never just one.

## 4. CUPAC delivers 31% variance reduction — far more than expected

```
theta = 1.0495   rho = 0.5563   variance reduction = 30.95%   ≈ 1.45× sample
```

The build guide predicts "a few percent at best" on the grounds that 12
randomly-projected anonymized features should not predict a 4.7% binary outcome
well. On this data they do: the cross-fitted control-only prediction correlates
0.556 with `visit`, and variance reduction equals ρ² = 0.309 as the identity
requires.

This was checked for the obvious artifact. With 9% exact-duplicate rows, a
duplicated row can land in both the training and the held-out fold of the
cross-fitting, letting the model memorise its outcome and inflating ρ. Rerunning
the whole procedure on the deduplicated file gives ρ = 0.5518 and 30.44% —
essentially unchanged. **The variance reduction is real.**

## 5. Compliance is very low, so ITT and CACE are far apart

```
P(exposed | treated) = 0.0361      P(exposed | control) = 0.0000
```

Only 3.6% of users assigned to the campaign were actually shown an ad. Since
`CACE = ITT / first stage`, the complier effect is roughly **28× the ITT**. This
makes the ITT-versus-CACE distinction quantitatively dramatic on this data
rather than academic, and it makes conditioning on `exposure` catastrophic
rather than merely sloppy. See §7.4 of the build guide and the IV module.

## 6. Design efficiency

```
allocation efficiency 4r(1−r) = 0.510 at r = 0.85
MDE (80% power, α=0.05): 0.945% relative at 85/15  vs  0.675% at 50/50
same MDE would need 1.96× the sample
```

The 85/15 split is 51% as statistically efficient as a balanced one — half the
sample is effectively wasted from a pure-precision standpoint. The design is
still correct: a business does not withhold ads from half its users to tighten a
confidence interval. The allocation buys revenue with precision.

---

## 7. Qini tie handling (an implementation finding, not a data one)

The reference Qini implementation initially integrated the curve at every unit.
That makes the result depend on the arbitrary ordering *within* a block of equal
scores, and the standard property test — reversing the ranking must flip the
sign — failed by 0.025 on a coefficient of 0.15.

The cause is that a Qini curve is defined by **thresholds** on the score, so
within a tied block there is no defined ordering and the curve should
interpolate linearly across the block. The test case `tau = 0.05 * (x > 0)` has
exactly two distinct score values, so the entire sample is two enormous ties —
a worst case that a continuous score would never have exposed.

Integrating at block boundaries instead fixes it: symmetry improves from 0.025
to 6e-5, and the change is a no-op on continuous scores (verified). Kept as
`tie_aware=True` with the alternative still reachable, and guarded by
`test_reversed_ranking_flips_the_sign` and `test_tie_aware_curve_collapses_tied_blocks`.

---

## 8. The secondary outcome — `conversion`

Revenue lives here, so it is reported despite the 0.29% base rate that makes it
a poor ranking target.

| Estimator | ATE | 95% CI | Relative | n |
|---|---|---|---|---|
| Difference-in-means | +0.001152 | [+0.001085, +0.001219] | **+59.4%** | 14.0M |
| Difference-in-means (same 2M) | +0.001116 | [+0.000936, +0.001296] | +56.1% | 2M |
| Lin regression-adjusted | +0.000980 | [+0.000803, +0.001156] | **+49.3%** | 2M |
| CUPAC-adjusted | +0.000998 | [+0.000828, +0.001167] | +51.0% | 2M |

Adjustment moves the estimate down here too, from +56.1% to +49.3% — the same
direction as `visit`, as the shared mechanism in section 3 requires. But the gap
is only **1.5 standard errors** against `visit`'s 7.1, because a 0.29% outcome
carries far less information. Do not present the conversion adjustment as
resolved; present it as consistent in sign with a much better-resolved result.

CUPAC on `conversion`: rho = 0.3322, variance reduction **11.03%** (≈1.12x
sample) — materially lower than `visit`'s 31%, exactly as the rho-squared bound
predicts for a rarer outcome.

Design MDE for `conversion` is **4.76% relative** at 80% power, against 1.05%
for `visit`. That is the honest statement of what this experiment can and cannot
resolve on the revenue metric.

## 9. What the numbers mean commercially

Assumptions, stated because every figure below is a function of them:
**$25.00 per conversion**, **$0.01 per treatment**. Both live in
`src/uplift/config.py` and are exposed as dashboard sliders.

| | Unadjusted | Adjusted (defensible) |
|---|---|---|
| Incremental visits caused | 122,895 | **92,214** |
| — as a share of the 576,824 visits in the treated arm | 21.3% | **16.0%** |
| Incremental conversions caused | 13,687 | **11,641** |
| — as a share of 36,711 conversions in the treated arm | 37.3% | **31.7%** |
| Campaign cost (11,882,655 × $0.01) | $118,827 | $118,827 |
| Incremental revenue | $342,183 | **$291,015** |
| Net | $223,356 | **$172,188** |
| ROAS | 2.88x | **2.45x** |
| Cost per incremental visit | $0.97 | **$1.29** |

The headline reframe: of the 576,824 visits a marketing dashboard would credit
to this campaign, roughly **92,000 were actually caused by it**. Six out of seven
would have happened without spending a cent.

**The operational finding is bigger than the modelling one.** Only 3.6% of
targeted users were ever shown an ad — 11.9M "targeted", ~428,000 reached. The
entire measured lift comes from that 3.6%. Delivery, not targeting, is where the
leverage is: taking delivery from 3.6% to 7% would roughly double impact before
anyone touches a model. Caveat that honestly — the next tranche of reachable
users is probably less responsive than the current one, so "roughly double" is
an upper bound.

**On CACE.** ITT / 0.036 = +28.7 percentage points is arithmetically right and
stays in the repo as a secondary estimand, but it describes 3.6% of the
population and leans on an exclusion restriction that is shaky here — being
*reachable* correlates with actively browsing, which correlates with visiting.
It is not a stakeholder number and will be misquoted within a day if presented
as one.


---

## 10. The X-learner's blending weight is backwards on an 85/15 design

The first full evaluation produced a result that could not be a real effect:

```
x_learner   Qini = -0.0016   calibration slope 0.003   decile rank correlation -0.44
```

Not merely weak — **anti-ranked**. Its lowest-predicted decile carried the second
*highest* observed uplift (predicted −0.119, observed +0.0275).

The cause is the blending weight, not the model. The X-learner forms

```
tau(x) = g(x) * tau_0(x) + (1 - g(x)) * tau_1(x)
```

where `tau_0` is imputed through the CONTROL outcome model and `tau_1` through
the treated one. Künzel et al. suggest reusing the propensity score as `g`, and
EconML's default follows that. On an 85/15 design this sets g ≈ 0.85, putting
**85% of the weight on `tau_0` — the arm holding 15% of the data**, and therefore
the noisier of the two. The paper's own reasoning is that `g` should be *small*
when `tau_0` is noisy, so reusing the propensity gets the sign of the argument
backwards here.

Measured on a 400k train / 600k validation slice:

| blending weight `g` | Qini |
|---|---|
| `LogisticRegression` propensity (≈0.85) — EconML default | +0.0429 |
| g = 0.85 | +0.0459 |
| g = 0.15 (= 1 − treated share) | +0.0807 |
| **g = 0.50** | **+0.0854** |

Flipping the weight roughly doubles the Qini. `fit_x_learner` now defaults to
`blend=0.5`, with `blend=None` restoring the textbook propensity-weighted
behaviour so the comparison stays reproducible.

**Confirmed at full scale.** Re-running the whole pipeline with `blend=0.5`
moved the X-learner from **Qini -0.0016 to +0.0778** on the test split, and every
other model's Qini came back bit-identical (0.08283, 0.08932, 0.07118, 0.07492,
0.08772) - a clean control that exactly one thing changed. It now sits fourth of
six and is *just* distinguishable from the T-learner (+0.0066, CI
[+0.0004, +0.0125]).

Worth noting the subsample was *kinder* than the full run: the same
propensity-weighted configuration scored +0.0429 on 400k rows but −0.0016 when
fitted on 1.5M. A bug that looks like mild underperformance at development scale
became a total ranking failure at full scale, which is an argument for evaluating
at the scale you intend to ship.

## 11. Model results (`visit`, held-out test split, n = 2,796,413)

Point estimates on the full test split; bootstrap replicates draw 1M rows, so the
intervals are conservative (wider) by roughly sqrt(n/b). Reference for the paired
comparisons is the T-learner.

| Model | Qini | 95% CI | vs T-learner | Calib. slope | Response AUC* |
|---|---|---|---|---|---|
| S-learner | **0.0893** | [+0.0779, +0.1005] | +0.0182, **resolved** | 0.954 | 0.867 |
| Causal forest | 0.0877 | [+0.0762, +0.0980] | +0.0165, **resolved** | 1.086 | 0.887 |
| Class transformation | 0.0828 | [+0.0715, +0.0942] | +0.0114, **resolved** | 0.496 | 0.726 |
| X-learner (`blend=0.5`) | 0.0778 | [+0.0680, +0.0890] | +0.0066, **resolved** (barely) | 0.665 | 0.706 |
| DR-learner | 0.0749 | [+0.0651, +0.0860] | +0.0040, *overlapping* | 0.747 | 0.749 |
| T-learner | 0.0712 | [+0.0620, +0.0825] | (reference) | 0.541 | 0.690 |

\* Response AUC measures who RESPONDS, not who responds BECAUSE OF treatment. It
is reported only so it can be labelled as not the objective.

Two things worth stating plainly:

- **The DR-learner is not distinguishable from the T-learner** (+0.0038, CI
  [−0.0020, +0.0108]) even at 2.8M held-out rows. Reporting this leaderboard
  without intervals would have declared a winner over a gap the data cannot
  resolve.
- **The S-learner leads despite its degeneracy diagnostic firing.**
  `treatment_gain_share = 0.0023` — only 0.23% of total split gain came from the
  treatment column, the classic sign that the trees are barely modelling the
  treatment at all. It still ranks best, and its calibration slope of 0.954 is
  the second best of the six. The diagnostic flags a real pathology in how the
  model represents the effect; it does not by itself predict poor ranking, and
  conflating the two would be a mistake.

---

## 12. The bias table — observational estimators against RCT ground truth

Confounding injected at strength 1.0 on 2M rows; 1,052,060 survive the selection.
Ground truth is the s(X)-weighted ATE computed from the RCT, which is the correct
target for the slice under an 85/15 design (see `inject_confounding`).

| Estimator | Estimate | Bias | Relative error |
|---|---|---|---|
| Ground truth (from the RCT) | 0.005314 | — | — |
| **Naive difference-in-means** | −0.032148 | −0.037462 | **−705%** |
| IPW (clipped, stabilized) | 0.001404 | −0.003910 | −73.6% |
| **Propensity matching (1:1)** | 0.004590 | −0.000724 | **−13.6%** |
| AIPW / doubly robust | 0.002641 | −0.002673 | −50.3% |

The headline holds: the naive comparison is not merely biased but **sign-flipped**,
reporting a negative effect where the truth is positive, and adjustment recovers
the sign and most of the magnitude.

Two honest departures from what the build guide anticipates:

- **Matching wins here, not AIPW.** The guide expects the doubly-robust estimator
  to be best. At strength 1.0 matching lands within 13.6% and AIPW within 50.3%.
  AIPW is nonetheless the most *robust* — see the strength curve below, where it
  is the only estimator still improving at strength 4.0.
- **Relative errors are inflated by a small denominator.** The s(X)-weighting
  halves the target, from a population ATE of ~0.0105 to 0.0053, so a −0.0027
  absolute bias reads as −50%. Judge these on the absolute column too.

### Bias against confounding strength

| strength | truth | naive | IPW | matching | AIPW | treated outside control support |
|---|---|---|---|---|---|---|
| 0.0 | 0.010473 | **+0.000040** | −0.004864 | −0.002015 | −0.003163 | 1.6% |
| 0.5 | 0.006297 | −0.023955 | −0.002778 | −0.001231 | −0.002191 | 3.5% |
| 1.0 | 0.005314 | −0.037462 | −0.003910 | −0.000724 | −0.002673 | 9.4% |
| 2.0 | 0.004888 | −0.050614 | −0.004268 | −0.005621 | −0.002552 | 22.8% |
| 4.0 | 0.004778 | −0.057434 | −0.011023 | −0.004876 | **+0.001705** | 65.9% |

The strength-0 row is the setup's own validation: with no confounding injected
the naive bias is +0.000040, i.e. zero. If that row were not ~0 the whole
apparatus would be measuring its own bug.

As strength rises, overlap collapses — minimum propensity falls from 0.44 to
2.0×10⁻⁷ and the share of treated units with no control counterpart goes from
1.6% to 65.9%. **IPW degrades fastest** (−0.0039 → −0.0110), which is the
expected failure: it divides by propensities approaching zero. AIPW is the only
estimator whose bias does not grow. That is the concrete answer to "when should I
*not* trust a propensity-score analysis" — when this diagnostic, not the point
estimate, says overlap is gone.

### Negative controls

```
on the raw RCT            max|z| =  26.18     <- baseline for THIS file
on the confounded slice   max|z| = 331.47
after IPW reweighting     max|z| =  13.93     -> 104% of injected imbalance removed
```

The RCT baseline is **not** ~0, because Criteo v2.1 carries real covariate
imbalance (section 3). The negative-control check therefore fails on the raw
RCT — which is independent confirmation of that finding by a completely
different method than the SMD analysis that first found it. Adjustment must be
judged against 26.18, not against zero.

Reweighting overshoots slightly (13.93 is *below* the RCT baseline), because the
propensity model fitted on the confounded slice also absorbs part of the file's
native imbalance.

## 13. ITT vs CACE vs the invalid comparison

```
ITT                        +0.010473  [+0.009715, +0.011231]   always valid
first stage P(exposed|Z=1)  0.035989
CACE                       +0.291005  [+0.269943, +0.312066]   27.8x the ITT
naive exposed-vs-control   +0.378360                           1.30x the CACE - INVALID
```

Conditioning on `exposure` — a post-treatment variable — overstates the complier
effect by 30%, and overstates the ITT, the number a launch decision actually
needs, by a factor of 36.

## 14. From scores to a decision, and a pricing bug worth recording

Off-policy evaluation on the held-out randomized test split, where the propensity
is known by design.

| Policy | Treated | DR value | 95% CI |
|---|---|---|---|
| Treat nobody | 0% | 0.03796 | [0.03699, 0.03893] |
| Treat everybody | 100% | **0.04838** | [0.04793, 0.04884] |
| Model targeting | 18.1% | 0.04715 | [0.04651, 0.04778] |
| Random at same budget | 18.1% | 0.03990 | [0.03900, 0.04081] |

**Model targeting beats random at the same budget by +0.00725 (SE 0.00056),
about 12.8 standard errors.** That is the row most projects omit and the only one
that shows the model is doing work.

But **treat-everybody has the highest raw value**, and saying otherwise would be
dishonest. The model policy wins only once treatment cost is counted; on outcome
alone, with a positive effect nearly everywhere and a very cheap treatment,
blanket treatment is hard to beat. The targeting case here is efficiency, not
raw lift.

**The pricing bug.** The first run valued an incremental *visit* at
`value_per_conversion_usd` = $25 — but $25 is the price of a **conversion**. A
visit converts with probability 0.0621, so it is worth about **$1.55**. The error
was not cosmetic: it moved the optimal treated fraction from **52.8% to 18.1%**
and expected profit from **$191,860 to $9,443**, a 20× overstatement, because the
break-even threshold `cost / value` was 16× too low. `Settings.value_per_outcome`
now prices each outcome explicitly and the readout prints the conversion, the
derived per-visit value, and the multiplier that connects them.
