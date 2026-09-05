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

Three candidate explanations were tested and two were ruled out:

- **Duplicate rows?** No. The file contains 1,259,545 exact duplicate rows
  (9.0%), and the duplication rate is very different by arm — 10.2% of treated
  rows versus 2.3% of control rows. But removing them makes the imbalance
  **worse**, not better: max |SMD| goes from 0.0488 to 0.0849. The duplicates
  were partially masking the imbalance.
- **Outcome-dependent subsampling?** Not the whole story. Stratifying by the
  outcome gives max |SMD| of 0.024 among non-visitors and 0.250 among visitors,
  but `visit` is post-treatment, so conditioning on it *should* induce
  imbalance — that is collider stratification, and the visitor stratum is a
  textbook demonstration of it rather than evidence about the design.
- **Heavy tails inflating a mean-based statistic?** This is what the evidence
  supports: identical medians, significantly different means, |skew| up to 15.

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

The two covariate-adjusted estimators agree with each other and disagree with
the raw difference by about a quarter. Under clean randomization they should all
agree, and adjustment should only tighten the interval. They diverge here
because the covariates are both **imbalanced** (this section) and **strongly
predictive of the outcome** (section 4) — which is exactly the condition under
which adjustment moves a point estimate rather than merely sharpening it.

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
