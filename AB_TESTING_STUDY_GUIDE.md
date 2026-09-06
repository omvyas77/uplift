# Understanding Your Project: From A/B Testing to Causal Inference

*A study guide built around one worked example, carried all the way through.*

---

## How to use this

Read it in order. Every section builds on the last, and **every section uses the same example** — a coffee chain sending a discount coupon. By the end that one example will have taught you everything your repo does.

The boxes marked **Check yourself** are small questions. Try them before reading the answer. If you can answer all of them, you can defend this project in an interview.

At the end there's a map from each concept to the exact file in your repo, and a walkthrough of your real measured numbers.

---

## Table of contents

**Part I — Foundations**
1. [First: what is this project actually about?](#1-first-what-is-this-project-actually-about)
2. [The running example: Brew & Bean](#2-the-running-example-brew--bean)
3. [The fundamental problem: you can never see the counterfactual](#3-the-fundamental-problem-you-can-never-see-the-counterfactual)
4. [Randomization: the trick that solves it](#4-randomization-the-trick-that-solves-it)
5. [Your first result](#5-your-first-result)

**Part II — A/B testing properly**

6. [How sure are you? SE, confidence intervals, p-values](#6-how-sure-are-you-se-confidence-intervals-p-values)
7. [Checks you run before trusting any result](#7-checks-you-run-before-trusting-any-result)
8. [The 85/15 lesson: why your split cost you half your precision](#8-the-8515-lesson-why-your-split-cost-you-half-your-precision)
9. [Power and MDE: what to do *before* you run](#9-power-and-mde-what-to-do-before-you-run)
10. [Peeking: how to fool yourself daily](#10-peeking-how-to-fool-yourself-daily)
11. [Variance reduction: CUPED and CUPAC](#11-variance-reduction-cuped-and-cupac)
12. [When the treatment doesn't arrive: ITT and CACE](#12-when-the-treatment-doesnt-arrive-itt-and-cace)

**Part III — Beyond A/B testing (what your project actually does)**

13. [Not everyone responds the same: the four customer types](#13-not-everyone-responds-the-same-the-four-customer-types)
14. [Measuring an uplift model: the Qini curve](#14-measuring-an-uplift-model-the-qini-curve)
15. [Turning scores into decisions](#15-turning-scores-into-decisions)
16. [When you can't randomize](#16-when-you-cant-randomize)
17. [The answer-key trick: your project's centerpiece](#17-the-answer-key-trick-your-projects-centerpiece)

**Part IV — Your project**

18. [Map: concept → file in your repo](#18-map-concept--file-in-your-repo)
19. [Your real numbers, explained](#19-your-real-numbers-explained)
20. [Glossary](#20-glossary)
21. [Self-test](#21-self-test)

---

# Part I — Foundations

## 1. First: what is this project actually about?

You guessed A/B testing. That's about a quarter right, and the missing three quarters is the interesting part.

**An A/B test answers one question:** *does this thing work, on average, across everyone?* You get one number, and you decide launch or kill.

**Your project answers five:**

| Question | Field | Where in your repo |
|---|---|---|
| Does the campaign work on average? | A/B testing | `src/uplift/experiment/` |
| How precisely can we know that? | Statistics | `ate.py`, `power.py`, `cuped.py` |
| Does it work *for whom*? | Heterogeneous treatment effects | `src/uplift/models/` |
| So who should we send it to? | Policy learning | `src/uplift/policy/` |
| Can we answer any of this when we *couldn't* randomize? | Causal inference | `src/uplift/causal/` |

The umbrella term is **causal inference**. A/B testing is the easiest special case of it — the case where you controlled the assignment yourself. Everything else in your project is what you do when that's not enough or not available.

There's also a more specific industry name for what you built: **incrementality measurement**. That's the marketing-world term for "how much of this outcome did we actually cause," and it's the exact phrase used by the teams at Uber, DoorDash, and Amazon that this project targets.

> **Check yourself**
> **Q:** Your friend says "I ran an A/B test, variant B won, we're launching it." What has she *not* learned?
> **A:** Whether B is better *for everyone* or just on average (it might hurt a large subgroup while helping a larger one), who she should give B to if it's expensive, and whether the win was real or a result of stopping the test the day it looked good. Your project addresses all three.

---

## 2. The running example: Brew & Bean

Here is the example. Everything in this guide comes back to it.

**Brew & Bean** is a coffee chain with a mobile app. They have **100,000 app users**. Marketing wants to know whether sending a "**20% off your next coffee**" push notification actually brings people into stores.

So they run a test:

- **85,000 users** get the coupon push. Call this the **treatment group**.
- **15,000 users** get nothing. Call this the **control group**, or the **holdout**.
- Who goes in which group is decided **at random**.
- They then measure, over the next two weeks, **who walked into a store** (a *visit*) and **who bought something** (a *conversion*).

Why 85/15 and not 50/50? Because marketing doesn't want to withhold coupons from half its customer base — that's real revenue given up. They'll sacrifice some measurement precision to keep more people earning. Hold that thought; §8 is entirely about what that decision cost.

**This maps exactly onto your real project:**

| Brew & Bean | Criteo dataset | Column in your data |
|---|---|---|
| Sent the coupon push | Assigned to the ad campaign | `treatment` |
| Phone actually received & displayed it | Ad was actually served | `exposure` |
| Walked into a store | Visited the advertiser's site | `visit` |
| Bought something | Converted | `conversion` |
| What we know about the customer | 12 anonymized features | `f0`–`f11` |

Every concept below, I'll explain in coffee terms first, then point at your data.

---

## 3. The fundamental problem: you can never see the counterfactual

Here's the thing that makes this whole field necessary.

Take **one customer — Maya**. Brew & Bean sends her the coupon. Three days later she walks into a store.

**Did the coupon cause that visit?**

You cannot possibly know. To know, you'd need to compare:

- **The world where Maya got the coupon** → she visited. ✅ You saw this.
- **The world where Maya did NOT get the coupon** → did she visit? ❓ **You can never see this.**

That second one is called the **counterfactual** — literally "counter to fact," the thing that didn't happen. For any single person, you only ever observe one of the two worlds. The other is permanently invisible.

This has a formal name: **the fundamental problem of causal inference**. It's not a limitation of your data or your budget. You cannot fix it with more users or better tracking. It's structural.

> **Check yourself**
> **Q:** Maya visited after getting the coupon. Why isn't "the coupon worked" a safe conclusion?
> **A:** Because Maya might have visited anyway. She might buy coffee every Tuesday. The coupon might have been irrelevant. You saw a visit *and* a coupon, but you have no evidence the one produced the other.

**The two things people get wrong here:**

**Wrong idea #1: "Just ask her."** People are terrible at this. Ask Maya "did the coupon make you come in?" and she'll give you a plausible story either way. Self-reported causation is not evidence.

**Wrong idea #2: "Compare people who used the coupon to people who didn't."** This is the big one, and it's wrong in a way that feels right. People who *use* coupons are systematically different — they're more engaged, they open the app, they were already planning to come in. You'd be comparing keen customers to indifferent ones and calling the difference "the coupon effect." (This exact mistake has a name and a whole section: §12.)

**So what's the way out?**

You give up on individuals. You cannot know what the coupon did to *Maya*. But you *can* learn what it did **on average across a group** — if you build the group correctly. That's the next section, and it's the single most important idea in the field.

---

## 4. Randomization: the trick that solves it

You can't see Maya's counterfactual. But suppose you could find a group of people who are, collectively, **just like the treated group in every respect except that they didn't get the coupon**.

Then that group's visit rate is a stand-in for what the treated group *would have done* without the coupon. Not for any individual — as a group.

**How do you build such a group? You flip a coin.**

That's it. For each of the 100,000 users, run a random draw. 85% chance treatment, 15% chance control. Nothing about the person influences the draw.

**Why this works — and this is worth really absorbing:**

Because assignment is random, the two groups end up similar on **everything**, automatically:

- Same average age
- Same average past spending
- Same proportion of students, commuters, weekend regulars
- Same distribution of coffee preference
- Same proportion of people having a bad week
- **Same on things you never measured and never thought of**

That last point is the magic, and it's why randomization is called the "gold standard." Any other method controls only for things you *thought to measure*. Randomization balances the variables you don't know exist. There is no other technique that does this.

Two caveats worth knowing:
- It works *on average and in large samples*. With 30 people per group, random assignment can easily produce lopsided groups by chance. With 100,000, it's reliable. (§7 shows you how to check.)
- It balances things measured *before* assignment. Anything that happens *after* — like whether the phone displayed the notification — is not protected. This is the trap in §12.

> **Check yourself**
> **Q:** Marketing suggests: "Instead of random, let's send the coupon to our most loyal customers — they're most likely to respond." What breaks?
> **A:** Everything. Loyal customers visit more often *anyway*. You'd measure a huge "effect" that is almost entirely their pre-existing loyalty. You'd have no way to separate coupon effect from loyalty effect, because the two are now perfectly tangled. That tangling has a name: **confounding**, and §16 is about what to do when you're stuck with it.

> **Q:** Why doesn't "match the groups on age, gender, and past spend" work just as well?
> **A:** It balances those three things and nothing else. Randomization balances those three *plus* everything you didn't measure. You can't match on variables you don't have.

---

## 5. Your first result

Two weeks pass. Here's what Brew & Bean observes:

| Group | Users | Visited a store | Visit rate |
|---|---|---|---|
| **Treatment** (got coupon) | 85,000 | 4,250 | **5.0%** |
| **Control** (holdout) | 15,000 | 600 | **4.0%** |

### The calculation

**Absolute effect** (the "lift"):

```
5.0% − 4.0% = 1.0 percentage point
```

**Relative effect:**

```
1.0 / 4.0 = 25% lift
```

That's your headline: **the coupon increased store visits by 25%.**

### The part everyone skips, and shouldn't

Marketing will look at that table and say *"the campaign drove 4,250 visits."*

**That is wrong, and correcting it is the most valuable thing this project does.**

Use the control group to work out what the treated group would have done anyway:

```
85,000 treated users × 4.0% (the holdout's natural rate) = 3,400 visits
```

Those 3,400 visits **would have happened with no coupon at all**. So:

```
4,250 observed  −  3,400 that would have happened  =  850 incremental visits
```

**Only 850 of the 4,250 visits — 20% — were actually caused by the campaign. The other 80% would have happened anyway.**

This is *the* sentence to say to a business stakeholder. Almost every marketing dashboard in existence reports the 4,250 number. Reporting the 850 is what makes you useful.

> **Check yourself**
> **Q:** Brew & Bean spent $4,000 on the campaign. What's the cost per visit, and what's the honest cost per visit?
> **A:** The number marketing will report: $4,000 / 4,250 = **$0.94 per visit**. The honest number: $4,000 / 850 = **$4.71 per incremental visit**. Five times worse. Whether that's still a good deal depends on what a visit is worth — but now the question is being asked against the right number.

> **Q:** If Brew & Bean had no holdout group at all, could they have computed the 850?
> **A:** No. With no control group there is no estimate of "what would have happened anyway," so there's no way to separate caused visits from natural ones. **The holdout is not wasted budget — it is the entire measurement instrument.** That's the argument to use when someone proposes shrinking it.

### Naming what you just computed

This has a formal name you'll see everywhere: the **Average Treatment Effect (ATE)**.

- **Average** — across the whole group, not per person (§3: per person is impossible)
- **Treatment** — the coupon
- **Effect** — the change it caused

And the estimator you used — subtract one group's mean from the other's — is called **difference in means**. It's the simplest possible causal estimator, and because of randomization it's **unbiased**: run this experiment many times and the answers center on the truth.

Everything else in Part II is about making this number *more precise* or *more trustworthy*. The number itself is already correct.

---

# Part II — A/B testing properly

## 6. How sure are you? SE, confidence intervals, p-values

You measured 1.0 percentage point. But you only observed 100,000 people, and random assignment introduces randomness. Run the same experiment again next month with different people and you'd get a slightly different number.

So: **how much would it wobble?**

### Standard error — the wobble

The **standard error (SE)** is the typical amount your estimate would move if you re-ran the experiment. For a difference between two rates:

```
SE = √( p_t(1−p_t)/n_t  +  p_c(1−p_c)/n_c )
```

Read it as: *(wobble from the treatment arm) + (wobble from the control arm)*, added together, square-rooted.

For Brew & Bean:

```
Treatment arm:  0.05 × 0.95 / 85,000  = 0.000000559
Control arm:    0.04 × 0.96 / 15,000  = 0.00000256
                                        ─────────────
Total                                 = 0.00000312
SE = √0.00000312                      = 0.00177
```

**Now look hard at those two lines.** The control arm is **15% of your users** but contributes **82% of the total wobble**. The treatment arm, with 85% of the users, contributes 18%.

That is not a rounding artifact. It's the single most important intuition in experiment design, and §8 is entirely about it.

### Confidence interval — the range

Take the estimate ± about 2 standard errors:

```
1.0pp ± 1.96 × 0.177pp  =  [0.65pp, 1.35pp]
```

**How to say it in a meeting:** *"Our best estimate is a 1.0 point lift. The data is consistent with anything from 0.65 to 1.35 points. We can rule out zero."*

**How NOT to say it:** "There's a 95% chance the true effect is in this range." That's a common misreading. The correct version: if you re-ran this experiment many times and built an interval each time, about 95% of those intervals would contain the truth. Nobody will correct you in a business meeting, but a technical interviewer might.

**What a CI is really telling you** is precision. A CI of [0.65, 1.35] means you know the effect is positive and roughly a point. A CI of [−0.2, 2.2] would have the same midpoint but tell you almost nothing — the campaign might be useless or might be great. Same estimate, completely different decision.

### p-value — the "could this be nothing?" check

Divide the estimate by its SE:

```
t = 1.0 / 0.177 = 5.66
```

Your effect is **5.66 standard errors away from zero**. The p-value converts that to a probability: **p = 1.5 × 10⁻⁸**.

**What it means:** *if the coupon truly did nothing*, you'd see a result this extreme about 1.5 times in 100 million experiments. So "it did nothing" is a poor explanation of what you saw.

**What it does NOT mean** (and interviewers do ask):
- ❌ Not "a 1.5-in-100-million chance the coupon doesn't work"
- ❌ Not a measure of how *big* the effect is
- ❌ Not a measure of how *important* the effect is

A p-value only answers: *could random noise plausibly have produced this?* Nothing more.

### The trap: big samples make everything "significant"

With 14 million rows — your real project — you can detect effects far too small to care about. A 0.001pp lift with p < 0.001 is statistically significant and commercially worthless.

**The habit that separates good analysts from bad ones: lead with the effect size and its confidence interval. Mention the p-value last, if at all.**

This isn't abstract for you. It's exactly the reasoning behind your SRM finding in §7 and your Dagster check gating on effect size rather than p-value.

> **Check yourself**
> **Q:** Campaign A: +0.5pp, CI [0.4, 0.6]. Campaign B: +3.0pp, CI [−0.5, 6.5]. Which do you launch?
> **A:** Probably A. B has the bigger point estimate but the data can't rule out that it does nothing at all. A is small but nailed down. What you'd actually do: launch A now, and go get more data on B, because B might be excellent — you just don't know yet. "Not significant" means *not yet measured well enough*, not *proven to be zero*.

> **Q:** Your SE is 0.00177. You want to halve it. How many more users do you need?
> **A:** Four times as many. SE shrinks with **√n**, so halving the SE means 4× the sample. This is why precision gets expensive fast, and why §11's variance reduction is worth real effort — it buys precision without buying users.

---

## 7. Checks you run before trusting any result

You have a number and a CI. **Do not report it yet.** Two things can silently invalidate everything, and both are cheap to check.

### Check 1: SRM — did the randomization actually work?

You *intended* 85,000 / 15,000. What if you got 84,000 / 16,000?

That's a **Sample Ratio Mismatch (SRM)**, and it's an emergency. Not because the numbers are unbalanced — because **it means something is broken in a way you don't understand**, and whatever broke probably didn't break at random.

Real causes:
- The app crashes on launch for some users, so they never register as treated — and crash-prone phones are older phones, which means poorer users, who behave differently
- A bug drops users whose region field is empty
- Someone re-ran part of the assignment job

In every case, the users who went missing are **a specific kind of user**. Your groups are no longer comparable, and your effect estimate is contaminated by whatever made them differ.

**The test:** chi-square goodness of fit, comparing observed counts to designed counts.

```
Observed: 84,000 / 16,000
Expected: 85,000 / 15,000
χ² = 78.4   →   p = 8×10⁻¹⁹
```

That's a five-alarm fire. **Stop. Do not analyze. Find the bug.**

Compare a mild case — 84,950 / 15,050:

```
χ² = 0.196   →   p = 0.66
```

Fine. That's ordinary coin-flip variation.

**Two practical details that show up in your repo:**

**Use α = 0.001, not 0.05.** You run SRM checks constantly. At 0.05 you'd cry wolf on one healthy experiment in twenty and people would stop listening.

**At huge sample sizes, gate on the deviation, not the p-value.** This is what you found in your real data. Your allocation was off by **0.000000129** — and the test can resolve deviations down to **0.0004**. So the test *can't* fire, and if it did it would be flagging something a thousand times too small to matter. Your Dagster check gates on `abs_deviation < 0.005` for exactly this reason.

### Check 2: Balance — do the groups actually look alike?

Randomization *should* have made the groups similar. Verify it. Compare a pre-existing characteristic — say, average spend in the three months *before* the experiment:

| | Treatment | Control |
|---|---|---|
| Avg. prior 3-month spend | $47.20 | $46.90 |

Close. Good. But "close" needs a number, and the standard one is the **standardized mean difference (SMD)**:

```
SMD = (mean_treatment − mean_control) / pooled standard deviation
```

Dividing by the standard deviation makes it comparable across variables measured in different units — dollars, visits, age. Rules of thumb: **|SMD| < 0.1** is conventionally "balanced"; in a true RCT with 14M rows you'd expect **< 0.01**.

**Critical rule: only check variables measured BEFORE assignment.** Prior spend, signup date, age — fine. Anything from during or after the experiment is *caused by* the treatment, and "balancing" on it would destroy your comparison. (This is the same trap as §12.)

**Your real project found something here**, and it's worth understanding as a live example rather than a hypothetical. You measured max |SMD| = 0.049, with 7 of 12 features past the 0.02 threshold — but 11 of 12 features had *identical medians*. So the groups have the same middle and differ in the tails.

The likely explanation is worth knowing because it teaches a real concept: **the Criteo dataset is documented as several separate experiments pooled together.** Each individual experiment was properly randomized. But if experiment A drew from a different user population than experiment B, then stacking them produces a dataset where the *aggregate* looks imbalanced even though every component is clean. Same medians, different tails is exactly the fingerprint of a mixture of populations.

You handled this correctly — you split the test into a warning at 0.02 and a blocking failure at 0.10, rather than widening the tolerance to make the build go green. **Loosening a test until it passes is how you end up shipping a wrong number.**

> **Check yourself**
> **Q:** SRM passes and balance passes. Are you safe?
> **A:** Safer, not safe. These two catch the most common breakages. They can't catch a broken outcome metric, treatment leaking into the control group, or a bug that hits both arms equally.

> **Q:** Balance fails on one feature out of twelve. Panic?
> **A:** Not immediately. Test twelve things at the 5% level and you expect roughly one false alarm by chance. Look at the *size* of the imbalance and whether that feature predicts your outcome. Seven of twelve failing — your actual result — is a different story and does need explaining.

---

## 8. The 85/15 lesson: why your split cost you half your precision

Back to that observation from §6. Here it is again, because it deserves its own section:

| Arm | Share of users | Share of the wobble |
|---|---|---|
| Treatment | 85% | **18%** |
| Control | **15%** | **82%** |

**Why?** Precision comes from sample size in *each* group, and the total is limited by the **smaller** one. 85,000 people measure the treated rate very precisely. 15,000 measure the control rate much less precisely. And your answer is the *difference* of the two, so it inherits the sloppiness of the worse-measured one.

**The kitchen analogy:** you're weighing two bags of flour to find the difference. One scale is precise to a gram; the other to fifty grams. Buying an even better first scale is pointless — your answer is only as good as the 50-gram scale. The control group is the 50-gram scale.

### Quantifying it

There's a clean formula for how efficient an unbalanced split is, relative to a 50/50 split:

```
efficiency = 4 × r × (1 − r)        where r = share in treatment
```

For r = 0.85:

```
4 × 0.85 × 0.15 = 0.51
```

**The 85/15 split is 51% as efficient as 50/50.** Concretely, at 100,000 users:

| | 85/15 | 50/50 |
|---|---|---|
| Standard error | 0.00177 | 0.00131 |
| Detectable effect (MDE) | 12.15% relative | 8.68% relative |
| Users needed for equal precision | **196,000** | 100,000 |

**You need roughly twice the users to learn the same amount.** Half your sample is, in pure measurement terms, wasted.

### But the 85/15 split is still the right call

This is the part that makes it a good interview answer instead of a criticism.

Every user in the control group is a user **not receiving a coupon** — real revenue given up. Marketing chose to keep 85% of users earning rather than shrink the error bar. That's a legitimate business trade.

**The framing to use:**

> "Our 85/15 allocation is 51% as statistically efficient as a balanced design — we need about twice the sample for the same confidence. That was the right trade, because a balanced split would have meant withholding the campaign from 35,000 more customers. We chose revenue over precision, and here's exactly what the precision cost."

Naming the trade *and pricing it* is what senior looks like. Anyone can say "we used 85/15." Almost nobody can say what it cost.

Your real project reproduced this exactly: **efficiency 0.510**, MDE 1.053% at 85/15 versus 0.675% at 50/50, 1.96× the sample needed.

> **Check yourself**
> **Q:** You can add 20,000 users. Put them all in control, all in treatment, or split them?
> **A:** All in control. It's the undersized arm and it's dominating your variance. Going 85,000/35,000 helps far more than 105,000/15,000 — which would barely move the needle, because you'd be improving the arm that was already precise.

> **Q:** When would 50/50 be wrong?
> **A:** When treatment is expensive or risky. Testing a change that might break checkout? Start at 99/1 — you're buying safety with precision. When treatment is *cheaper* than control (like here, where control means forgone revenue), tilt the other way. The split is a business decision informed by statistics, not a statistical decision.

---

## 9. Power and MDE: what to do *before* you run

Everything so far was analysis after the fact. This section is the one that happens **before** — and it's the one people skip and regret.

**The question:** *given the sample we'll have, what's the smallest effect we could reliably detect?*

That's the **Minimum Detectable Effect (MDE)**. Get it wrong and you run a six-week experiment that was mathematically incapable of finding the effect you were looking for. That happens constantly.

### The Brew & Bean version

At 100,000 users, 85/15, a 4% baseline rate:

```
MDE = 0.49 percentage points  =  12.15% relative lift
```

**Read that carefully.** With this sample, Brew & Bean can only reliably detect lifts of **12% or larger**. If the coupon's true effect is a 6% lift — a real, valuable effect — this experiment will probably come back "not significant" and marketing will conclude the coupon doesn't work.

**It does work. The experiment was just too small to see it.**

### The three levers

| Lever | Effect | Cost |
|---|---|---|
| More users | 4× users → half the MDE | Time or reach |
| Balanced split | 85/15 → 50/50 cuts MDE from 12.15% to 8.68% | Forgone revenue |
| Variance reduction | See §11 — free precision | Engineering effort |

At 400,000 users the MDE drops to 6.08%. Note again: **quadrupling the sample only halves the MDE.** Precision is expensive.

### "Power" — the related term

**Statistical power** is the probability your experiment detects an effect *that is really there*. Convention is 80% — meaning if the effect is exactly your MDE, you have an 80% chance of finding it and a **20% chance of missing it entirely**.

Say that out loud: even a properly powered experiment misses a real effect one time in five.

### The conversation this enables

Before running anything, marketing says *"we think this will lift visits by about 5%."* You compute the MDE at 12% and say:

> "With our current user base we can only detect a 12% lift. If your 5% guess is right, this experiment will most likely come back inconclusive and we'll have burned six weeks. To detect 5% we need about 600,000 users, or we run for four times as long, or we go to a balanced split, or we use variance reduction. Which do you want?"

**That conversation is worth more than any model in your repo.** It's the difference between an analyst who runs tests and one who designs them.

> **Check yourself**
> **Q:** Test comes back +3% lift, p = 0.4, not significant. MDE was 12%. Conclusion?
> **A:** **You learned almost nothing.** You cannot say the coupon doesn't work — you built an experiment incapable of detecting anything under 12%. The honest write-up: "inconclusive; this design could not detect effects below 12%, and our point estimate of 3% is well inside the range we can't resolve." Reporting that as "no effect" would be a real error, and it's an extremely common one.

---

## 10. Peeking: how to fool yourself daily

Brew & Bean runs the test for two weeks. The marketing manager checks the dashboard every morning.

**Day 4:** *"We're significant! Let's call it and roll out!"*

**This is the single most common way real experiments produce wrong answers.**

### Why it breaks

A p-value of 0.05 means: *if there were no effect, you'd get a result this extreme 5% of the time.* That guarantee holds for **one look, at a pre-specified time**.

Look ten times and you get ten chances to catch a random fluctuation. Since you stop the moment you see significance, you're **systematically selecting the flukes**.

### The measured damage

Your project simulates this. Ten peeks at nominal 5%:

| | False positive rate |
|---|---|
| What you think you have | 5% |
| **What you actually have (10 peeks)** | **18.25%** |
| With O'Brien-Fleming correction | 6.25% |

**Roughly one in five of your "wins" is pure noise.**

### The business translation

> "If we check daily and stop the moment it looks good, about one in five of our winning campaigns is actually doing nothing. We'll roll it out, we'll pay for it, and we'll never know."

That number gets executive attention in a way "multiple comparisons problem" never will.

### The three fixes

1. **Pre-register.** Decide the sample size and end date up front, look once. Simplest, and unpopular.
2. **Alpha spending** (O'Brien-Fleming, Pocock). Plan a fixed number of interim looks and use a stricter threshold at each. Early looks require overwhelming evidence; the final look is near-normal. Standard in clinical trials.
3. **Always-valid p-values / sequential testing** (mSPRT). Built to be looked at continuously — check as often as you like without inflating error. What Optimizely and Statsig use.

**A cultural note that matters more than the math:** you usually can't stop people from looking. Dashboards exist. The realistic solution is to make the dashboard show a sequentially-valid boundary, so peeking is *safe by construction*. Fixing the tool beats fixing the humans.

> **Check yourself**
> **Q:** Is peeking with no intention to stop early also a problem?
> **A:** Looking is harmless. **Deciding based on what you saw** is the problem. If you'd stop early on a good result, you're peeking in the harmful sense — even if it doesn't happen this time. What matters is the decision rule, not the eyeballs.

---

## 11. Variance reduction: CUPED and CUPAC

§9 gave you three ways to get more precision. Two cost money. This one is nearly free, and it's the most practically useful technique in this guide.

### The idea

Some Brew & Bean customers spend $200/month. Some spend $5. That enormous variation makes your visit-rate measurement noisy — not because of the coupon, but because customers differ.

But here's the thing: **you knew they differed before the experiment started.** You have three months of prior spend data.

So: **subtract off the variation you could have predicted anyway.** What's left is closer to pure treatment effect, and it's much less noisy.

### The mechanics

**CUPED** — *Controlled experiment Using Pre-Experiment Data*:

```
Y_adjusted = Y − θ × (X − mean(X))

where X = the pre-experiment covariate (prior spend)
      θ = Cov(Y, X) / Var(X)
```

**Plain English:** if a customer spent more than average *before* the experiment, we expected more visits from them regardless. So we subtract that expectation off. High-prior-spend customers get adjusted down, low get adjusted up. Everyone is put on a level field.

### The one formula to memorize

```
variance reduction = ρ²
```

where ρ is the correlation between your covariate and your outcome. That's it.

| Correlation ρ | Variance reduction | Equivalent extra sample |
|---|---|---|
| 0.3 | 9% | 1.10× |
| 0.5 | **25%** | **1.33×** |
| 0.7 | 49% | 1.96× |
| 0.9 | 81% | 5.26× |

At ρ = 0.5 your SE drops from 0.00177 to 0.00153. Same users, same experiment, better answer — **equivalent to having recruited 33,000 more people, for free.**

**Two properties that make it safe:**
- The adjustment **preserves the mean**, so your effect estimate is unchanged in expectation. Only the noise shrinks.
- It requires **only that the covariate is pre-treatment**. It doesn't need to be a good predictor — a bad one just helps less.

### The one rule you must never break

**The covariate must be measured BEFORE treatment.**

Use post-treatment data and you'll "adjust away" part of the actual effect, biasing your answer toward zero. If someone suggests adjusting for "app opens during the experiment" — that's caused by the coupon. Refuse.

### CUPAC: what to do with no pre-period

Your Criteo data has **no time dimension at all.** No before-period, no dates. So textbook CUPED is unavailable.

The fix is **CUPAC** — *Control Using Predictions As Covariates*, from DoorDash:

1. Take only the **control group** users
2. Train a model predicting the outcome from `f0`–`f11`
3. Use that model's prediction as the CUPED covariate

**Why control-only?** So the covariate can't contain any treatment information. The features are pre-treatment, and the model never sees treated outcomes, so the prediction is a legitimate pre-treatment quantity. Cross-fitting (train on some folds, predict on the held-out one) prevents the model memorizing rather than predicting.

### Your real result, and the flag on it

Your project measured **ρ = 0.5563, variance reduction 30.95%** — an effective sample 1.45× larger.

Your build guide predicted "a few percent." Getting 31% is a *pleasant* surprise, and pleasant surprises in statistics deserve one extra check.

**Why it's suspicious:** it means 12 anonymized, randomly-projected features explain 31% of the variance of a 4.7% binary outcome. That's strong prediction from features that look like noise.

**Why it might be completely real:** §7's mixture hypothesis. If the Criteo file pools several separate campaigns, and the features partly encode *which campaign* a user came from, and campaigns differ in baseline visit rate — then predicting visits *is* easy, with no bug involved. That would explain the high ρ, the tail imbalance, and the adjusted-estimate shift, all from one cause.

**The check that settles it:** you found 1.26M exact duplicate rows (9% of the file). Plain K-fold splitting shuffles rows independently, so a duplicate's twin can land in a different fold — and the model *memorizes* it rather than predicting. Confirm that **ρ itself** stays near 0.556 after deduplication, not just that variance reduction stays positive. If it holds up, group your cross-validation folds by a hash of the feature vector so near-duplicates can't leak either.

> **Check yourself**
> **Q:** Someone proposes using "number of app opens during the experiment" as a CUPED covariate. It correlates 0.8 with visits. Do it?
> **A:** **No.** App opens during the experiment are *caused by* the coupon. You'd subtract off part of the treatment effect and bias your estimate toward zero. High correlation makes it tempting and makes the damage worse. Pre-treatment only, no exceptions.

> **Q:** Your CUPED gives 2% variance reduction. Wasted effort?
> **A:** Not wasted, just honest. 2% means ρ ≈ 0.14 — your covariate barely predicts the outcome. That's information about your data, not a failure of the method. Report the 2%. Claiming more than ρ² allows is the actual failure.

---

## 12. When the treatment doesn't arrive: ITT and CACE

This is where your Criteo data has a **trap** that will silently wreck a naive analysis. It's also the single best interview story in your project.

### The setup

Brew & Bean sends the coupon to 85,000 users. But not everyone gets it:

- Some have notifications disabled
- Some have the app uninstalled
- Some have their phone off for days
- Some have a full notification tray and never see it

**Only 30% of the treated group — 25,500 people — actually received and saw the coupon.**

So which comparison do you make?

### Option A: ITT — Intention To Treat

Compare **everyone assigned to treatment** vs. **everyone assigned to control**. Include the people who never saw it.

```
ITT = 5.0% − 4.0% = 1.0 percentage point
```

**Feels wrong** — you're counting people who never got the coupon as "treated."

**It's the right headline number**, for two reasons:

1. **It's what actually happens when you launch.** In the real world, when you fire this campaign at everyone, only 30% will see it. The ITT already includes that reality. It answers *"what happens if we press the button?"* — the actual business question.
2. **It's the only comparison protected by randomization.** Assignment was random. Everything downstream wasn't.

### Option B: The tempting one, which is wrong

Compare **people who actually saw the coupon** (25,500) vs. **the control group**.

**This is broken**, and it's worth understanding exactly why.

Who has notifications enabled? People who are **engaged with the app**. Who is engaged with the app? People who **like Brew & Bean and visit often anyway.**

So you'd be comparing engaged customers to a mix of engaged and disengaged ones, and calling the gap "the coupon effect." Most of that gap is pre-existing engagement.

**The formal statement, worth memorizing:** *exposure is a post-treatment variable. Conditioning on it breaks randomization.*

Your `f0`–`f11` are pre-treatment — safe. `exposure` happens **after** assignment and is influenced by the user's own behavior — **not safe**. Filtering on it, or using it as a model feature, silently reintroduces exactly the selection bias randomization eliminated.

Your repo has a comment marking this in `features.py`. Good — that comment is worth more to a code reviewer than a model.

### Option C: CACE — the right way to ask the question

You still might genuinely want *"what's the effect on people who actually see the ad?"* There's a valid way, using the randomization as a lever.

```
CACE = ITT / compliance rate
```

For Brew & Bean:

```
CACE = 1.0pp / 0.30 = 3.33 percentage points
```

**The intuition:** the whole 1.0pp effect was produced by only 30% of the group — the other 70% never saw anything and can't have been affected. So concentrate the effect onto the people who could have responded. 1.0 ÷ 0.30 = 3.33.

This is an **instrumental variables** estimate. Random assignment is the *instrument*: it shifts exposure without being connected to anything else about the person.

**It needs an assumption you can't test — the exclusion restriction:** being *assigned* affects your visiting **only through actually seeing the coupon**. If assignment changed behavior some other way, CACE is wrong. Naming an untestable assumption instead of pretending it holds is exactly the habit these teams hire for.

### Your real numbers

| | Brew & Bean | Your Criteo project |
|---|---|---|
| Compliance | 30% | **3.6%** |
| ITT | 1.0pp | **1.03pp** |
| CACE | 3.33pp | **28.7pp** |

**Only 3.6% of the "targeted" users were ever actually shown an ad.** You targeted 11.9M people and reached about 428,000.

**Two things follow, and they're both worth saying out loud:**

**The business insight:** the entire 27% lift comes from 3.6% of the audience. **Delivery, not targeting, is where the leverage is.** Push delivery from 3.6% to 7% and impact roughly doubles — before anyone touches a model. (Caveat it: the next tranche of reachable users is probably less responsive than the current one.)

**The reporting discipline:** do **not** put "ads have a 28.7 point effect" in front of stakeholders. It's arithmetically correct, it describes 3.6% of people, it leans on an assumption that's shaky here (being reachable correlates with actively browsing, which correlates with visiting anyway), and it will be misquoted within a day. Keep it as a secondary estimand in the repo.

> **Check yourself**
> **Q:** Compliance is 3.6% and CACE is 28.7pp. Why not report the bigger number — it's more impressive?
> **A:** Because it answers a question nobody is asking. Nobody can choose to launch a campaign "only for people who'll be reachable" — reachability isn't a targetable attribute. The decision on the table is "run it or don't," and ITT is the number that answers it. CACE is diagnostic: it tells you the *mechanism* is strong and delivery is the bottleneck.

> **Q:** Why is CACE always bigger than ITT?
> **A:** You're dividing by a number less than 1. ITT spreads the effect across everyone including those who never got treated; CACE concentrates it on those who did. They're only equal at 100% compliance.

---

# Part III — Beyond A/B testing

Everything so far produced **one number** for **everyone**. That's A/B testing, and you now know it properly.

Part III is what your project actually does.

## 13. Not everyone responds the same: the four customer types

The coupon lifts visits by 1.0pp **on average**. But an average hides its ingredients. Consider four Brew & Bean customers:

**Maya** — comes in every Tuesday, coupon or not. Gets the coupon, visits, redeems it. Brew & Bean **lost 20% margin on a visit they were getting for free.**
→ Effect of the coupon on Maya: **zero.**

**Devon** — used to come in, hasn't in two months. Gets the coupon, remembers he liked the place, comes in.
→ Effect on Devon: **large and positive.** This is who the campaign is *for*.

**Priya** — moved cities last year, never coming back. Gets the coupon, ignores it.
→ Effect on Priya: **zero.** Harmless but wasted.

**Sam** — hates marketing push notifications. Gets one, is annoyed, **mutes the app.** Now he can't be reached again and comes in less.
→ Effect on Sam: **negative.**

These are the four types, and the names are industry-standard:

| Type | Visits if treated? | Visits if not? | Effect | What to do |
|---|---|---|---|---|
| **Sure Thing** (Maya) | Yes | Yes | **0** | Don't send — you're discounting a sale you had |
| **Persuadable** (Devon) | Yes | No | **+** | **Send. This is the entire value.** |
| **Lost Cause** (Priya) | No | No | **0** | Don't send — wasted spend |
| **Sleeping Dog** (Sam) | No | Yes | **−** | **Definitely don't send — you're causing harm** |

**Your 1.0pp average is a blend of all four.** It could be a modest effect on everyone, or a huge effect on 10% of people and nothing on the rest. The average can't tell you which, and **the two situations call for completely different actions**.

### The number you actually want

Instead of one ATE, you want a per-person effect:

```
τ(x) = the effect of the coupon on a person with characteristics x
```

This is the **CATE** — Conditional Average Treatment Effect. "Conditional" = depends on who they are. Also called **uplift**, **incremental effect**, or **heterogeneous treatment effect**. Same thing, four names, all used in job descriptions.

### The catch (and it's the same one as §3)

**You can't observe τ for any individual.** Maya either got the coupon or didn't; you never see both. So you can't train a model with per-person labels — there are none.

**The workaround:** you can't predict individual effects, but you can predict effects for *groups defined by characteristics*. If you can find groups where treated and untreated outcomes differ a lot, you've found the persuadables.

The models in your `src/uplift/models/` do this in various ways:

| Model | The idea |
|---|---|
| **S-learner** | One model on [features + treatment flag], predict twice — once with the flag on, once off, take the difference |
| **T-learner** | Two separate models, one per group, take the difference of their predictions |
| **X-learner** | T-learner refined to handle unequal group sizes — relevant to you, since control is only 15% |
| **DR-learner** | Doubly robust and cross-fitted; the strongest theoretical guarantees |
| **Causal forest** | Random forest that splits to maximize *effect difference* rather than outcome accuracy — and gives confidence intervals on τ(x) |

You don't need to master the internals to use them. What you need is the next section: **how to tell whether any of them worked.**

> **Check yourself**
> **Q:** Why is sending a coupon to a Sure Thing actively bad, not just neutral?
> **A:** You pay the discount on a sale you were getting anyway. A Lost Cause costs the send. A Sure Thing costs the **margin**. Sure Things are usually your best customers and the easiest to accidentally target — a plain response model ranks them first, because they're the most likely to visit. That's precisely why response models are the wrong tool.

> **Q:** Sleeping Dogs sound like an edge case. Do they matter?
> **A:** They're the reason "just send to everyone" can lose money. Email fatigue, notification muting, and unsubscribes are all real Sleeping Dog behavior. A campaign with a positive average can still be destroying long-term value in a subgroup, and only a CATE model surfaces it.

---

## 14. Measuring an uplift model: the Qini curve

Your model outputs a score per person: *"predicted uplift = 0.03."* **How do you know if it's any good?** You can't check per-person predictions — there's no ground truth (§13).

### Why accuracy metrics don't work

The instinct is AUC — how well does the model predict who visits? **That's the wrong question.** A model that perfectly predicts *who visits* would rank Maya (Sure Thing) at the top, because she's the most likely visitor. But she's the worst person to send to.

**AUC measures who responds. Uplift measures who responds *because of* the treatment.** Different quantities. Report AUC if you like, but label it clearly as not the objective — your build guide is right about that.

### The Qini curve, built by hand

Here's the whole idea in one small table. Twelve customers, six treated (`w=1`), six control (`w=0`), sorted by **predicted uplift, best first**. `y=1` means they visited.

Walk down the list. At each depth, ask: *among everyone down to here, how many extra visits did treatment produce?*

The comparison needs a scaling factor, since the treated and control counts differ as you walk:

```
Qini(k) = (treated visits so far) − (control visits so far) × (treated count / control count)
```

| Rank | w | y | Treated so far | Control so far | T visits | C visits | Scaled C | **Qini** |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0.00 | **1.00** |
| 2 | 0 | 0 | 1 | 1 | 1 | 0 | 0.00 | **1.00** |
| 3 | 1 | 1 | 2 | 1 | 2 | 0 | 0.00 | **2.00** |
| 4 | 0 | 0 | 2 | 2 | 2 | 0 | 0.00 | **2.00** |
| 5 | 1 | 1 | 3 | 2 | 3 | 0 | 0.00 | **3.00** ← peak |
| 6 | 0 | 1 | 3 | 3 | 3 | 1 | 1.00 | 2.00 |
| 7 | 1 | 1 | 4 | 3 | 4 | 1 | 1.33 | 2.67 |
| 8 | 0 | 1 | 4 | 4 | 4 | 2 | 2.00 | 2.00 |
| 9 | 1 | 0 | 5 | 4 | 4 | 2 | 2.50 | 1.50 |
| 10 | 0 | 1 | 5 | 5 | 4 | 3 | 3.00 | 1.00 |
| 11 | 1 | 0 | 6 | 5 | 4 | 3 | 3.60 | 0.40 |
| 12 | 0 | 1 | 6 | 6 | 4 | 4 | 4.00 | **0.00** |

**Read the shape:**

- **Ranks 1–5 (rises to 3.0):** the top-ranked people are **Persuadables**. Treated ones visit, control ones don't. Every one added gains incremental visits.
- **Ranks 6–8 (flattens):** **Sure Things.** Treated and control both visit. Adding them gains nothing.
- **Ranks 9–12 (falls to 0):** **Sleeping Dogs.** Control visits, treated doesn't. Adding them *destroys* value.

### The punchline

**The curve ends at 0.00. Across all twelve people, the campaign had zero net effect.** An A/B test on this population reports "no effect, don't launch."

**But targeting only the top five yields +3 incremental visits.**

That is the entire case for uplift modeling in one table. The average was zero *because the Persuadables and the Sleeping Dogs cancelled out.* Only by ranking and targeting do you find the value hiding inside a null result.

### Reading real Qini curves

- **Higher and earlier peak** = better model
- **Where the peak sits** tells you your **budget**: peak at rank 5 of 12 means treat the top ~40%
- **A diagonal line** = random ranking, no value
- **The Qini coefficient** summarizes the curve as one number (area above the random line, normalized)

**Three warnings:**

1. **Qini values aren't comparable across implementations.** Different libraries normalize differently. State which you used. (Your repo uses `scikit-uplift` — say so.)
2. **Always report confidence intervals.** Qini is noisy. Your build guide is right to insist on bootstrap CIs — without them you'll crown a winner out of noise. When CIs overlap, say the models are indistinguishable. That sentence is worth more than a higher number.
3. **The control arm limits you.** With a 15% control group, the scaling factor gets unstable in thin slices. Related to §8.

> **Check yourself**
> **Q:** Model A: Qini 0.045, CI [0.021, 0.069]. Model B: Qini 0.038, CI [0.014, 0.062]. Which is better?
> **A:** You can't tell. The intervals overlap heavily. The honest report: "indistinguishable at this sample size." Pick on other grounds — simplicity, speed, stability. Declaring A the winner is manufacturing a result.

> **Q:** Your Qini curve is flat along the diagonal. What happened?
> **A:** Your model isn't finding heterogeneity. Either there genuinely isn't much (everyone responds similarly), or your features don't capture what drives differential response. Both are real findings — and the first one is a legitimate result to publish, not a failure.

---

## 15. Turning scores into decisions

A Qini curve is a model evaluation. It is not yet a business decision. This section closes that gap, and it's where most portfolio projects stop short.

### The rule

Send the coupon when the expected gain beats the cost:

```
Send if:  predicted uplift × value per visit  >  cost per coupon
```

For Brew & Bean: a visit is worth $3 in margin, a coupon costs $0.60 in discount.

```
Send if:  τ̂ × $3 > $0.60
Send if:  τ̂ > 0.20     (a 20% predicted chance of an incremental visit)
```

Sweep that threshold across all customers, and you get a profit curve: treat too few and you leave money on the table; treat too many and you're paying Sure Things and annoying Sleeping Dogs. Somewhere in between is the optimum.

### The trap in that calculation

The profit number above comes from **the model's own predictions**. If the model is overconfident, so is the profit. You'd be marking your own homework.

### The fix: off-policy evaluation

You have held-out randomized data where you know exactly how treatment was assigned — you assigned it. So you can ask: *for the people my policy would have treated, what actually happened?*

The estimators (**IPW** and **doubly robust**) reweight the held-out data to simulate your policy, giving an honest value estimate with a confidence interval.

### The comparison table that matters

| Policy | Value | 95% CI | Treated |
|---|---|---|---|
| Treat nobody | baseline | | 0% |
| Treat everybody | ... | ... | 100% |
| **Model targeting** | ... | ... | 40% |
| **Random targeting, same 40%** | ... | ... | 40% |

**That last row is the one almost everyone omits, and it's the one that matters.** If your model doesn't beat *random selection at the same budget*, the model adds nothing — you could have flipped coins.

Given how weak the uplift signal is in Criteo, be ready for the gap to be small. **Report it either way.** "My model did not beat random targeting at equal budget" is a real, publishable finding and a far better interview answer than a fabricated win.

### Translating to money

```
Incremental visits per 1M targeted:  X
Value per visit (ASSUMPTION):        $3.00
Cost per coupon (ASSUMPTION):        $0.60
Estimated incremental profit:        $Y  [CI: $A – $B]
```

**Never state a dollar figure without the assumption line directly above it.** Your repo puts these in config and exposes them as dashboard sliders — that's the right pattern, because it makes the sensitivity of your dollar figure visible instead of hiding it behind one hardcoded number.

> **Check yourself**
> **Q:** Your model's policy value beats treat-everybody but not random-at-same-budget. What's happening?
> **A:** Your gain comes from **treating fewer people**, not from *choosing better people*. Cutting spend helped; the ranking didn't. The honest conclusion: "we should target less broadly, but my model isn't identifying who." Useful — and completely different from what a Qini number alone would have suggested.

---

## 16. When you can't randomize

Everything so far assumed a proper randomized experiment. Often you don't have one:

- The campaign already ran, to whoever marketing picked
- You can't ethically withhold a treatment
- You're analyzing historical data

### What goes wrong

Suppose Brew & Bean's marketing team sent the coupon to their **Gold tier** customers — their best ones. Now:

| Group | Visit rate |
|---|---|
| Got coupon (Gold tier) | 12.0% |
| No coupon (everyone else) | 4.0% |

**"The coupon caused an 8 point lift!"** Obviously false. Gold customers visit more because they're Gold customers. The coupon effect and the loyalty effect are **completely tangled**.

That tangling is **confounding**. A confounder is anything that affects both *who gets treated* and *the outcome*. Here, loyalty does both.

### The fix, and its price

**The idea:** compare like with like. Find untreated customers who *look like* the treated ones — same spend, same visit history, same tenure — and compare within those matched groups.

**The methods:**

| Method | How it works |
|---|---|
| **Propensity score** | Model P(treated \| characteristics). Compare people with similar propensity. |
| **Matching** | Pair each treated person with a similar untreated person. |
| **IPW** | Reweight so the groups look alike — upweight underrepresented types. |
| **Doubly robust (AIPW)** | Combine an outcome model with a propensity model. Correct if **either** is right. |

**The price:** all of them require an assumption you cannot test — **no unmeasured confounding**. You must have measured *every* variable that affects both treatment and outcome. Miss one and your estimate is wrong, with no warning sign in the data.

Compare that to randomization, which balances even the variables you never thought of (§4). **This is why an RCT is worth so much:** not because the math is fancier, but because it removes an untestable assumption.

### Two more things you need

**Overlap (positivity).** For every kind of person, there must be both treated and untreated examples. If *every single* Gold customer got the coupon, there is no comparison group for Gold customers and no amount of statistics rescues you. Check overlap **before** estimating.

**Sensitivity analysis.** Since "no unmeasured confounding" is untestable, ask instead: *how strong would a hidden confounder need to be to overturn my conclusion?* If a tiny one would flip it, be humble. If it'd take an implausibly strong one, you're on firmer ground. The **E-value** is the standard summary.

> **Check yourself**
> **Q:** You match on age, gender, tenure, and past spend. The estimate looks reasonable. Are you done?
> **A:** No. You've handled four confounders. The assumption is that there are **no others** — and you can't test it. Maybe treated customers were picked partly on "engagement score," which you don't have. Run sensitivity analysis and state the assumption explicitly. Confidence here should come from domain knowledge about the assignment process, not from the fit statistics.

---

## 17. The answer-key trick: your project's centerpiece

Here's the problem with §16: you run IPW on observational data, get a number, and **have no way to check whether it's right.** That's true of essentially every observational study ever published.

**Your project solves this**, and it's the best thing in your repo.

### The trick

You have a **randomized experiment**. So you already know the true answer. Which means you can grade the observational methods:

1. **Compute the truth from the RCT.** This is your answer key.
2. **Deliberately break the randomization.** Throw away units in a way that depends on their characteristics — simulate marketing having cherry-picked Gold customers.
3. **Run the observational toolkit** on the corrupted data: naive comparison, IPW, matching, doubly robust.
4. **Compare each answer to the key.**

You now have something almost no portfolio has: **measured evidence about when causal methods work and when they fail**, on real data, with a known answer.

### The result

From the validation run in your build guide:

| Method | Estimate | Error vs. truth |
|---|---|---|
| **Truth** (from the RCT) | 0.0401 | — |
| Naive comparison | 0.0726 | **+81%** ❌ |
| IPW | 0.0378 | −5.8% ✅ |
| Doubly robust | 0.0378 | −5.9% ✅ |

**The naive comparison is off by 81%. Adjustment recovers the truth within 6%.**

**How to say it:**

> "I used the randomized experiment as an answer key. I deliberately corrupted it to look like observational data — the way real marketing data arrives — and checked which methods recovered the truth. Naive comparison was off by 81%. Doubly robust estimation got within 6%. So when I use these methods on data where I *don't* have the answer, I have evidence about how far to trust them."

**Then push it further, which is where it gets genuinely interesting:** turn the confounding strength up gradually. At some point **even the good methods break** — because extreme selection destroys overlap (§16). Finding that breaking point gives you a real answer to *"when should I not trust a propensity analysis?"*, grounded in your own experiment rather than a textbook.

### Negative controls: the same idea, cheaper

A **pre-treatment** feature cannot possibly be affected by treatment. So regressing one on treatment must give zero. That gives you a free diagnostic:

- **On the RCT** → effects ≈ 0. ✅ Passes, as it must.
- **On the corrupted data** → large fake "effects." ❌ Correctly detects confounding you injected and know is there.
- **After IPW adjustment** → back toward zero. ✅ Evidence your adjustment worked — *without needing to know the true answer.*

That last one is the useful part in practice: a check you can run on real observational data where there is no answer key.

> **Check yourself**
> **Q:** Why not just always use doubly robust and skip randomizing?
> **A:** Because in real observational data you never see the answer key. DR worked here because you *know* the confounders — you injected them yourself. In the wild you can't be sure you've measured them all, and nothing in the output warns you. The exercise measures how well these methods work *when their assumptions hold*. It says nothing about whether they hold in your next project.

---

# Part IV — Your project

## 18. Map: concept → file in your repo

| Concept | Section | File in your repo | Status |
|---|---|---|---|
| Randomization check (SRM) | §7 | `src/uplift/experiment/srm.py` | ✅ run |
| Balance / SMD | §7 | `src/uplift/experiment/balance.py` + `mart_covariate_balance` | ✅ run |
| Difference in means, Lin estimator | §5, §6 | `src/uplift/experiment/ate.py` | ✅ run |
| CUPED / CUPAC | §11 | `src/uplift/experiment/cuped.py` | ✅ run |
| Power, MDE, allocation efficiency | §8, §9 | `src/uplift/experiment/power.py` | ✅ run |
| Peeking simulation | §10 | `src/uplift/experiment/sequential.py` | ✅ run |
| The four customer types / CATE | §13 | `src/uplift/models/learners.py` | ⏳ written, not run |
| Qini / AUUC + bootstrap CIs | §14 | `src/uplift/evaluation/qini.py` | ⏳ written, not run |
| Targeting policy | §15 | `src/uplift/policy/targeting.py` | ⏳ written, not run |
| Off-policy evaluation | §15 | `src/uplift/policy/ope.py` | ⏳ written, not run |
| Confounding, IPW, DR | §16 | `src/uplift/causal/estimators.py` | ⏳ written, not run |
| ITT vs CACE | §12 | `src/uplift/causal/iv.py` | ⏳ written, not run |
| The answer-key trick | §17 | `src/uplift/causal/confounding.py` | ⏳ written, not run |
| Negative controls, E-value | §17, §16 | `src/uplift/causal/sensitivity.py` | ⏳ written, not run |

**You have fully finished and validated everything in Part II.** Parts III's code exists and is tested but hasn't been run on the full data. That's a completely normal place to be at week 4 of a 10-week plan.

---

## 19. Your real numbers, explained

Every number below is from your actual run. Here's what each one means and how to say it.

### The headline

| Metric | Your value | Plain English |
|---|---|---|
| ATE (unadjusted) | **+0.0103** (+27.07%) | Ads lifted site visits 27% |
| ATE (Lin-adjusted) | **+0.0078** (+20.24%) | After correcting for group differences: 20% |
| ATE (CUPAC) | **+0.0079** (+20.97%) | Second correction agrees: 21% |
| 95% CI (unadjusted) | [+0.0101, +0.0106] | Very precisely measured |
| t-statistic | ~71 | The effect is not in doubt |

**The incrementality version — the one for stakeholders:**

```
11.9M users treated × 3.82% (holdout rate) = ~454,000 visits that would have happened anyway
Observed treated visits:                     ~577,000
Incremental visits caused by the ads:        ~123,000  (about 21%)
```

> "Of the 577,000 visits we credited to this campaign, about 123,000 were actually caused by it. The other four out of five would have happened without spending a cent."

### The number to actually quote: 20.2%, not 27.1%

This follows directly from §7 and §11 together.

In a clean RCT, adjustment **tightens the interval and leaves the estimate alone**. Yours **moved by 7.1 standard errors** — and you already ruled out subsampling as the cause by running difference-in-means on the same 2M rows (+0.010204, essentially the full-data value).

So the movement is real adjustment, driven by the covariate imbalance in §7. **The honest headline is +20.2%**, with 27.1% shown as the unadjusted comparison.

That's a 25% haircut on the impact claim — a material business number. **Being the person who found it beats being the person who shipped 27%.**

### The operational finding

| | Value |
|---|---|
| P(exposed \| treated) | **3.6%** |
| Users "targeted" | 11.9M |
| Users actually reached | ~428,000 |

**The entire 27% lift comes from 3.6% of the audience.** Delivery, not targeting, is the bottleneck (§12). Doubling delivery roughly doubles impact — before touching a model.

### The design finding

| | Your design (85/15) | Balanced (50/50) |
|---|---|---|
| Allocation efficiency | **0.510** | 1.000 |
| MDE (relative) | 1.053% | 0.675% |
| Sample for equal precision | **1.96×** | 1× |

Frame as a priced trade (§8), not a mistake.

### The measurement-quality findings

| Finding | Value | Meaning |
|---|---|---|
| SRM | None. Deviation 1.3×10⁻⁷ vs. test resolution 4×10⁻⁴ | Allocation is essentially perfect — §7's large-sample point |
| Balance | max \|SMD\| 0.049; 7 of 12 past WARN, 0 past the 0.10 gate | Real imbalance, not blocking. Likely the pooled-experiments effect |
| CUPAC | ρ = 0.556, **31% variance reduction**, 1.45× sample | Genuine — but verify ρ survives dedup (§11) |
| Duplicates | 1,259,545 (9.0%) | Not in the build guide; your finding |
| Peeking | 18.25% false positives at nominal 5% | One in five "wins" is noise (§10) |
| Build health | dbt PASS=52 WARN=1 ERROR=0; 27 tests; ruff + mypy clean | Solid |

### The one hypothesis that ties it together

Criteo v2.1 is documented as **several incrementality trials pooled into one file**. That single fact predicts all four of your "contradictions":

| Your finding | Explained by pooling? |
|---|---|
| Allocation exact to 1.3×10⁻⁷ | ✅ Each sub-trial randomized cleanly → aggregate ratio exact |
| Covariates imbalanced, identical medians, fat tails | ✅ Mixture of populations: same bulk, different mixture weights |
| ρ = 0.556, far above prediction | ✅ If features encode *which campaign*, and campaigns differ in baseline rate, predicting visits is easy — no leakage needed |
| Adjustment moves the estimate 7.1 SEs | ✅ Adjustment is removing campaign-mix confounding, not just noise |

**The cheap test:** cluster on `f0`–`f11` (start with the near-discrete features — with 11 of 12 sharing identical medians, several are probably low-cardinality), then compute the treatment ratio and SMDs **within** each cluster. If within-cluster balance is clean while ratios differ across clusters, confirmed. The right estimator then becomes a **stratified ATE** — which is roughly what Lin adjustment already approximates, and why the adjusted number is the defensible one.

If it holds up, that's your blog post: *"the most-used public uplift dataset is a pool of experiments, and analyzing it as one experiment inflates the measured effect by 25%."* That's a finding, not a tutorial.

### Two open items

**`conversion` ATE is missing.** You've measured `visit`. **Revenue lives in `conversion`**, and it's the first thing a CFO asks for. Run it even at the 0.29% base rate, with the precision caveat attached.

**The `io.py` schema bug.** dbt materializes to `main_marts`; `read_arm_summary()` and `read_balance()` query unqualified names. Your Dagster asset checks call those helpers, which means **the SRM and balance gates would currently fail open** — the worst failure mode for a blocking check. Fix before the causal and policy reports run. Good catch that it hasn't produced a wrong number yet.

---

## 20. Glossary

| Term | Plain English |
|---|---|
| **ATE** | Average Treatment Effect — the average effect across everyone |
| **CATE / uplift / HTE** | The effect for a *particular kind* of person. Four names, one thing |
| **Counterfactual** | What would have happened otherwise. Never observable for an individual |
| **Control group / holdout** | Users deliberately not treated, so you can estimate the counterfactual |
| **Confounder** | Something affecting both who gets treated and the outcome |
| **Randomization** | Assigning treatment by chance — balances known *and unknown* variables |
| **SRM** | Sample Ratio Mismatch — arm sizes don't match the design. Usually a bug |
| **SMD** | Standardized mean difference — balance measured in standard deviations |
| **SE** | Standard error — how much your estimate would wobble on a re-run |
| **CI** | Confidence interval — the range consistent with your data |
| **p-value** | Probability of a result this extreme *if there were no effect* |
| **Power** | Chance of detecting a real effect. Convention: 80% |
| **MDE** | Minimum Detectable Effect — the smallest effect your design can find |
| **CUPED / CUPAC** | Using pre-treatment data to remove predictable noise. Reduction = ρ² |
| **ITT** | Intention To Treat — compare by *assignment*. The launch decision number |
| **CACE / LATE** | Effect on those who actually received treatment. ITT ÷ compliance |
| **Compliance** | Fraction of assigned who actually got treated |
| **Exclusion restriction** | Assignment affects the outcome *only* via actual treatment. Untestable |
| **Post-treatment variable** | Anything happening after assignment. Never condition on it |
| **Persuadable** | Responds only if treated. The whole point |
| **Sure Thing** | Responds either way. Sending is wasted margin |
| **Sleeping Dog** | Treatment makes them *worse*. Sending causes harm |
| **Qini / AUUC** | How much incremental outcome you capture by targeting top-ranked users |
| **Propensity score** | Modeled probability of being treated, given characteristics |
| **IPW** | Reweight units to make groups comparable |
| **Doubly robust / AIPW** | Combines outcome + propensity models. Right if *either* is right |
| **Overlap / positivity** | Every kind of person must appear in both arms |
| **E-value** | How strong a hidden confounder would need to be to overturn your result |
| **Off-policy evaluation** | Estimating a policy's value using data collected under a different one |

---

## 21. Self-test

If you can answer these without looking, you can defend this project.

**Foundations**
1. Why can't you tell whether the coupon caused Maya's visit?
2. What does randomization do that matching on age/gender/spend cannot?
3. Brew & Bean saw 4,250 visits in the treated group. Why isn't that the campaign's impact, and what is?

**A/B testing**
4. Your CI is [0.65pp, 1.35pp]. Say what that means without saying "95% chance."
5. Control is 15% of users but 82% of the variance. Why?
6. Result: +3% lift, p=0.4, MDE was 12%. What did you learn?
7. Why does peeking daily turn a 5% error rate into 18%?
8. Why does variance reduction equal ρ², and why must the covariate be pre-treatment?

**The trap**
9. Why is comparing people who *saw* the ad to the control group wrong?
10. Compliance is 3.6% and CACE is 28.7pp. Why report ITT instead?

**Beyond A/B**
11. Name the four customer types. Which two should never be treated, and why is one of them worse?
12. Why is AUC the wrong metric for an uplift model?
13. In the §14 table the campaign's overall effect is zero. Why is the model still valuable?
14. Why is "beats random targeting at the same budget" the comparison that matters?

**Causal inference**
15. What untestable assumption does every observational method require?
16. Explain the answer-key trick and why the 81% vs 6% result matters.
17. Why does a negative control work as a confounding diagnostic on real data?

**Your project**
18. Why quote 20.2% rather than 27.1%?
19. What single hypothesis explains all four of your "contradictions"?
20. Why is the `io.py` bug worse than an ordinary bug?

---

## One last thing: how to talk about this

Don't say *"I built an uplift model on the Criteo dataset."* Every candidate says that.

Say:

> "I had a randomized experiment, so I used it as an answer key to test which causal methods actually recover the truth when you take the randomization away — then built the targeting system on top of what survived.
>
> Along the way I found the dataset is a pool of separate experiments rather than one, which inflates the naive effect estimate by about 25%. And that only 3.6% of targeted users were ever actually reached — so delivery, not targeting, is where the leverage is."

The models are the easy part and everyone has them. **The validation is what almost nobody does, and it's what makes the rest credible.**
