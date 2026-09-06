# Pre-registration — preprocessing ablation

**Written 2026-09-06, before the ablation is run.**

This fixes the design, the decision rule and the tolerance *before* any ablation
result is seen. It is written because this project has repeatedly reached
conclusions that did not survive scrutiny, and in every case the conclusion was
formed after looking at the numbers.

**Honest status:** this is *prospective registration of a new analysis, informed
by exploratory results.* The paired comparison (`paired_rerun.py`) is already
known: corn +0.089, soybeans +0.079, spring wheat +0.033, oats −0.000. That
motivates the ablation. Nothing in the ablation itself has been run.

---

## 1. Question

The current dataset predicts the same county-years better than the poster-era
dataset. Many things differ between them. **Which change is responsible?**

Candidate changes, to be ablated one at a time:

| # | change | poster-era | current |
|---|---|---|---|
| 1 | vegetation crop mask | none (whole county) | stable cropland mask |
| 2 | soil product & depth | SoilGrids 0–5 cm | OpenLandMap 0–30 cm, thickness-weighted |
| 3 | season definition | fixed Apr–Sep, all states | state/crop-specific window |
| 4 | reduction scale | `GRIDMET_SCALE = 4000` | native 4638.31 m |

---

## 2. What is held fixed

Everything except the input being ablated:

- **Rows.** The 11,786 shared county-years from `paired_rerun.py`, with the same
  train/test split per crop. No re-derivation of eligibility.
- **Outcome.** `yield_value` from the current table (verified identical to the
  poster-era table on these rows).
- **Features.** The poster's 14, in the same order.
- **Model.** `RandomForestRegressor`, `n_estimators=200`, `max_depth=None`,
  `min_samples_leaf=5`, `max_features=0.5`.
- **Seeds.** Paired: the same seed list `(25, 0, 1, 2, 3)` for both arms of every
  comparison, each fit compared against its own seed-matched counterpart.

---

## 3. Estimand and primary metric

The estimand is **the performance of a single randomly seeded forest**, not an
averaged ensemble. Accordingly, RMSE is the square root of the mean squared
error taken over observations *and* seeds.

Primary metric is the **RMSE ratio**

```
Q = RMSE(ablated) / RMSE(intact)
```

**Not ΔR².** The same ΔR² means different things at different baselines — a
0.02 R² gain is a 1.7% RMSE reduction at R²=0.39 but 4.4% at R²=0.77 — so a
single ΔR² threshold would impose inconsistent practical tolerances across crops.
Q has the same meaning everywhere: a proportional change in prediction error.

---

## 4. Equivalence margin, declared in advance

**Primary: Q within [0.95, 1.05]** — prediction error changes by less than 5%.
At an intact RMSE of 20 BU/AC that tolerates 1 BU/AC.

**This is a declared methodological tolerance, not an agronomic standard.**
Without a specified decision and its loss function — insurance pricing, a
production forecast — no uniquely correct margin can be derived. 5% is chosen for
transparency and cross-crop comparability, and the paper must say so rather than
implying an established threshold.

Secondary tolerances of 2% and 10% will be reported. **The primary conclusion
will not be switched to whichever tolerance is convenient.**

---

## 5. Decision rule

Using a paired 95% confidence interval on Q (paired by seed and by observation,
with uncertainty accounting for county-level dependence):

| interval | conclusion |
|---|---|
| entirely inside [0.95, 1.05] | equivalence supported within the declared tolerance |
| entirely above 1.05 | ablation materially worsens prediction — the change matters |
| entirely below 0.95 | ablation materially improves prediction — the change hurts |
| anything else | **inconclusive** |

An inconclusive result will be reported as inconclusive. Failing to detect a
difference is not evidence of equivalence (Lakens, equivalence testing).

---

## 6. Reported alongside the pooled result, always

`paired_rerun.py` established that pooled scores hide year-level structure. Oats
moved −0.0001 overall while 2023 deteriorated by 1,960 SSE and 2024 improved by
1,945 on just **eight observations** — and the sign of the pooled oats delta
flips across seeds. Corn and soybeans, by contrast, improved in *every* test year.

So every ablation result reports:

1. pooled Q with its interval;
2. **per-year** loss, so cancellation is visible;
3. per-state loss, since the irrigation work already showed Kansas and Nebraska
   behaving differently;
4. the seed spread, and whether the sign is stable across seeds.

A pooled result whose sign flips across seeds will not be described as an effect.

---

## 7. Scope of any resulting claim

The claim concerns **these fixed test years**, not generalisation to future
years. Three test years support the latter weakly at best. Wording that implies
out-of-sample generality is out of scope for this design.

Interactions are expected. If the joint ablation of all four changes differs
materially from the sum of individual ablations, that is **cancellation, not
insensitivity**, and will be reported as such.

---

## 8. What would make this uninteresting

Stated in advance so it cannot be rationalised later:

- If no single change produces an interval outside [0.95, 1.05], there is no
  sharp finding about preprocessing, and the poster-line paper should not be
  written. The irrigation work becomes the paper.
- If the effect is entirely crop-dependent with no common mechanism, report it as
  a crop-dependent improvement and nothing more.

**Do not force the ablation to rescue an earlier thesis.** Two claims from this
project have already been retracted after being shaped to fit a narrative
(`ASSESSOR_BRIEF.md` §6).
