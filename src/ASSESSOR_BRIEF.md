# Reviewer brief — CropCast

**Date:** 2026-09-06
**Repo:** `ScienceAndBeyond/CropCast`, branch `main`
**Role:** adversarial scientific audit. Not a code review — the code has been
through seven audit rounds. What has *not* been independently checked is whether
the claims are **warranted by the evidence**, and whether the headline claim is
confounded.

Your job is to try to break the claims in §3. Where you cannot break one, say
so. Where a claim is overstated but salvageable, say what it should narrow to.
Assume every number below might be wrong and check it against the committed CSVs.

---

## 1. What the project is

County-level US crop yield modelling (corn, soybeans, oats, sorghum, spring
wheat), 2008–2025, 11 Corn Belt / Northern Plains states, from four sources:
USDA NASS yields, gridMET climate, MODIS NDVI/EVI, OpenLandMap soil.

It began as an AGU 2025 poster (GC13F-0713) reporting R² = 0.76 and "+74%
improvement over climate-only". Defects were later found in the pipeline that
produced those numbers. The poster material is preserved under `archive/`.

The paper being written is an **evaluation-protocol paper**, not a better-model
paper. The AGU analysis is retained deliberately, as the motivating example.

---

## 2. Reproduce it

```
cd <repo>/src                                       # paths are relative to here
python -X utf8 ml.py                                # ~25 min -> ../results
python -X utf8 evaluate.py --detrend none county    # ~25 min -> ../results
python -X utf8 irrigation_contrast.py               # ~1 min  -> ../results_split
```

**Do not run two of these at once** — they write to fixed paths and will race.
Committed outputs in `results/` and `results_split/` are from a verified single
run; you should reproduce them. `data/` and `data_raw/` are gitignored (162 MB);
rebuilding needs about an hour of Earth Engine time and GEE credentials, and is
not required to audit anything below.

Poster-era outputs are in `archive/results_agu/`, in the same format.

---

## 3. The claims to audit

Each says where its numbers live, so you can check them directly.

### Claim A — the headline metric is insensitive to serious data defects

*Source: `archive/results_agu/model_performance.csv` vs `results/model_performance.csv`.*

Correcting an inverted CDL crop mask (vegetation indices averaged over
*non*-farmland for 2013–2023), season-total rather than per-day precipitation,
and a soil mask covering essentially the whole county, changes corn's
`all_features` test R² from **0.764 to 0.772**. Mean "+% improvement over
climate-only" goes from 74.1% to 81.9%.

**This is the central claim of the paper and the one most likely to be wrong.**
See §4.1 — it is not a controlled comparison, and we know it.

### Claim B — the apparent skill decomposes into county identity, trend, and a same-season outcome

*Source: `results/model_performance.csv`, `results/bootstrap_ci.csv`,
`results/variance_decomposition.csv`.*

For corn: the county-mean baseline alone scores R² 0.412; 57.1% of total yield
variance is between-county; the feature ladder runs `climate_only` 0.355 to
`climate_soil` 0.573 to `climate_veg` 0.713 to `all_features` 0.772; and
`skill_vs_county_mean` for `all_features` goes from **+0.612 undetrended to
[−0.233, −0.046] county-detrended**.

### Claim C — detrended, most crops do not beat a county average

*Source: `results/bootstrap_ci.csv`, `detrend=county`.* 95% CIs on
`skill_vs_county_mean`, county-clustered bootstrap:

| crop | n | all_features | county_mean_trend |
|---|---|---|---|
| CORN | 13,809 | [−0.233, −0.046] | [0.006, 0.010] |
| SOYBEANS | 13,233 | [−0.200, −0.063] | [0.005, 0.006] |
| OATS | 3,315 | [−0.046, 0.114] | [0.008, 0.013] |
| SORGHUM | 1,389 | [−0.040, 0.124] | [−0.077, −0.063] |
| WHEAT_SPRING | 1,706 | [0.081, 0.257] | [0.005, 0.015] |

### Claim D — irrigation decouples yield from weather

*Source: `results_split/`.* Within the same county-year, where NASS reports both
irrigated and non-irrigated corn: weather explains **68.5%** of the rainfed yield
anomaly and **16.6%** of the irrigated one. 867 pairs, 114 counties, 2008–2018.
Mean gap +81.4 BU/AC. Framed as associational, not causal.

### Claim E — soil is not merely a county fingerprint

*Source: `results/placebo_test.csv`.* Against a placebo of climate plus four
random county-constant features, county identity explains 7–13% of soil's
contribution (21% for spring wheat; undefined for sorghum, where soil does not
beat climate).

---

## 4. Where to attack — known weak points

Do not stop at these, but do not skip them.

**4.1 Claim A is confounded, and adjudicating it is the most valuable thing you
can do.** The two runs differ in more than data quality:

| | poster | corrected |
|---|---|---|
| train / test years | 2010–2021 / 2022–2024 | 2008–2021 / 2022–2025 |
| n features (`all_features`) | 14 | 20 |
| crop panel | 6 (incl. upland cotton) | 5 |
| seeds averaged | 1 | 5 |

So "the corrections did not matter" is *suggested*, not demonstrated. Is the
claim salvageable as stated? What is the minimum experiment that would establish
it? We believe the decisive one is the mirror image — run the **current
evaluation protocol on the poster-era data** in `archive/data_agu/processed/` —
but we have not run it. Would that settle it, or is a matched re-run of the
corrected data on the poster's exact split also needed? Or is the comparison
unsalvageable, so the claim should be dropped?

**4.2 Is `skill_vs_county_mean` the right null?** Defined as
`1 − SSE(model)/SSE(county mean)`. Alternatives: climatology, county-mean-plus-
trend, last year's yield. `evaluate.py` computes a ladder including
`county_mean_trend`. Is the ladder complete? Is the headline comparison the
informative one?

**4.3 Is county-level linear detrending the right operation?** Results are
reported raw and county-detrended and the two tell opposite stories. Should the
trend be per-county, per-state, or global? A spline? Does removing a linear
trend also remove real weather signal that happens to be trended, such as
warming? **This step carries the paper's conclusion**, so it deserves the
harshest scrutiny.

**4.4 Vegetation is a same-season outcome.** NDVI/EVI is measured *during* the
season being predicted, so it cannot support a pre-harvest forecast and is an
effect of crop growth rather than a cause. It is excluded from the "causal"
feature set but included in `all_features` — the headline number — and supplies
most of the R² (0.355 to 0.713). Is reporting that as the headline defensible
because it is labelled, or misleading regardless of labelling?

**4.5 Irrigation selection.** Counties appear only if NASS published *both*
practices — plausibly counties with substantial irrigated *and* rainfed acreage.
Which way does that bias the contrast? Counties with fewer than 4 paired years
are dropped; is that defensible? The window is 2008–2018 (NASS stopped
publishing the split) against a 2008–2025 main analysis — does that undermine
the comparison?

**4.6 Sample construction.** `MIN_SAMPLES = 1000`, `MIN_YEARS = 10`,
`MIN_VEG_MONTHS` = the state's full season, `MIN_CLIMATE_COVERAGE = 1.0`. These
decide which crops appear; sorghum sits near the boundary. Principled or
convenient? Spring wheat has **no NASS county estimates for 2024**, a hole in its
2021–2025 test window — and it is the only crop with positive detrended skill.
How much does that undermine Claim C's one exception?

**4.7 Does the sample match the claim?** `STUDY_STATES` is described as "rainfed
Corn Belt + Northern Plains", but western Kansas is Ogallala-irrigated and
Nebraska was added deliberately to enable Claim D.

---

## 5. Documented limitations — do not spend time rediscovering

Do say if any is more serious than its current treatment implies.

- Crop masks are generic cropland, not crop-specific — corn NDVI is averaged
  with soybean pixels.
- The soil stability mask uses CDL across 2008–2025 including held-out years.
  A stated retrospective-design choice.
- Climate, vegetation and soil are averaged at three native resolutions
  (4638 m, 927 m, 232 m) over the same county polygons.
- `COVERAGE_*` detects missing months, not missing days within a month. Verified
  non-binding on six sampled month-years spanning 2008–2025.
- `TMAX_MAX` is "the hottest month's county-average peak", not a county absolute
  maximum.
- `GDD_TMAX` / `EDD_TMAX` are Tmax-based per-day indices, **not** conventional
  growing degree days and **not** Schlenker & Roberts degree-days.

---

## 6. Calibration — where this project has already been wrong

All were caught by external audit or late self-checking, never by the original
analysis. Weight the rest accordingly.

1. **"Soil is a county fingerprint; its advantage vanishes when detrended."**
   Wrong — the placebo arm was random features *alone*, compared against
   climate plus soil. Figures quoted at the time (38% / 70% / 134%) are invalid.
2. **"Anomaly R² ≤ 0 means no skill beyond county identity."** Wrong — that
   metric was `r2_score` on residuals, centred on the mean *test* anomaly rather
   than zero. A predictor tying the county-mean baseline exactly scored −33.8.
3. **"The poster's figures do not survive correction."** Wrong, and corrected
   only on 2026-09-06 — they do survive (0.764 to 0.772). The error conflated
   data corrections with the change of evaluation protocol. This is the origin
   of Claim A.
4. **A README reporting 3 of 5 crops**, with a "57–61%" between-county range
   that was the corn-and-soy range quoted as general (true range 40–61%).
5. **An inverted CDL crop mask** averaging non-farmland for 2013–2023, and a
   column-order bug that silently corrupted 258,066 of 305,046 climate rows.

Note the pattern: the recurring failure is a *metric or comparison that does not
measure what it is described as measuring*, and it has survived multiple review
rounds each time. Claim A is a comparison of exactly that kind.

---

## 7. What to report back

1. **Claim A**: established, salvageable-if-narrowed, or unsupported? If
   salvageable, state the exact experiment needed.
2. Any claim in §3 whose numbers you cannot reproduce from the committed CSVs.
3. A view on §4.2 and §4.3 — the metric and the detrending. These carry the
   paper.
4. Whether the framing *"conventional R² on county-year panels is dominated by
   between-county variance, a technology trend and a same-season outcome, and is
   insensitive to data quality"* is supported, overclaimed, or underclaimed.
5. Literature positioning: skill-relative-to-climatology is standard in
   operational forecasting. Is "it is routinely absent from ML-for-agriculture
   work, and here is a runnable protocol plus a demonstration of what it catches"
   a fair and novel contribution, or has this been done?
6. Anything in §5 more serious than its current treatment.

Please distinguish throughout between *"this is wrong"*, *"this is unsupported
as stated"*, and *"this is a judgement call I would have made differently"*.
