# CropCast

County-level crop yield analysis for the US Corn Belt and Northern Plains,
combining USDA NASS yields, gridMET climate, MODIS vegetation indices and
OpenLandMap soil properties.

**Arit Prince** · [AGU 2025 poster GC13F-0713](poster/) · New Orleans

---

## What this repository now contains

The work presented at AGU 2025 has been substantially reworked: different crop
masking, soil product and depth, season definitions, spatial scales, and a
different evaluation protocol. The poster-era code, data and results are
preserved under [`archive/`](archive/) and tagged `agu2025-poster`.

The poster reported its own results accurately, and its data pipeline was not
corrupted — see [Errors](#errors-found-and-corrected) for a correction to an
earlier version of this README that said otherwise. What the current work
revisits is chiefly how the models are *scored*.

**Current code is in [`src/`](src/).**

| [`src/ABLATION_PREREGISTRATION.md`](src/ABLATION_PREREGISTRATION.md) | Design and decision rule for the preprocessing ablation, fixed in advance |
| [`archive/README.md`](archive/README.md) | What is wrong with the poster-era material |

---

## Findings

All intervals are 95% confidence intervals from a county-clustered bootstrap.
2008–2025, 11 states.

### 1. Under county detrending, corn and soybean models underperform a county-mean reference

Yields rise over time. Detrended, and scored against predicting each county's
historical mean yield:

| crop | n | full model (climate + soil + satellite) | county mean + linear trend |
|---|---|---|---|
| Corn | 13,809 | **[−0.233, −0.046]** | [0.006, 0.010] |
| Soybeans | 13,233 | **[−0.200, −0.063]** | [0.005, 0.006] |
| Oats | 3,315 | [−0.046, 0.114] | [0.008, 0.013] |
| Sorghum | 1,389 | [−0.040, 0.124] | [−0.077, −0.063] |
| Spring wheat | 1,706 | **[0.081, 0.257]** | [0.005, 0.015] |

All five crops are shown. Corn and soybeans are *significantly worse* than a
county mean, while county mean plus a straight line is significantly better.
Oats and sorghum are **inconclusive** — positive point estimates with intervals
spanning zero, which is insufficient evidence of improvement rather than evidence
of failure. Spring wheat has positive skill, and is also the crop with a missing
test year (see Limitations).

Two cautions on reading this. The removed linear component is **not necessarily a
technology trend** — it can absorb warming, irrigation expansion, cultivar
change, or anything else gradual. And the detrended comparison refits the model
and changes the reference prediction, so it is not simply exposing the raw
model's "true" skill: undetrended, corn and soybean models do beat both a shared
and a county-specific trend baseline (corn 0.772 vs 0.683 and 0.676).

Sorghum is the weakest case in the panel: it is the smallest sample, the only
crop where `county_mean_trend` is itself negative, and the only one where soil
adds nothing over climate. Its inclusion does not change any conclusion above.

Conventional R² on a county-year panel looks far more impressive than this,
because 40–61% of yield variance is persistent differences *between* counties
that county identity alone explains (corn 57%, soybeans 61%, spring wheat 54%,
oats 52%, sorghum 40%).

### 2. Weather associations are weaker under irrigation

Where NASS reports both irrigated and non-irrigated corn for the **same county
in the same year**, county, soil and season are held constant and only
management differs. 867 such pairs, 114 counties, 2008–2018.

Weather explains **68.5%** of the rainfed yield anomaly and **16.6%** of the
irrigated one. Irrigated corn retains roughly a tenth of rainfed precipitation
sensitivity:

| weather | rainfed slope | irrigated slope | ratio (unit-free correlation) |
|---|---|---|---|
| Precipitation | +32.95 | +3.31 | 0.19 |
| Extreme heat (EDD) | −21.61 | −5.74 | 0.51 |
| Vapour pressure deficit | −73.44 | −17.02 | 0.45 |

Mean irrigated-minus-rainfed gap: **+81.4 BU/AC**. This is an *association*
within counties, not a randomised treatment effect.

**The direction is robust; the magnitude is not.** These are in-sample fits, and
the variance-explained gap depends heavily on one year: dropping 2012 moves the
detrended rainfed R² from 0.699 to 0.300 while the irrigated figure barely moves
(0.253 → 0.256). The precipitation contrast also differs sharply by state —
rainfed/irrigated correlation is 0.771/0.373 in Kansas but 0.711/−0.051 in
Nebraska. Read this as *weaker weather association under irrigation in this
selected sample*, not as a general sensitivity ratio or a causal variance
fraction. Counties enter only if NASS published both practices, and the direction
of that selection bias cannot be determined from the selection rule alone.

### 3. Soil features outperform a matched random-feature control

Soil takes one value per county, so a model given soil could in principle
recover county identity without learning any agronomy. Against a control — the
same climate model plus four random numbers held constant per county — soil's
gain is 7–13× the control's (21% for spring wheat; undefined for sorghum, where
soil does not beat climate alone).

**This does not identify how much of soil's contribution is county identity.**
Four random county-constant numbers are one particular encoding of identity, and
a random forest need not exploit arbitrary coordinates as efficiently as
geographically structured soil variables. The surviving claim is only that soil
beats this control. A stronger test would permute whole soil vectors between
counties, preserving their covariance, and compare against coordinates and
climate normals.

---

## Errors found and corrected

Documented in full in [`src/HANDOFF.md`](src/HANDOFF.md).

### Correction, 2026-09-06 — read this first

**An earlier version of this section attributed defects to the AGU poster that
were not in the poster's code.** It claimed an inverted CDL crop mask, precipitation
totals confounded with state, and an inert soil mask. Checking
[`archive/src_agu/`](archive/src_agu/) directly:

| claimed defect | actually in the poster? |
|---|---|
| Inverted CDL mask, 2013–2023 | **No.** `download_vegetation.py` has no crop mask at all; `cultivated.eq(1)` appears only in a later WIP file that never produced these results |
| Season totals confounded with state | **No.** A fixed April–September window is used for every state, so totals are comparable |
| Inert soil "cropland union" mask | **No.** SoilGrids API at 0–5 cm; there is no Earth Engine masking |
| Non-native reduction scale | **Yes.** `GRIDMET_SCALE = 4000` against a 4638.31 m native grid |

Those three belong to the project's later development code. Attributing them to
the poster was wrong, and it was repeated in the README, the archive README and a
commit message before being caught in independent review.

### Genuine issues in the poster analysis

Mostly design and evaluation choices rather than corrupted data:

- **No crop mask on vegetation** — NDVI/EVI averaged over whole counties,
  including non-agricultural land.
- **Soil at 0–5 cm**, a surface value rather than a rooting-zone depth.
- **A fixed Apr–Sep season for all states and crops**, which fits neither spring
  wheat in North Dakota nor the southern end of the sample.
- **Non-native reduction scale**, forcing resampling.
- **Test R² on a county-year panel with no county-mean baseline and no
  detrending.** This is the substantive issue, and what the current work addresses.
- **"+74%" is a mean of per-crop ratios**, dominated by crops with weak
  baselines: oats +137% off R² 0.189, spring wheat +162% off 0.265, while corn —
  the largest crop — improved 28%.

### Errors in the reworked pipeline, found and fixed

- **A metric that did not measure what it claimed.** An R² computed on residuals
  was described as skill against a county-mean baseline. A predictor tying that
  baseline exactly scored −33.8.
- **A placebo that tested the wrong contrast**, which reversed a headline
  conclusion once corrected.
- **An inverted CDL mask and a column-order bug** in development code, caught
  before any result depended on them.

### Does the newer data predict better? Yes — measured on identical rows

An earlier version of this README claimed the corrections "barely move" the
headline number, based on [`src/matched_rerun.py`](src/matched_rerun.py). That
experiment matched the poster's *configuration* but not its *rows* — corn trained
on 9,374 observations against the poster's 5,448 — so it compared R² across
different test sets. **That claim is retracted.**

The paired version intersects the two datasets on (crop, county_fips, year),
giving 11,786 shared county-years with identical yield values, and trains the
poster's own model on the same rows using each dataset's features (5 seeds):

| crop | train / test | poster-era data | current data | Δ |
|---|---|---|---|---|
| Corn | 4,020 / 900 | 0.543 | 0.632 | **+0.089** |
| Soybeans | 3,829 / 865 | 0.640 | 0.719 | **+0.079** |
| Spring wheat | 413 / 110 | 0.175 | 0.208 | +0.033 |
| Oats | 1,199 / 156 | 0.387 | 0.387 | −0.000 |

Seed spread is ≤0.011, so corn and soybeans sit well outside noise — about a 20%
reduction in squared error, and they improve in *every* test year rather than
one. **Which** change is responsible (masking, soil depth and product, season
definition, native scales) has not been isolated; that needs an ablation with
rows held fixed, pre-registered in
[`src/ABLATION_PREREGISTRATION.md`](src/ABLATION_PREREGISTRATION.md).

**Oats shows why pooled scores need a per-year breakdown.** Its −0.0001 is not
stability: 2023 deteriorates by 1,960 SSE while 2024 improves by 1,945 on just
*eight* observations, and the sign of the pooled delta flips across seeds. Full
breakdown in `results_matched/paired_rerun_by_year.csv`.

Similar headline scores across *unmatched* samples therefore concealed a material
difference in predictive performance. That, rather than any insensitivity of the
metric, is what the comparison actually shows.

On positioning: trend-aware benchmarking in agricultural ML is **not** new —
[Paudel et al. (2022)](https://doi.org/10.1016/j.fcr.2021.108377) compares
regional ML forecasts against a linear-trend model across 35 crop-country cases,
and [CY-Bench](https://doi.org/10.5194/essd-18-3997-2026) provides reproducible
sub-national benchmarking infrastructure. Neither settles whether a controlled
preprocessing ablation is novel, but any claim that trend baselines are
"routinely absent" would be wrong.

---

## Reproducing

Requires Python 3.14, [uv](https://docs.astral.sh/uv/), a Google Earth Engine
account, and a [USDA NASS QuickStats API key](https://quickstats.nass.usda.gov/api).

```bash
uv sync
cp src/.env.example src/.env    # add your own keys
cd src                          # paths are relative to this directory

python download_yield.py        # ~5 min
python download_soil.py         # ~5 min
python download_vegetation.py   # ~20 min
python download_climate.py      # ~40 min

python ml.py                    # ~25 min
python evaluate.py --detrend none county
python irrigation_contrast.py   # ~1 min
```

Run the downloads one at a time — Earth Engine rate-limits concurrent requests,
and `ml.py` / `evaluate.py` write to fixed paths and will race each other.

Raw and processed data are not committed (162 MB); the download scripts
regenerate them.

---

## Data sources

| | Source | Resolution |
|---|---|---|
| Yields | [USDA NASS QuickStats](https://quickstats.nass.usda.gov/) | county, annual |
| Climate | [gridMET](https://www.climatologylab.org/gridmet.html) via Earth Engine | 4638 m, daily |
| Vegetation | [MODIS MOD13A3](https://lpdaac.usgs.gov/products/mod13a3v061/) NDVI/EVI | 927 m, monthly |
| Soil | [OpenLandMap](https://openlandmap.org/) | 232 m, static |
| Crop mask | [USDA CDL](https://nassgeodata.gmu.edu/CropScape/) | 30 m, annual |

---

## Limitations

- Vegetation indices are measured *during* the season being predicted. They are
  strong predictors but cannot support a forecast made before harvest, and they
  are outcomes of crop growth rather than causes of yield.
- Crop masks are generic cropland, not crop-specific — corn NDVI is averaged
  with soybean pixels.
- The soil mask uses land cover from across the full study period, including
  held-out years. A retrospective-design choice.
- The irrigation analysis covers 2008–2018 only; NASS stopped publishing the
  county-level irrigation split after 2018.
- Spring wheat has no NASS county estimates for 2024, leaving a gap in its test
  window — and it is the only crop showing positive detrended skill.
- `GDD_TMAX` and `EDD_TMAX` are maximum-temperature-based per-day indices, not
  conventional growing degree days and not Schlenker & Roberts degree-days.

---

## Citation

```
Prince, A. (2025). CropCast: county-level crop yield analysis with climate,
vegetation and soil data. AGU Fall Meeting 2025, GC13F-0713.
https://github.com/ScienceAndBeyond/CropCast
```
