# CropCast

County-level crop yield analysis for the US Corn Belt and Northern Plains,
combining USDA NASS yields, gridMET climate, MODIS vegetation indices and
OpenLandMap soil properties.

**Arit Prince** · [AGU 2025 poster GC13F-0713](poster/) · New Orleans

---

## What this repository now contains

The work presented at AGU 2025 has been substantially reworked. Several errors
were found in the original pipeline, and the headline poster figures do not
survive their correction. The poster-era code, data and results are preserved
under [`archive/`](archive/) and tagged `agu2025-poster`; they should not be
used as current.

**Current code is in [`src/`](src/).**

| Document | What it holds |
|---|---|
| [`src/HANDOFF.md`](src/HANDOFF.md) | Findings, methods, and a log of retracted claims |
| [`src/ASSESSOR_BRIEF.md`](src/ASSESSOR_BRIEF.md) | Open design questions for independent review |
| [`archive/README.md`](archive/README.md) | What is wrong with the poster-era material |

---

## Findings

All intervals are 95% confidence intervals from a county-clustered bootstrap.
2008–2025, 11 states.

### 1. Once the technology trend is removed, the models lose to a county average

Yields rise over time for reasons unrelated to weather. Detrended, and scored
against simply predicting each county's historical mean yield:

| crop | n | full model (climate + soil + satellite) | county mean + linear trend |
|---|---|---|---|
| Corn | 13,809 | **[−0.233, −0.046]** | [0.006, 0.010] |
| Soybeans | 13,233 | **[−0.200, −0.063]** | [0.005, 0.006] |
| Oats | 3,315 | [−0.046, 0.114] | [0.008, 0.013] |
| Sorghum | 1,389 | [−0.040, 0.124] | [−0.077, −0.063] |
| Spring wheat | 1,706 | **[0.081, 0.257]** | [0.005, 0.015] |

All five crops in the study are shown. For the two highest-volume crops the full
model is *significantly worse* than a county average, while a county average plus
a straight line is significantly better. Oats and sorghum straddle zero — not
distinguishable from a county average either way. Spring wheat is the only crop
with skill significantly above zero, and it is also the crop with a missing test
year (see Limitations).

Sorghum is the weakest case in the panel: it is the smallest sample, the only
crop where `county_mean_trend` is itself negative, and the only one where soil
adds nothing over climate. Its inclusion does not change any conclusion above.

Conventional R² on a county-year panel looks far more impressive than this,
because 40–61% of yield variance is persistent differences *between* counties
that county identity alone explains (corn 57%, soybeans 61%, spring wheat 54%,
oats 52%, sorghum 40%).

### 2. Irrigation decouples yield from weather

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

### 3. Soil is not merely a county fingerprint

Soil takes one value per county, so a model given soil could in principle
recover county identity without learning any agronomy. Tested against a placebo
— the same climate model plus four random numbers held constant per county —
county identity explains only **7–13%** of soil's contribution (21% for spring
wheat). Sorghum is excluded from this comparison: soil does not improve on
climate alone for it, so the ratio is undefined.

---

## Errors found and corrected

Documented in full in [`src/HANDOFF.md`](src/HANDOFF.md). The most
consequential:

- **Inverted crop mask.** The CDL `cultivated` band is `1 = Non-cultivated`,
  `2 = Cultivated`. The original code used `.eq(1)`, so for 2013–2023 every
  vegetation index was averaged over land that was *not* farmland. Story County,
  Iowa reads 0.772 cropland under the corrected mask and 0.216 under the old one
  — the exact complement.
- **Season-total precipitation.** Growing-season windows differ by state
  (Minnesota 153 days vs 183), so totals were not comparable across states and
  the model could learn "low rainfall = Minnesota". Now per-day rates.
- **Soil masking that did nothing.** The "cropland union" mask covered
  essentially the whole county; masked and unmasked values agreed to r = 0.9999.
- **A metric that did not measure what it claimed.** An R² computed on residuals
  was described as skill against a county-mean baseline. A predictor tying that
  baseline exactly scored −33.8.
- **A placebo that tested the wrong contrast**, which reversed a headline
  conclusion once corrected.

**For the two largest crops, these corrections barely move the poster's headline
number.** Re-running the poster's exact configuration — its 14 features, its
2010 start, its split rule, its hyperparameters and seed — on fully corrected
data ([`src/matched_rerun.py`](src/matched_rerun.py), so that only the *data*
differs):

| crop | n | poster | corrected data | Δ |
|---|---|---|---|---|
| Corn | 11,570 | 0.764 | **0.753** | −0.011 |
| Soybeans | 11,091 | 0.692 | **0.769** | +0.077 |
| Spring wheat | 1,400 | 0.694 | 0.589 | −0.105 |
| Oats | 2,716 | 0.448 | 0.389 | −0.059 |
| Sorghum | 1,134 | 0.480 | 0.228 | −0.252 |

Corn moves by 0.011 after correcting a crop mask that had averaged vegetation
indices over *non*-farmland for 11 of 18 years. The poster reported its own
results accurately (both its figures reconcile exactly to
[`archive/results_agu/`](archive/results_agu/)).

A metric that barely notices defects of that size, on the crops with the most
data, is not measuring what it appears to measure — see Findings below. What
overturns the poster's conclusion is not the data corrections but the change of
evaluation protocol.

The small-sample crops behave differently: sorghum (n = 1,134) moves by 0.252.
The insensitivity claim is therefore made **for the large-sample crops only**,
and the divergence at small n is reported rather than averaged away.

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
