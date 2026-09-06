# CropCast

County-level crop yield analysis for the US Corn Belt and Northern Plains,
combining USDA NASS yields, gridMET climate, MODIS vegetation indices and
OpenLandMap soil properties.

**Arit Prince & Arya Prince** · [AGU 2025 poster GC13F-0713](archive/poster/) · New Orleans

---

## About this repository

This is a continuation of the work presented at AGU 2025. The original project —
its code, data, results and README — is preserved unchanged under
[`archive/`](archive/) and tagged `agu2025-poster`.

The current work rebuilds the data pipeline and changes how models are
evaluated. Both changes matter, and the second changes the conclusions.

**Current code is in [`src/`](src/).**

---

## What changed since the original

### Data pipeline

| | original | current |
|---|---|---|
| Vegetation | MODIS averaged over whole county polygons | restricted to stable cropland |
| Soil | SoilGrids, 0–5 cm surface value | OpenLandMap, thickness-weighted 0–30 cm |
| Growing season | fixed April–September for all states and crops | state- and crop-specific windows |
| Spatial reduction | 4000 m | each asset's native scale (4638 m / 927 m / 232 m) |
| Coverage checks | none | per-variable completeness enforced |
| Years | 2010–2024 | 2008–2025 |

Precipitation and reference ET are now per-day rates rather than season totals,
which matters once season length varies by state.

### The new pipeline predicts better

Measured on identical county-years, so the comparison is not affected by the two
datasets covering different samples. Intersecting the two processed tables gives
11,786 shared county-years with identical yields; the same model is trained on
the same rows using each dataset's features in turn
([`src/paired_rerun.py`](src/paired_rerun.py)):

| crop | train / test | original data | current data | Δ R² |
|---|---|---|---|---|
| Corn | 4,020 / 900 | 0.543 | 0.632 | **+0.089** |
| Soybeans | 3,829 / 865 | 0.640 | 0.719 | **+0.079** |
| Spring wheat | 413 / 110 | 0.175 | 0.208 | +0.033 |
| Oats | 1,199 / 156 | 0.387 | 0.387 | −0.000 |

Seed spread is ≤0.011. Corn and soybeans improve in every test year, not just on
average. Which single change is responsible has **not** been isolated — that
needs an ablation rebuilding one input at a time.

Oats is a caution against reading pooled scores alone: its −0.000 is not
stability but cancellation, with 2023 worsening by 1,960 SSE and 2024 improving
by 1,945 on eight observations. Per-year detail is in
`results_comparison/paired_rerun_by_year.csv`.

### Evaluation

The original reported test R² on a county-year panel. On such a panel, 40–61% of
yield variance is persistent differences *between* counties, so a model can score
well by learning which county it is looking at. The current work adds a
county-mean baseline, a county-mean-plus-trend baseline, county-clustered
bootstrap intervals, and results with and without detrending.

---

## Findings

95% confidence intervals from a county-clustered bootstrap, 2008–2025, 11 states.

### 1. Under county detrending, corn and soybean models underperform a county mean

Scored against predicting each county's historical mean yield:

| crop | n | full model | county mean + linear trend |
|---|---|---|---|
| Corn | 13,809 | **[−0.233, −0.046]** | [0.006, 0.010] |
| Soybeans | 13,233 | **[−0.200, −0.063]** | [0.005, 0.006] |
| Oats | 3,315 | [−0.046, 0.114] | [0.008, 0.013] |
| Sorghum | 1,389 | [−0.040, 0.124] | [−0.077, −0.063] |
| Spring wheat | 1,706 | **[0.081, 0.257]** | [0.005, 0.015] |

Oats and sorghum are **inconclusive** — positive point estimates with intervals
spanning zero, which is insufficient evidence rather than evidence of failure.

Two cautions. The removed linear component is not necessarily a technology
trend; it can absorb warming, irrigation expansion or cultivar change.
And undetrended, corn and soybean models do beat both shared and county-specific
trend baselines (corn 0.772 vs 0.683 and 0.676), so this is not a claim that the
models are uninformative.

### 2. Weather associations are weaker under irrigation

Where NASS reports both irrigated and non-irrigated corn for the **same county in
the same year**, county, soil and season are held constant and only management
differs. 867 pairs, 114 counties, 2008–2018.

Weather explains **68.5%** of the rainfed yield anomaly and **16.6%** of the
irrigated one.

| weather | rainfed slope | irrigated slope | ratio |
|---|---|---|---|
| Precipitation | +32.95 | +3.31 | 0.19 |
| Extreme heat (EDD) | −21.61 | −5.74 | 0.51 |
| Vapour pressure deficit | −73.44 | −17.02 | 0.45 |

Mean irrigated-minus-rainfed gap: **+81.4 BU/AC**.

The direction is robust; the magnitude is not. These are in-sample fits, and the
gap depends heavily on one year — dropping 2012 moves the detrended rainfed R²
from 0.699 to 0.300 while the irrigated figure barely moves. The precipitation
contrast also differs sharply between Kansas and Nebraska. Read this as a weaker
weather association under irrigation in this selected sample, not as a general
sensitivity ratio or a causal effect.

### 3. Soil features outperform a matched random-feature control

Soil takes one value per county, so a model given soil could in principle recover
county identity without learning agronomy. Against a control of the same climate
model plus four random county-constant numbers, soil's gain is 7–13× the
control's (21% for spring wheat; undefined for sorghum, where soil does not beat
climate alone).

This shows soil beats that particular control. It does **not** identify what
share of soil's contribution is county identity — four random numbers are only
one encoding of identity, and a random forest need not exploit them as
efficiently as geographically structured variables.

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
python paired_rerun.py          # ~3 min
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
  strong predictors but are outcomes of crop growth rather than causes of yield,
  and using them in a forecast requires checking product latency against the
  intended issuance date.
- Crop masks are generic cropland, not crop-specific — corn NDVI is averaged with
  soybean pixels.
- The soil mask uses land cover from across the full study period, including
  held-out years. A retrospective-design choice.
- The irrigation analysis covers 2008–2018 only; NASS stopped publishing the
  county-level irrigation split after 2018. Counties appear only where both
  practices were reported, and the direction of that selection is not known.
- Spring wheat has no NASS county estimates for 2024, leaving a gap in its test
  window — and it is the only crop showing positive detrended skill.
- `GDD_TMAX` and `EDD_TMAX` are maximum-temperature-based per-day indices, not
  conventional growing degree days and not Schlenker & Roberts degree-days.
- Three test years support claims about these years, not about generalisation to
  future ones.

---

## Related work

Trend-aware benchmarking is established in this field: Paudel et al. (2022),
[*Machine learning for regional crop yield forecasting in Europe*](https://doi.org/10.1016/j.fcr.2021.108377),
compares regional ML forecasts against a linear-trend model. Kallenberg et al.
(2026), [*CY-Bench*](https://doi.org/10.5194/essd-18-3997-2026), provides
reproducible sub-national crop-yield benchmarking infrastructure.

---

## Citation

```
Prince, A. & Prince, A. (2025). CropCast: county-level crop yield analysis with
climate, vegetation and soil data. AGU Fall Meeting 2025, GC13F-0713.
https://github.com/ScienceAndBeyond/CropCast
```
