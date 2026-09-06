# CropCast

Predicting county-level crop yields in the US Corn Belt and Northern Plains from
climate, soil and satellite data.

**Arit Prince & Arya Prince** · [AGU 2025 poster GC13F-0713](archive/poster/) · New Orleans

---

## About

This project started as an AGU 2025 poster asking whether soil properties and
satellite vegetation indices improve crop yield prediction over climate alone.
Since then the data pipeline has been rebuilt and the models are scored
differently. The original project is kept unchanged in [`archive/`](archive/).

Crops covered: corn, soybeans, spring wheat, oats and sorghum, across 11 states
from 2008 to 2025.

---

## What the program does

Four download scripts pull the raw data, one merges it and trains models, and
three more analyse the results.

**Downloading.** `download_yield.py` pulls county yields from the USDA NASS
QuickStats API. The three Earth Engine scripts — `download_climate.py`,
`download_vegetation.py` and `download_soil.py` — average gridded data over
county boundaries: daily gridMET weather, monthly MODIS NDVI and EVI, and static
OpenLandMap soil properties. Each masks to cropland using the USDA Cropland Data
Layer, so a county's numbers describe its farmland rather than its parks and
towns, and each reduces at its source data's native resolution to avoid
resampling.

Climate is downloaded monthly and aggregated into growing-season features
afterwards, so the season can be redefined without re-downloading. Alongside
temperature and rainfall the aggregation computes heat-stress measures: growing
and extreme degree days, the fraction of hot days, and vapour pressure deficit.

**Modelling.** `ml.py` joins the four sources into one county-year table and
trains a random forest per crop. It fits four feature sets — climate only,
climate plus soil, climate plus vegetation, and everything — so each source's
contribution can be read off. Models are tested on held-out years rather than
random rows, since predicting a year you have already partly seen is easier than
forecasting a new one.

**Analysis.** `evaluate.py` scores those models against baselines and computes
confidence intervals. `irrigation_contrast.py` compares irrigated and rainfed
corn in the same county and year. `paired_rerun.py` checks whether the rebuilt
data actually predicts better than the original.

---

## What changed since the original

**Cropland masking.** The original averaged satellite and soil data over entire
county polygons, including land that was not farmland. Both are now restricted to
cropland.

**Soil depth.** The original used SoilGrids at 0–5 cm, the surface layer. Roots
go deeper, so soil is now averaged over 0–30 cm from OpenLandMap, weighted by
layer thickness.

**Growing season.** The original used April to September for every state and
crop. Spring wheat in North Dakota does not grow on the same calendar as corn in
Illinois, so seasons are now set per state and crop. Once season length varies,
season *totals* stop being comparable, so rainfall and reference ET are stored as
per-day rates.

**Resolution.** Everything was previously reduced at 4 km. Each source now uses
its own native scale.

**Scoring.** This is the change that matters most. The original reported test R²
on a county-year panel. The trouble is that 40–61% of yield variance is
persistent differences *between* counties — some counties simply grow more corn
than others, every year. A model can score well on that panel largely by
recognising which county it is looking at, without predicting weather effects at
all. The current work therefore compares each model against a baseline that
predicts each county's own historical average, and reports results with and
without the long-run upward trend removed.

### Does the rebuilt data predict better?

Yes, for the two main crops. Comparing datasets is harder than it sounds: the
rebuilt pipeline keeps more county-years, so scoring each on its own rows would
compare numbers computed over different samples. `paired_rerun.py` instead finds
the 11,786 county-years present in both, confirms the yields match, and trains
the same model on the same rows using each dataset's features in turn.

| crop | original data | rebuilt data | change |
|---|---|---|---|
| Corn | 0.543 | 0.632 | +0.089 |
| Soybeans | 0.640 | 0.719 | +0.079 |
| Spring wheat | 0.175 | 0.208 | +0.033 |
| Oats | 0.387 | 0.387 | −0.000 |

Corn and soybeans improve in every test year, so this is not one unusual season.
Which specific change earns the improvement is still an open question — masking,
soil depth, season and resolution all changed together.

Oats is a useful warning. Its flat result is not stability: 2023 got noticeably
worse and 2024 got better by almost exactly as much, on only eight counties. A
single pooled number hid both. Per-year figures are in
`results_comparison/paired_rerun_by_year.csv`.

---

## Results

Confidence intervals are 95%, from a bootstrap that resamples whole counties.

### Weather adds little once the trend is removed

Yields have risen steadily for decades from better seed and management. Remove
that trend and score against each county's own average:

| crop | full model | county average + trend |
|---|---|---|
| Corn | −0.233 to −0.046 | 0.006 to 0.010 |
| Soybeans | −0.200 to −0.063 | 0.005 to 0.006 |
| Oats | −0.046 to 0.114 | 0.008 to 0.013 |
| Sorghum | −0.040 to 0.124 | −0.077 to −0.063 |
| Spring wheat | 0.081 to 0.257 | 0.005 to 0.015 |

For corn and soybeans the full model does worse than a county average plus a
straight line. Oats and sorghum land on both sides of zero, so there is not
enough evidence either way. Spring wheat is the one crop with a clear gain.

This is not the same as saying the models are useless. Without detrending they do
beat a trend baseline. It says that most of what they get right is *where* yields
are high rather than *which years* are good, and that a lot of the rest is the
long-run trend rather than weather.

The removed trend is also not purely technology. Warming, irrigation and changing
varieties all move slowly enough to be absorbed by a straight line.

### Irrigation weakens the link between weather and yield

USDA reports irrigated and non-irrigated corn separately for some counties. Where
both appear in the same county and year, soil, weather and season are identical
and only management differs — 867 such pairs across 114 counties, 2008–2018.

Weather explains 68.5% of the year-to-year variation in rainfed yields, but only
16.6% for irrigated. Irrigated corn responds far less to rainfall in particular:

| | rainfed | irrigated | ratio |
|---|---|---|---|
| Precipitation | +32.95 | +3.31 | 0.19 |
| Extreme heat | −21.61 | −5.74 | 0.51 |
| Vapour pressure deficit | −73.44 | −17.02 | 0.45 |

Irrigated fields yielded 81.4 BU/AC more on average.

The direction holds up, but the size of the gap should be read carefully. Drought
years dominate it: dropping 2012 alone moves the rainfed figure from 0.699 to
0.300 while irrigated barely moves. Kansas and Nebraska also behave differently.
And farmers choose whether to irrigate, so this is an association within
counties, not an experiment.

### Soil is doing real work

Soil has one value per county and never changes, so a model given soil could in
principle just use it to identify the county. To test that, the soil features
were swapped for four random numbers held constant per county — anything the
model gains from those is pure county-labelling. Real soil beat that control by 7
to 13 times.

That is evidence soil contributes more than a county label, though it does not
put a precise number on how much.

---

## Running it

Needs Python 3.14, [uv](https://docs.astral.sh/uv/), a Google Earth Engine
account and a [NASS API key](https://quickstats.nass.usda.gov/api).

```bash
uv sync
cp src/.env.example src/.env    # add your keys
cd src

python download_yield.py        # ~5 min
python download_soil.py         # ~5 min
python download_vegetation.py   # ~20 min
python download_climate.py      # ~40 min

python ml.py                    # ~25 min
python evaluate.py --detrend none county
python irrigation_contrast.py
python paired_rerun.py
```

Run the downloads one at a time — Earth Engine limits concurrent requests, and
`ml.py` and `evaluate.py` write to the same paths.

Data is not committed (162 MB); the download scripts regenerate it.

---

## Data sources

| | Source | Resolution |
|---|---|---|
| Yields | [USDA NASS QuickStats](https://quickstats.nass.usda.gov/) | county, annual |
| Climate | [gridMET](https://www.climatologylab.org/gridmet.html) | 4638 m, daily |
| Vegetation | [MODIS MOD13A3](https://lpdaac.usgs.gov/products/mod13a3v061/) | 927 m, monthly |
| Soil | [OpenLandMap](https://openlandmap.org/) | 232 m, static |
| Crop mask | [USDA CDL](https://nassgeodata.gmu.edu/CropScape/) | 30 m, annual |

---

## Limitations

- NDVI and EVI are measured during the season being predicted. They predict well
  but they are a result of crop growth, not a cause of it, so a genuine forecast
  would need to check when each product actually becomes available.
- The crop mask is generic cropland, not crop-specific, so corn NDVI includes
  soybean fields.
- The soil mask uses land cover from the whole study period, including test
  years.
- The irrigation comparison ends in 2018, when USDA stopped publishing the
  county-level split, and only covers counties that reported both practices.
- Spring wheat has no county estimates for 2024, leaving a gap in its test years,
  and it is the only crop with a clear detrended gain.
- `GDD_TMAX` and `EDD_TMAX` are based on maximum temperature only. They are not
  conventional growing degree days.

---

## Related work

Comparing crop yield models against a trend baseline is established practice —
see Paudel et al. (2022),
[*Machine learning for regional crop yield forecasting in Europe*](https://doi.org/10.1016/j.fcr.2021.108377).
Kallenberg et al. (2026), [*CY-Bench*](https://doi.org/10.5194/essd-18-3997-2026),
provides a reproducible benchmark dataset for sub-national yield forecasting.

---

## Citation

```
Prince, A. & Prince, A. (2025). CropCast: county-level crop yield analysis with
climate, vegetation and soil data. AGU Fall Meeting 2025, GC13F-0713.
https://github.com/ScienceAndBeyond/CropCast
```
