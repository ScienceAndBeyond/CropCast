# CropCast

Predicting U.S. crop yields using machine learning with climate, satellite vegetation indices, and soil data.

---

## What this project does

This project evaluates whether soil properties and satellite-derived vegetation indices improve county-level crop yield prediction relative to climate-only models.

It was presented at AGU 2025. Since then we rebuilt the data pipeline and changed how the models are scored, so the results below differ from the poster. The original version is kept unchanged in [`archive/`](archive/).

We cover corn, soybeans, spring wheat, oats, and sorghum across 11 states from 2008 to 2025.

---

## What changed since AGU 2025

**Cropland masking.** Satellite and soil values were averaged over whole county polygons, including land that was not farmland. Both are now restricted to cropland using the USDA Cropland Data Layer.

**Soil depth.** Soil came from SoilGrids at 0 to 5 cm, the surface layer. Roots go deeper, so we now use OpenLandMap averaged over 0 to 30 cm and weighted by layer thickness.

**Growing season.** One April to September season was applied to every state and crop. Spring wheat in North Dakota does not follow the same calendar as corn in Illinois, so seasons are now set per state and crop. Because season length now varies, rainfall and reference ET are stored as per-day rates instead of season totals.

**Resolution.** Every source was reduced at 4 km. Each now uses its own native resolution.

**Scoring.** This is the most important change. Test R² on a county-year panel is flattering, because 40 to 61% of yield variance is persistent differences between counties rather than year-to-year variation. A model can score well largely by learning which county it is looking at. We now compare every model against a baseline that predicts each county's own historical average, and report results with and without the long-run upward trend removed.

### Does the rebuilt data predict better?

Yes, for the two main crops. The rebuilt pipeline keeps more county-years, so scoring each dataset on its own rows would compare numbers computed over different samples. `paired_rerun.py` instead finds the 11,786 county-years present in both, checks that the yields match, and trains the same model on the same rows using each dataset's features in turn.

| Crop | Original data | Rebuilt data | Change |
|------|---------------|--------------|--------|
| Corn | 0.543 | 0.632 | +0.089 |
| Soybeans | 0.640 | 0.719 | +0.079 |
| Spring wheat | 0.175 | 0.208 | +0.033 |
| Oats | 0.387 | 0.387 | −0.000 |

Corn and soybeans improve in every test year. We have not yet isolated which change earns the improvement, since masking, soil depth, season, and resolution all changed together.

---

## Results

Confidence intervals are 95%, from a bootstrap that resamples whole counties.

### Weather adds little once the trend is removed

Yields have risen for decades from better seed and management. Removing that trend and scoring against each county's own average:

| Crop | Full model | County average + trend |
|------|------------|------------------------|
| Corn | −0.233 to −0.046 | 0.006 to 0.010 |
| Soybeans | −0.200 to −0.063 | 0.005 to 0.006 |
| Oats | −0.046 to 0.114 | 0.008 to 0.013 |
| Sorghum | −0.040 to 0.124 | −0.077 to −0.063 |
| Spring wheat | 0.081 to 0.257 | 0.005 to 0.015 |

For corn and soybeans the full model does worse than a county average plus a straight line. Oats and sorghum fall on both sides of zero, so there is not enough evidence either way. Spring wheat is the one crop with a clear gain.

Most of what these models get right is where yields are high rather than which years are good.

### Irrigation weakens the link between weather and yield

USDA reports irrigated and non-irrigated corn separately for some counties. Where both appear in the same county and year, soil and weather are identical and only management differs. We found 867 such pairs across 114 counties from 2008 to 2018.

Weather explains 68.5% of the year-to-year variation in rainfed yields but only 16.6% for irrigated. Holding out whole counties in cross-validation barely changes this, so the fit is not simply flattering itself: rainfed drops to 68.0% and irrigated to 15.2%.

Irrigated corn responds far less to rainfall in particular.

| Weather | Rainfed | Irrigated | Ratio |
|---------|---------|-----------|-------|
| Precipitation | +32.95 | +3.31 | 0.19 |
| Extreme heat | −21.61 | −5.74 | 0.51 |
| Vapor pressure deficit | −73.44 | −17.02 | 0.45 |

Irrigated fields yielded 81.4 BU/AC more on average.

**Drought years carry most of the gap.** Refitting with each year dropped in turn, and recomputing the anomalies each time, every year leaves the result roughly unchanged except 2012. Dropping 2012 alone takes the rainfed figure from 0.699 to 0.321 while irrigated barely moves, closing the gap from 0.45 to 0.05. The 2012 drought is doing most of the work, which fits the mechanism but means this is really a statement about how irrigation protects yields in a severe drought, not in an average year.

**The two states differ.** Nebraska supplies 665 of the 867 pairs and Kansas 202.

| State | Pairs | Rainfed R² | Irrigated R² | Rainfed PRCP corr | Irrigated PRCP corr |
|-------|-------|------------|--------------|-------------------|---------------------|
| Kansas | 202 | 0.728 | 0.491 | +0.771 | +0.373 |
| Nebraska | 665 | 0.703 | 0.247 | +0.711 | −0.051 |

Irrigated corn in Nebraska shows essentially no rainfall response, while Kansas retains some. Full tables are in `results_split/irrigation_leave_one_year_out.csv` and `results_split/irrigation_by_state.csv`.

### Soil is doing real work

Soil has one value per county and never changes, so a model given soil could in principle use it to identify the county instead of learning agronomy. To test this we replaced the soil features with four random numbers held constant per county, since anything the model gains from those is pure county labeling. Real soil beat that control by 7 to 13 times.

---

## Data sources

| Data | Source | Resolution |
|------|--------|------------|
| Crop yields | USDA NASS QuickStats | County, annual |
| Climate | gridMET via Google Earth Engine | 4638 m, daily |
| Vegetation | MODIS MOD13A3 NDVI/EVI via Google Earth Engine | 927 m, monthly |
| Soil | OpenLandMap via Google Earth Engine | 232 m, static |
| Crop mask | USDA Cropland Data Layer | 30 m, annual |

---

## Project structure

```
CropCast/
├── src/
│   ├── config.py                # Paths, states, seasons, settings
│   ├── utils.py                 # Crop masks and helpers
│   ├── download_yield.py        # USDA NASS crop yields
│   ├── download_climate.py      # gridMET via GEE, monthly
│   ├── download_vegetation.py   # MODIS via GEE, monthly
│   ├── download_soil.py         # OpenLandMap via GEE
│   ├── ml.py                    # Merge and train
│   ├── evaluate.py              # Baselines and bootstrap intervals
│   ├── irrigation_contrast.py   # Irrigated vs rainfed corn
│   └── paired_rerun.py          # Old vs new data on matched rows
│
├── data/                        # Not committed, regenerate with the scripts
├── results/                     # Model outputs
├── results_split/               # Irrigation analysis
├── results_comparison/          # Old vs new data comparison
└── archive/                     # AGU 2025 version, unchanged
```

Climate and vegetation are downloaded monthly and aggregated into growing-season features afterward, so the season can be redefined without downloading again.

---

## How to run

```bash
git clone https://github.com/ScienceAndBeyond/CropCast.git
cd CropCast
uv sync
cp src/.env.example src/.env    # add your API keys
cd src
```

To download fresh data:
```bash
python download_yield.py
python download_soil.py
python download_vegetation.py
python download_climate.py
```

Run these one at a time. Earth Engine limits concurrent requests.

To train and evaluate:
```bash
python ml.py
python evaluate.py --detrend none county
python irrigation_contrast.py
python paired_rerun.py
```

---

## Requirements

- Python 3.14 and [uv](https://docs.astral.sh/uv/)
- pandas, scikit-learn, numpy
- Google Earth Engine account (for climate, vegetation, and soil)
- [USDA NASS QuickStats API key](https://quickstats.nass.usda.gov/api)

---

## Poster

AGU 2025, New Orleans poster presentation (GC13F-0713).

📄 [View Poster (PDF)](archive/poster/AGU2025_CropCast_Poster.pdf)

---

## Limitations

- NDVI and EVI are measured during the season being predicted. They predict well, but they are a result of crop growth rather than a cause of it, and a real forecast would need to check when each product becomes available.
- The crop mask is generic cropland and not crop-specific, so corn NDVI includes soybean fields.
- The soil mask uses land cover from the whole study period, including test years.
- The irrigation comparison ends in 2018, when USDA stopped publishing the county-level split, and only covers counties that reported both practices. Farmers choose whether to irrigate, so this is an association rather than an experiment.
- The trend removed from yields is not purely technology. Warming, irrigation, and changing varieties all move slowly enough to be absorbed by a straight line.
- The oats result in the comparison table is flat overall but not stable underneath. 2023 got worse and 2024 got better by almost the same amount, on only eight counties. Per-year figures are in `results_comparison/paired_rerun_by_year.csv`.
- Spring wheat has no county estimates for 2024, leaving a gap in its test years, and it is the only crop with a clear detrended gain.
- `GDD_TMAX` and `EDD_TMAX` are based on maximum temperature only and are not conventional growing degree days.

---

## Roadmap

- [ ] Isolate which pipeline change earns the prediction improvement
- [ ] Extend the irrigation comparison beyond corn and beyond two states
- [ ] Crop-specific vegetation masking
- [ ] Extend the irrigation comparison past 2018 with another data source

---

## Related work

Comparing crop yield models against a trend baseline is established practice. See Paudel et al. (2022), [Machine learning for regional crop yield forecasting in Europe](https://doi.org/10.1016/j.fcr.2021.108377). Kallenberg et al. (2026), [CY-Bench](https://doi.org/10.5194/essd-18-3997-2026), provides a reproducible benchmark dataset for sub-national yield forecasting.

---

## Authors

- Arit Prince
- Arya Prince

---

## Contact

Questions or ideas?
- Open an [Issue](https://github.com/ScienceAndBeyond/CropCast/issues)
- Or reach us through our [GitHub profile](https://github.com/ScienceAndBeyond)

---

If you use this, a citation would be appreciated:

```
Prince, Arit. & Prince, Arya. (2025). CropCast: Multi-source crop yield prediction.
GitHub: https://github.com/ScienceAndBeyond/CropCast
```
