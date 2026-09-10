# CropCast

Predicting U.S. crop yields with climate, satellite vegetation, and soil data.

---

## What this project does

This project tests whether soil data and satellite vegetation data help predict county-level crop yields better than climate data alone.

It was presented at AGU 2025. The AGU version was a first version of the study. It documented several limits, including county-level modeling, a general growing season, static soil data, and the need for more precise vegetation masking. This version is a follow-up that addresses several of those limits by adding deeper soil information, state-specific growing seasons, cropland-level masking, and a stricter scoring check against each county's own average yield. The original AGU version is kept unchanged in [`archive/`](archive/).

We cover corn, soybeans, spring wheat, oats, and sorghum across 11 states from 2008 to 2025.

---

## What this version adds

**Cropland masking.** The AGU README noted that this project used county-level data and listed more precise vegetation masking as future work. This version takes a first step in that direction by restricting satellite and soil averages to cropland using the USDA Cropland Data Layer. It is still not crop-specific.

**Soil depth.** The original version used the top 0 to 5 cm of soil. This version uses OpenLandMap soil data from 0 to 30 cm, which better reflects the soil layer crop roots use.

**Growing season.** The original version used April to September everywhere. This version sets the season by state, because crops in Texas and Minnesota do not grow on the same calendar. Most states still use April to September. This is not yet crop-specific.

**Resolution.** Each data source has its own natural pixel size. This version uses the native scale for climate, vegetation, and soil.

**Scoring.** This is the most important change. A model can look good just by learning which counties usually have high yields. That does not mean it is good at predicting unusually good or bad years. We now compare every model against a simple baseline: each county's own average yield. We also report results after removing the long-term upward yield trend.

### Does the extended data predict better?

Yes, for corn and soybeans. The updated pipeline keeps more county-years than the original one, so we cannot compare the two versions on different rows. `paired_rerun.py` finds the 11,786 county-years that exist in both versions, checks that the yield values match, and trains the same model on those same rows.

| Crop | Original data | Updated data | Change |
|------|---------------|--------------|--------|
| Corn | 0.543 | 0.632 | +0.089 |
| Soybeans | 0.640 | 0.719 | +0.079 |
| Spring wheat | 0.175 | 0.208 | +0.033 |
| Oats | 0.387 | 0.387 | −0.000 |

Corn and soybeans improve in every test year.

### Which update mattered most?

Vegetation, almost entirely. `ablation.py` changes one data source at a time back to the original version while keeping the rows and model the same. (We call this an "ablation" because that word is common in machine learning, but nothing is actually removed here. Each test just swaps one input back to its original version and leaves the rest alone.) The numbers below are error ratios. A value above 1 means the update reduced prediction accuracy. The range in parentheses is a 95% uncertainty range made by resampling whole counties. The 5% cutoff and decision rule were chosen before running this check.

| Reverted source | Corn | Soybeans |
|-----------------|------|----------|
| Climate | 1.007 (0.994 to 1.020) | 1.010 (0.998 to 1.023) |
| Soil | 1.016 (1.003 to 1.030) | 1.027 (1.007 to 1.046) |
| **Vegetation** | **1.100 (1.065 to 1.134)** | **1.103 (1.075 to 1.133)** |
| Everything | 1.114 (1.077 to 1.153) | 1.131 (1.096 to 1.167) |

Changing vegetation back to the original version causes about 10% more prediction error. That is almost the same as changing everything back. Climate does not show a clear effect: its ranges include 1.0. Soil is different. For corn and soybeans, soil shows a real but small effect; the ranges do not include 1.0, but the effect stays below the 5% cutoff. For spring wheat, the soil range also misses 1.0 but crosses the 5% cutoff, so that result is inconclusive. The vegetation result holds in every test year rather than depending on one season.

The "Vegetation" row uses the original vegetation table as it was. That table differs from the updated one in more than one way. The cleaner test below separates crop masking from pixel size.

Vegetation was extended in two main ways: it gained a cropland mask, and its pixel size moved from 1 km to MODIS's native 927 m. To separate those updates, we rebuilt vegetation three more times and changed only one option at a time. `veg_ablation.py` runs this comparison.

| Vegetation version | Mask | Scale | Corn | Soybeans |
|--------------------|------|-------|------|----------|
| Production | on | 927 m (native) | reference | reference |
| Scale reverted only | on | 1 km | 1.007 (1.000 to 1.015) | 1.012 (1.005 to 1.019) |
| **Mask removed only** | off | 927 m (native) | **1.086 (1.052 to 1.120)** | **1.102 (1.076 to 1.129)** |
| **Mask and scale reverted** | off | 1 km | **1.088 (1.054 to 1.121)** | **1.102 (1.077 to 1.128)** |

Removing the cropland mask causes almost all of the vegetation loss. Removing the mask alone gives nearly the same result as removing both the mask and the scale change. Changing only the scale has a much smaller effect, around 1% or less. For corn, that small scale-only result is borderline because the lower end of the range is almost exactly 1.0. Soybeans show clearer evidence of a small scale effect.

We also compared against the archived original vegetation data directly. That result is close to the mask-removed rows, but it is not a clean test. The original pipeline also handled clouds, snow, and growing seasons differently.

So the useful change was masking to cropland. The original county-level approach averaged vegetation across broader county areas, while this version focuses the satellite signal on cropland. That improves prediction error by roughly 9 to 10% for corn and soybeans combined. The pixel-size change is small by comparison.

**This comparison covers 4 states, not all 11.** The matched rows exist only in Iowa, Illinois, Minnesota, and Nebraska, for test years 2022-2024. Removing the mask hurts corn prediction in Iowa, Illinois, and Nebraska, but it improves Minnesota corn prediction by about 7%. Soybeans do not reverse in Minnesota, but the effect is smaller there. So "masking to cropland helps" is a pooled result, not a rule that holds the same way in every state.

---

## Results

Ranges in parentheses are 95% uncertainty ranges. Where ranges are produced by resampling, the resampling is done by whole county, not by single rows.

### Weather adds little once the trend is removed

Yields have risen for decades because of better seed, equipment, and management. When we remove that long-term trend and compare against each county's own average:

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

USDA reports irrigated and non-irrigated corn separately for some counties. When both appear in the same county and year, they share the same reported weather. The actual fields may still differ in soil, slope, seed variety, or management. We found 867 such pairs across 114 counties from 2008 to 2018.

The weather model accounts for 68.5% of the year-to-year movement in rainfed yields, but only 16.6% for irrigated yields. A check that holds out whole counties gives almost the same result: 68.0% for rainfed and 15.2% for irrigated.

Irrigated corn responds far less to rainfall in particular.

| Weather | Rainfed slope | Irrigated slope | Slope ratio | Correlation ratio |
|---------|---------------|------------------|-------------|--------------------|
| Precipitation | +32.95 | +3.31 | 0.10 | 0.19 |
| Extreme heat | −21.61 | −5.74 | 0.27 | 0.51 |
| Vapor pressure deficit (how dry the air is) | −73.44 | −17.02 | 0.23 | 0.45 |
| Maximum temperature | −13.39 | −3.35 | 0.25 | 0.48 |

The slope ratio compares irrigated to rainfed on the units shown. The correlation ratio makes the same comparison without units. Both tell the same story.

Irrigated fields yielded 81.4 BU/AC more on average.

**Drought years carry most of the gap.** We reran the analysis while dropping one year at a time. Every dropped year leaves the result roughly the same except 2012. These are the trend-removed numbers: dropping 2012 cuts the rainfed number from 0.699 to 0.321, while the irrigated number barely changes. The gap falls from 0.45 to 0.05. The headline 68.5% and 16.6% numbers above are not trend-removed; in that version, dropping 2012 moves the gap from 0.52 to 0.16. Across these 114 counties, 2012 is the driest and hottest year in the available 2008-2025 climate record used here. So the result is real, but it depends strongly on one historic drought year. This is an association, not a randomized experiment.

**The two states differ, and Nebraska has more say in the combined number because it has more data.** Nebraska supplies 665 of the 867 pairs (77%) and Kansas 202.

| State | Pairs | Rainfed R² | Irrigated R² | Rainfed rain link | Irrigated rain link |
|-------|-------|------------|--------------|-------------------|---------------------|
| Kansas | 202 | 0.728 | 0.491 | +0.771 | +0.373 |
| Nebraska | 665 | 0.703 | 0.247 | +0.711 | −0.051 |

Irrigated corn in Nebraska shows almost no rainfall response, while Kansas still shows some. Nebraska has more than three times as many pairs as Kansas, so the pooled result is closer to Nebraska than to a balanced two-state result. If Kansas and Nebraska are given equal weight, rainfed is 69.7% and irrigated is 26.9%. The direction stays the same, but the irrigated number is higher.

This is also a selected sample. Before requiring at least 4 paired years per county, 989 county-years across Kansas, Nebraska, North Dakota, and South Dakota report both practices. North Dakota and South Dakota drop out after that filter. The final result describes counties with repeated reporting of both irrigated and non-irrigated corn, not irrigated versus rainfed corn everywhere.

Full tables are in `irrigation_results/irrigation_leave_one_year_out.csv`, `irrigation_results/irrigation_by_state.csv`, and `irrigation_results/irrigation_state_balanced.csv`.

### Soil is doing real work

Soil has one value per county and does not change over time. That means a model might use soil partly as a county label, not just as an agronomy signal. To test this, we replaced the soil data with four random numbers that stay fixed for each county. Real soil beat that random control by 4.7x to 14.2x across the crops where soil helped.

**This is only one simple check.** The random numbers do not look like real soil maps. Nearby counties often have similar soils, but the random control does not copy that pattern. So read this as "real soil beats this simple random county label," not as a final answer about how much of soil's value is truly agronomic.

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

```text
CropCast/
  src/                 Code for download, training, evaluation, and checks
  data/processed/      Processed CSV inputs committed for reproducibility
  data/raw/            Local raw downloads, not committed
  results/             Main model outputs
  irrigation_results/  Irrigation study CSV outputs
  irrigation_weather_yield/
                       Irrigation study tables, figures, and helper scripts
  results_comparison/  Original-vs-updated pipeline comparison outputs
  archive/             AGU 2025 version, kept unchanged
```

The committed processed CSVs are enough to run the default yield-prediction pipeline, rerun the irrigation analysis, and rebuild the reported irrigation tables and figures. `ml.py` rebuilds `data/processed/merged.csv` locally. Raw and processed data paths are set in `src/config.py`. Climate and vegetation are downloaded monthly first when rebuilding from raw sources. The growing-season values are built afterward, so the season can be changed without downloading everything again.

---

## How to run

```bash
git clone https://github.com/ScienceAndBeyond/CropCast.git
cd CropCast
uv sync
cp src/.env.example src/.env    # add your API keys
cd src
```

To rebuild fresh data from the original sources:
```bash
python download_yield.py
python download_soil.py
python download_vegetation.py
python download_climate.py
```

Run these one at a time. Earth Engine limits concurrent requests.

To train and evaluate the default yield-prediction pipeline from the committed processed CSVs:
```bash
python ml.py
python evaluate.py --detrend none county
python paired_rerun.py
python ablation.py
```

To rerun the irrigation analysis from the committed processed CSVs:
```bash
python irrigation_contrast.py
```

To rebuild the irrigation study tables and figures from the repo root:
```bash
cd ..
python irrigation_weather_yield/make_irrigation_tables.py
python irrigation_weather_yield/make_irrigation_figures.py
```

Before running the vegetation mask/scale check, build the three extra vegetation versions:
```bash
python download_vegetation.py --study --scale 1000 --variant masked_1km
python download_vegetation.py --study --no-crop-mask --variant unmasked_native
python download_vegetation.py --study --no-crop-mask --scale 1000 --variant unmasked_1km
python veg_ablation.py
```

If the monthly vegetation raw file already exists locally, rebuild only the processed yearly vegetation table with:
```bash
python download_vegetation.py --aggregate-only
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

- NDVI and EVI are measured during the growing season. They predict well, but they partly measure crop growth that has already happened. A true early forecast would need to check when these satellite products become available.
- The crop mask is generic cropland, not crop-specific. Corn NDVI can still include soybean fields.
- The soil mask uses land cover from the whole study period, including test years.
- The irrigation comparison ends in 2018, when USDA stopped publishing the county-level split. It only covers counties that reported both irrigated and non-irrigated corn. Farmers choose whether to irrigate, so this is an association rather than an experiment.
- The trend removed from yields is not only technology. Warming, irrigation, and changing seed varieties can also move slowly over time.
- The oats result looks flat overall, but it is unstable underneath. 2023 got worse and 2024 got better by almost the same amount, on only eight counties. Per-year figures are in `results_comparison/paired_rerun_by_year.csv`.
- Spring wheat has no county estimates for 2024. It is still the only crop with a clear gain after removing the long-term trend.
- `GDD_TMAX` and `EDD_TMAX` are based on maximum temperature only and are not conventional growing degree days.
- The original-vs-updated data comparison and the mask/scale test only cover the 4 states and 3 test years present in both datasets: Iowa, Illinois, Minnesota, and Nebraska from 2022-2024. They do not cover the full 11-state, 2008-2025 study. Corn also behaves differently in Minnesota; see the vegetation section above.

---

## Progress Since AGU

Some items from the original roadmap are partly addressed in this version:

- The single April-September growing season was replaced with state-specific growing seasons.
- Vegetation and soil averages are now restricted to cropland, but not yet to a crop-specific mask.
- Irrigation is now tested using paired USDA NASS irrigated and non-irrigated corn reports for Kansas and Nebraska. It does not use USGS irrigation data and does not go past 2018.
- Soil is checked against a simple county-level placebo control. A stronger repeated placebo check is still open.

---

## Roadmap

These are still open items.

- [ ] Understand why Minnesota corn reverses (mask removal helps there, hurts elsewhere)
- [ ] Crop-specific masks, so corn NDVI excludes soybean fields
- [ ] Extend the irrigation comparison beyond corn and beyond Kansas and Nebraska
- [ ] Stronger soil placebo checks that better match county-to-county patterns, instead of one random draw
- [ ] Extend the irrigation comparison past 2018 with another data source

---

## Related work

Comparing crop yield models against a trend baseline is established practice. See Paudel et al. (2022), [Machine learning for regional crop yield forecasting in Europe](https://doi.org/10.1016/j.fcr.2021.108377). Kallenberg et al. (2026), [CY-Bench](https://doi.org/10.5194/essd-18-3997-2026), provides a reproducible benchmark dataset for sub-national yield forecasting.

---

## License

Code in this repository is released under the MIT License. The processed data
tables are derived from public data sources listed above; users should also
cite those original sources when reusing the data.

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
Prince, Arit. & Prince, Arya. (2026). CropCast: Multi-source crop yield prediction.
GitHub: https://github.com/ScienceAndBeyond/CropCast
```
