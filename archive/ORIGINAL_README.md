# CropCast

Predicting U.S. crop yields using machine learning with climate, satellite vegetation indices, and soil data.

---

## What this project does

This project evaluates whether incorporating soil properties and satellite-derived vegetation indices improves county-level crop yield prediction relative to climate-only models.

Across the crops studied, adding soil and satellite vegetation data improved test R² by about 70% on average compared to using climate data alone.

## Results

![Model Performance](results/figures/model_performance.png)

<sub>
* R² comparison across model variants (climate-only, climate + vegetation, climate + soil, all features). 
Results reflect county-level modeling across 734 counties in 11 U.S. states (2010–2024) using an 80:20 temporal train/test split.
</sub>

---

## Data sources

We used 4 data sources covering 734 counties across 11 states from 2010 to 2024:

| Data | Source | What It Provides |
|------|--------|------------------|
| Crop yields | USDA NASS QuickStats | County-level crop yield by year (crop-specific units) |
| Climate | gridMET via Google Earth Engine | Temperature, precipitation, solar radiation (4km grid, derived from NOAA stations) |
| Vegetation | MODIS NDVI/EVI via Google Earth Engine | Satellite-based crop health indices |
| Soil | SoilGrids (ISRIC) | Bulk density, pH, organic carbon, clay content |

See `data/README.md` for details on each file.

---

## Project structure

```
CropCast/
├── src/
│   ├── config.py              # File paths and settings
│   ├── utils.py               # Helper functions
│   ├── download_yields.py     # USDA NASS crop yield data
│   ├── download_climate.py    # gridMET via GEE
│   ├── download_vegetation.py # MODIS via GEE
│   ├── download_soil.py       # SoilGrids API
│   └── ml.py                  # ML pipeline
│
├── data/
│   ├── raw/                   # Downloaded data before cleaning
│   └── processed/             # ML-ready datasets
│
├── results/                   # Model outputs
│   ├── model_performance.csv
│   └── feature_importance/
│
└── poster/                    # AGU 2025 materials
```

---

## How to run

```bash
git clone https://github.com/ScienceAndBeyond/CropCast.git
cd CropCast
pip install -r requirements.txt
```

To download fresh data (needs API keys in `.env`):
```bash
python src/download_yields.py --states MN IA IL --start-year 2010 --end-year 2024
python src/download_climate.py --states MN IA IL --start-year 2010 --end-year 2024
python src/download_vegetation.py --states MN IA IL --start-year 2010 --end-year 2024
python src/download_soil.py --states MN IA IL
```

To train models:
```bash
python src/ml.py
```

---

## Requirements

- Python 3.9+
- pandas, scikit-learn, numpy
- Google Earth Engine account (for climate and vegetation data)

---

## Poster

AGU 2025, New Orleans poster presentation (GC13F-0713).

📄 [View Poster (PDF)](poster/AGU2025_CropCast_Poster.pdf)

---

## Limitations

- The model uses county-level data, so it does not capture field-level differences or individual farm management practices.
- Soil data is treated as static, meaning it does not account for year-to-year changes in soil health or management.
- A general April–September growing season was applied across all crops, which may not perfectly reflect each crop’s actual growing period.
- NDVI/EVI are used as predictive indicators of crop condition and are not interpreted as direct causal drivers of yield.
  
## Roadmap

- [ ] Causal interpretation model (climate + soil only, excluding vegetation indices as outcomes)
- [ ] Add irrigation data from USGS
- [ ] Crop-specific vegetation masking using AI or other sources
- [ ] Crop-specific growing season

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
Prince, Arya. & Prince, Arit. (2025). CropCast: Multi-source crop yield prediction.
GitHub: https://github.com/ScienceAndBeyond/CropCast
```
