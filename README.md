# CropCast

Predicting U.S. crop yields using machine learning with climate, satellite vegetation indices, and soil data.

---

## What this project does

We wanted to see if adding soil and vegetation data could improve crop yield predictions beyond just using weather/climate data. Turns out it really does — our models improved by ~57% on average when we combined everything.

The surprising finding? **Soil bulk density** (basically how compacted the soil is) ended up being more important than temperature or rainfall for predicting corn, wheat, and cotton yields. We didn't expect that.

## Data sources

We used 4 data sources covering 742 counties across 11 states from 2010-2024:

- **Crop yields**: USDA NASS QuickStats
- **Climate**: GridMET via Google Earth Engine
- **Vegetation**: MODIS NDVI/EVI via Google Earth Engine  
- **Soil**: SoilGrids (bulk density, pH, organic carbon, clay)

See `data/README.md` for details on each file.

## Project structure

```
CropCast/
├── src/
│   ├── config.py              # File paths and other configuration
│   ├── utils.py               # Helper functions used across other codes
│   ├── climate_utils.py       # Helper functions used mainly for the climate data download when using NOAA. No longer used
│   ├── download_yields.py     # USDA NASS data for the crop yield
│   ├── download_climate.py    # GridMET via GEE
│   ├── download_vegetation.py # MODIS via GEE
│   ├── download_soil.py       # SoilGrids API
│   └── ml.py                  # ML pipeline
│
├── data/
│   ├── raw/                   # Downloaded data before cleaning for ML-ready pipeline. Can be used for debugging
│   └── processed/             # Cleansed and ML-ready datasets
│
├── results/                   # Model outputs
│   ├── model_performance.csv
│   └── feature_importance/
│
└── poster/                    # AGU 2025 materials
```

## How to run

```bash
git clone https://github.com/scienceAndBeyond/CropCast.git
cd CropCast
pip install -r requirements.txt
```

To download fresh data (needs API keys in `.env`):
```bash
python src/download_yields.py  --states CA MN --start-year 2020 --end-yer 2022
python src/download_climate.py  --states CA MN --start-year 2020 --end-yer 2022
python src/download_vegetation.py --states CA MN --start-year 2020 --end-yer 2022
python src/download_soil.py --states CA MN
```

To train models:
```bash
python src/ml.py
```

## Requirements

- Python 3.9+
- pandas, scikit-learn, numpy
- Google Earth Engine account (for climate and vegetation data)

## Notes

- Code could be cleaner, but it works. More work is needed to clean up
- Work in progress to download data more efficiently through  cloud sources

## Authors

- Arit Prince
- Arya Prince

---

If you use this, a citation would be appreciated:

```
Prince, Arit. & Prince, Arya. (2025). CropCast: Multi-source crop yield prediction. 
GitHub: https://github.com/scienceAndBeyond/CropCast
```

