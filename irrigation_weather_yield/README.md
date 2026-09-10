# Irrigation Weather-Yield Study Files

This folder contains the tables, figures, references, and helper scripts for the irrigation weather-yield analysis.

The main analysis is run by `src/irrigation_contrast.py`. Its CSV outputs are in `irrigation_results/`. The scripts in this folder turn those CSV outputs into the tables and figures used to check and report the results.

## Files

- `make_irrigation_tables.py` rebuilds the table CSVs and the short reproducibility summary.
- `make_irrigation_figures.py` rebuilds the SVG figures.
- `tables/` contains the table exports used for checking the reported numbers.
- `figures/` contains the figure SVGs.
- `references.bib` lists the references used by the manuscript.
- `irrigation_reproducibility_summary.md` gives a quick check of the main sample size and headline results.

## Rebuild

From the repository root:

```bash
python irrigation_weather_yield/make_irrigation_tables.py
python irrigation_weather_yield/make_irrigation_figures.py
```

To rerun the paired irrigation analysis first:

```bash
cd src
python irrigation_contrast.py
cd ..
python irrigation_weather_yield/make_irrigation_tables.py
python irrigation_weather_yield/make_irrigation_figures.py
```

The yield panel is 2008-2018 because USDA NASS stopped county estimates by irrigated and non-irrigated practice beginning with the 2019 crop year. The 2012 drought ranking uses the available 2008-2025 local climate record for the selected counties.
