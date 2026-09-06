# Archive — AGU 2025 material

Everything here backs the AGU 2025 poster (GC13F-0713) and is **superseded**.
Retained for provenance; tagged `agu2025-poster`.

`src_agu/`, `results_agu/`, `data_agu/` are the poster pipeline, its outputs and
its processed inputs. They are not reproducible with the current code, which
differs in data sources, spatial masking, season definitions and evaluation.

## The poster reported its own results accurately

Both headline figures reconcile exactly to `results_agu/improvement_summary.csv`
as committed here: CORN `best_r2` = 0.764, and the mean of `pct_improvement`
across the six crops is 74.1%.

## Correction, 2026-09-06

**An earlier version of this file listed defects that were not in this code.**
It claimed the poster pipeline had an inverted CDL crop mask, state-confounded
precipitation totals, and an inert soil mask. Checking `src_agu/` directly:

- `src_agu/download_vegetation.py` contains **no crop mask of any kind**. MODIS
  NDVI/EVI is averaged over whole county polygons. The inverted
  `cultivated.eq(1)` selector exists only in a later `download_vegetation-wip.py`
  that never produced these results.
- `src_agu/download_climaate.py` uses a **fixed April–September season for every
  state** (`GROWING_SEASON_START_MONTH = 4`, `END_MONTH = 9`). Precipitation and
  ETO are sums over that fixed window, so they are comparable across states. The
  state-varying season windows that make totals misleading came later.
- `src_agu/download_soil.py` uses the **SoilGrids API at 0–5 cm**. There is no
  Earth Engine masking to be inert.

Those three defects belong to the project's later development code, not to the
poster. Attributing them here was wrong.

## What is actually different about the poster pipeline

These are design choices and limitations, not corrupted data:

- **No crop mask.** Vegetation indices are county-wide averages including
  non-agricultural land. The current pipeline restricts to stable cropland.
- **Soil at 0–5 cm** from SoilGrids, a surface value rather than a rooting-zone
  depth. The current pipeline uses OpenLandMap thickness-weighted over 0–30 cm.
- **A fixed Apr–Sep season for all states and crops**, which does not match
  spring wheat in North Dakota or the southern end of the sample.
- **Non-native reduction scale**: `GRIDMET_SCALE = 4000` against a 4638.31 m
  native grid, forcing resampling. This one is a genuine poster-era defect.
- **Evaluation.** Test R² on a county-year panel, with no county-mean baseline
  and no detrending. This, not the data, is what the current work revisits.
- **"+74%" is a mean of per-crop ratios**, so crops with weak baselines dominate:
  oats (+137%, baseline R² 0.189) and spring wheat (+162%, baseline 0.265) pull
  the average up while corn, the largest crop, improved 28%.

## Does the newer data predict better?

Yes, and this was measured on identical rows rather than inferred. Intersecting
`data_agu/processed/merged.csv` with the current `data/processed/merged.csv` on
(crop, county_fips, year) gives 11,786 shared county-years with identical yield
values. Training the poster's own model on the same rows, using each dataset's
version of the poster's 14 features (mean of 5 seeds):

| crop | train / test | poster-era data | current data | Δ |
|---|---|---|---|---|
| CORN | 4,020 / 900 | 0.543 | 0.632 | **+0.089** |
| SOYBEANS | 3,829 / 865 | 0.640 | 0.719 | **+0.079** |
| WHEAT_SPRING | 413 / 110 | 0.175 | 0.208 | +0.033 |
| OATS | 1,199 / 156 | 0.387 | 0.387 | −0.000 |

Seed spread is ≤0.011, so corn and soybeans are well outside noise — roughly a
20% reduction in squared error. Which of the many changes (masking, soil depth
and product, season definition, native scales, added features) is responsible has
**not** been isolated; that would need an ablation holding rows fixed.

See `../src/HANDOFF.md`.
