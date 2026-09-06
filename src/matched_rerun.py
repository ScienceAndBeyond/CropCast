"""
SUPERSEDED 2026-09-06 -- DO NOT CITE THIS SCRIPT'S RESULT.

This experiment matched the poster's CONFIGURATION but not its ROWS. The
corrected dataset contains more eligible county-years, so corn trained on 9,374
observations against the poster's 5,448 and was scored on a different test set
with a different variance denominator. Comparing R2 across different samples
does not isolate the effect of the data.

The conclusion drawn from it -- that the corrections "barely move" the headline
metric -- is RETRACTED. On identical rows (see paired_rerun.py) the corrected
data scores +0.089 higher on corn and +0.079 on soybeans.

Kept for provenance and because its per-crop scores are still a valid record of
"poster configuration on current data". Use paired_rerun.py for any claim about
what the data changes did.

Original docstring follows.
--------------------------------------------------------------------------

Matched re-run: the AGU 2025 poster's EXACT analysis, on CORRECTED data.

WHY THIS EXISTS
---------------
The repository claims that correcting serious data defects (an inverted CDL crop
mask, season-total rather than per-day precipitation, an inert soil mask) barely
moves the headline metric: corn all_features test R2 0.764 -> 0.772.

That comparison is CONFOUNDED. The poster run and the current run differ in more
than data quality:

                      poster              current
  train / test        2010-2021/2022-2024 2008-2021/2022-2025
  n features (all)    14                  20
  crop panel          6 (incl. cotton)    5
  seeds averaged      1                   5

So the raw 0.764 vs 0.772 comparison cannot separate "the corrections did not
matter" from "the extra features and the longer window happened to compensate".

This script removes that confound. It applies the poster's configuration --
its 14 features, its 2010 start, its split rule, its RF hyperparameters, its
single seed 25 -- to the CORRECTED data. The ONLY difference from the poster
run is then the data itself.

READ THE RESULT AS
------------------
  corn lands near 0.77  -> the corrections genuinely did not move the metric;
                           the insensitivity claim is established.
  corn drops toward 0.60 -> the apparent insensitivity was an artefact of the
                           extra features / longer window; the claim is dead.

Either outcome is informative. Writes to ../results_matched/ and touches
nothing that ml.py or evaluate.py produce.

Run from src/:  python -X utf8 matched_rerun.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# ---------------------------------------------------------------------------
# The poster's configuration, copied verbatim from archive/src_agu/ml.py.
# Do not "improve" any of these -- the whole point is that they match.
# ---------------------------------------------------------------------------
CLIMATE_FEATURES = ["TMIN", "TMAX", "PRCP", "VPD", "ETO", "SRAD"]
VEG_FEATURES = ["evi_min_year", "evi_max_year", "ndvi_min_year", "ndvi_max_year"]
SOIL_FEATURES = ["clay_mean", "ph_mean", "soc_mean", "bdod_mean"]
ALL_FEATURES = CLIMATE_FEATURES + VEG_FEATURES + SOIL_FEATURES

FEATURE_SETS = {
    "climate_only": CLIMATE_FEATURES,
    "climate_soil": CLIMATE_FEATURES + SOIL_FEATURES,
    "climate_veg": CLIMATE_FEATURES + VEG_FEATURES,
    "all_features": ALL_FEATURES,
}

START_YEAR = 2010
END_YEAR = 2024        # cap so the 80:20 rule lands on the poster's own split
MIN_SAMPLES = 1000
MIN_YEARS = 10
TEST_FRACTION = 0.20
MIN_TEST_YEARS = 2
RANDOM_SEED = 25

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_leaf": 5,
    "max_features": 0.5,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

DATA_PATH = Path("../data")
OUT_DIR = Path("../results_matched")
POSTER_RESULTS = Path("../archive/results_agu/model_performance.csv")

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def temporal_split(df: pd.DataFrame):
    """The poster's split rule, verbatim."""
    years = sorted(df["year"].unique())
    n_years = len(years)
    if n_years < MIN_YEARS:
        return None
    n_test = max(MIN_TEST_YEARS, int(np.ceil(n_years * TEST_FRACTION)))
    n_test = min(n_test, n_years - 3)
    test_years = years[-n_test:]
    train_years = years[:-n_test]
    train_df = df[df["year"].isin(train_years)]
    test_df = df[df["year"].isin(test_years)]
    if len(train_df) < 100 or len(test_df) < 30:
        return None
    return train_df, test_df, train_years, test_years


def main() -> None:
    merged_path = DATA_PATH / "processed" / "merged.csv"
    if not merged_path.exists():
        raise FileNotFoundError(
            f"{merged_path} not found. Run ml.py first, or run this from src/."
        )

    df = pd.read_csv(merged_path)
    df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)

    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        raise KeyError(f"corrected data is missing poster features: {missing}")

    # Poster's year window.
    df = df[(df["year"] >= START_YEAR) & (df["year"] <= END_YEAR)]

    # Poster drops rows missing ANY feature, before splitting, so every feature
    # set is trained and scored on identical rows.
    df = df.dropna(subset=ALL_FEATURES)

    log.info("=" * 70)
    log.info("MATCHED RE-RUN - poster configuration, corrected data")
    log.info("=" * 70)
    log.info(f"  years {START_YEAR}-{END_YEAR} | {len(df):,} rows | "
             f"{df['crop'].nunique()} crops | {df['county_fips'].nunique()} counties")

    rows = []
    for crop in sorted(df["crop"].unique()):
        cdf = df[df["crop"] == crop]
        if len(cdf) < MIN_SAMPLES:
            log.info(f"\n  {crop}: skipped ({len(cdf):,} rows < {MIN_SAMPLES})")
            continue
        split = temporal_split(cdf)
        if split is None:
            log.info(f"\n  {crop}: skipped (insufficient years after split)")
            continue
        train_df, test_df, train_years, test_years = split

        log.info(f"\n  {crop}")
        log.info(f"    train {train_years[0]}-{train_years[-1]} (n={len(train_df):,})  "
                 f"test {test_years[0]}-{test_years[-1]} (n={len(test_df):,})")

        for name, feats in FEATURE_SETS.items():
            model = RandomForestRegressor(**RF_PARAMS)
            model.fit(train_df[feats], train_df["yield_value"])
            pred = model.predict(test_df[feats])
            r2 = r2_score(test_df["yield_value"], pred)
            train_r2 = r2_score(train_df["yield_value"],
                                model.predict(train_df[feats]))
            rows.append({
                "crop": crop,
                "feature_set": name,
                "train_r2": round(train_r2, 3),
                "test_r2": round(r2, 3),
                "n_features": len(feats),
                "n_train": len(train_df),
                "n_test": len(test_df),
                "train_years": f"{train_years[0]}-{train_years[-1]}",
                "test_years": f"{test_years[0]}-{test_years[-1]}",
            })
            log.info(f"    {name:<14} n_feat={len(feats):<3} test_r2={r2:.3f}")

    out = pd.DataFrame(rows)
    OUT_DIR.mkdir(exist_ok=True)
    out.to_csv(OUT_DIR / "matched_rerun.csv", index=False)

    # ---- head-to-head against the poster --------------------------------
    if POSTER_RESULTS.exists():
        poster = pd.read_csv(POSTER_RESULTS)
        p = poster[poster.feature_set == "all_features"].set_index("crop").test_r2
        m = out[out.feature_set == "all_features"].set_index("crop").test_r2

        log.info("\n" + "=" * 70)
        log.info("HEAD TO HEAD - all_features test R2, poster config throughout")
        log.info("=" * 70)
        log.info(f"  {'crop':<26} {'poster':>8} {'matched':>9} {'delta':>8}")
        comp = []
        for crop in m.index:
            if crop in p.index:
                d = m[crop] - p[crop]
                log.info(f"  {crop:<26} {p[crop]:>8.3f} {m[crop]:>9.3f} {d:>+8.3f}")
                comp.append({"crop": crop, "poster_test_r2": p[crop],
                             "matched_test_r2": m[crop], "delta": round(d, 3)})
            else:
                log.info(f"  {crop:<26} {'--':>8} {m[crop]:>9.3f} {'--':>8}")
        if comp:
            cdf = pd.DataFrame(comp)
            cdf.to_csv(OUT_DIR / "matched_vs_poster.csv", index=False)
            log.info(f"\n  mean |delta| = {cdf.delta.abs().mean():.3f}")
            log.info("\n  Only the DATA differs between these two columns.")

    log.info(f"\n  Written to {OUT_DIR}")


if __name__ == "__main__":
    main()
