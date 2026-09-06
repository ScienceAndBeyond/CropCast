"""
Does the current data pipeline predict better than the archived 2025 one?

Compares the two processed datasets on IDENTICAL county-years. Both pipelines
cover overlapping counties and years but produce different feature values, and
the current one admits more eligible rows. Scoring each on its own sample would
compare R2 values with different denominators, so this script intersects them
on (crop, county_fips, year), checks the yield values agree, and trains the
same model on the SAME rows using each dataset's features in turn. The only
thing that differs is which dataset supplied the feature values.

Model and features are the archived pipeline's, so the comparison is not
flattered by later modelling changes.

WHAT IT DOES NOT SHOW
---------------------
It does not attribute the difference to any single change. Crop masking, soil
product and depth, season definition and spatial scale all differ between the
two datasets. Isolating them requires an ablation that rebuilds one input at a
time with rows held fixed.

Pooled scores are reported alongside a per-year breakdown, because a near-zero
pooled difference can hide large opposing year-level changes.

Run from src/:  python -X utf8 paired_rerun.py     (~3 min)
Writes ../results_comparison/ and touches nothing else.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Features and hyperparameters as used in the archived 2025 pipeline
# (archive/src/ml.py), so the comparison is not flattered by later changes.
CLIMATE_FEATURES = ["TMIN", "TMAX", "PRCP", "VPD", "ETO", "SRAD"]
VEG_FEATURES = ["evi_min_year", "evi_max_year", "ndvi_min_year", "ndvi_max_year"]
SOIL_FEATURES = ["clay_mean", "ph_mean", "soc_mean", "bdod_mean"]
FEATURES = CLIMATE_FEATURES + VEG_FEATURES + SOIL_FEATURES

RF_PARAMS = {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 5,
             "max_features": 0.5, "n_jobs": -1}
SEEDS = (25, 0, 1, 2, 3)          # 25 is the poster's; the rest bound seed noise

MIN_YEARS = 10
TEST_FRACTION = 0.20
MIN_TEST_YEARS = 2

OLD_MERGED = Path("../archive/data/processed/merged.csv")
NEW_MERGED = Path("../data/processed/merged.csv")
OUT_DIR = Path("../results_comparison")

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    for f in (OLD_MERGED, NEW_MERGED):
        if not f.exists():
            raise FileNotFoundError(f"{f} not found. Run this from src/.")

    old = pd.read_csv(OLD_MERGED)
    new = pd.read_csv(NEW_MERGED)
    for d in (old, new):
        d["county_fips"] = d["county_fips"].astype(str).str.zfill(5)

    missing = [c for c in FEATURES if c not in old.columns]
    if missing:
        raise KeyError(f"poster-era table is missing features: {missing}")

    key = ["crop", "county_fips", "year"]
    old = old.dropna(subset=FEATURES + ["yield_value"])
    new = new.dropna(subset=FEATURES + ["yield_value"])
    m = old[key + FEATURES + ["yield_value"]].merge(
        new[key + FEATURES + ["yield_value"]], on=key, suffixes=("_old", "_new"))

    dy = (m["yield_value_old"] - m["yield_value_new"]).abs().max()
    if dy > 1e-6:
        raise ValueError(
            f"yield values disagree on shared rows (max |diff| = {dy}); "
            "the two tables are not describing the same observations")

    log.info("=" * 74)
    log.info("PAIRED RE-RUN - poster-era vs current data on IDENTICAL rows")
    log.info("=" * 74)
    log.info(f"  {len(m):,} shared county-years | yields agree exactly")
    log.info(f"  {len(SEEDS)} seeds per fit; the poster used seed 25 alone\n")
    log.info(f"  {'crop':<24}{'train':>8}{'test':>7}{'poster':>9}"
             f"{'current':>9}{'delta':>9}{'seed sd':>9}")

    rows, year_sse = [], {}
    for crop in sorted(m["crop"].unique()):
        c = m[m["crop"] == crop]
        years = sorted(c["year"].unique())
        if len(years) < MIN_YEARS:
            continue
        n_test = max(MIN_TEST_YEARS, int(np.ceil(len(years) * TEST_FRACTION)))
        n_test = min(n_test, len(years) - 3)
        test_years = years[-n_test:]
        tr = c[~c["year"].isin(test_years)]
        te = c[c["year"].isin(test_years)]
        if len(tr) < 100 or len(te) < 30:
            continue

        row = {"crop": crop, "n_train": len(tr), "n_test": len(te),
               "train_years": f"{years[0]}-{years[-n_test - 1]}",
               "test_years": f"{test_years[0]}-{test_years[-1]}"}
        for tag in ("old", "new"):
            cols = [f + f"_{tag}" for f in FEATURES]
            scores = []
            for seed in SEEDS:
                model = RandomForestRegressor(random_state=seed, **RF_PARAMS)
                model.fit(tr[cols], tr["yield_value_new"])
                scores.append(r2_score(te["yield_value_new"],
                                       model.predict(te[cols])))
            row[f"r2_{tag}"] = round(float(np.mean(scores)), 4)
            row[f"sd_{tag}"] = round(float(np.std(scores)), 4)
        row["delta"] = round(row["r2_new"] - row["r2_old"], 4)
        rows.append(row)

        # Per-year squared error. A near-zero pooled delta can hide large
        # opposing year-level changes -- oats is exactly that case.
        for tag in ("old", "new"):
            cols = [f + f"_{tag}" for f in FEATURES]
            per_seed = []
            for seed in SEEDS:
                mdl = RandomForestRegressor(random_state=seed, **RF_PARAMS)
                mdl.fit(tr[cols], tr["yield_value_new"])
                e2 = (te["yield_value_new"].to_numpy() - mdl.predict(te[cols])) ** 2
                per_seed.append(pd.Series(e2, index=te["year"].to_numpy())
                                .groupby(level=0).sum())
            year_sse[(crop, tag)] = pd.concat(per_seed, axis=1).mean(axis=1)

        log.info(f"  {crop:<24}{len(tr):>8,}{len(te):>7,}{row['r2_old']:>9.4f}"
                 f"{row['r2_new']:>9.4f}{row['delta']:>+9.4f}"
                 f"{max(row['sd_old'], row['sd_new']):>9.4f}")

    out = pd.DataFrame(rows)
    OUT_DIR.mkdir(exist_ok=True)
    out.to_csv(OUT_DIR / "paired_rerun.csv", index=False)

    # ---- per-year breakdown -------------------------------------------
    yrows = []
    for (crop, tag), ser in year_sse.items():
        for y, v in ser.items():
            yrows.append({"crop": crop, "year": int(y), "dataset": tag,
                          "sse": round(float(v), 1)})
    ydf = (pd.DataFrame(yrows)
             .pivot_table(index=["crop", "year"], columns="dataset", values="sse")
             .reset_index())
    ydf["sse_change"] = (ydf["new"] - ydf["old"]).round(1)
    counts = m.groupby(["crop", "year"]).size().rename("n").reset_index()
    ydf = ydf.merge(counts, on=["crop", "year"], how="left")
    ydf.to_csv(OUT_DIR / "paired_rerun_by_year.csv", index=False)

    log.info("\n  PER-YEAR squared error (pooled deltas can hide this):")
    log.info(f"  {'crop':<24}{'year':>6}{'n':>6}{'SSE old':>12}{'SSE new':>12}{'change':>12}")
    for _, r in ydf.iterrows():
        log.info(f"  {r['crop']:<24}{int(r['year']):>6}{int(r['n']):>6}"
                 f"{r['old']:>12.1f}{r['new']:>12.1f}{r['sse_change']:>+12.1f}")

    worst_sd = out[["sd_old", "sd_new"]].to_numpy().max()
    log.info(f"\n  largest seed sd across all fits: {worst_sd:.4f}")
    log.info("  READ AS: the current dataset predicts these same county-years")
    log.info("  better or equally well. Which change is responsible is NOT")
    log.info("  isolated here - that needs an ablation with rows held fixed.")
    log.info(f"\n  Written to {OUT_DIR / 'paired_rerun.csv'}")


if __name__ == "__main__":
    main()
