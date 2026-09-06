"""
Which data source earns the prediction improvement?

The rebuilt pipeline predicts the same county-years better than the archived one
(paired_rerun.py). Masking, soil product and depth, season definition and
spatial scale all changed at once, so that result does not say which change was
responsible.

This isolates by SOURCE. Rows, model, hyperparameters and seeds are held fixed;
one source's feature values are reverted to the archived version at a time:

    all_new       every source rebuilt          (reference)
    climate_old   climate reverted, rest new
    veg_old       vegetation reverted, rest new
    soil_old      soil reverted, rest new
    all_old       everything reverted

It does NOT separate changes made within one source. Vegetation changed both
its crop mask and its reduction scale, so a large veg_old effect says
"vegetation processing mattered", not "the crop mask mattered". Splitting those
requires rebuilding inputs through Earth Engine one option at a time.

Design, metric and decision rule are fixed in ABLATION_PREREGISTRATION.md and
were set before this ran. In brief: the metric is the RMSE ratio
Q = RMSE(ablated) / RMSE(intact), the primary equivalence margin is
[0.95, 1.05], and an interval spanning the margin is inconclusive rather than
evidence of equivalence.

Run from src/:  python -X utf8 ablation.py      (~10 min)
Writes ../results_comparison/ablation*.csv
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

CLIMATE_FEATURES = ["TMIN", "TMAX", "PRCP", "VPD", "ETO", "SRAD"]
VEG_FEATURES = ["evi_min_year", "evi_max_year", "ndvi_min_year", "ndvi_max_year"]
SOIL_FEATURES = ["clay_mean", "ph_mean", "soc_mean", "bdod_mean"]
FEATURES = CLIMATE_FEATURES + VEG_FEATURES + SOIL_FEATURES

SOURCES = {"climate": CLIMATE_FEATURES, "veg": VEG_FEATURES, "soil": SOIL_FEATURES}

RF_PARAMS = {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 5,
             "max_features": 0.5, "n_jobs": -1}
SEEDS = (25, 0, 1, 2, 3)

MIN_YEARS = 10
TEST_FRACTION = 0.20
MIN_TEST_YEARS = 2

# Pre-registered. Primary first; the others are reported but never promoted.
MARGIN = 0.05
SECONDARY_MARGINS = (0.02, 0.10)
N_BOOT = 2000
BOOT_SEED = 12345

OLD_MERGED = Path("../archive/data/processed/merged.csv")
NEW_MERGED = Path("../data/processed/merged.csv")
OUT_DIR = Path("../results_comparison")

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def build_pairs() -> pd.DataFrame:
    old = pd.read_csv(OLD_MERGED)
    new = pd.read_csv(NEW_MERGED)
    for d in (old, new):
        d["county_fips"] = d["county_fips"].astype(str).str.zfill(5)
    key = ["crop", "county_fips", "year"]
    old = old.dropna(subset=FEATURES + ["yield_value"])
    new = new.dropna(subset=FEATURES + ["yield_value"])
    m = old[key + FEATURES + ["yield_value"]].merge(
        new[key + FEATURES + ["yield_value"]], on=key, suffixes=("_old", "_new"))
    dy = (m["yield_value_old"] - m["yield_value_new"]).abs().max()
    if dy > 1e-6:
        raise ValueError(f"yields disagree on shared rows (max {dy})")
    return m


def arm_columns(arm: str) -> list[str]:
    """Feature columns for one arm: '_new' everywhere except the reverted source."""
    if arm == "all_new":
        reverted = set()
    elif arm == "all_old":
        reverted = set(FEATURES)
    else:
        reverted = set(SOURCES[arm.removesuffix("_old")])
    return [f + ("_old" if f in reverted else "_new") for f in FEATURES]


def split(c: pd.DataFrame):
    years = sorted(c["year"].unique())
    if len(years) < MIN_YEARS:
        return None
    n_test = max(MIN_TEST_YEARS, int(np.ceil(len(years) * TEST_FRACTION)))
    n_test = min(n_test, len(years) - 3)
    test_years = years[-n_test:]
    tr, te = c[~c["year"].isin(test_years)], c[c["year"].isin(test_years)]
    if len(tr) < 100 or len(te) < 30:
        return None
    return tr, te


def squared_errors(tr, te, cols) -> np.ndarray:
    """Per-observation squared error, averaged over the seed set."""
    y_tr, y_te = tr["yield_value_new"].to_numpy(), te["yield_value_new"].to_numpy()
    acc = np.zeros(len(te))
    for seed in SEEDS:
        model = RandomForestRegressor(random_state=seed, **RF_PARAMS)
        model.fit(tr[cols], y_tr)
        acc += (y_te - model.predict(te[cols])) ** 2
    return acc / len(SEEDS)


def verdict(lo: float, hi: float, margin: float) -> str:
    if lo >= 1 - margin and hi <= 1 + margin:
        return "equivalent"
    if lo > 1 + margin:
        return "reverting HURTS (change matters)"
    if hi < 1 - margin:
        return "reverting HELPS (change hurt)"
    return "inconclusive"


def main() -> None:
    m = build_pairs()
    OUT_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(BOOT_SEED)

    log.info("=" * 78)
    log.info("SOURCE ABLATION - which data source earns the improvement?")
    log.info("=" * 78)
    log.info(f"  {len(m):,} shared county-years | metric Q = RMSE(ablated)/RMSE(all_new)")
    log.info(f"  pre-registered margin [{1-MARGIN:.2f}, {1+MARGIN:.2f}], "
             f"{N_BOOT:,} county-clustered bootstrap resamples\n")

    arms = ["all_new", "climate_old", "veg_old", "soil_old", "all_old"]
    rows, year_rows = [], []

    for crop in sorted(m["crop"].unique()):
        s = split(m[m["crop"] == crop])
        if s is None:
            continue
        tr, te = s
        se = {a: squared_errors(tr, te, arm_columns(a)) for a in arms}
        counties = te["county_fips"].to_numpy()
        uniq = np.unique(counties)
        idx_by_county = {c: np.where(counties == c)[0] for c in uniq}

        log.info(f"  {crop}   train {len(tr):,}  test {len(te):,}  "
                 f"({len(uniq)} counties)")
        log.info(f"    {'arm':<14}{'RMSE':>8}{'Q':>8}{'95% CI':>18}   verdict")

        # One resample index set, shared across arms, so Q is properly paired.
        boots = [np.concatenate([idx_by_county[c]
                                 for c in rng.choice(uniq, len(uniq), replace=True)])
                 for _ in range(N_BOOT)]

        ref = se["all_new"]
        for a in arms:
            rmse = float(np.sqrt(se[a].mean()))
            q = rmse / float(np.sqrt(ref.mean()))
            if a == "all_new":
                log.info(f"    {a:<14}{rmse:>8.2f}{q:>8.3f}{'reference':>18}")
                rows.append({"crop": crop, "arm": a, "rmse": round(rmse, 3),
                             "q": 1.0, "q_lo": np.nan, "q_hi": np.nan,
                             "verdict": "reference", "n_test": len(te)})
                continue
            qs = np.array([np.sqrt(se[a][b].mean() / ref[b].mean()) for b in boots])
            lo, hi = np.percentile(qs, [2.5, 97.5])
            v = verdict(lo, hi, MARGIN)
            log.info(f"    {a:<14}{rmse:>8.2f}{q:>8.3f}"
                     f"{f'[{lo:.3f}, {hi:.3f}]':>18}   {v}")
            row = {"crop": crop, "arm": a, "rmse": round(rmse, 3), "q": round(q, 4),
                   "q_lo": round(float(lo), 4), "q_hi": round(float(hi), 4),
                   "verdict": v, "n_test": len(te)}
            years = te["year"].to_numpy()
            for y in sorted(np.unique(years)):
                k = years == y
                year_rows.append({
                    "crop": crop, "arm": a, "year": int(y), "n": int(k.sum()),
                    "q": round(float(np.sqrt(se[a][k].mean() / ref[k].mean())), 4)})
            for sm in SECONDARY_MARGINS:
                row[f"verdict_{int(sm*100)}pct"] = verdict(lo, hi, sm)
            rows.append(row)
        log.info("")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "ablation.csv", index=False)

    ydf = pd.DataFrame(year_rows)
    ydf.to_csv(OUT_DIR / "ablation_by_year.csv", index=False)
    log.info("  PER-YEAR Q (pre-registered: a pooled Q can hide opposing years)")
    log.info(f"  {'crop':<24}{'arm':<14}" + "".join(f"{y:>9}" for y in sorted(ydf.year.unique())))
    for (crop, arm), g in ydf.groupby(["crop", "arm"], sort=False):
        cells = "".join(f"{g[g.year == y].q.iloc[0]:>9.3f}" if (g.year == y).any() else f"{'-':>9}"
                        for y in sorted(ydf.year.unique()))
        log.info(f"  {crop:<24}{arm:<14}{cells}")
    log.info("")

    log.info("=" * 78)
    log.info("  Q > 1 means reverting that source made prediction WORSE, so the")
    log.info("  rebuild of that source is what earned the improvement.")
    log.info("  Secondary margins (2%, 10%) are in the CSV and do not override")
    log.info("  the primary 5% verdict.")
    log.info(f"\n  Written to {OUT_DIR / 'ablation.csv'}")


if __name__ == "__main__":
    main()
