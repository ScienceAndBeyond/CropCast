"""
Compare the two vegetation changes: cropland masking and reduction scale.

The source ablation shows that the rebuilt vegetation table accounts for most
of the prediction improvement. This script separates the two main vegetation
changes by fitting the same model on the same rows while swapping only the
vegetation columns.

Arms:
    production        masked, native MODIS scale, current QA/season rules
    masked_1km        masked, 1 km scale
    unmasked_native   unmasked, native MODIS scale
    unmasked_1km      unmasked, 1 km scale
    archived          historical table, included as context only

The historical table also differs in QA filtering and season handling, so it is
not used as the clean mask/scale comparison. The metric is Q = RMSE(arm) /
RMSE(production), with [0.95, 1.05] as the primary practical-equivalence range.

Requires the three variant downloads first:
    python -X utf8 download_vegetation.py --study --scale 1000 --variant masked_1km
    python -X utf8 download_vegetation.py --study --no-crop-mask --variant unmasked_native
    python -X utf8 download_vegetation.py --study --no-crop-mask --scale 1000 --variant unmasked_1km

Run from src/:  python -X utf8 veg_ablation.py
Writes ../results_comparison/veg_ablation*.csv
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

RF_PARAMS = {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 5,
             "max_features": 0.5, "n_jobs": -1}
SEEDS = (25, 0, 1, 2, 3)
MIN_YEARS, TEST_FRACTION, MIN_TEST_YEARS = 10, 0.20, 2
MARGIN, SECONDARY_MARGINS = 0.05, (0.02, 0.10)
N_BOOT, BOOT_SEED = 2000, 12345

NEW_MERGED = Path("../data/processed/merged.csv")
OLD_MERGED = Path("../archive/data/processed/merged.csv")
VARIANTS = {
    "masked_1km":      Path("../data/processed/vegetation_masked_1km.csv"),
    "unmasked_native": Path("../data/processed/vegetation_unmasked_native.csv"),
    "unmasked_1km":    Path("../data/processed/vegetation_unmasked_1km.csv"),
}
OUT_DIR = Path("../results_comparison")

# Arm -> where its vegetation columns come from. The historical table is kept
# as context because it also differs in QA filtering and season handling.
ARMS = {
    "production":      None,               # masked, native, current QA/season   (reference)
    "masked_1km":      "masked_1km",       # masked, 1 km,   current QA/season   -> scale only
    "unmasked_native": "unmasked_native",  # unmasked, native, current QA/season -> mask only
    "unmasked_1km":    "unmasked_1km",     # unmasked, 1 km, current QA/season   -> mask+scale, clean
    "archived":        "old",              # historical table, not an isolating arm
}

# Arm metadata for the printed table.
META = {
    "production":      ("on",  "926.63 m (native)"),
    "masked_1km":       ("on",  "1000 m"),
    "unmasked_native":  ("off", "926.63 m (native)"),
    "unmasked_1km":     ("off", "1000 m"),
    "archived":         ("off", "1000 m*"),
}

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def build() -> pd.DataFrame:
    new = pd.read_csv(NEW_MERGED)
    old = pd.read_csv(OLD_MERGED)
    for d in (new, old):
        d["county_fips"] = d["county_fips"].astype(str).str.zfill(5)
    key = ["crop", "county_fips", "year"]

    new = new.dropna(subset=FEATURES + ["yield_value"])
    old = old.dropna(subset=VEG_FEATURES)

    df = new[key + FEATURES + ["yield_value"]].copy()
    df = df.rename(columns={v: f"{v}__production" for v in VEG_FEATURES})

    df = df.merge(old[key + VEG_FEATURES].rename(
        columns={v: f"{v}__old" for v in VEG_FEATURES}), on=key, how="inner")

    for name, path in VARIANTS.items():
        if not path.exists():
            raise FileNotFoundError(f"{path} missing. Run the variant download first.")
        v = pd.read_csv(path)
        v["county_fips"] = v["county_fips"].astype(str).str.zfill(5)
        v = v.dropna(subset=VEG_FEATURES)
        df = df.merge(
            v[["county_fips", "year"] + VEG_FEATURES].rename(
                columns={c: f"{c}__{name}" for c in VEG_FEATURES}),
            on=["county_fips", "year"], how="inner")
    return df


def cols_for(arm: str) -> list[str]:
    """
    Feature columns for an arm, ordered as climate, vegetation, soil.

    Must match ablation.py's column order exactly. The model picks columns
    partly by position, so a different order gives different results even
    with the same data and the same seed.
    """
    src = ARMS[arm] or "production"
    return (CLIMATE_FEATURES + [f"{v}__{src}" for v in VEG_FEATURES]
            + SOIL_FEATURES)


def split(c: pd.DataFrame):
    years = sorted(c["year"].unique())
    if len(years) < MIN_YEARS:
        return None
    n_test = min(max(MIN_TEST_YEARS, int(np.ceil(len(years) * TEST_FRACTION))),
                 len(years) - 3)
    ty = years[-n_test:]
    tr, te = c[~c["year"].isin(ty)], c[c["year"].isin(ty)]
    return None if len(tr) < 100 or len(te) < 30 else (tr, te)


def squared_errors(tr, te, cols) -> np.ndarray:
    y_tr, y_te = tr["yield_value"].to_numpy(), te["yield_value"].to_numpy()
    acc = np.zeros(len(te))
    for seed in SEEDS:
        m = RandomForestRegressor(random_state=seed, **RF_PARAMS)
        m.fit(tr[cols], y_tr)
        acc += (y_te - m.predict(te[cols])) ** 2
    return acc / len(SEEDS)


def verdict(lo: float, hi: float, margin: float) -> str:
    if lo >= 1 - margin and hi <= 1 + margin:
        return "equivalent"
    if lo > 1 + margin:
        return "materially worse"
    if hi < 1 - margin:
        return "materially better"
    return "inconclusive"


STATE_NAMES = {"19": "IA", "17": "IL", "31": "NE", "27": "MN"}


def per_state(crop: str, te: pd.DataFrame, se: dict) -> list[dict]:
    """Per-state Q for checking whether pooled results hide reversals.

    The pooled result can hide state-level differences, including a sign change.
    """
    state_fips = te["county_fips"].str[:2].to_numpy()
    ref = se["production"]
    rows = []
    for fips, name in STATE_NAMES.items():
        m = state_fips == fips
        if m.sum() < 10:
            continue
        for arm in ARMS:
            if arm == "production":
                continue
            q = float(np.sqrt(se[arm][m].mean() / ref[m].mean()))
            rows.append({"crop": crop, "state": name, "arm": arm,
                         "n": int(m.sum()), "q": round(q, 4)})
    return rows


def main() -> None:
    df = build()
    OUT_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(BOOT_SEED)

    log.info("=" * 82)
    log.info("VEGETATION ABLATION - crop mask vs reduction scale")
    log.info("=" * 82)
    log.info(f"  {len(df):,} county-years present in all five vegetation versions")
    log.info(f"  Q = RMSE(arm)/RMSE(production); margin [{1-MARGIN:.2f}, {1+MARGIN:.2f}]")
    log.info("  'equivalent' means the interval sits INSIDE the margin - this can")
    log.info("  still be a real, statistically detectable effect (CI excluding 1.0),")
    log.info("  just one small enough not to matter at the declared 5% tolerance.\n")

    rows, state_rows = [], []
    for crop in sorted(df["crop"].unique()):
        s = split(df[df["crop"] == crop])
        if s is None:
            continue
        tr, te = s
        se = {a: squared_errors(tr, te, cols_for(a)) for a in ARMS}
        state_rows.extend(per_state(crop, te, se))
        counties = te["county_fips"].to_numpy()
        uniq = np.unique(counties)
        idx = {c: np.where(counties == c)[0] for c in uniq}
        boots = [np.concatenate([idx[c] for c in rng.choice(uniq, len(uniq), replace=True)])
                 for _ in range(N_BOOT)]
        ref = se["production"]

        log.info(f"  {crop}   train {len(tr):,}  test {len(te):,}  ({len(uniq)} counties)")
        log.info(f"    {'arm':<18}{'mask':>6}{'scale':>20}{'RMSE':>8}{'Q':>8}"
                 f"{'95% CI':>18}   verdict")
        for a in ARMS:
            rmse = float(np.sqrt(se[a].mean()))
            q = rmse / float(np.sqrt(ref.mean()))
            mk, sc = META[a]
            if a == "production":
                log.info(f"    {a:<18}{mk:>6}{sc:>20}{rmse:>8.2f}{q:>8.3f}{'reference':>18}")
                rows.append({"crop": crop, "arm": a, "mask": mk, "scale_m": 926.63,
                             "rmse": round(rmse, 3), "q": 1.0, "q_lo": np.nan,
                             "q_hi": np.nan, "verdict": "reference", "n_test": len(te)})
                continue
            qs = np.array([np.sqrt(se[a][b].mean() / ref[b].mean()) for b in boots])
            lo, hi = np.percentile(qs, [2.5, 97.5])
            v = verdict(lo, hi, MARGIN)
            log.info(f"    {a:<18}{mk:>6}{sc:>20}{rmse:>8.2f}{q:>8.3f}"
                     f"{f'[{lo:.3f}, {hi:.3f}]':>18}   {v}")
            row = {"crop": crop, "arm": a, "mask": mk,
                   "scale_m": 1000.0 if "1000" in sc else 926.63,
                   "rmse": round(rmse, 3), "q": round(q, 4),
                   "q_lo": round(float(lo), 4), "q_hi": round(float(hi), 4),
                   "verdict": v, "n_test": len(te)}
            for sm in SECONDARY_MARGINS:
                row[f"verdict_{int(sm*100)}pct"] = verdict(lo, hi, sm)
            rows.append(row)
        log.info("")

    pd.DataFrame(rows).to_csv(OUT_DIR / "veg_ablation.csv", index=False)

    sdf = pd.DataFrame(state_rows)
    sdf.to_csv(OUT_DIR / "veg_ablation_by_state.csv", index=False)
    log.info("  PER-STATE Q (pooled results can hide state-level reversals):")
    for crop in sdf.crop.unique():
        for arm in ["unmasked_native", "unmasked_1km"]:
            g = sdf[(sdf.crop == crop) & (sdf.arm == arm)]
            if g.empty:
                continue
            cells = "  ".join(f"{r.state}(n={r.n}): {r.q:.3f}" for _, r in g.iterrows())
            flag = "  <-- reversal (Q<1 while pooled Q>1)" if (g.q < 1.0).any() else ""
            log.info(f"    {crop:<26}{arm:<18}{cells}{flag}")
    log.info("")

    log.info("=" * 82)
    log.info(f"  SCOPE: this comparison covers only the states/years present in")
    log.info(f"  BOTH the archived and current datasets - 4 states (IA/IL/MN/NE),")
    log.info(f"  test years 2022-2024. It is not a claim about all 11 study states.")
    log.info("  Interpretation: masked_1km isolates the scale (1 km -> 926.63 m),")
    log.info("  unmasked_native isolates the mask, unmasked_1km isolates")
    log.info("  both jointly - all three built through CURRENT QA/season code, so")
    log.info("  they are clean. 'archived' is the real historical data and differs")
    log.info("  by QA filtering and season too, not just mask/scale - read it as")
    log.info("  context, not as a clean isolator. 'Equivalent' means inside the")
    log.info("  margin; the interval can still exclude 1.0 and be a real effect.")
    log.info(f"\n  Written to {OUT_DIR / 'veg_ablation.csv'} and")
    log.info(f"  {OUT_DIR / 'veg_ablation_by_state.csv'}")


if __name__ == "__main__":
    main()
