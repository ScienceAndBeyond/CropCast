"""
irrigation_contrast.py - does irrigation decouple yield from weather?

THE DESIGN
----------
NASS publishes county corn yield separately for IRRIGATED and NON-IRRIGATED
land. Where both appear for the SAME county in the SAME year, the two
observations share a county, a soil map and a growing season - they differ in
management. That removes the between-county variance that made the AGU poster's
"29 C heat threshold" a geography contrast.

The test: regress each practice's WITHIN-COUNTY yield anomaly on the same
within-county weather anomaly. If irrigation decouples yield from weather, the
irrigated slope is SMALLER IN MAGNITUDE than the rainfed one.

WHAT THIS IS, AND IS NOT
------------------------
These are WITHIN-COUNTY ASSOCIATIONS, not isolated causal irrigation effects.
County-demeaning removes each county's level. It does NOT remove:
  - year shocks unrelated to weather (prices, policy) that happen to correlate
    with weather over an 11-year window;
  - the technology trend - which is why a linear-detrended sensitivity is
    reported beside the raw within-county estimate;
  - changes in which fields are irrigated from year to year.
Year fixed effects are deliberately NOT used: within a state, weather in a given
year is shared across counties, so year dummies would absorb most of the very
signal being measured. The reported weather R2 is an IN-SAMPLE fit, not
predictive validation.

Why not a Random Forest per stratum: NASS stopped publishing the county split
after 2018 and coverage thins throughout (105 counties in 2008 to 51 in 2018),
so a temporal holdout would land in the sparsest years. The paired design needs
no train/test split.

WHAT THIS CANNOT SUPPORT
------------------------
- 2008-2018 only; cannot extend to the main study's 2025 endpoint.
- Effectively Nebraska and Kansas. Corn only.
- Irrigation is a management choice, not a randomised treatment.

Outputs (RESULTS_DIR):
  irrigation_contrast.csv             slopes, scale-free companions and
                                      county-clustered bootstrap CIs, for
                                      trend=none and trend=linear
  irrigation_variance_explained.csv   in-sample weather R2 per practice
  irrigation_pairs.csv                the paired county-years used

Usage:
    python download_yield.py --practice-mode split --reuse-raw
    python irrigation_contrast.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from config import DATA_PATH, STUDY_STATES, yield_filename, results_dirname, logging
from utils import save_df
from ml import REQUIRE_COVERAGE

PROCESSED_DIR = DATA_PATH / "processed"
CLIMATE_FILE = PROCESSED_DIR / "climate.csv"
SPLIT_YIELD_FILE = PROCESSED_DIR / yield_filename("split")
RESULTS_DIR = Path(results_dirname("split"))

CROP = "CORN"
PRACTICES = ["irrigated", "non_irrigated"]
WEATHER = ["PRCP", "EDD_TMAX", "VPD", "TMAX"]

N_BOOTSTRAP = 2000
CI_LEVEL = 95
RANDOM_SEED = 25
MIN_YEARS_PER_COUNTY = 4

# Same rule ml.py applies: a county-year built on a partial climate season is
# not the same quantity as one built on a full season.
MIN_CLIMATE_COVERAGE = 1.0

# Column suffix -> label for the two estimates reported side by side.
TREND_MODES = {"_a": "none", "_dt": "linear"}


def load_pairs() -> pd.DataFrame:
    """County-years where BOTH practices are reported, joined to climate."""
    if not SPLIT_YIELD_FILE.exists():
        raise FileNotFoundError(
            f"{SPLIT_YIELD_FILE} not found. Build it with:\n"
            f"  python download_yield.py --practice-mode split --reuse-raw"
        )

    y = pd.read_csv(SPLIT_YIELD_FILE, dtype={"county_fips": str})
    y = y[(y["commodity_desc"] == CROP) & y["state_alpha"].isin(STUDY_STATES)]

    wide = y.pivot_table(
        index=["state_alpha", "county_fips", "year"],
        columns="irrigation", values="yield_value", aggfunc="first",
    )
    missing = [p for p in PRACTICES if p not in wide.columns]
    if missing:
        raise KeyError(f"split yield table has no {missing} rows for {CROP}")
    wide = wide.dropna(subset=PRACTICES).reset_index()

    clim = pd.read_csv(CLIMATE_FILE, dtype={"county_fips": str})
    need = [c for c in WEATHER if c not in clim.columns]
    if need:
        raise KeyError(f"climate.csv missing {need}")
    cols = ["county_fips", "year"] + WEATHER
    if "COVERAGE_MIN" in clim.columns:
        cols.append("COVERAGE_MIN")
    df = wide.merge(clim[cols], on=["county_fips", "year"])

    # Enforce climate completeness and drop missing weather BEFORE any slope is
    # computed. A NaN regressor would otherwise be handled differently here than
    # in ml.py, so the two analyses would silently use different row sets.
    n0 = len(df)
    if "COVERAGE_MIN" in df.columns:
        df = df[df["COVERAGE_MIN"].fillna(0) >= MIN_CLIMATE_COVERAGE]
    else:
        message = ("COVERAGE_MIN is absent from the climate data, so season completeness "
                   "cannot be enforced. Rebuild with:  python download_climate.py --aggregate-only")
        if REQUIRE_COVERAGE:
            raise KeyError(message)
        logging.warning(message)
    df = df.dropna(subset=WEATHER + PRACTICES)
    if len(df) < n0:
        logging.info(f"Dropped {n0 - len(df):,} pairs for incomplete or missing weather")

    n_years = df.groupby("county_fips")["year"].transform("nunique")
    dropped = int((n_years < MIN_YEARS_PER_COUNTY).sum())
    df = df[n_years >= MIN_YEARS_PER_COUNTY].copy()
    if dropped:
        logging.info(
            f"Dropped {dropped:,} pairs in counties with < {MIN_YEARS_PER_COUNTY} paired years"
        )

    # Within-county anomalies: county fixed effect removed from both sides.
    for col in PRACTICES + WEATHER:
        df[col + "_a"] = df[col] - df.groupby("county_fips")[col].transform("mean")

    # Sensitivity check: additionally remove a pooled linear year trend from
    # every anomaly, so a technology trend cannot masquerade as weather response.
    yr = df["year"].to_numpy(dtype=float)
    # Global centring attenuates trends on unbalanced panels and reintroduces
    # county means. Use the same within-county estimator as evaluate.py.
    yrm = df.groupby("county_fips")["year"].transform("mean").to_numpy(dtype=float)
    yr_c = yr - yrm
    denom = float(np.sum(yr_c ** 2))
    for col in PRACTICES + WEATHER:
        a = df[col + "_a"].to_numpy(dtype=float)
        slope = float(np.sum(yr_c * a) / denom) if denom > 0 else 0.0
        df[col + "_dt"] = a - slope * yr_c

    df["gap"] = df["irrigated"] - df["non_irrigated"]
    return df


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    """OLS slope through the origin - both inputs are already centred."""
    d = float(np.sum(x * x))
    return float(np.sum(x * y) / d) if d > 0 else np.nan


def contrast(df: pd.DataFrame, suffix: str, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Slopes per practice with county-clustered bootstrap CIs.

    THE DECOUPLING TEST IS ON MAGNITUDE:
        p = fraction of resamples where |irrigated slope| >= |rainfed slope|.

    An earlier version compared the SIGN of the slope difference against the
    sign of the rainfed slope. That reported "decoupled" (p = 0) for a rainfed
    slope of -1 against an irrigated slope of +10, where irrigated sensitivity
    was ten times LARGER. Direction is not magnitude.

    Resamples are shared between the two practices, so the comparison is paired.
    Clustering by county matters: a county contributes up to 11 correlated
    paired years, and resampling rows would give intervals far too narrow.
    """
    rng = np.random.default_rng(seed)
    counties = df["county_fips"].unique()
    cf = df["county_fips"].to_numpy()
    idx_by_county = {c: np.flatnonzero(cf == c) for c in counties}
    lo_q, hi_q = (100 - CI_LEVEL) / 2, 100 - (100 - CI_LEVEL) / 2
    rows: List[Dict] = []

    if suffix == "_dt":
        # Whole-county resampling preserves each county mean, including when a
        # county is drawn repeatedly; only the pooled trend must be re-fitted.
        yr_c = (df["year"] - df.groupby("county_fips")["year"].transform("mean")).to_numpy(dtype=float)
        anomalies = {c: df[c + "_a"].to_numpy(dtype=float) for c in PRACTICES + WEATHER}

    for w in WEATHER:
        xw = df[w + suffix].to_numpy()
        ys = {p: df[p + suffix].to_numpy() for p in PRACTICES}
        obs = {p: _slope(xw, ys[p]) for p in PRACTICES}

        draws = {p: [] for p in PRACTICES}
        for _ in range(N_BOOTSTRAP):
            pick = rng.choice(counties, size=len(counties), replace=True)
            idx = np.concatenate([idx_by_county[c] for c in pick])
            xb = xw[idx]
            yb = {p: ys[p][idx] for p in PRACTICES}
            if suffix == "_dt":
                # Refit on this paired county resample to include uncertainty
                # in the estimated trend; never reuse the original _dt values.
                yr_b = yr_c[idx]
                denom_b = float(np.sum(yr_b ** 2))
                adjusted = {}
                for col in [w] + PRACTICES:
                    a = anomalies[col][idx]
                    slope_b = float(np.sum(yr_b * a) / denom_b) if denom_b > 0 else 0.0
                    adjusted[col] = a - slope_b * yr_b
                xb = adjusted[w]
                yb = {p: adjusted[p] for p in PRACTICES}
            if np.sum(xb * xb) <= 0:
                continue
            for p in PRACTICES:
                draws[p].append(_slope(xb, yb[p]))

        irr = np.asarray(draws["irrigated"])
        rain = np.asarray(draws["non_irrigated"])

        def ci(v):
            if len(v) <= 50:
                return (None, None)
            return (round(float(np.percentile(v, lo_q)), 3),
                    round(float(np.percentile(v, hi_q)), 3))

        # Scale-free companions. Irrigated and rainfed yields differ in mean
        # (190 vs 109 BU/AC) and spread (sd 15.3 vs 29.5), so an absolute slope
        # is mechanically larger for the more variable series. Correlation is
        # unit-free; pct is the slope as a share of that practice's own mean.
        corr = {p: float(np.corrcoef(xw, ys[p])[0, 1]) for p in PRACTICES}
        pct = {p: obs[p] / float(df[p].mean()) * 100 for p in PRACTICES}

        row = {
            "trend": TREND_MODES[suffix],
            "weather": w,
            "slope_rainfed": round(obs["non_irrigated"], 3),
            "slope_irrigated": round(obs["irrigated"], 3),
            "corr_rainfed": round(corr["non_irrigated"], 3),
            "corr_irrigated": round(corr["irrigated"], 3),
            "pct_of_mean_rainfed": round(pct["non_irrigated"], 3),
            "pct_of_mean_irrigated": round(pct["irrigated"], 3),
            "ratio_abs_slope": (round(abs(obs["irrigated"]) / abs(obs["non_irrigated"]), 3)
                                if obs["non_irrigated"] else None),
            "ratio_abs_corr": (round(abs(corr["irrigated"]) / abs(corr["non_irrigated"]), 3)
                               if corr["non_irrigated"] else None),
            "ratio_abs_pct": (round(abs(pct["irrigated"]) / abs(pct["non_irrigated"]), 3)
                              if pct["non_irrigated"] else None),
        }
        row["rainfed_lo"], row["rainfed_hi"] = ci(rain)
        row["irrigated_lo"], row["irrigated_hi"] = ci(irr)
        row["abs_diff_irr_minus_rain"] = round(
            abs(obs["irrigated"]) - abs(obs["non_irrigated"]), 3
        )
        row["abs_diff_lo"], row["abs_diff_hi"] = ci(np.abs(irr) - np.abs(rain))
        # One-sided on MAGNITUDE. Small = the irrigated response is reliably
        # weaker, i.e. decoupling.
        row["p_no_decoupling"] = (round(float(np.mean(np.abs(irr) >= np.abs(rain))), 4)
                                  if len(irr) else None)
        rows.append(row)

    return pd.DataFrame(rows)


def variance_explained(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """IN-SAMPLE share of each practice's yield anomaly fitted by weather jointly."""
    X = np.column_stack([df[w + suffix].to_numpy() for w in WEATHER] + [np.ones(len(df))])
    out = []
    for p in PRACTICES:
        y = df[p + suffix].to_numpy()
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        out.append({
            "trend": TREND_MODES[suffix],
            "practice": p,
            "r2_weather_in_sample": (round(1 - float(np.sum(resid ** 2)) / ss_tot, 3)
                                     if ss_tot else np.nan),
            "sd_anomaly": round(float(np.std(y)), 2),
            "mean_yield": round(float(df[p].mean()), 1),
        })
    return pd.DataFrame(out)


def run() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_pairs()

    logging.info("=" * 70)
    logging.info("IRRIGATION CONTRAST - paired within county-year (associational)")
    logging.info("=" * 70)
    logging.info(f"  {len(df):,} pairs | {df.county_fips.nunique()} counties | "
                 f"{df.year.nunique()} years ({df.year.min()}-{df.year.max()})")
    logging.info(f"  states: {df.groupby('state_alpha').size().to_dict()}")
    logging.info(f"  mean irrigated-minus-rainfed gap: {df.gap.mean():+.1f} BU/AC "
                 f"(sd {df.gap.std():.1f}, {100 * (df.gap < 0).mean():.1f}% negative)")

    ve = pd.concat([variance_explained(df, s) for s in TREND_MODES], ignore_index=True)
    cs = pd.concat([contrast(df, s) for s in TREND_MODES], ignore_index=True)

    for label in TREND_MODES.values():
        logging.info(f"\n  --- trend={label} ---")
        for _, r in ve[ve.trend == label].iterrows():
            logging.info(f"    {r['practice']:14s} in-sample weather R2="
                         f"{r['r2_weather_in_sample']:.3f}  sd(anomaly)={r['sd_anomaly']:.1f}")
        for _, r in cs[cs.trend == label].iterrows():
            logging.info(
                f"    {r['weather']:9s} rainfed {r['slope_rainfed']:+8.2f} "
                f"[{r['rainfed_lo']:+.2f},{r['rainfed_hi']:+.2f}]  "
                f"irrigated {r['slope_irrigated']:+7.2f} "
                f"[{r['irrigated_lo']:+.2f},{r['irrigated_hi']:+.2f}]  "
                f"|ratio| slope {r['ratio_abs_slope']} corr {r['ratio_abs_corr']} "
                f"%mean {r['ratio_abs_pct']}  p(|irr|>=|rain|)={r['p_no_decoupling']}"
            )

    save_df(cs, RESULTS_DIR / "irrigation_contrast.csv")
    save_df(ve, RESULTS_DIR / "irrigation_variance_explained.csv")
    save_df(df, RESULTS_DIR / "irrigation_pairs.csv")
    logging.info(f"\n  Written to {RESULTS_DIR}")
    logging.info("  READ AS: within-county associations, in-sample. 2008-2018, corn, "
                 "effectively NE+KS; irrigation is a management choice, not a treatment.")


def main():
    ap = argparse.ArgumentParser(
        description="Paired irrigated-vs-rainfed weather sensitivity (within county-year)"
    )
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args()
    global RESULTS_DIR
    if args.results_dir:
        RESULTS_DIR = Path(args.results_dir)
    run()


if __name__ == "__main__":
    main()
