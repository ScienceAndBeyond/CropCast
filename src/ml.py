"""
ml.py - Crop Yield Prediction Pipeline

Trains Random Forest models with multiple feature sets:
- climate_only: Baseline climate features
- climate_soil: Climate + Soil (which is also used for causal model for interpretation)
- climate_veg: Climate + vegetation indices
- all_features: All features including non-causal features like vegetation index
that may help with prediction not necessarily actual yield

Interpretation (ALE, permutation importance) is run ONLY on the causal model
(climate_soil) since vegetation indices are outcomes, not causes.

HOW TO READ THE SCORES
----------------------
Raw test R2 is NOT the headline number. Soil is static per county (one unique
soil tuple per county in this dataset), so a model given soil can recover county
identity and reproduce a county-mean predictor without learning any agronomy.
Three numbers are therefore reported together for every feature set:

  baseline_county_mean_r2  R2 of predicting each county's TRAINING-mean yield.
                           Uses no climate, soil or satellite data at all.
  test_r2                  Conventional R2. Dominated by between-county variance.
  skill_vs_county_mean     1 - SSE(model)/SSE(county-mean predictor). 0 means
                           exactly as good as the county mean, negative means
                           worse, 1 is perfect. This is what a climate-impacts
                           claim actually rests on. NOTE: an earlier version
                           reported an R2 on residuals under this heading, which
                           is a different quantity - it scored -33.8 for a
                           predictor that tied the baseline exactly.

A feature set that does not clearly exceed baseline_county_mean_r2 has
demonstrated nothing, whatever its test_r2 looks like.

All models are fitted over N_SEEDS random seeds; means and standard deviations
are reported so that between-model gaps can be compared against fitting noise.

Outputs:
- model_performance.csv:    test/skill/spatial R2, RMSE, MAE, county-mean baseline
- feature_importance.csv:   permutation importance on the test set (causal model),
                            with the old impurity value alongside for audit
- category_importance.csv:  Climate/Soil totals (causal model)
- ale_data.csv:             Accumulated Local Effects - PREFERRED over PDP here,
                            because the climate predictors are highly collinear
- pdp_data.csv:             1D partial dependence (retained for comparison only)
- pdp_2d_data.csv:          2D PDP for feature interactions
- feature_collinearity.csv: feature pairs with |r| >= 0.7, the caveat list
- sensitivity_analysis.csv: Sensitivity derived from 1D PDP
- optimal_conditions.csv:   Feature values for top-yielding conditions
                            (descriptive and confounded; NOT causal optima)
- improvement_summary.csv:  delta R2 over climate-only, plus the county-mean
                            baseline and skill_vs_county_mean for context
"""

import re
from typing import Dict, List, Optional
from itertools import combinations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error as rmse
from sklearn.model_selection import GroupKFold

from config import (
    DATA_PATH, STUDY_STATES, STUDY_CROPS, ANALYSIS_YEARS, get_growing_season,
    IRRIGATION_COLUMN, NASS_PRACTICE_MODE, describe_years,
    yield_filename, merged_filename, results_dirname, logging,
)
from utils import save_df

PROCESSED_DIR = DATA_PATH / "processed"
# Both are mode-dependent: the aggregate and split studies use different yield
# tables and different results directories so neither can overwrite the other.
YIELD_FILE = PROCESSED_DIR / yield_filename()
MERGED_FILE = PROCESSED_DIR / merged_filename()
CLIMATE_FILE = PROCESSED_DIR / "climate.csv"
VEG_FILE = PROCESSED_DIR / "vegetation.csv"
SOIL_FILE = PROCESSED_DIR / "soil.csv"

RESULTS_DIR = Path(results_dirname())
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Thresholds
MIN_YEARS = 10
MIN_SAMPLES = 1000
TEST_FRACTION = 0.20
MIN_TEST_YEARS = 2
RANDOM_SEED = 25

# Repeat every model over several seeds and report mean +/- sd. A single
# random_state gives a point estimate with no indication of how much of a
# between-model difference is just Random Forest sampling noise.
N_SEEDS = 5
SEEDS = [RANDOM_SEED + i for i in range(N_SEEDS)]

# Vegetation completeness is judged against EACH STATE'S OWN season length, not
# a flat count. A flat 5 accepted an incomplete 6-month Iowa season and an
# incomplete 7-month Arkansas season while being exactly right for Minnesota's
# 5-month window. Measured impact on this dataset is small (5 rows of 31,504),
# but the flat rule was correct only by luck.
# MAX_MISSING_VEG_MONTHS = 0 requires a complete season.
MAX_MISSING_VEG_MONTHS = 0

# Minimum climate coverage (observed days / expected season days, worst
# variable) for a county-year to be modelled. Recording coverage without
# enforcing it would let partial-season rates sit beside full-season ones as if
# they were the same quantity.
MIN_CLIMATE_COVERAGE = 1.0
REQUIRE_COVERAGE = True

# Retained for callers that want the old flat floor (evaluate.py imports it).
MIN_VEG_MONTHS = 5

# Renamed on merge so it cannot collide with climate.csv's own n_months.
VEG_MONTHS_COL = "veg_n_months"

# Random Forest parameters (random_state is injected per-seed at fit time)
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_leaf": 5,
    "max_features": 0.5,
    "n_jobs": -1,
}


# GDD/EDD/HOT_DAY_FRAC/TMAX_MAX come from the daily-resolution climate download.
# TMAX alone is a season mean of daily maxima and mostly encodes WHERE a county
# is (85% of its variance is between-county); EDD_TMAX and HOT_DAY_FRAC are the
# variables that actually carry heat stress.
#
# DELIBERATELY EXCLUDED from the feature list, though present in climate.csv:
#   SEASON_DAYS - constant per state, so it is a state fingerprint, not weather
#   HOT_DAYS    - a raw count scales with window length; use HOT_DAY_FRAC
# GDD_TMAX / EDD_TMAX are Tmax-based per-day heat indices, NOT conventional
# growing degree days and NOT Schlenker & Roberts degree-days. See the naming
# note in download_climate.py before describing them in writing.
CLIMATE_FEATURES = [
    "TMIN", "TMAX", "PRCP", "VPD", "ETO", "SRAD",
    "GDD_TMAX", "EDD_TMAX", "HOT_DAY_FRAC", "TMAX_MAX",
]
SOIL_FEATURES = ["clay_mean", "ph_mean", "soc_mean", "bdod_mean"]
VEG_FEATURES = ["evi_min_year", "evi_mean_year", "evi_max_year", "ndvi_min_year", "ndvi_mean_year", "ndvi_max_year"]
#VEG_FEATURES = ["evi_min_year", "evi_max_year", "ndvi_min_year", "ndvi_max_year"]

# All feature sets to evaluate
FEATURE_SETS = {
    "climate_only": CLIMATE_FEATURES,
    "climate_soil": CLIMATE_FEATURES + SOIL_FEATURES,
    "climate_veg": CLIMATE_FEATURES + VEG_FEATURES,
    "all_features": CLIMATE_FEATURES + SOIL_FEATURES + VEG_FEATURES,
}

CAUSAL_MODEL = "climate_soil"   # Features that has influence on actual yield

CATEGORY_MAP = {
    **{f: "climate" for f in CLIMATE_FEATURES},
    **{f: "soil" for f in SOIL_FEATURES},
    **{f: "vegetation" for f in VEG_FEATURES},
}


def standardize_fips(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure county_fips is 5-digit zero-padded string."""
    df = df.copy()
    if "county_fips" in df.columns:
        df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
    return df


def filter_by_year(df: pd.DataFrame, min_year: int, max_year: int = None) -> pd.DataFrame:
    """
    Restrict to the configured analysis window.

    max_year is enforced as well as min_year: filtering only the start meant
    that narrowing ANALYSIS_YEARS left later observations in the data, so the
    modelled period could silently exceed the configured one.
    """
    if "year" not in df.columns:
        return df
    n_before = len(df)
    df = df[df["year"] >= min_year]
    if max_year is not None:
        df = df[df["year"] <= max_year]
    df = df.copy()
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logging.info(f"    Filtered to {min_year}-{max_year if max_year else '...'}: dropped {n_dropped:,} rows")
    return df


def sanitize_class_desc(desc) -> str:
    """Clean class_desc: remove special chars, uppercase, underscore-separated."""
    if pd.isna(desc):
        return ""
    desc = str(desc).strip().upper()
    if desc in ("", "ALL CLASSES"):
        return ""
    desc = re.sub(r"[^A-Z0-9 ]+", " ", desc)
    desc = re.sub(r"\s+", "_", desc).strip("_")
    return desc


def make_crop_name(row: pd.Series) -> str:
    """
    Analysis-unit identifier: commodity + class, plus irrigation status when
    the yield data was built in "split" mode.

    In split mode the same biological crop under different management is a
    DIFFERENT analysis unit - CORN__IRRIGATED and CORN__NON_IRRIGATED are
    scored separately. That is the point: irrigation decouples yield from
    growing-season weather, so the two strata should show very different
    anomaly R2 despite sharing a state, a climate and a soil map.

    Keeping them as one unit is what lets a model read "hot and productive"
    off irrigated counties and call it a heat response.
    """
    base = str(row.get("commodity_desc", "")).strip().upper()
    cls = sanitize_class_desc(row.get("class_desc"))
    name = f"{base}_{cls}" if cls else base

    irr = row.get(IRRIGATION_COLUMN)
    if pd.notna(irr) and str(irr) != "all":
        name = f"{name}__{str(irr).upper()}"
    return name


def load_data():
    """Load all data files and apply year filter."""
    start_year, end_year = ANALYSIS_YEARS
    logging.info("Loading data files...")

    yield_df = standardize_fips(pd.read_csv(YIELD_FILE))
    logging.info(f"  Yield:   {len(yield_df):,} rows")
    yield_df = filter_by_year(yield_df, start_year, end_year)
    if STUDY_CROPS:
        yield_df = yield_df[yield_df['commodity_desc'].isin(STUDY_CROPS)]

    if STUDY_STATES:
        yield_df = yield_df[yield_df['state_alpha'].isin(STUDY_STATES)]

    climate_df = standardize_fips(pd.read_csv(CLIMATE_FILE))
    logging.info(f"  Climate: {len(climate_df):,} rows")
    climate_df = filter_by_year(climate_df, start_year, end_year)

    veg_df = standardize_fips(pd.read_csv(VEG_FILE))
    logging.info(f"  Veg:     {len(veg_df):,} rows")
    veg_df = filter_by_year(veg_df, start_year, end_year)

    soil_df = standardize_fips(pd.read_csv(SOIL_FILE))
    logging.info(f"  Soil:    {len(soil_df):,} rows (no year filter)")

    return yield_df, climate_df, veg_df, soil_df


def merge_datasets(yield_df: pd.DataFrame, climate_df: pd.DataFrame,
                   veg_df: pd.DataFrame, soil_df: pd.DataFrame) -> pd.DataFrame:
    """Merge all datasets using inner joins."""
    logging.info("Merging datasets...")

    yield_df = yield_df.copy()

    if IRRIGATION_COLUMN not in yield_df.columns:
        # Yield file predates the irrigation split; treat every row as the
        # aggregate so downstream behaviour is unchanged.
        yield_df[IRRIGATION_COLUMN] = "all"
        logging.warning(
            f"'{IRRIGATION_COLUMN}' not in yield.csv - assuming aggregate rows. "
            f"Re-run download_yield.py to pick up the practice split."
        )

    yield_df["crop"] = yield_df.apply(make_crop_name, axis=1)

    # Keep only essential columns from each dataset
    yield_cols = ["county_fips", "year", "crop", IRRIGATION_COLUMN, "yield_value"]
    yield_df = yield_df[[c for c in yield_cols if c in yield_df.columns]]

    # GUARD: NASS publishes the same county-year-crop as an aggregate AND as
    # irrigated / non-irrigated components. Under practice mode "both" those
    # collide on (county_fips, year, crop) and every downstream count, split and
    # score is silently inflated. Fail loudly instead.
    dup = yield_df.duplicated(subset=["county_fips", "year", "crop"]).sum()
    if dup:
        msg = (
            f"{dup:,} duplicate (county_fips, year, crop) yield rows under "
            f"NASS_PRACTICE_MODE='{NASS_PRACTICE_MODE}'. The same county-year is "
            f"present as more than one production practice, which would multiply "
            f"the sample. Use mode 'aggregate' or 'split', not 'both'."
        )
        if NASS_PRACTICE_MODE == "both":
            raise ValueError(msg)
        logging.error(msg)

    # Climate: county_fips, year + feature columns only
    # climate.csv also carries an n_months (season month count). It must be
    # excluded or the merge produces n_months_x / n_months_y and the vegetation
    # completeness filter below silently never fires.
    climate_exclude = ["county_fips", "year", "county_name", "state_abbr", "state_alpha",
                       "state_fips", "n_months"]
    climate_cols = ["county_fips", "year"] + [c for c in climate_df.columns if c not in climate_exclude]
    climate_df = climate_df[[c for c in climate_cols if c in climate_df.columns]]
    climate_df = climate_df.drop_duplicates(subset=["county_fips", "year"])

    # Vegetation: county_fips, year + feature columns only
    # n_months is DELIBERATELY kept, and renamed so it cannot collide with the
    # climate column of the same name. It is the data-quality field used by
    # build_study_baseline() to drop county-years with incomplete vegetation
    # coverage: ndvi_min/max over 3 months is not the same quantity as over 6.
    veg_exclude = ["county_fips", "year", "county_name", "state_fips", "state_abbr"]
    veg_df = veg_df.rename(columns={"n_months": VEG_MONTHS_COL})
    veg_cols = ["county_fips", "year"] + [c for c in veg_df.columns if c not in veg_exclude]
    veg_df = veg_df[[c for c in veg_cols if c in veg_df.columns]]
    veg_df = veg_df.drop_duplicates(subset=["county_fips", "year"])

    # Soil: county_fips + state_abbr (for reference) + feature columns only
    soil_exclude = ["county_fips", "state_abbr", "county_name", "state_fips", "mask_start_year", "mask_end_year",
                    "sand_mean"]
    soil_cols = ["county_fips", "state_abbr", "county_name"] + [c for c in soil_df.columns if c not in soil_exclude]
    soil_df = soil_df[[c for c in soil_cols if c in soil_df.columns]]
    soil_df = soil_df.drop_duplicates(subset=["county_fips"])

    # Merge all datasets
    merged = yield_df.merge(climate_df, on=["county_fips", "year"], how="inner")
    merged = merged.merge(veg_df, on=["county_fips", "year"], how="inner")
    merged = merged.merge(soil_df, on=["county_fips"], how="inner")

    logging.info(f"  Merged: {len(merged):,} rows")
    logging.info(f"  Years:  {merged['year'].min()}-{merged['year'].max()}")

    # Surface a download/analysis mismatch instead of letting it pass silently.
    # A source that stopped short of the configured window shows up here as a
    # truncated year range rather than as quietly missing rows much later.
    y0, y1 = int(merged["year"].min()), int(merged["year"].max())
    if (y0, y1) != tuple(ANALYSIS_YEARS):
        logging.warning(
            f"  Merged window {y0}-{y1} != configured ANALYSIS_YEARS "
            f"{ANALYSIS_YEARS[0]}-{ANALYSIS_YEARS[1]}. Some source is short - "
            f"check which one before trusting the split."
        )
    logging.info(f"  Counties: {merged['county_fips'].nunique()}")
    logging.info(f"  States: {merged['state_abbr'].nunique()}")
    logging.info(f"  Crops: {merged['crop'].nunique()}")

    # NO state filtering - use all available data

    return merged


def filter_veg_complete(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep county-years whose vegetation season is complete for THEIR state.

    ndvi_min/max over a partial season is not the same quantity as over a full
    one, and "partial" depends on the state: 5 months is complete for Minnesota
    and two months short for Arkansas.
    """
    if VEG_MONTHS_COL not in df.columns or "state_abbr" not in df.columns:
        logging.warning(
            "filter_veg_complete: need both %s and state_abbr; skipping filter",
            VEG_MONTHS_COL,
        )
        return df
    bounds = df["state_abbr"].map(get_growing_season)
    expected = pd.Series([b[1] - b[0] for b in bounds], index=df.index)
    return df[df[VEG_MONTHS_COL].fillna(0) >= (expected - MAX_MISSING_VEG_MONTHS)]


def filter_climate_complete(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep county-years whose climate season is complete (COVERAGE_MIN, the worst
    variable's observed/expected days). ONE definition, imported by evaluate.py
    and irrigation_contrast.py, so the modelling and evaluation paths cannot
    disagree about which rows exist.
    """
    if "COVERAGE_MIN" not in df.columns:
        message = ("COVERAGE_MIN is absent from the climate data, so season completeness "
                   "cannot be enforced. Rebuild with:  python download_climate.py --aggregate-only")
        if REQUIRE_COVERAGE:
            raise KeyError(message)
        logging.warning(message)
        return df
    n0 = len(df)
    out = df[df["COVERAGE_MIN"].fillna(0) >= MIN_CLIMATE_COVERAGE]
    if len(out) < n0:
        logging.info(f"Dropped {n0 - len(out):,} county-years with climate coverage "
                     f"< {MIN_CLIMATE_COVERAGE:.0%}")
    return out


def build_study_baseline(
    merged_df: pd.DataFrame,
    all_features: list,
) -> pd.DataFrame:
    """
    Build a clean, reusable baseline dataset for a study.

    Steps:
      1) Drop county-years with incomplete vegetation coverage
      2) Drop rows missing any feature in all_features or target
      3) Return the single clean table used by ALL feature sets

    Every model shares this row set, so differences in score reflect the
    features and not which rows happened to survive. (State/crop/year filtering
    is applied earlier, in load_data().)
    """

    df = merged_df.copy()

    before = len(df)

    # --- Vegetation completeness ---
    # ndvi_min_year/ndvi_max_year over a partial growing season are not
    # comparable with the same statistics over a full one, so county-years with
    # cloud or mask gaps must be dropped rather than silently mixed in.
    if VEG_MONTHS_COL in df.columns:
        n_before_veg = len(df)
        df = filter_veg_complete(df)
        n_veg_dropped = n_before_veg - len(df)
        if n_veg_dropped > 0:
            logging.info(
                f"Study baseline: dropped {n_veg_dropped:,} county-years with "
                f"an incomplete vegetation season "
                f"(> {MAX_MISSING_VEG_MONTHS} month(s) missing vs the state's window)"
            )
    else:
        logging.warning(
            f"Study baseline: '{VEG_MONTHS_COL}' not present - cannot "
            "verify growing-season completeness. Re-run download_vegetation.py."
        )

    # --- Climate season completeness (shared rule with evaluate.py) ---
    df = filter_climate_complete(df)

    # --- Drop NaNs based on feature universe ---
    required = [f for f in all_features if f in df.columns] + ["yield_value"]
    df = df.dropna(subset=required)

    after = len(df)

    dropped = before - after
    logging.info(
        f"Study baseline: {after:,} rows kept "
        f"({dropped:,} dropped: incomplete vegetation or missing features)"
    )

    return df


def temporal_split(df: pd.DataFrame):
    """Split data: train on earlier years, test on last N years."""
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

    #  Minimum samples for reliable Random Forest training and evaluation
    if len(train_df) < 100 or len(test_df) < 30:
        return None

    return train_df, test_df, train_years, test_years


def county_mean_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    """
    Naive reference predictor: each county's mean yield over the TRAINING years.
    Counties unseen in training fall back to the global training mean.

    This is the benchmark every model in this study must beat. Because soil is
    static per county (one unique soil tuple per county in this dataset), a
    Random Forest given soil can in principle recover county identity and
    reproduce this baseline without learning any agronomy. Reporting raw test R2
    without it makes memorised county means look like climate skill.
    """
    county_means = train_df.groupby("county_fips")["yield_value"].mean()
    global_mean = train_df["yield_value"].mean()
    return test_df["county_fips"].map(county_means).fillna(global_mean).to_numpy()


def train_model(X_train, y_train, X_test, y_test, features: List[str],
                baseline_test: np.ndarray) -> Dict:
    """
    Fit a Random Forest once per seed; report mean and sd across seeds.

    Two skill scores are returned:
      test_r2      - conventional R2 against the test-set mean. Dominated by
                     BETWEEN-county variance (some counties are simply more
                     productive), so it flatters any model that can identify
                     the county.
      skill_vs_county_mean
                   - 1 - SSE(model)/SSE(county-mean predictor). 0 means exactly
                     as good as predicting each county's training mean, negative
                     means worse, 1 is perfect. This is the quantity a
                     climate-impacts claim actually rests on.

    This was previously an r2_score on residuals, which centres on the mean TEST
    anomaly rather than on zero. The two only agree when the test period's mean
    anomaly is zero; otherwise they diverge badly - a predictor that tied the
    county-mean baseline exactly scored -33.8 under the old definition.
    """
    per_seed = {"train_r2": [], "test_r2": [], "skill_vs_county_mean": [], "mae": [], "rmse": []}
    models, importances = [], []

    y_test_anom = y_test - baseline_test

    for seed in SEEDS:
        model = RandomForestRegressor(random_state=seed, **RF_PARAMS)
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        per_seed["train_r2"].append(r2_score(y_train, y_pred_train))
        per_seed["test_r2"].append(r2_score(y_test, y_pred_test))
        # 1 - SSE(model)/SSE(county-mean). Replaces an r2_score on residuals,
        # which centres on the mean TEST anomaly and therefore does NOT answer
        # "did we beat the county mean" - it scored -33.8 for a predictor that
        # tied the baseline exactly.
        sse_ref = float(np.sum((y_test - baseline_test) ** 2))
        sse_mod = float(np.sum((y_test - y_pred_test) ** 2))
        per_seed["skill_vs_county_mean"].append(
            1.0 - sse_mod / sse_ref if sse_ref > 0 else float("nan"))
        per_seed["mae"].append(mean_absolute_error(y_test, y_pred_test))
        per_seed["rmse"].append(rmse(y_test, y_pred_test))

        models.append(model)
        importances.append(model.feature_importances_)

    out = {k: float(np.mean(v)) for k, v in per_seed.items()}
    out.update({k + "_sd": float(np.std(v)) for k, v in per_seed.items()})

    # Keep the median-performing model as the representative one for
    # interpretation, rather than an arbitrary first seed.
    median_idx = int(np.argsort(per_seed["test_r2"])[len(SEEDS) // 2])

    out["model"] = models[median_idx]
    out["baseline_r2"] = float(r2_score(y_test, baseline_test))
    # Impurity importance is retained only for backward comparison; it is biased
    # toward continuous high-cardinality features. compute_permutation_importance()
    # is what the reported figures should use.
    out["importances_impurity"] = pd.Series(np.mean(importances, axis=0), index=features)
    return out


def train_and_test(crop_df: pd.DataFrame,
                       features: List[str], feature_set_name: str) -> Optional[Dict]:
    """Train and evaluate a single model with specified features."""
    # Filter to available features
    available_features = [f for f in features if f in crop_df.columns]

    # A feature set that quietly loses columns keeps its label while modelling
    # something else - "climate_soil" scored without any soil, for instance.
    # Any comparison against it then measures the wrong contrast.
    missing = [f for f in features if f not in crop_df.columns]
    if missing:
        raise KeyError(
            f"{feature_set_name}: {len(missing)} feature(s) absent from the data: "
            f"{missing}. Re-run the relevant downloader, or edit the feature list "
            f"in ml.py - do not score a feature set under a label it no longer matches."
        )

    if len(available_features) < 2:
        logging.warning(f"    {feature_set_name}: Not enough features available")
        return None

    # Temporal split
    split_result = temporal_split(crop_df)
    if split_result is None:
        logging.warning(f"    {feature_set_name}: Insufficient data after split")
        return None

    train_df, test_df, train_years, test_years = split_result

    X_train = train_df[available_features].values
    X_test = test_df[available_features].values
    y_train = train_df["yield_value"].values
    y_test = test_df["yield_value"].values

    baseline_test = county_mean_baseline(train_df, test_df)

    result = train_model(X_train, y_train, X_test, y_test, available_features, baseline_test)

    out = {
        "model": result["model"],
        "features": available_features,
        "feature_set": feature_set_name,
        "train_df": train_df,
        "test_df": test_df,
        "train_years": train_years,
        "test_years": test_years,
        "importances": result["importances_impurity"],
        "baseline_test": baseline_test,
    }
    for k in ("train_r2", "test_r2", "skill_vs_county_mean", "mae", "rmse", "baseline_r2"):
        out[k] = result[k]
    for k in ("train_r2", "test_r2", "skill_vs_county_mean", "mae", "rmse"):
        out[k + "_sd"] = result[k + "_sd"]
    return out


def spatial_cv_r2(crop_df: pd.DataFrame, features: List[str], n_splits: int = 5) -> Optional[float]:
    """
    Mean R2 across county-grouped folds: can the model generalise to counties it
    has never seen? A temporal split alone leaves every test county present in
    training, so a model can score well purely by memorising county means.

    NOTE: folds are not time-separated, so this is an OPTIMISTIC bound - it is
    diagnostic of spatial transfer, not a forecast score. Report it alongside
    the temporal split, never instead of it.
    """
    available = [f for f in features if f in crop_df.columns]
    d = crop_df.dropna(subset=available + ["yield_value"])
    if d["county_fips"].nunique() < n_splits * 2 or len(d) < 200:
        return None

    scores = []
    for tr_idx, te_idx in GroupKFold(n_splits=n_splits).split(
        d[available], d["yield_value"], groups=d["county_fips"]
    ):
        model = RandomForestRegressor(random_state=RANDOM_SEED, **RF_PARAMS)
        model.fit(d[available].iloc[tr_idx], d["yield_value"].iloc[tr_idx])
        scores.append(
            r2_score(d["yield_value"].iloc[te_idx], model.predict(d[available].iloc[te_idx]))
        )
    return float(np.mean(scores))


def compute_permutation_importance(model, test_df: pd.DataFrame, features: List[str],
                                   n_repeats: int = 10) -> pd.Series:
    """
    Permutation importance measured on the TEST set.

    Replaces `model.feature_importances_` (mean impurity decrease), which is
    biased toward continuous, high-cardinality predictors and is computed on
    training data, so it rewards overfitting. Soil features take only ~1 value
    per county while climate varies continuously, so the two are not on a
    comparable footing under impurity importance.

    Values are clipped at 0 (a negative score means the feature is no better
    than noise) and normalised to sum to 1 for category aggregation.
    """
    result = permutation_importance(
        model, test_df[features].values, test_df["yield_value"].values,
        n_repeats=n_repeats, random_state=RANDOM_SEED, n_jobs=-1,
    )
    imp = pd.Series(result.importances_mean, index=features).clip(lower=0)
    total = imp.sum()
    return imp / total if total > 0 else imp


def compute_pdp_data(model, train_df: pd.DataFrame, crop: str,
                     features: List[str], importances: pd.Series,
                     top_n: int = 5) -> List[Dict]:
    """Compute 1D Partial Dependence Plot data for top features."""
    pdp_rows = []

    top_features = importances.nlargest(top_n).index.tolist()
    X_train = train_df[features].values

    for feature_name in top_features:
        if feature_name not in features:
            continue
        feature_idx = features.index(feature_name)

        try:
            pdp_result = partial_dependence(
                model, X_train, features=[feature_idx],
                kind="average", grid_resolution=20
            )

            grid_values = pdp_result["grid_values"][0]
            avg_response = pdp_result["average"][0]

            for val, resp in zip(grid_values, avg_response):
                pdp_rows.append({
                    "crop": crop,
                    "feature": feature_name,
                    "category": CATEGORY_MAP.get(feature_name, "other"),
                    "feature_value": round(float(val), 4),
                    "predicted_yield": round(float(resp), 2),
                })
        except Exception as e:
            logging.warning(f"    PDP failed for {crop}/{feature_name}: {e}")

    return pdp_rows


def compute_ale_data(model, train_df: pd.DataFrame, crop: str,
                     features: List[str], importances: pd.Series,
                     top_n: int = 5, n_bins: int = 20) -> List[Dict]:
    """
    1D Accumulated Local Effects.

    WHY THIS EXISTS ALONGSIDE PDP
    -----------------------------
    A partial dependence plot varies one feature while holding the others at
    their observed values, which forces the model to predict on combinations
    that never occur. In this dataset the climate predictors are severely
    collinear (VPD-ETO ~ 0.95, TMAX-ETO ~ 0.86, TMAX-TMIN ~ 0.84), so a PDP
    sweep of TMAX evaluates hot temperatures paired with humid, low-ET
    conditions that do not exist anywhere in the sample. Any "threshold" read
    off such a curve may be an extrapolation artefact.

    ALE instead accumulates the model's LOCAL differences inside narrow bins,
    using only points that genuinely fall in each bin, so it stays on the data
    manifold. Where PDP and ALE disagree, believe ALE.

    Returned effects are centred on zero (relative change in yield), not
    absolute predicted yield.
    """
    ale_rows: List[Dict] = []
    top_features = importances.nlargest(top_n).index.tolist()
    X = train_df[features].to_numpy(dtype=float)

    for feature_name in top_features:
        if feature_name not in features:
            continue
        j = features.index(feature_name)
        x = X[:, j]

        try:
            edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
            if len(edges) < 3:
                continue

            # Bin index of each row (1..n_bins), clipped into range
            idx = np.clip(np.searchsorted(edges, x, side="left"), 1, len(edges) - 1)

            local_effects = np.zeros(len(edges) - 1)
            for b in range(1, len(edges)):
                in_bin = idx == b
                if not in_bin.any():
                    continue
                # Same rows, feature forced to the bin's lower then upper edge:
                # the difference is a LOCAL effect, not an extrapolation.
                lo = X[in_bin].copy()
                hi = X[in_bin].copy()
                lo[:, j] = edges[b - 1]
                hi[:, j] = edges[b]
                local_effects[b - 1] = np.mean(model.predict(hi) - model.predict(lo))

            ale = np.concatenate([[0.0], np.cumsum(local_effects)])
            # Centre so the mean effect over the data distribution is zero
            counts = np.bincount(idx, minlength=len(edges))[1:len(edges)]
            if counts.sum() > 0:
                centre = np.sum(((ale[:-1] + ale[1:]) / 2.0) * counts) / counts.sum()
                ale = ale - centre

            for edge, eff in zip(edges, ale):
                ale_rows.append({
                    "crop": crop,
                    "feature": feature_name,
                    "category": CATEGORY_MAP.get(feature_name, "other"),
                    "feature_value": round(float(edge), 4),
                    "ale_effect": round(float(eff), 3),
                })
        except Exception as e:
            logging.warning(f"    ALE failed for {crop}/{feature_name}: {e}")

    return ale_rows


def compute_feature_correlations(train_df: pd.DataFrame, features: List[str],
                                 crop: str, threshold: float = 0.7) -> List[Dict]:
    """
    Record strongly collinear feature pairs so every importance and dependence
    figure can be read with the right caveat attached.
    """
    rows = []
    corr = train_df[features].corr()
    for a, b in combinations(features, 2):
        r = corr.loc[a, b]
        if pd.notna(r) and abs(r) >= threshold:
            rows.append({
                "crop": crop, "feature1": a, "feature2": b,
                "correlation": round(float(r), 3),
            })
    return rows


def compute_pdp_2d(model, train_df: pd.DataFrame, crop: str,
                   features: List[str], importances: pd.Series,
                   top_n_pairs: int = 3) -> List[Dict]:
    """Compute 2D Partial Dependence for top feature pairs (interactions)."""
    pdp_rows = []

    # Get top features
    top_features = importances.nlargest(4).index.tolist()
    X_train = train_df[features].values

    # Analyze pairs
    for feat1, feat2 in list(combinations(top_features, 2))[:top_n_pairs]:
        if feat1 not in features or feat2 not in features:
            continue

        idx1 = features.index(feat1)
        idx2 = features.index(feat2)

        try:
            pdp_result = partial_dependence(
                model, X_train,
                features=[(idx1, idx2)],
                kind="average",
                grid_resolution=10
            )

            grid1 = pdp_result["grid_values"][0]
            grid2 = pdp_result["grid_values"][1]
            pdp_values = pdp_result["average"][0]

            for i, v1 in enumerate(grid1):
                for j, v2 in enumerate(grid2):
                    pdp_rows.append({
                        "crop": crop,
                        "feature1": feat1,
                        "feature2": feat2,
                        "value1": round(float(v1), 3),
                        "value2": round(float(v2), 3),
                        "predicted_yield": round(float(pdp_values[i, j]), 2),
                    })

        except Exception as e:
            logging.warning(f"    2D PDP failed for {crop}/{feat1}×{feat2}: {e}")

    return pdp_rows


def compute_sensitivity_from_pdp(pdp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive sensitivity from PDP data (more reliable than IQR-shift method).
    Shows actual yield range across the feature's observed values.
    """
    results = []

    for (crop, feature), group in pdp_df.groupby(["crop", "feature"]):
        # MUST sort by feature_value for direction to be meaningful
        group = group.sort_values("feature_value").reset_index(drop=True)

        yields = group["predicted_yield"].values
        feature_values = group["feature_value"].values

        min_idx = yields.argmin()
        max_idx = yields.argmax()

        results.append({
            "crop": crop,
            "feature": feature,
            "category": group["category"].iloc[0],
            "yield_min": round(yields.min(), 1),
            "yield_max": round(yields.max(), 1),
            "yield_range": round(yields.max() - yields.min(), 1),
            "feature_at_min_yield": round(feature_values[min_idx], 3),
            "feature_at_max_yield": round(feature_values[max_idx], 3),
            # Now correctly compares low feature value → high feature value
            "direction": "positive" if yields[-1] > yields[0] else "negative",
            # Bonus: detect if there's an optimal peak in the middle
            "has_peak": bool(0 < max_idx < len(yields) - 1),
        })

    return pd.DataFrame(results).sort_values(
        ["crop", "yield_range"], ascending=[True, False]
    )


def compute_optimal_conditions(df: pd.DataFrame, crop: str,
                               percentile: float = 80,
                               features: List[str] = None) -> Dict:
    """Analyze feature values for top-performing yields."""
    if features is None:
        features = FEATURE_SETS[CAUSAL_MODEL]

    threshold = np.percentile(df["yield_value"], percentile)
    top_df = df[df["yield_value"] >= threshold]

    optimal = {
        "crop": crop,
        "threshold_yield": round(float(threshold), 1),
        "n_top_samples": len(top_df),
        "n_total_samples": len(df),
    }

    for feature in features:
        if feature in top_df.columns:
            optimal[f"{feature}_mean"] = round(float(top_df[feature].mean()), 3)
            optimal[f"{feature}_std"] = round(float(top_df[feature].std()), 3)

    return optimal


def evaluate_crop(crop_df: pd.DataFrame, crop_name: str) -> Dict:
    """Evaluate all feature sets for a crop and run interpretation on causal model."""

    results = {
        "perf_rows": [],
        "feat_rows": [],
        "cat_rows": [],
        "pdp_rows": [],
        "ale_rows": [],
        "corr_rows": [],
        "pdp_2d_rows": [],
        "sens_rows": [],
        "optimal": None,
    }

    causal_result = None

    for set_name, features in FEATURE_SETS.items():
        logging.info(f"  Training: {set_name}")

        model_result = train_and_test(crop_df, features, set_name)

        if model_result is None:
            continue

        spatial_r2 = spatial_cv_r2(crop_df, model_result["features"])

        # Store performance metrics
        results["perf_rows"].append({
            "crop": crop_name,
            "feature_set": set_name,
            "train_r2": round(model_result["train_r2"], 3),
            "test_r2": round(model_result["test_r2"], 3),
            "test_r2_sd": round(model_result["test_r2_sd"], 3),
            # The benchmark: predict each county's training-mean yield.
            "baseline_county_mean_r2": round(model_result["baseline_r2"], 3),
            "r2_vs_baseline": round(model_result["test_r2"] - model_result["baseline_r2"], 3),
            # Skill on year-to-year deviation. <= 0 means the model adds nothing
            # beyond knowing which county the observation came from.
            "skill_vs_county_mean": round(model_result["skill_vs_county_mean"], 3),
            "skill_sd": round(model_result["skill_vs_county_mean_sd"], 3),
            # Generalisation to unseen counties (optimistic; see spatial_cv_r2)
            "spatial_cv_r2": round(spatial_r2, 3) if spatial_r2 is not None else None,
            "r2_gap": round(model_result["train_r2"] - model_result["test_r2"], 3),
            "mae": round(model_result["mae"], 2),
            "rmse": round(model_result["rmse"], 2),
            "n_train": len(model_result["train_df"]),
            "n_test": len(model_result["test_df"]),
            "n_features": len(model_result["features"]),
            "n_seeds": N_SEEDS,
            "train_years": f"{model_result['train_years'][0]}-{model_result['train_years'][-1]}",
            "test_years": f"{model_result['test_years'][0]}-{model_result['test_years'][-1]}",
        })

        logging.info(
            f"    test R2 = {model_result['test_r2']:.3f} +/- {model_result['test_r2_sd']:.3f}"
            f" | county-mean baseline = {model_result['baseline_r2']:.3f}"
            f" | skill vs county mean = {model_result['skill_vs_county_mean']:.3f}"
            f" (train={model_result['train_r2']:.3f}, n={len(model_result['train_df']):,})"
        )
        if model_result["test_r2"] <= model_result["baseline_r2"]:
            logging.warning(
                f"    ^ {set_name} does NOT beat the county-mean baseline for {crop_name}"
            )

        # Save causal model for interpretation
        if set_name == CAUSAL_MODEL:
            causal_result = model_result

    if causal_result is not None:
        logging.info(f"  Running interpretation on {CAUSAL_MODEL} model...")

        model = causal_result["model"]
        features = causal_result["features"]
        train_df = causal_result["train_df"]
        test_df = causal_result["test_df"]

        # Permutation importance on the TEST set, not impurity importance on
        # the training set. See compute_permutation_importance() for why.
        importances = compute_permutation_importance(model, test_df, features)
        impurity = causal_result["importances"]

        total_imp = importances.sum() or 1.0
        for feat, imp in importances.items():
            cat = CATEGORY_MAP.get(feat, "other")
            results["feat_rows"].append({
                "crop": crop_name,
                "feature": feat,
                "category": cat,
                "importance": round(imp, 4),
                "importance_pct": round(100 * imp / total_imp, 2),
                "method": "permutation_test",
                # kept side by side so the change in method is auditable
                "importance_impurity": round(float(impurity.get(feat, np.nan)), 4),
            })

        # Category importance
        cat_totals = {}
        for feat, imp in importances.items():
            cat = CATEGORY_MAP.get(feat, "other")
            cat_totals[cat] = cat_totals.get(cat, 0) + imp

        for cat, val in cat_totals.items():
            results["cat_rows"].append({
                "crop": crop_name,
                "category": cat,
                "importance": round(val, 4),
                "importance_pct": round(100 * val / total_imp, 1),
                "method": "permutation_test",
            })

        # Collinearity audit - context for every figure below
        results["corr_rows"].extend(
            compute_feature_correlations(train_df, features, crop_name)
        )

        # 1D PDP (kept for comparison) and ALE (preferred under collinearity)
        pdp_data = compute_pdp_data(model, train_df, crop_name, features, importances)
        results["pdp_rows"].extend(pdp_data)

        results["ale_rows"].extend(
            compute_ale_data(model, train_df, crop_name, features, importances)
        )

        if pdp_data:
            pdp_df = pd.DataFrame(pdp_data)
            sens_data = compute_sensitivity_from_pdp(pdp_df)
            results["sens_rows"].extend(sens_data.to_dict("records"))

        # 2D PDP (interactions)
        pdp_2d_data = compute_pdp_2d(model, train_df, crop_name, features, importances)
        results["pdp_2d_rows"].extend(pdp_2d_data)

        # Optimal conditions
        results["optimal"] = compute_optimal_conditions(train_df, crop_name, features=features)

    return results


def run_pipeline():
    """Main pipeline to train models and generate all outputs."""
    logging.info(f"{'=' * 70}")
    logging.info("CROP YIELD PREDICTION PIPELINE")
    logging.info(f"Years: {describe_years()}")
    logging.info(f"States: {len(STUDY_STATES)} | Crops: {len(STUDY_CROPS)} | Practice mode: {NASS_PRACTICE_MODE}")
    logging.info(f"{'=' * 70}")

    # Load and merge data
    yield_df, climate_df, veg_df, soil_df = load_data()
    merged = merge_datasets(yield_df, climate_df, veg_df, soil_df)

    save_df(merged, MERGED_FILE)
    logging.info(f"\nMerged data saved to: {MERGED_FILE}")

    all_features = sorted(
        {f for feature_list in FEATURE_SETS.values() for f in feature_list}
    )

    baseline_df = build_study_baseline(
        merged_df=merged,
        all_features=all_features,
    )

    # Get crop statistics
    crop_stats = baseline_df.groupby("crop").agg({
        "year": "nunique",
        "yield_value": "count"
    }).rename(columns={"year": "n_years", "yield_value": "n_records"})

    valid_crops = crop_stats[
        (crop_stats["n_years"] >= MIN_YEARS) &
        (crop_stats["n_records"] >= MIN_SAMPLES)
    ].index.tolist()

    logging.info(f"Crops meeting thresholds (years>={MIN_YEARS}, samples>={MIN_SAMPLES}):")
    for crop in sorted(valid_crops):
        stats = crop_stats.loc[crop]
        logging.info(f"  {crop}: {stats['n_records']:,} records, {stats['n_years']} years")

    # Process each crop
    all_perf = []
    all_feat = []
    all_cat = []
    all_pdp = []
    all_ale = []
    all_corr = []
    all_pdp_2d = []
    all_sens = []
    all_optimal = []

    for crop in sorted(valid_crops):
        logging.info("=" * 70)
        logging.info(f"CROP: {crop}")
        logging.info("=" * 70)

        crop_df = baseline_df[baseline_df["crop"] == crop].copy()
        results = evaluate_crop(crop_df, crop)

        all_perf.extend(results["perf_rows"])
        all_feat.extend(results["feat_rows"])
        all_cat.extend(results["cat_rows"])
        all_pdp.extend(results["pdp_rows"])
        all_ale.extend(results["ale_rows"])
        all_corr.extend(results["corr_rows"])
        all_pdp_2d.extend(results["pdp_2d_rows"])
        all_sens.extend(results["sens_rows"])
        if results["optimal"]:
            all_optimal.append(results["optimal"])

    logging.info("=" * 70)
    logging.info("SAVING RESULTS")
    logging.info("=" * 70)

    perf_df = pd.DataFrame(all_perf) if all_perf else None
    cat_df = pd.DataFrame(all_cat) if all_cat else None
    feat_df = pd.DataFrame(all_feat) if all_feat else None
    pdp_df = pd.DataFrame(all_pdp) if all_pdp else None
    ale_df = pd.DataFrame(all_ale) if all_ale else None
    corr_df = pd.DataFrame(all_corr) if all_corr else None
    pdp_2d_df = pd.DataFrame(all_pdp_2d) if all_pdp_2d else None
    sens_df = pd.DataFrame(all_sens) if all_sens else None
    opt_df = pd.DataFrame(all_optimal) if all_optimal else None

    if perf_df is not None:
        save_df(perf_df, RESULTS_DIR / "model_performance.csv")

    if feat_df is not None:
        save_df(feat_df, RESULTS_DIR / "feature_importance.csv")

    if cat_df is not None:
        save_df(cat_df, RESULTS_DIR / "category_importance.csv")

    if pdp_df is not None:
        save_df(pdp_df, RESULTS_DIR / "pdp_data.csv")

    if ale_df is not None:
        save_df(ale_df, RESULTS_DIR / "ale_data.csv")

    if corr_df is not None:
        save_df(corr_df, RESULTS_DIR / "feature_collinearity.csv")

    if pdp_2d_df is not None:
        save_df(pdp_2d_df, RESULTS_DIR / "pdp_2d_data.csv")

    if sens_df is not None:
        save_df(sens_df, RESULTS_DIR / "sensitivity_analysis.csv")

    if opt_df is not None:
        save_df(opt_df, RESULTS_DIR / "optimal_conditions.csv")

    impr_rows = []
    if perf_df is not None:
        for crop in perf_df["crop"].unique():
            crop_perf = perf_df[perf_df["crop"] == crop]

            baseline = crop_perf[crop_perf["feature_set"] == "climate_only"]["test_r2"].values
            causal = crop_perf[crop_perf["feature_set"] == CAUSAL_MODEL]["test_r2"].values
            best = crop_perf[crop_perf["feature_set"] == "all_features"]["test_r2"].values

            county_base = crop_perf["baseline_county_mean_r2"].dropna().values
            anom = crop_perf[crop_perf["feature_set"] == "all_features"]["skill_vs_county_mean"].values

            if len(baseline) > 0:
                baseline_r2 = baseline[0]
                # `is not None` matters: a legitimate R2 of exactly 0.0 is falsy,
                # so `if causal_r2` silently turned real results into None.
                causal_r2 = causal[0] if len(causal) > 0 else None
                best_r2 = best[0] if len(best) > 0 else None

                def _delta(x):
                    return round(x - baseline_r2, 3) if x is not None else None

                def _pct(x):
                    # Percent-of-R2 explodes when the baseline is near zero:
                    # an oats baseline of 0.086 turned a +0.27 gain into
                    # "+320%". Reported only when the baseline is large enough
                    # to make a ratio meaningful; delta_r2 is the honest metric.
                    if x is None or baseline_r2 is None or baseline_r2 < 0.2:
                        return None
                    return round((x - baseline_r2) / baseline_r2 * 100, 1)

                impr_rows.append({
                    "crop": crop,
                    "climate_only_r2": baseline_r2,
                    "causal_r2": causal_r2,
                    "best_r2": best_r2,
                    # The reference that actually matters, carried alongside:
                    "county_mean_baseline_r2": round(float(county_base[0]), 3) if len(county_base) else None,
                    "best_skill_vs_county_mean": round(float(anom[0]), 3) if len(anom) else None,
                    "causal_improvement": _delta(causal_r2),
                    "best_improvement": _delta(best_r2),
                    "causal_pct_improvement": _pct(causal_r2),
                    "best_pct_improvement": _pct(best_r2),
                    "pct_suppressed_low_baseline": bool(baseline_r2 < 0.2),
                })

        if impr_rows:
            impr_df = pd.DataFrame(impr_rows)
            save_df(impr_df, RESULTS_DIR / "improvement_summary.csv")

    col_order = ["climate_only", "climate_soil", "climate_veg", "all_features"]

    if perf_df is not None:
        def _pivot(metric: str) -> pd.DataFrame:
            t = perf_df.pivot_table(index="crop", columns="feature_set",
                                    values=metric, aggfunc="first")
            t = t[[c for c in col_order if c in t.columns]]
            t.index.name = None
            t.columns.name = None
            return t.round(3)

        logging.info("=" * 70)
        logging.info("TEST R2 BY FEATURE SET (vs county-mean baseline)")
        logging.info("=" * 70)
        summary = _pivot("test_r2")
        base_col = perf_df.groupby("crop")["baseline_county_mean_r2"].first().round(3)
        summary.insert(0, "county_mean", base_col)
        logging.info(f"\n{summary.to_string()}")
        logging.info(
            "\n  'county_mean' predicts each county's training-mean yield and uses NO"
            "\n  climate, soil or satellite data. Any feature set that does not clearly"
            "\n  exceed it has demonstrated no value."
        )

        logging.info("=" * 70)
        logging.info("SKILL vs COUNTY MEAN - 1 - SSE(model)/SSE(county mean); 0 = tie, <0 = worse")
        logging.info("=" * 70)
        logging.info(f"\n{_pivot('skill_vs_county_mean').to_string()}")
        logging.info(
            "\n  This is the number a climate-impacts claim rests on. Raw test R2 above"
            "\n  is inflated by between-county variance that county identity alone explains."
        )

        if perf_df["spatial_cv_r2"].notna().any():
            logging.info("=" * 70)
            logging.info("SPATIAL CV R2 - generalisation to UNSEEN counties (optimistic)")
            logging.info("=" * 70)
            logging.info(f"\n{_pivot('spatial_cv_r2').to_string()}")

    if cat_df is not None:
        logging.info("=" * 70)
        logging.info(f"CATEGORY IMPORTANCE (permutation, test set, {CAUSAL_MODEL} model)")
        logging.info("=" * 70)
        cat_summary = cat_df.pivot_table(
            index="crop", columns="category", values="importance_pct", aggfunc="first"
        )
        cat_summary.index.name = None       # Remove "crop" label from display
        cat_summary.columns.name = None     # remove "category" as column header for crops
        logging.info(f"\n{cat_summary.round(1).to_string()}")

    if corr_df is not None and len(corr_df):
        logging.info("=" * 70)
        logging.info("COLLINEARITY WARNING - |r| >= 0.7 among causal-model features")
        logging.info("=" * 70)
        worst = corr_df.reindex(
            corr_df["correlation"].abs().sort_values(ascending=False).index
        ).head(10)
        for _, r in worst.iterrows():
            logging.info(f"  {r['crop']}: {r['feature1']} ~ {r['feature2']}  r={r['correlation']:+.2f}")
        logging.info(
            "\n  Read PDP curves and single-feature importances for these pairs with care;"
            "\n  prefer ale_data.csv, which stays on the observed data distribution."
        )

    if impr_rows:
        logging.info("=" * 70)
        logging.info("IMPROVEMENT OVER CLIMATE-ONLY BASELINE (delta R2)")
        logging.info("=" * 70)

        def _fmt(v, spec=".3f"):
            # A missing value is missing; never format None as a float.
            return "  n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else format(v, spec)

        for row in impr_rows:
            logging.info(f"  {row['crop']}:  (county-mean baseline = {_fmt(row['county_mean_baseline_r2'])})")
            for label, r2_key, d_key, p_key in (
                (f"Causal ({CAUSAL_MODEL})", "causal_r2", "causal_improvement", "causal_pct_improvement"),
                ("Best (all_features) ", "best_r2", "best_improvement", "best_pct_improvement"),
            ):
                pct = row[p_key]
                pct_txt = f"  ({pct:+.0f}%)" if pct is not None else "  (% omitted: baseline < 0.2)"
                logging.info(
                    f"    {label}: {_fmt(row['climate_only_r2'])} -> {_fmt(row[r2_key])}"
                    f"  delta={_fmt(row[d_key], '+.3f')}{pct_txt}"
                )
            logging.info(f"    best skill vs county mean = {_fmt(row['best_skill_vs_county_mean'])}")

    logging.info("=" * 70)
    logging.info("PIPELINE COMPLETE")
    logging.info(f"Results saved to: {RESULTS_DIR}")
    logging.info("=" * 70)

    return perf_df if perf_df is not None else pd.DataFrame()


def main():
    import argparse
    global YIELD_FILE, MERGED_FILE, RESULTS_DIR, NASS_PRACTICE_MODE
    ap = argparse.ArgumentParser(description="Crop yield modelling pipeline")
    ap.add_argument("--practice-mode", default=NASS_PRACTICE_MODE,
                    choices=["aggregate", "split", "both"],
                    help="Which yield table to model. 'split' reads yield_split.csv "
                         "and writes ../results_split, leaving the aggregate study intact.")
    ap.add_argument("--results-dir", default=None,
                    help="Override the results directory.")
    args = ap.parse_args()

    NASS_PRACTICE_MODE = args.practice_mode
    YIELD_FILE = PROCESSED_DIR / yield_filename(args.practice_mode)
    MERGED_FILE = PROCESSED_DIR / merged_filename(args.practice_mode)
    RESULTS_DIR = Path(args.results_dir or results_dirname(args.practice_mode))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not YIELD_FILE.exists():
        raise FileNotFoundError(
            f"{YIELD_FILE} not found. Build it first:  "
            f"python download_yield.py --practice-mode {args.practice_mode}"
        )
    logging.info(f"Practice mode: {args.practice_mode} | yield={YIELD_FILE.name} | merged={MERGED_FILE.name} | results={RESULTS_DIR}")
    run_pipeline()


if __name__ == "__main__":
    main()