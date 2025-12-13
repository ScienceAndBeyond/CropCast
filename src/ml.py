"""
ml.py - Crop Yield Prediction Pipeline

Loads climate, vegetation, and soil data, merges them with USDA yield data,
and trains Random Forest models to predict crop yields by county.

Uses temporal train/test split (train on older years, test on recent years)

Outputsr:
- model_performance.csv: R², RMSE, MAE for all feature sets
- feature_importance.csv: Per-feature importance
- category_importance.csv: Climate/Soil/Vegetation totals
- improvement_summary.csv: % improvement over climate-only baseline
- pdp_data.csv: Partial dependence plot data for top features
- sensitivity_analysis.csv: Yield sensitivity to feature changes  (TBD)
- optimal_conditions.csv: Feature values for top-yielding conditions  (TBD)
"""

import re
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence
from sklearn.metrics import r2_score, mean_absolute_error

try:
    from sklearn.metrics import root_mean_squared_error as rmse
except ImportError:
    from sklearn.metrics import mean_squared_error
    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

from config import DATA_PATH
from utils import save_df, logging

PROCESSED_DIR = DATA_PATH / "processed"
YIELD_FILE = PROCESSED_DIR / "yield.csv"
CLIMATE_FILE = PROCESSED_DIR / "climate.csv"
VEG_FILE = PROCESSED_DIR / "vegetation.csv"
SOIL_FILE = PROCESSED_DIR / "soil.csv"

RESULTS_DIR = Path("../results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Year filter - change this to adjust the study period
START_YEAR = 2010  # Filter all data to >= this year

# Thresholds
MIN_YEARS = 10
MIN_SAMPLES = 1000
TEST_FRACTION = 0.20  # 80:20 split
MIN_TEST_YEARS = 2    # Minimum test years regardless of fraction
RANDOM_SEED = 25

# Features (some redundant and derived features are removed)
CLIMATE_FEATURES = ["TMIN", "TMAX", "PRCP", "VPD", "ETO", "SRAD"]
#VEG_FEATURES = ["evi_min_year", "evi_max_year", "evi_mean_year",
# "ndvi_min_year", "ndvi_max_year", "ndvi_mean_year"]  # mean did not influence much, so excluding
VEG_FEATURES = ["evi_min_year", "evi_max_year", "ndvi_min_year", "ndvi_max_year"]  # similar results as above
SOIL_FEATURES = ["clay_mean", "ph_mean", "soc_mean", "bdod_mean"]
ALL_FEATURES = CLIMATE_FEATURES + VEG_FEATURES + SOIL_FEATURES

FEATURE_SETS = {
    "climate_only": CLIMATE_FEATURES,
    "climate_veg": CLIMATE_FEATURES + VEG_FEATURES,
    "climate_soil": CLIMATE_FEATURES + SOIL_FEATURES,
    "all_features": ALL_FEATURES,
}

CATEGORY_MAP = {
    **{f: "climate" for f in CLIMATE_FEATURES},
    **{f: "vegetation" for f in VEG_FEATURES},
    **{f: "soil" for f in SOIL_FEATURES},
}

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_leaf": 5,
    "max_features": 0.5,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}


def standardize_fips(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure county_fips is 5-digit zero-padded string."""
    df = df.copy()
    if "county_fips" in df.columns:
        df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
    return df


def filter_by_year(df: pd.DataFrame, min_year: int) -> pd.DataFrame:
    """Filter dataframe to rows where year >= min_year."""
    if "year" not in df.columns:
        return df
    n_before = len(df)
    df = df[df["year"] >= min_year].copy()
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logging.info(f"    Filtered to year >= {min_year}: dropped {n_dropped:,} rows")
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
    # desc = desc.replace("_EXCL_DURUM", "").replace("_INCL_CHICKPEAS", "")
    return desc


def make_crop_name(row: pd.Series) -> str:
    """Create crop identifier from commodity + class."""
    base = str(row.get("commodity_desc", "")).strip().upper()
    cls = sanitize_class_desc(row.get("class_desc"))
    return f"{base}_{cls}" if cls else base


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all data files and apply year filter."""
    logging.info("Loading data files...")

    yield_df = standardize_fips(pd.read_csv(YIELD_FILE))
    logging.info(f"  Yield:   {len(yield_df):,} rows")
    yield_df = filter_by_year(yield_df, START_YEAR)

    climate_df = standardize_fips(pd.read_csv(CLIMATE_FILE))
    logging.info(f"  Climate: {len(climate_df):,} rows")
    climate_df = filter_by_year(climate_df, START_YEAR)

    veg_df = standardize_fips(pd.read_csv(VEG_FILE))
    logging.info(f"  Veg:     {len(veg_df):,} rows")
    veg_df = filter_by_year(veg_df, START_YEAR)

    soil_df = standardize_fips(pd.read_csv(SOIL_FILE))
    logging.info(f"  Soil:    {len(soil_df):,} rows (no year filter)")

    return yield_df, climate_df, veg_df, soil_df


def merge_datasets(
    yield_df: pd.DataFrame,
    climate_df: pd.DataFrame,
    veg_df: pd.DataFrame,
    soil_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge all datasets using inner joins."""
    logging.info("Merging datasets...")

    yield_df = yield_df.copy()
    yield_df["crop"] = yield_df.apply(make_crop_name, axis=1)

    yield_cols = ["county_fips", "year", "crop", "yield_value"]
    yield_df = yield_df[[c for c in yield_cols if c in yield_df.columns]]

    climate_cols = ["county_fips", "year"] + [c for c in climate_df.columns 
                   if c not in ["county_fips", "year", "county_name", "state_abbr"]]
    climate_df = climate_df[[c for c in climate_cols if c in climate_df.columns]]
    climate_df = climate_df.drop_duplicates(subset=["county_fips", "year"])

    veg_cols = ["county_fips", "year"] + [c for c in veg_df.columns 
               if c not in ["county_fips", "year", "county_name", "state_fips"]]
    veg_df = veg_df[[c for c in veg_cols if c in veg_df.columns]]
    veg_df = veg_df.drop_duplicates(subset=["county_fips", "year"])

    soil_cols = ["county_fips", "state_abbr", "county_name"] + [c for c in soil_df.columns
                if c not in ["county_fips", "county_name", "state_abbr"]]
    soil_df = soil_df[[c for c in soil_cols if c in soil_df.columns]]
    soil_df = soil_df.drop_duplicates(subset=["county_fips"])

    merged = yield_df.merge(climate_df, on=["county_fips", "year"], how="inner")
    merged = merged.merge(veg_df, on=["county_fips", "year"], how="inner")
    merged = merged.merge(soil_df, on=["county_fips"], how="inner")

    logging.info(f"  Merged: {len(merged):,} rows")
    logging.info(f"  Years:  {merged['year'].min()}-{merged['year'].max()}")
    logging.info(f"  Counties: {merged['county_fips'].nunique()}")
    n_states = merged['state_abbr'].nunique() if 'state_abbr' in merged.columns else "N/A"
    logging.info(f"  States: {n_states}")
    logging.info(f"  Crops: {merged['crop'].nunique()}")

    return merged


def drop_missing_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing values in any feature column."""
    existing = [f for f in ALL_FEATURES if f in df.columns]
    n_before = len(df)
    df = df.dropna(subset=existing)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logging.info(f"  Dropped {n_dropped:,} rows with missing features")
    return df


def temporal_split(df: pd.DataFrame) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, List, List]]:
    """Split data: train on earlier years, test on last N years (dynamic 80:20)."""
    years = sorted(df["year"].unique())
    n_years = len(years)
    
    if n_years < MIN_YEARS:
        return None
    
    # Dynamic calculation: at least MIN_TEST_YEARS, up to TEST_FRACTION of total
    n_test = max(MIN_TEST_YEARS, int(np.ceil(n_years * TEST_FRACTION)))
    n_test = min(n_test, n_years - 3)  # Ensure at least 3 training years
    
    test_years = years[-n_test:]
    train_years = years[:-n_test]
    
    train_df = df[df["year"].isin(train_years)]
    test_df = df[df["year"].isin(test_years)]
    
    if len(train_df) < 100 or len(test_df) < 30:
        logging.warning(f"Skipping: Crop has insufficient data after split")
        return None

    return train_df, test_df, train_years, test_years


def train_model(X_train, y_train, X_test, y_test, features: List[str]) -> Dict:
    """Train Random Forest and return metrics."""
    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    return {
        "model": model,
        "train_r2": r2_score(y_train, y_pred_train),
        "test_r2": r2_score(y_test, y_pred_test),
        "mae": mean_absolute_error(y_test, y_pred_test),
        "rmse": rmse(y_test, y_pred_test),
        "importances": pd.Series(model.feature_importances_, index=features),
    }


def compute_pdp_data(model, train_df: pd.DataFrame, crop: str, 
                     features: List[str], importances: pd.Series, 
                     top_n: int = 3) -> List[Dict]:
    """Compute Partial Dependence Plot data for top features."""
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
                    "feature_value": round(float(val), 4),
                    "predicted_yield": round(float(resp), 2),
                })
        except Exception as e:
            logging.warning(f"  PDP failed for {crop}/{feature_name}: {e}")
    
    return pdp_rows



def evaluate_crop(crop_df: pd.DataFrame, crop_name: str) -> Dict:
    """Evaluate all feature sets for a single crop and compute analytics."""
    results = {
        "perf_rows": [],
        "feat_rows": [],
        "cat_rows": [],
        "pdp_rows": [],
        "sens_rows": [],
        "optimal": None,
    }

    split_result = temporal_split(crop_df)
    if split_result is None:
        logging.info(f"  Skipping: insufficient data")
        return results

    train_df, test_df, train_years, test_years = split_result
    logging.info(f"  Train: {train_years[0]}-{train_years[-1]} ({len(train_df):,} rows)")
    logging.info(f"  Test:  {test_years[0]}-{test_years[-1]} ({len(test_df):,} rows)")

    y_train = train_df["yield_value"].values
    y_test = test_df["yield_value"].values

    best_model = None
    best_importances = None
    best_features = None

    for set_name, feature_list in FEATURE_SETS.items():
        features = [f for f in feature_list if f in crop_df.columns]
        if not features:
            continue

        X_train = train_df[features].values
        X_test = test_df[features].values

        result = train_model(X_train, y_train, X_test, y_test, features)

        results["perf_rows"].append({
            "crop": crop_name,
            "feature_set": set_name,
            "train_r2": round(result["train_r2"], 3),
            "test_r2": round(result["test_r2"], 3),
            "r2_gap": round(result["train_r2"] - result["test_r2"], 3),
            "mae": round(result["mae"], 2),
            "rmse": round(result["rmse"], 2),
            "n_train": len(train_df),
            "n_test": len(test_df),
            "train_years": f"{train_years[0]}-{train_years[-1]}",
            "test_years": f"{test_years[0]}-{test_years[-1]}",
        })

        if set_name == "all_features":
            best_model = result["model"]
            best_importances = result["importances"]
            best_features = features

    # Feature and category importance from all_features model
    if best_importances is not None:
        total = best_importances.sum() or 1.0
        
        for feat, imp in best_importances.items():
            cat = CATEGORY_MAP.get(feat, "other")
            results["feat_rows"].append({
                "crop": crop_name,
                "feature": feat,
                "category": cat,
                "importance": round(imp, 4),
                "importance_pct": round(100 * imp / total, 2),
            })

        cat_totals = {}
        for feat, imp in best_importances.items():
            cat = CATEGORY_MAP.get(feat, "other")
            cat_totals[cat] = cat_totals.get(cat, 0) + imp

        for cat, val in cat_totals.items():
            results["cat_rows"].append({
                "crop": crop_name,
                "category": cat,
                "importance": round(val, 4),
                "importance_pct": round(100 * val / total, 1),
            })

    # Advanced analytics for poster
    if best_model is not None and best_features is not None:
        pdp_data = compute_pdp_data(
            best_model, train_df, crop_name, best_features, best_importances
        )
        results["pdp_rows"].extend(pdp_data)
        
    return results


def run_pipeline():
    logging.info("=" * 60)
    logging.info("AGU 2025 - CROP YIELD PREDICTION PIPELINE")
    logging.info(f"Study period: {START_YEAR} onwards")
    logging.info(f"Train/Test split: {int((1-TEST_FRACTION)*100)}:{int(TEST_FRACTION*100)} (dynamic)")
    logging.info("=" * 60)

    # Load and merge data
    yield_df, climate_df, veg_df, soil_df = load_data()
    merged = merge_datasets(yield_df, climate_df, veg_df, soil_df)
    merged = drop_missing_features(merged)

    save_df(merged, PROCESSED_DIR / "merged.csv")
    logging.info(f"  Combined file saved to: {PROCESSED_DIR}/merged.csv")

    # Get crop statistics
    crop_stats = merged.groupby("crop").agg({
        "year": "nunique",
        "yield_value": "count"
    }).rename(columns={"year": "n_years", "yield_value": "n_records"})
    
    valid_crops = crop_stats[
        (crop_stats["n_years"] >= MIN_YEARS) & 
        (crop_stats["n_records"] >= MIN_SAMPLES)
    ].index.tolist()
    
    logging.info(f"\nCrops meeting thresholds (years>={MIN_YEARS}, samples>={MIN_SAMPLES}):")
    for crop in sorted(valid_crops):
        stats = crop_stats.loc[crop]
        logging.info(f"  {crop}: {stats['n_records']:,} records, {stats['n_years']} years")

    # Process each crop
    all_perf = []
    all_feat = []
    all_cat = []
    all_pdp = []
    all_sens = []
    all_optimal = []

    for crop in sorted(valid_crops):
        logging.info(f"\n{'='*40}")
        logging.info(f"CROP: {crop}")
        logging.info("=" * 40)

        crop_df = merged[merged["crop"] == crop].copy()
        results = evaluate_crop(crop_df, crop)

        all_perf.extend(results["perf_rows"])
        all_feat.extend(results["feat_rows"])
        all_cat.extend(results["cat_rows"])
        all_pdp.extend(results["pdp_rows"])

    # Save all results
    logging.info("\n" + "=" * 60)
    logging.info("SAVING RESULTS")
    logging.info("=" * 60)

    if all_perf:
        perf_df = pd.DataFrame(all_perf)
        save_df(perf_df, RESULTS_DIR / "model_performance.csv")

    if all_feat:
        feat_df = pd.DataFrame(all_feat)
        save_df(feat_df, RESULTS_DIR / "feature_importance.csv")

    if all_cat:
        cat_df = pd.DataFrame(all_cat)
        save_df(cat_df, RESULTS_DIR / "category_importance.csv")

    if all_pdp:
        pdp_df = pd.DataFrame(all_pdp)
        save_df(pdp_df, RESULTS_DIR / "pdp_data.csv")

    # Compute improvement summary
    impr_rows = []
    if all_perf:
        perf_df = pd.DataFrame(all_perf)
        for crop in perf_df["crop"].unique():
            crop_perf = perf_df[perf_df["crop"] == crop]
            baseline = crop_perf[crop_perf["feature_set"] == "climate_only"]["test_r2"].values
            best = crop_perf[crop_perf["feature_set"] == "all_features"]["test_r2"].values
            
            if len(baseline) > 0 and len(best) > 0:
                baseline_r2 = baseline[0]
                best_r2 = best[0]
                improvement = best_r2 - baseline_r2
                pct_impr = (improvement / baseline_r2) * 100 if baseline_r2 > 0 else 0
                impr_rows.append({
                    "crop": crop,
                    "baseline_r2": baseline_r2,
                    "best_r2": best_r2,
                    "improvement": round(improvement, 3),
                    "pct_improvement": round(pct_impr, 1),
                })
        
        if impr_rows:
            impr_df = pd.DataFrame(impr_rows)
            save_df(impr_df, RESULTS_DIR / "improvement_summary.csv")

    # Print summaries
    if all_perf:
        perf_df = pd.DataFrame(all_perf)
        
        logging.info("\n" + "=" * 70)
        logging.info("PERFORMANCE SUMMARY - Test R² (Train R²)")
        logging.info("=" * 70)

        summary_rows = []
        for crop in perf_df["crop"].unique():
            crop_data = perf_df[perf_df["crop"] == crop]
            row = {"crop": crop}
            for _, r in crop_data.iterrows():
                row[r["feature_set"]] = f"{r['test_r2']:.3f} ({r['train_r2']:.3f})"
            summary_rows.append(row)

        summary = pd.DataFrame(summary_rows).set_index("crop")
        col_order = ["climate_only", "climate_veg", "climate_soil", "all_features"]
        summary = summary[[c for c in col_order if c in summary.columns]]
        logging.info(f"\n{summary.to_string()}")

    if all_cat:
        cat_df = pd.DataFrame(all_cat)
        logging.info("\n" + "=" * 60)
        logging.info("CATEGORY IMPORTANCE (% by crop)")
        logging.info("=" * 60)
        cat_summary = cat_df.pivot_table(
            index="crop", columns="category", values="importance_pct", aggfunc="first"
        )
        logging.info(f"\n{cat_summary.round(1).to_string()}")

    if impr_rows:
        logging.info("\n" + "=" * 60)
        logging.info("IMPROVEMENT OVER CLIMATE-ONLY BASELINE")
        logging.info("=" * 60)
        for row in impr_rows:
            logging.info(f"  {row['crop']}: {row['baseline_r2']:.3f} → {row['best_r2']:.3f} "
                        f"(+{row['pct_improvement']:.0f}%)")
        avg_impr = np.mean([r["pct_improvement"] for r in impr_rows])
        logging.info(f"\n  Average improvement: +{avg_impr:.0f}%")

    logging.info("\n" + "=" * 60)
    logging.info("PIPELINE COMPLETE")
    logging.info(f"Results saved to: {RESULTS_DIR}")
    logging.info("=" * 60)

    return perf_df if all_perf else pd.DataFrame()


if __name__ == "__main__":
    run_pipeline()
