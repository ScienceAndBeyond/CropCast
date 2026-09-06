"""
evaluate.py - Skill evaluation protocol for county-level crop yield models

This is the analysis that the paper rests on. ml.py fits the models; this
module asks whether their skill is real.

THE PROBLEM THIS ADDRESSES
--------------------------
County-year yield panels contain two very different sources of variance:

  BETWEEN counties  some counties are simply more productive, every year
  WITHIN counties   some years are better than others in the same county

Conventional R2 on a county-year panel is dominated by the between-county part,
which is typically 60-90% of total variance. A model that learns nothing except
which county a row came from can therefore post a high R2. Static soil
covariates make this easy: soil takes exactly one value per county (verified:
860 unique soil tuples for 860 counties in this dataset), so it is a county
fingerprint that a tree ensemble can memorise.

A climate-impacts claim, however, is a WITHIN-county claim: it says weather in a
given year moves yield in that place. `skill_vs_county_mean` scores exactly
that - 1 - SSE(model)/SSE(county-mean predictor) - and it is usually far smaller
than the headline number. `residual_r2` is reported alongside it but is a
DIFFERENT quantity; see score().

WHAT THIS MODULE PRODUCES
-------------------------
1. Variance decomposition - how much of each crop's yield variance is even
   available to a within-county model.

2. A BASELINE LADDER, so every model is scored against progressively stronger
   nulls rather than against zero:
      global_mean         one number for the whole crop
      county_mean         each county's training mean          <- the key null
      county_mean_trend   county mean plus a linear year trend <- adds technology
      climate / +soil / +veg / all                             <- the real models

3. THREE SPLIT SCHEMES, which answer three different questions:
      temporal   train on early years, test on later ones   - forecasting
      loyo       leave one year out                         - interannual skill
      spatial    leave county groups out                    - transfer to new places
   They are not interchangeable and a model can pass one while failing another.

4. A PLACEBO CONTROL. `climate_placebo` is the climate model plus four random
   numbers drawn once per county and held fixed across years - structurally
   identical to soil (static, county-level) but carrying zero agronomic
   information. It differs from `climate_soil` in exactly one respect: soil
   replaced by noise. If real soil beats climate-only by about as much as the
   placebo does, then "soil improves prediction" is a statement about county
   identity, not about soil.

Outputs (RESULTS_DIR):
  evaluation_protocol.csv   - every (crop, split, predictor set) score
  variance_decomposition.csv
  placebo_test.csv          - the county-identity control

Usage:
    python evaluate.py
    python evaluate.py --crops CORN SOYBEANS --splits temporal loyo
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error as rmse
from sklearn.model_selection import GroupKFold

from config import DATA_PATH, merged_filename, results_dirname, logging
from utils import save_df
from ml import (
    CLIMATE_FEATURES,
    SOIL_FEATURES,
    VEG_FEATURES,
    MIN_YEARS,
    MIN_SAMPLES,
    MIN_VEG_MONTHS,
    VEG_MONTHS_COL,
    filter_veg_complete,
    filter_climate_complete,
    RF_PARAMS,
    RANDOM_SEED,
    TEST_FRACTION,
    MIN_TEST_YEARS,
)

PROCESSED_DIR = DATA_PATH / "processed"
MERGED_FILE = PROCESSED_DIR / merged_filename()
RESULTS_DIR = Path(results_dirname())
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Placebo: county-constant random features with no agronomic content.
N_PLACEBO_FEATURES = 4

# Seeds. The temporal split is cheap so it gets several; leave-one-year-out
# refits once per year and spatial refits once per fold, so those use one seed
# and report the fold spread instead.
SEEDS_TEMPORAL = [RANDOM_SEED + i for i in range(5)]
SEEDS_RESAMPLING = [RANDOM_SEED]

SPLIT_SCHEMES = ["temporal", "loyo", "spatial"]

# Bootstrap resamples for confidence intervals. Applied to the TEMPORAL split
# only: it is the headline result and has a single fold, so a CI is well
# defined. loyo/spatial report fold-to-fold spread instead.
N_BOOTSTRAP = 1000
CI_LEVEL = 95

# Detrending. Yields rise over time for reasons unrelated to weather (genetics,
# management), and a Random Forest cannot extrapolate a trend beyond its
# training range, so it systematically under-predicts a later holdout.
#   "none"    raw yields
#   "global"  subtract one linear year trend shared by all counties
#   "county"  subtract a per-county linear trend (falls back to global where a
#             county has too few training years)
# The trend is ALWAYS fitted on training rows only and then applied to the test
# rows; fitting it on the full panel would leak the test period's level.
DETREND_MODES = ["none", "global", "county"]
MIN_YEARS_FOR_COUNTY_TREND = 8


# ---------------------------------------------------------------------------
# PREDICTOR SETS - the baseline ladder
# ---------------------------------------------------------------------------

def build_predictor_sets(df: pd.DataFrame) -> Dict[str, Optional[List[str]]]:
    """
    Ordered ladder from weakest null to the full model. A None value marks an
    analytic baseline (no features, handled directly in fit_and_score).
    """
    def avail(fs):
        """
        Resolve a feature list, refusing to silently shrink it.

        Dropping absent columns keeps the label ("climate_soil") while modelling
        something else, so every comparison against it measures the wrong
        contrast. ml.py raises on this; evaluate.py must agree or the two
        scripts can disagree about what a named feature set contains.
        """
        missing = [f for f in fs if f not in df.columns]
        if missing:
            raise KeyError(
                f"Predictor set is missing {len(missing)} column(s): {missing}. "
                f"Re-run the relevant downloader, or update the feature lists in ml.py."
            )
        return list(fs)

    return {
        "global_mean": None,
        "county_mean": None,
        "county_mean_trend": None,
        # CLIMATE + placebo, not placebo alone. The comparison this supports is
        # "does REAL soil add more than random county-constant noise added to
        # the SAME climate model" - so the placebo arm must differ from
        # climate_soil in exactly one respect: soil replaced by noise. An
        # earlier version used the placebo features by themselves, which made
        # soil_gain and placebo_gain measure different things and inflated the
        # apparent contribution of soil.
        "climate_placebo": avail(CLIMATE_FEATURES) + [f"placebo_{i}" for i in range(N_PLACEBO_FEATURES)],
        "climate_only": avail(CLIMATE_FEATURES),
        "climate_soil": avail(CLIMATE_FEATURES + SOIL_FEATURES),
        "climate_veg": avail(CLIMATE_FEATURES + VEG_FEATURES),
        "all_features": avail(CLIMATE_FEATURES + SOIL_FEATURES + VEG_FEATURES),
    }


def add_placebo_features(df: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Attach N_PLACEBO_FEATURES random values, drawn ONCE PER COUNTY and constant
    across that county's years. Structurally identical to soil (static, county
    level) but with no agronomic content whatsoever.
    """
    df = df.copy()
    rng = np.random.default_rng(seed)
    counties = df["county_fips"].unique()
    draws = {c: rng.normal(size=N_PLACEBO_FEATURES) for c in counties}
    mat = np.vstack([draws[c] for c in df["county_fips"]])
    for i in range(N_PLACEBO_FEATURES):
        df[f"placebo_{i}"] = mat[:, i]
    return df


# ---------------------------------------------------------------------------
# BASELINES
# ---------------------------------------------------------------------------

def predict_baseline(kind: str, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Analytic (non-ML) predictors in the ladder."""
    global_mean = train["yield_value"].mean()

    if kind == "global_mean":
        return np.full(len(test), global_mean)

    county_means = train.groupby("county_fips")["yield_value"].mean()

    if kind == "county_mean":
        return test["county_fips"].map(county_means).fillna(global_mean).to_numpy()

    if kind == "county_mean_trend":
        # County level plus a single shared linear technology trend. Yields rise
        # over time for reasons unrelated to weather (genetics, management), and
        # a Random Forest cannot extrapolate a trend beyond its training range.
        # Any model that fails to beat THIS has not demonstrated weather skill.
        #
        # WITHIN (fixed-effects) ESTIMATOR: year must be centred per county, not
        # just yield. Regressing county-demeaned yield on RAW year conflates the
        # trend with which years each county happens to be observed in. On an
        # unbalanced panel with a true shared trend of 2.0/yr, the uncentred
        # version predicted 114.2 and 134.2 where both truths were 140.0.
        dev = (train["yield_value"] - train["county_fips"].map(county_means)).to_numpy(dtype=float)
        year_means = train.groupby("county_fips")["year"].mean()
        yr_dev = (train["year"] - train["county_fips"].map(year_means)).to_numpy(dtype=float)

        denom = float(np.sum(yr_dev ** 2))
        slope = float(np.sum(yr_dev * dev) / denom) if denom > 0 else 0.0

        # Extrapolate from each county's own mean observation year, so a county
        # observed only in early years is carried forward the right distance.
        global_year_mean = float(train["year"].mean())
        county_part = test["county_fips"].map(county_means).fillna(global_mean).to_numpy()
        test_yr_dev = (
            test["year"] - test["county_fips"].map(year_means).fillna(global_year_mean)
        ).to_numpy(dtype=float)
        return county_part + slope * test_yr_dev

    raise ValueError(f"Unknown baseline: {kind}")


# ---------------------------------------------------------------------------
# DETRENDING
# ---------------------------------------------------------------------------

def _fit_linear(years: np.ndarray, values: np.ndarray):
    """Least-squares line; (0, mean) when there is not enough spread to fit."""
    if len(np.unique(years)) < 2:
        return 0.0, float(np.mean(values)) if len(values) else 0.0
    slope, intercept = np.polyfit(years.astype(float), values.astype(float), 1)
    return float(slope), float(intercept)


def detrend(train: pd.DataFrame, test: pd.DataFrame, mode: str):
    """
    Remove a technology trend from the target, fitted on TRAINING ROWS ONLY.

    Returns (train_out, test_out, trend_test) where the frames carry a detrended
    `yield_value` and `trend_test` is the trend component subtracted from the
    test rows (kept so predictions can be returned to the original scale).

    Fitting the trend on the whole panel would leak the test period's mean level
    into training and inflate every score, so the fit is strictly in-sample.
    """
    if mode == "none":
        return train, test, np.zeros(len(test))

    train = train.copy()
    test = test.copy()

    if mode == "global":
        slope, intercept = _fit_linear(train["year"].to_numpy(), train["yield_value"].to_numpy())
        tr_trend = slope * train["year"].to_numpy() + intercept
        te_trend = slope * test["year"].to_numpy() + intercept

    elif mode == "county":
        # Per-county line, falling back to the global line where a county has
        # too few training years to support its own slope.
        g_slope, g_intercept = _fit_linear(train["year"].to_numpy(), train["yield_value"].to_numpy())
        fits = {}
        for cty, grp in train.groupby("county_fips"):
            if grp["year"].nunique() >= MIN_YEARS_FOR_COUNTY_TREND:
                fits[cty] = _fit_linear(grp["year"].to_numpy(), grp["yield_value"].to_numpy())
            else:
                fits[cty] = (g_slope, g_intercept)

        def _trend(df: pd.DataFrame) -> np.ndarray:
            pars = df["county_fips"].map(lambda c: fits.get(c, (g_slope, g_intercept)))
            slopes = np.array([p[0] for p in pars])
            inters = np.array([p[1] for p in pars])
            return slopes * df["year"].to_numpy(dtype=float) + inters

        tr_trend = _trend(train)
        te_trend = _trend(test)

    else:
        raise ValueError(f"Unknown detrend mode: {mode}")

    train["yield_value"] = train["yield_value"].to_numpy() - tr_trend
    test["yield_value"] = test["yield_value"].to_numpy() - te_trend
    return train, test, te_trend


# ---------------------------------------------------------------------------
# BOOTSTRAP CONFIDENCE INTERVALS
# ---------------------------------------------------------------------------

def bootstrap_cis(y_true: np.ndarray, preds: Dict[str, np.ndarray],
                  county_ref: np.ndarray, groups: np.ndarray,
                  compute_anomaly: bool = True,
                  n_boot: int = N_BOOTSTRAP, seed: int = RANDOM_SEED) -> Dict[str, Dict]:
    """
    Cluster bootstrap over COUNTIES, shared across predictor sets.

    Two design choices that matter:

    1. Resampling counties, not rows. A county contributes many correlated
       county-year observations, so a naive row bootstrap treats ~15 repeated
       measures as 15 independent draws and produces intervals that are far too
       narrow.

    2. One set of resample indices reused for every predictor set. That makes
       the comparisons PAIRED, so a difference like
       (climate_soil - placebo) has a valid interval — which is exactly the
       claim the placebo control needs to support.

    Returns per-set CIs plus CIs on differences against climate_only, and on
    the difference-of-differences (soil gain minus placebo gain).
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_group = {g: np.flatnonzero(groups == g) for g in uniq}

    names = list(preds)
    draws: Dict[str, Dict[str, list]] = {n: {"r2": [], "skill_vs_county_mean": []} for n in names}

    for _ in range(n_boot):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in chosen])
        yt, ref = y_true[idx], county_ref[idx]
        # A resample can be degenerate (all one value); skip rather than emit nan
        if np.ptp(yt) == 0:
            continue
        for n in names:
            yp = preds[n][idx]
            draws[n]["r2"].append(r2_score(yt, yp))
            sse_ref = float(np.sum((yt - ref) ** 2))
            if compute_anomaly and sse_ref > 0:
                draws[n]["skill_vs_county_mean"].append(
                    1.0 - float(np.sum((yt - yp) ** 2)) / sse_ref)

    lo_q, hi_q = (100 - CI_LEVEL) / 2, 100 - (100 - CI_LEVEL) / 2

    def _ci(vals):
        if len(vals) < 20:
            return (None, None)
        return (round(float(np.percentile(vals, lo_q)), 3),
                round(float(np.percentile(vals, hi_q)), 3))

    out: Dict[str, Dict] = {n: {"r2_ci": _ci(draws[n]["r2"]),
                                "skill_ci": _ci(draws[n]["skill_vs_county_mean"])}
                            for n in names}

    # Paired differences against the climate-only reference
    if "climate_only" in draws:
        base = np.asarray(draws["climate_only"]["r2"], dtype=float)
        for n in names:
            if n == "climate_only":
                continue
            arr = np.asarray(draws[n]["r2"], dtype=float)
            if len(arr) == len(base) and len(arr) >= 20:
                d = arr - base
                out[n]["gain_vs_climate_ci"] = _ci(d)
                # P(gain <= 0): how often the resample shows no improvement
                out[n]["p_gain_le_0"] = round(float(np.mean(d <= 0)), 4)

        # The headline test: is the SOIL gain distinguishable from what random
        # county-constant noise achieves? An interval spanning 0 means the
        # "soil helps" result is not separable from county-identity memorisation.
        if {"climate_soil", "climate_placebo"} <= set(draws):
            soil = np.asarray(draws["climate_soil"]["r2"], dtype=float)
            plac = np.asarray(draws["climate_placebo"]["r2"], dtype=float)
            if len(soil) == len(plac) == len(base) and len(soil) >= 20:
                dd = (soil - base) - (plac - base)
                out["_soil_vs_placebo"] = {
                    "diff_of_diffs": round(float(np.mean(dd)), 3),
                    "ci": _ci(dd),
                    "p_soil_not_better_than_placebo": round(float(np.mean(dd <= 0)), 4),
                }

    return out


# ---------------------------------------------------------------------------
# SCORING
# ---------------------------------------------------------------------------

def score(y_true: np.ndarray, y_pred: np.ndarray, county_ref: np.ndarray) -> Dict[str, float]:
    """
    Three distinct quantities. They are NOT interchangeable.

    r2            Conventional R2 against the test-set mean. Inflated by
                  between-county variance that county identity alone explains.

    skill_vs_county_mean
                  1 - SSE(model) / SSE(county-mean predictor).
                  THIS is "does the model beat the county mean": 0 means exactly
                  as good, negative means worse, 1 means perfect. Use this for
                  any claim about improving on the baseline.

    residual_r2   R2 computed on deviations from the county mean. Measures how
                  much of the ANOMALY pattern is captured, but it is scored
                  against the mean of the test anomalies, not against zero.

    Why both exist: when the test period sits above the training era the mean
    anomaly is non-zero, and the two diverge sharply. Predicting the county mean
    exactly scores 0.000 on skill_vs_county_mean but -33.8 on residual_r2 in one
    verified case. An earlier version reported residual_r2 while describing it
    as the county-mean benchmark, which is wrong in both directions.
    """
    sse_model = float(np.sum((y_true - y_pred) ** 2))
    sse_ref = float(np.sum((y_true - county_ref) ** 2))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "skill_vs_county_mean": float(1.0 - sse_model / sse_ref) if sse_ref > 0 else float("nan"),
        "residual_r2": float(r2_score(y_true - county_ref, y_pred - county_ref)),
        "rmse": float(rmse(y_true, y_pred)),
    }


def fit_predict(name: str, features: Optional[List[str]],
                train: pd.DataFrame, test: pd.DataFrame,
                seeds: List[int]) -> Optional[np.ndarray]:
    """
    Predictions for one predictor set on one split, averaged over seeds.

    Returns predictions rather than scores so that the bootstrap can reuse ONE
    set of resample indices across every predictor set, making the comparisons
    paired.
    """
    if features is None:
        return predict_baseline(name, train, test)
    if not features:
        return None

    preds = []
    for seed in seeds:
        model = RandomForestRegressor(random_state=seed, **RF_PARAMS)
        model.fit(train[features], train["yield_value"])
        preds.append(model.predict(test[features]))
    return np.mean(preds, axis=0)


# ---------------------------------------------------------------------------
# SPLIT SCHEMES
# ---------------------------------------------------------------------------

def iter_splits(df: pd.DataFrame, scheme: str, n_spatial_folds: int = 5):
    """
    Yield (train, test, label) for one scheme.

    temporal  - one split: earliest years train, latest test. The forecasting
                question, and the only scheme that respects causality.
    loyo      - leave-one-year-out. Isolates interannual skill with maximum
                data per fit, but uses future years to predict the past, so it
                is a skill diagnostic and NOT a forecast estimate.
    spatial   - leave county groups out. Transfer to unseen places. Folds are
                not time-separated, so it is optimistic as a forecast score.
    """
    years = sorted(df["year"].unique())

    if scheme == "temporal":
        n_test = max(MIN_TEST_YEARS, int(np.ceil(len(years) * TEST_FRACTION)))
        n_test = min(n_test, len(years) - 3)
        if n_test < 1:
            return
        test_years = years[-n_test:]
        tr = df[~df["year"].isin(test_years)]
        te = df[df["year"].isin(test_years)]
        if len(tr) >= 100 and len(te) >= 30:
            yield tr, te, f"{test_years[0]}-{test_years[-1]}"

    elif scheme == "loyo":
        for y in years:
            tr, te = df[df["year"] != y], df[df["year"] == y]
            if len(tr) >= 100 and len(te) >= 20:
                yield tr, te, str(y)

    elif scheme == "spatial":
        if df["county_fips"].nunique() < n_spatial_folds * 2:
            return
        gkf = GroupKFold(n_splits=n_spatial_folds)
        for i, (tr_i, te_i) in enumerate(
            gkf.split(df, df["yield_value"], groups=df["county_fips"])
        ):
            yield df.iloc[tr_i], df.iloc[te_i], f"fold{i + 1}"

    else:
        raise ValueError(f"Unknown split scheme: {scheme}")


# ---------------------------------------------------------------------------
# VARIANCE DECOMPOSITION
# ---------------------------------------------------------------------------

def decompose_variance(df: pd.DataFrame, crop: str) -> Dict[str, float]:
    """
    Split total yield variance into between-county and within-county parts.

    within_county_pct  share of total variance from YEAR-TO-YEAR variation inside
                       counties (the part a weather anomaly could move).
    between_county_pct share from PERSISTENT differences between counties. A
                       climatological weather model, soil, or plain county
                       identity can all explain this part - so a high
                       conventional R2 says little about weather RESPONSE.
    Neither number is a ceiling on any model's R2 (a weather model explains
    between-county variance too, via climatology). They describe where the
    variance sits, which is what makes conventional R2 uninformative here.
    """
    county_mean = df.groupby("county_fips")["yield_value"].transform("mean")
    total = df["yield_value"].var()
    between = county_mean.var()
    within = (df["yield_value"] - county_mean).var()
    return {
        "crop": crop,
        "n_obs": len(df),
        "n_counties": df["county_fips"].nunique(),
        "n_years": df["year"].nunique(),
        "total_variance": round(float(total), 2),
        "between_county_pct": round(100 * float(between) / float(total), 1) if total else np.nan,
        "within_county_pct": round(100 * float(within) / float(total), 1) if total else np.nan,
        "soil_tuples_per_county": (
            f"{df[[f for f in SOIL_FEATURES if f in df.columns]].drop_duplicates().shape[0]}"
            f"/{df['county_fips'].nunique()}"
        ),
    }


# ---------------------------------------------------------------------------
# DRIVER
# ---------------------------------------------------------------------------

def evaluate_crop(df: pd.DataFrame, crop: str, schemes: List[str],
                  detrend_modes: List[str]) -> tuple:
    """Returns (score_rows, bootstrap_rows) for one crop."""
    rows, boot_rows = [], []
    predictor_sets = build_predictor_sets(df)

    for mode in detrend_modes:
        for scheme in schemes:
            compute_anomaly = scheme != "spatial"
            seeds = SEEDS_TEMPORAL if scheme == "temporal" else SEEDS_RESAMPLING

            per_set: Dict[str, List[Dict]] = {k: [] for k in predictor_sets}
            n_folds = 0

            for train_raw, test_raw, label in iter_splits(df, scheme):
                # Trend fitted in-sample, then removed from both sides
                train, test, _ = detrend(train_raw, test_raw, mode)
                n_folds += 1

                y_test = test["yield_value"].to_numpy()
                county_ref = (
                    predict_baseline("county_mean", train, test)
                    if compute_anomaly
                    else np.full(len(test), train["yield_value"].mean())
                )

                preds: Dict[str, np.ndarray] = {}
                for name, feats in predictor_sets.items():
                    p = fit_predict(name, feats, train, test, seeds)
                    if p is None:
                        continue
                    preds[name] = p
                    per_set[name].append(score(y_test, p, county_ref))

                # CIs on the single-fold headline split only
                if scheme == "temporal" and preds:
                    cis = bootstrap_cis(
                        y_test, preds, county_ref,
                        groups=test["county_fips"].to_numpy(),
                        compute_anomaly=compute_anomaly,
                    )
                    sp = cis.pop("_soil_vs_placebo", None)
                    for name, c in cis.items():
                        boot_rows.append({
                            "crop": crop, "detrend": mode, "predictor_set": name,
                            "r2_lo": c["r2_ci"][0], "r2_hi": c["r2_ci"][1],
                            "skill_lo": c["skill_ci"][0],
                            "skill_hi": c["skill_ci"][1],
                            "gain_vs_climate_lo": c.get("gain_vs_climate_ci", (None, None))[0],
                            "gain_vs_climate_hi": c.get("gain_vs_climate_ci", (None, None))[1],
                            "p_gain_le_0": c.get("p_gain_le_0"),
                        })
                    if sp:
                        boot_rows.append({
                            "crop": crop, "detrend": mode,
                            "predictor_set": "SOIL_MINUS_PLACEBO",
                            "r2_lo": sp["ci"][0], "r2_hi": sp["ci"][1],
                            "gain_vs_climate_lo": sp["ci"][0],
                            "gain_vs_climate_hi": sp["ci"][1],
                            "p_gain_le_0": sp["p_soil_not_better_than_placebo"],
                        })

            if n_folds == 0:
                logging.warning(f"  {crop}/{scheme}/detrend={mode}: no usable folds")
                continue

            for name, scores in per_set.items():
                if not scores:
                    continue
                rows.append({
                    "crop": crop,
                    "detrend": mode,
                    "split_scheme": scheme,
                    "predictor_set": name,
                    "n_folds": len(scores),
                    "r2": round(float(np.mean([s["r2"] for s in scores])), 3),
                    "r2_fold_sd": round(float(np.std([s["r2"] for s in scores])), 3),
                    "rmse": round(float(np.mean([s["rmse"] for s in scores])), 2),
                    # Reported alongside skill because the docstring promises it
                    # and because the two diverging is itself diagnostic.
                    "residual_r2": round(float(np.mean([s["residual_r2"] for s in scores])), 3),
                    "skill_vs_county_mean": (round(float(np.mean([s["skill_vs_county_mean"] for s in scores])), 3)
                                   if compute_anomaly else None),
                })

    return rows, boot_rows


def run(crops: Optional[List[str]] = None, schemes: Optional[List[str]] = None,
        detrend_modes: Optional[List[str]] = None) -> None:
    schemes = schemes or SPLIT_SCHEMES
    detrend_modes = detrend_modes or ["none"]

    if not MERGED_FILE.exists():
        raise FileNotFoundError(f"{MERGED_FILE} not found - run ml.py first to build it")

    merged = pd.read_csv(MERGED_FILE, dtype={"county_fips": str})
    logging.info(f"Loaded {len(merged):,} rows from {MERGED_FILE}")

    # Same column and same rule as ml.py. merged.csv is written BEFORE ml.py
    # applies its completeness filter, so evaluate.py must re-apply it or it
    # scores rows the model never trained on. The column is veg_n_months (the
    # rename that stops it colliding with climate's own n_months); checking the
    # old name here silently filtered nothing.
    if VEG_MONTHS_COL in merged.columns:
        n0 = len(merged)
        merged = filter_veg_complete(merged)   # same rule as ml.py
        merged = filter_climate_complete(merged)   # same rule as ml.py
        if n0 != len(merged):
            logging.info(f"Dropped {n0 - len(merged):,} rows with incomplete vegetation seasons")
    else:
        raise KeyError(
            f"merged.csv has no '{VEG_MONTHS_COL}' column, so vegetation completeness "
            f"cannot be enforced and evaluate.py would score rows ml.py excluded. "
            f"Re-run ml.py to regenerate merged.csv."
        )

    all_feats = [f for f in CLIMATE_FEATURES + SOIL_FEATURES + VEG_FEATURES
                 if f in merged.columns]
    merged = merged.dropna(subset=all_feats + ["yield_value"])
    merged = add_placebo_features(merged)

    stats = merged.groupby("crop").agg(n_years=("year", "nunique"),
                                       n_records=("yield_value", "size"))
    eligible = stats[(stats.n_years >= MIN_YEARS) & (stats.n_records >= MIN_SAMPLES)].index.tolist()
    if crops:
        eligible = [c for c in eligible if c in crops]

    logging.info(f"Evaluating {len(eligible)} crops: {sorted(eligible)}")
    logging.info(f"Split schemes: {schemes}")
    logging.info(f"Detrend modes: {detrend_modes}")
    logging.info(f"Bootstrap: {N_BOOTSTRAP} county-clustered resamples, {CI_LEVEL}% CI (temporal split)")
    logging.info(f"Features in use ({len(all_feats)}): {all_feats}")

    all_rows, var_rows, boot_all = [], [], []
    for crop in sorted(eligible):
        logging.info("=" * 70)
        logging.info(f"CROP: {crop}")
        d = merged[merged["crop"] == crop].copy()
        var_rows.append(decompose_variance(d, crop))
        r, b = evaluate_crop(d, crop, schemes, detrend_modes)
        all_rows.extend(r); boot_all.extend(b)

    if not all_rows:
        logging.warning("No results produced")
        return

    res = pd.DataFrame(all_rows)
    var = pd.DataFrame(var_rows)
    save_df(res, RESULTS_DIR / "evaluation_protocol.csv")
    save_df(var, RESULTS_DIR / "variance_decomposition.csv")
    boot = pd.DataFrame(boot_all) if boot_all else None
    if boot is not None:
        save_df(boot, RESULTS_DIR / "bootstrap_ci.csv")

    # Placebo comparison: how much of the "soil helps" gain is county identity?
    pl = []
    for crop in res["crop"].unique():
        t = res[(res.crop == crop) & (res.split_scheme == "temporal")
                & (res.detrend == detrend_modes[0])].set_index("predictor_set")
        if not {"climate_only", "climate_soil", "climate_placebo"} <= set(t.index):
            continue
        clim = t.loc["climate_only", "r2"]
        soil_gain = t.loc["climate_soil", "r2"] - clim
        plac_gain = t.loc["climate_placebo", "r2"] - clim
        pl.append({
            "crop": crop,
            "climate_only_r2": clim,
            "climate_soil_r2": t.loc["climate_soil", "r2"],
            "placebo_r2": t.loc["climate_placebo", "r2"],
            "soil_gain": round(float(soil_gain), 3),
            "placebo_gain": round(float(plac_gain), 3),
            # Share of the soil gain that random county-constant noise reproduces
            "pct_gain_explained_by_county_id": (
                round(100 * float(plac_gain) / float(soil_gain), 1)
                if soil_gain > 0.001 else None
            ),
        })
    if pl:
        pl_df = pd.DataFrame(pl)
        save_df(pl_df, RESULTS_DIR / "placebo_test.csv")

    _report(res, var, pd.DataFrame(pl) if pl else None, schemes, detrend_modes, boot)


def _report(res: pd.DataFrame, var: pd.DataFrame,
            placebo: Optional[pd.DataFrame], schemes: List[str],
            detrend_modes: List[str], boot: Optional[pd.DataFrame] = None) -> None:
    order = ["global_mean", "county_mean", "county_mean_trend", "climate_placebo",
             "climate_only", "climate_soil", "climate_veg", "all_features"]

    def pivot(sub: pd.DataFrame, metric: str) -> str:
        t = sub.pivot_table(index="predictor_set", columns="crop", values=metric, aggfunc="first")
        t = t.reindex([o for o in order if o in t.index])
        t.index.name = None
        t.columns.name = None
        return t.to_string()

    logging.info("=" * 70)
    logging.info("VARIANCE DECOMPOSITION")
    logging.info("=" * 70)
    logging.info(f"\n{var[['crop','n_obs','n_counties','between_county_pct','within_county_pct','soil_tuples_per_county']].to_string(index=False)}")
    logging.info(
        "\n  within_county_pct = year-to-year variance inside counties;"
        "\n  between_county_pct = persistent differences that climatology, soil"
        "\n  or county identity can all explain. Neither is a ceiling on any R2."
        "\n  soil_tuples_per_county at 1:1 means soil uniquely identifies the county."
    )

    for mode in detrend_modes:
      for scheme in schemes:
        sub = res[(res.split_scheme == scheme) & (res.detrend == mode)]
        if sub.empty:
            continue
        logging.info("=" * 70)
        logging.info(f"SPLIT: {scheme.upper()}   detrend={mode}   - conventional R2")
        logging.info("=" * 70)
        logging.info(f"\n{pivot(sub, 'r2')}")
        if scheme != "spatial":
            logging.info(f"\n  SKILL vs COUNTY MEAN (0 = tie, <0 = worse)\n\n{pivot(sub, 'skill_vs_county_mean')}")

    if placebo is not None and len(placebo):
        logging.info("=" * 70)
        logging.info("PLACEBO CONTROL - is the soil gain really county identity?")
        logging.info("=" * 70)
        logging.info(f"\n{placebo.to_string(index=False)}")
        logging.info(
            "\n  placebo_gain comes from 4 random numbers held constant per county."
            "\n  Whatever share of soil_gain it reproduces is county-identity"
            "\n  memorisation, not soil science."
        )

    logging.info("=" * 70)
    logging.info(f"Results written to {RESULTS_DIR}")
    logging.info("=" * 70)


def main():
    ap = argparse.ArgumentParser(description="Skill evaluation protocol for crop yield models")
    ap.add_argument("--crops", nargs="+", default=None, help="Subset of crops (default: all eligible)")
    ap.add_argument("--splits", nargs="+", default=SPLIT_SCHEMES, choices=SPLIT_SCHEMES)
    ap.add_argument("--practice-mode", default=None,
                    choices=["aggregate", "split", "both"],
                    help="Which study to evaluate. Selects the matching merged table "
                         "AND results directory, so the two studies cannot cross.")
    ap.add_argument("--results-dir", default=None,
                    help="Where to read merged.csv's study from and write outputs. "
                         "Must match the ml.py run being evaluated.")
    ap.add_argument("--detrend", nargs="+", default=["none"], choices=DETREND_MODES,
                    help="Target detrending; pass several to compare (default: none)")
    args = ap.parse_args()
    global RESULTS_DIR, MERGED_FILE
    if args.practice_mode:
        MERGED_FILE = PROCESSED_DIR / merged_filename(args.practice_mode)
        RESULTS_DIR = Path(args.results_dir or results_dirname(args.practice_mode))
    elif args.results_dir:
        RESULTS_DIR = Path(args.results_dir)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    logging.info(f"merged={MERGED_FILE.name} -> results={RESULTS_DIR}")
    run(crops=args.crops, schemes=args.splits, detrend_modes=args.detrend)


if __name__ == "__main__":
    main()
