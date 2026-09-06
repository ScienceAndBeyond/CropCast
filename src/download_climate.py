
# download_climate.py
# =============================================================================
# gridMET climate downloader (county-year)
# Outputs ML-aligned columns: TMIN, TMAX, PRCP, VPD, ETO, SRAD, WIND
# Uses growing season from config.py via get_growing_season(state_abbr)
#
# Notes:
# - BASE_TEMP_C is kept (you set it to 18.3), but degree-days are NOT computed
#   because your ML features don't include them.
# - Heavy work is server-side (GEE). Client only fetches final county tables.
# - Year chunking reduces getInfo() size/timeouts.
# - Checkpointing prevents re-downloading.
# =============================================================================

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd

# Imported lazily: aggregate_to_season() / build_season_table() (and therefore
# --aggregate-only) need no Earth Engine at all, and must stay runnable on a
# machine that has never authenticated to GEE. `from __future__ import
# annotations` above keeps the ee.* type hints below from being evaluated.
try:
    import ee
except ImportError:
    ee = None

from config import (
    DATA_PATH,
    DATA_PATH_RAW,
    DEFAULT_STATES,
    ANALYSIS_YEARS,
    GEE_PROJECT_ID,
    get_growing_season,  # returns (start_month, end_month_exclusive)
    get_required_months,
    resolve_states,
)
from utils import (
    native_scale,
    fetch_state_fips,
    get_tiger_counties_fc,
    save_df,
    logging,
)

# -----------------------------------------------------------------------------
# ML climate feature names (must match your model)
# -----------------------------------------------------------------------------
# Months fetched from Earth Engine, derived from the configured growing seasons
# (widest season 3..9, plus one month of slack either side).
#
# This was list(range(1, 13)) — all 12 months — on the theory that it kept any
# future season buildable offline. It did not pay for itself: no configured
# season uses Jan, Feb, Nov or Dec, and winter crops (the only reason to want
# them) are excluded upstream AND span two calendar years, which a single-year
# month window cannot express anyway. Five of twelve months were pure waste.
DOWNLOAD_MONTHS = get_required_months(slack=1)

# Monthly bands, split by how they aggregate to a season.
MEAN_BANDS = ["TMIN", "TMAX", "VPD", "SRAD", "WIND"]   # days-weighted mean
SUM_BANDS = ["PRCP", "ETO", "GDD_TMAX", "EDD_TMAX"]              # sum, then -> per-day rate
MONTHLY_BANDS = MEAN_BANDS + SUM_BANDS + ["HOT_DAYS", "TMAX_MAX", "N_DAYS"]

# Season-level columns produced by aggregate_to_season() and consumed by ml.py.
# PRCP/ETO/GDD/EDD are per-day RATES, not totals: growing-season windows differ
# by state (config.GROWING_SEASON_EXCEPTIONS), so a season SUM is not comparable
# across states and lets the model learn "low PRCP => Minnesota". SEASON_DAYS is
# exported so totals can be recovered as PRCP * SEASON_DAYS if ever needed.
CLIMATE_FEATURES = MEAN_BANDS + SUM_BANDS + [
    "HOT_DAYS", "HOT_DAY_FRAC", "TMAX_MAX", "SEASON_DAYS",
]

# -----------------------------------------------------------------------------
# Output + checkpoint
# -----------------------------------------------------------------------------
PROCESSED_DIR = DATA_PATH / "processed"
RAW_DIR = DATA_PATH_RAW

MONTHLY_OUTPUT_FILE = RAW_DIR / "climate_monthly.csv"   # re-windowable source
OUTPUT_FILE = PROCESSED_DIR / "climate.csv"             # season table for ml.py
CHECKPOINT_FILE = RAW_DIR / "climate_checkpoint.json"

# -----------------------------------------------------------------------------
# gridMET params
# -----------------------------------------------------------------------------
GRIDMET_COLLECTION = "IDAHO_EPSCOR/GRIDMET"
# Resolved from the collection itself; 4000 m was off gridMET's 4638.31 m grid
# and forced resampling (+0.018 C on TMIN, -0.23% on PRCP, systematically).
GRIDMET_SCALE_M = None  # set by _resolve_scale() on first use
GRIDMET_TILE_SCALE = 4

BASE_TEMP_C = 18.3  # legacy HDD/CDD base, retained for reference

# Heat indices derived from DAILY MAXIMUM temperature only.
#
# NAMING: these are NOT conventional growing degree days, and NOT the
# Schlenker & Roberts (2009) degree-day measures an earlier comment claimed.
#   - Conventional GDD uses (Tmax+Tmin)/2 - Tbase.
#   - Schlenker & Roberts integrate the WITHIN-DAY temperature curve (sine
#     interpolation between Tmin and Tmax) to get degree-HOURS above a threshold.
# Both of ours use Tmax alone, and aggregate_to_season divides by the number of
# days, so they are per-day RATES rather than seasonal accumulations.
# They remain useful heat-exposure indices; they must not be reported as GDD/EDD
# in the literature sense. Columns are named GDD_TMAX / EDD_TMAX accordingly.
GDD_BASE_C = 10.0    # beneficial growth accumulates above this
GDD_CAP_C = 29.0     # ...and stops accumulating here
EDD_THRESHOLD_C = 29.0  # extreme/killing degree days above this


# -----------------------------------------------------------------------------
# EE init
# -----------------------------------------------------------------------------
def init_gee(project_id: Optional[str] = GEE_PROJECT_ID) -> None:
    if ee is None:
        raise RuntimeError(
            "earthengine-api is not installed, so climate data cannot be downloaded. "
            "Install with: pip install earthengine-api  "
            "(--aggregate-only works without it.)"
        )
    try:
        if project_id:
            ee.Initialize(project=project_id)
        else:
            ee.Initialize()
        logging.info("✓ Google Earth Engine initialized")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Earth Engine: {e}") from e


# -----------------------------------------------------------------------------
# Checkpoint helpers
# -----------------------------------------------------------------------------
def load_checkpoint(path: Path) -> set[str]:
    """
    Stored as {"completed": ["IA_2005_2010", ...]}
    """
    if not path.exists():
        return set()
    try:
        obj = json.loads(path.read_text())
        if isinstance(obj, dict):
            return set(obj.get("completed", []))
        if isinstance(obj, list):
            return set(obj)
        return set()
    except Exception:
        return set()


def save_checkpoint(path: Path, completed: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"completed": sorted(completed)}, indent=2))


def chunk_key(state_abbr: str, y0: int, y1: int) -> str:
    return f"{state_abbr.upper()}_{int(y0)}_{int(y1)}"


# -----------------------------------------------------------------------------
# Cache county FCs per state
# -----------------------------------------------------------------------------
_COUNTY_FC_CACHE: Dict[str, ee.FeatureCollection] = {}


def _standardize_county_props(fc: ee.FeatureCollection) -> ee.FeatureCollection:
    """
    Ensure we always have:
      state_fips, county_fips, county_name
    regardless of what utils.get_tiger_counties_fc() returns.
    TIGER/2018/Counties typically has: STATEFP, COUNTYFP, GEOID, NAME.
    """
    def _add_props(f: ee.Feature) -> ee.Feature:
        return f.set({
            "state_fips": f.get("STATEFP"),
            "county_fips": f.get("GEOID"),
            "county_name": f.get("NAME"),
        })

    return fc.map(_add_props)


def _resolve_scale() -> float:
    """gridMET's native pixel size, resolved once and memoised."""
    global GRIDMET_SCALE_M
    if GRIDMET_SCALE_M is None:
        GRIDMET_SCALE_M = native_scale(GRIDMET_COLLECTION)
    return GRIDMET_SCALE_M


def get_counties_fc(state_abbr: str) -> ee.FeatureCollection:
    st = state_abbr.upper()
    if st in _COUNTY_FC_CACHE:
        return _COUNTY_FC_CACHE[st]

    state_fips = fetch_state_fips([st])[0]
    fc = get_tiger_counties_fc(state_fips)
    fc = _standardize_county_props(fc)

    _COUNTY_FC_CACHE[st] = fc
    return fc


# -----------------------------------------------------------------------------
# Server-side climate feature engineering (ML-aligned names)
# -----------------------------------------------------------------------------
def build_monthly_climate_image(year: int, month: int) -> ee.Image:
    """
    One month of gridMET reduced to a single ee.Image.

    Coverage assumes a complete daily collection: N_DAYS is calendar length.
    At the 2026-09-06 handoff, image counts matched calendar days in six sampled
    month-years spanning 2008-2025, including February 2020. Missing days within
    a month are not detected. Future fix (requires re-download): emit col.size()
    as N_IMAGES alongside N_DAYS and incorporate it into coverage.

    WHY MONTHLY AND NOT SEASONAL
    ----------------------------
    Aggregating to a growing season during download bakes one season definition
    into the data: changing it later (crop-specific windows, a sensitivity test,
    winter crops) means re-downloading everything from Earth Engine. Monthly
    rows are re-windowable offline, so the season becomes an analysis choice
    rather than a download choice. Vegetation is already monthly; this matches.

    Every band aggregates cleanly from months to any season:
      MEANS (weight by N_DAYS): TMIN, TMAX, VPD, SRAD, WIND
      SUMS  (add, then / days): PRCP, ETO, GDD, EDD
      COUNT (add):              HOT_DAYS
      MAX   (take max):         TMAX_MAX
      N_DAYS                    days actually present in the month

    Note there is deliberately no percentile band: a season percentile cannot be
    recovered from monthly percentiles. HOT_DAYS (a count above a threshold) and
    EDD carry the same hot-tail information and DO aggregate additively.
    """
    start = ee.Date.fromYMD(int(year), int(month), 1)
    end = start.advance(1, "month")

    col = ee.ImageCollection(GRIDMET_COLLECTION).filterDate(start, end)
    n_days = ee.Number(end.difference(start, "day"))

    tmin = col.select("tmmn").mean().subtract(273.15).rename("TMIN")
    tmax = col.select("tmmx").mean().subtract(273.15).rename("TMAX")
    vpd = col.select("vpd").mean().rename("VPD")
    srad = col.select("srad").mean().rename("SRAD")
    wind = col.select("vs").mean().rename("WIND")

    # Monthly TOTALS. Converted to per-day rates at season aggregation time so
    # that states with different window lengths stay comparable.
    prcp = col.select("pr").sum().rename("PRCP")
    eto = col.select("eto").sum().rename("ETO")

    # --- Daily heat exposure -------------------------------------------------
    def _daily_tmax_c(img: ee.Image) -> ee.Image:
        return img.select("tmmx").subtract(273.15).rename("tmax_c")

    tmax_c_col = col.map(_daily_tmax_c)

    gdd = tmax_c_col.map(
        lambda i: i.select("tmax_c").min(GDD_CAP_C).subtract(GDD_BASE_C).max(0).rename("GDD_TMAX")
    ).sum().rename("GDD_TMAX")

    edd = tmax_c_col.map(
        lambda i: i.select("tmax_c").subtract(EDD_THRESHOLD_C).max(0).rename("EDD_TMAX")
    ).sum().rename("EDD_TMAX")

    hot_days = tmax_c_col.map(
        lambda i: i.select("tmax_c").gt(EDD_THRESHOLD_C).rename("HOT_DAYS")
    ).sum().rename("HOT_DAYS")

    # TMAX_MAX DEFINITION (it is not the obvious quantity):
    #   per pixel: the month's hottest daily maximum
    #   then reduceRegions takes the COUNTY MEAN of those pixel maxima
    #   then aggregate_to_season takes the MAX across months
    # So it is "the hottest month's county-average peak temperature", NOT the
    # county's absolute seasonal maximum (which would be a max over pixels too)
    # and NOT the county mean of pixel-level SEASONAL maxima (max and mean do
    # not commute). Report it with that wording or not at all.
    tmax_max = tmax_c_col.max().rename("TMAX_MAX")

    days = ee.Image.constant(n_days).toFloat().rename("N_DAYS")

    return ee.Image.cat([
        tmin, tmax, vpd, srad, wind,
        prcp, eto, gdd, edd, hot_days, tmax_max, days,
    ])


def build_year_stack(year: int) -> ee.Image:
    """
    All DOWNLOAD_MONTHS for one year stacked as BANDS of a single image, named
    "{BAND}_{MM}" (e.g. TMIN_04, GDD_07).

    WHY STACK INSTEAD OF REDUCING PER MONTH
    ---------------------------------------
    Reducing each month separately meant one reduceRegions per month, so a
    2-year chunk issued 18 aggregations in a single request. Earth Engine
    rejects that with "Too many concurrent aggregations" and the retries then
    burn the request budget. Stacking months into bands gives ONE aggregation
    per state-year instead - the shape download_vegetation.py already uses
    successfully via toBands().

    NOTE: .select(MONTHLY_BANDS) before .rename() is load-bearing.
    build_monthly_climate_image() concatenates bands in a different order than
    MONTHLY_BANDS lists them (PRCP/ETO sit third and fifth there, but after the
    means here), so renaming positionally without selecting first would silently
    mislabel every variable.
    """
    imgs = []
    for m in DOWNLOAD_MONTHS:
        mm = f"{int(m):02d}"
        img = build_monthly_climate_image(year, int(m)).select(MONTHLY_BANDS)
        imgs.append(img.rename([f"{b}_{mm}" for b in MONTHLY_BANDS]))
    return ee.Image.cat(imgs)


def reduce_year_to_counties(state_abbr: str, year: int,
                            counties_fc: ee.FeatureCollection) -> ee.FeatureCollection:
    """ONE aggregation covering every month of a year, for all counties."""
    fc = build_year_stack(year).reduceRegions(
        collection=counties_fc,
        reducer=ee.Reducer.mean(),
        scale=_resolve_scale(),
        tileScale=GRIDMET_TILE_SCALE,
    )

    def _decorate(f: ee.Feature) -> ee.Feature:
        return f.set({"state_abbr": state_abbr.upper(), "year": int(year)}).setGeometry(None)

    fc = fc.map(_decorate)

    keep = (["state_abbr", "year", "state_fips", "county_fips", "county_name"]
            + [f"{b}_{int(m):02d}" for m in DOWNLOAD_MONTHS for b in MONTHLY_BANDS])
    return fc.select(keep)


def process_state_years_chunk(state_abbr: str, years: List[int]) -> pd.DataFrame:
    """
    Server-side compute for several years, ONE getInfo() for the chunk, then
    unpivot the wide {BAND}_{MM} columns back into long monthly rows.
    """
    counties_fc = get_counties_fc(state_abbr)

    fcs = [reduce_year_to_counties(state_abbr, yr, counties_fc) for yr in years]
    combined = ee.FeatureCollection(fcs).flatten()

    info = combined.getInfo()
    ids = ["state_abbr", "year", "state_fips", "county_fips", "county_name"]

    rows = []
    for feat in info.get("features", []):
        props = feat.get("properties", {}) or {}
        base = {k: props.get(k) for k in ids}
        for m in DOWNLOAD_MONTHS:
            mm = f"{int(m):02d}"
            row = {**base, "month": int(m)}
            for b in MONTHLY_BANDS:
                row[b] = props.get(f"{b}_{mm}")
            # A month with no imagery yields all-None; keep it out rather than
            # letting a phantom row reach the season aggregation.
            if any(row[b] is not None for b in MONTHLY_BANDS):
                rows.append(row)

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# MONTHLY -> GROWING SEASON (county-year)
# -----------------------------------------------------------------------------
def aggregate_to_season(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse monthly rows to one county-year row using each state's growing
    season from config.get_growing_season().

    Re-run this alone (via --aggregate-only) to change the season definition;
    no Earth Engine access required.

    PRCP/ETO/GDD/EDD are emitted as PER-DAY RATES so that states with different
    window lengths remain comparable - MN's May-Sep window is 153 days against
    183 for an Apr-Sep state, which would otherwise understate MN totals by 16%
    purely by construction. HOT_DAYS is kept as a count (days above the
    threshold) and also as a fraction of the season.
    """
    if monthly_df.empty:
        return pd.DataFrame()

    df = monthly_df.copy()

    # Keep only each state's growing-season months
    bounds = df["state_abbr"].map(lambda s: get_growing_season(s))
    df["_gs_start"] = [b[0] for b in bounds]
    df["_gs_end"] = [b[1] for b in bounds]
    df = df[(df["month"] >= df["_gs_start"]) & (df["month"] < df["_gs_end"])]

    keys = ["state_abbr", "state_fips", "county_fips", "county_name", "year"]

    # Expected season length from the CALENDAR, not from the rows present.
    # Summing N_DAYS over surviving rows makes a missing month invisible: one
    # remaining month would report 100% coverage of a six-month season.
    import calendar as _cal

    def _expected_days(row) -> int:
        gs0, gs1 = get_growing_season(row["state_abbr"])
        return sum(_cal.monthrange(int(row["year"]), m)[1] for m in range(gs0, gs1))

    expected = (
        df[["state_abbr", "year"]].drop_duplicates()
        .assign(_EXPECTED_DAYS=lambda d: d.apply(_expected_days, axis=1))
    )

    # A month with a missing value must not contribute its days to the
    # denominator: pandas skips NaN in the numerator, so one observed month at
    # 10 C plus one missing month previously averaged to 5 C - a plausible,
    # entirely wrong number. Weights are computed PER VARIABLE from the months
    # where that variable is actually present.
    w = df["N_DAYS"]

    # Days-weighted means
    # EVERY accumulating variable needs its own observed-days denominator, not
    # just the means. Summed variables were the worse case: pandas sum() treats
    # NaN as 0, so one observed month of 30 mm beside one missing month gave
    # 0.5 mm/day instead of 1.0, and a fully missing variable gave 0.0 - a FALSE
    # ZERO that reads as drought rather than as absent data. min_count=1 makes
    # an all-missing group produce NaN instead.
    accumulating = SUM_BANDS + ["HOT_DAYS"]
    for c in MEAN_BANDS:
        df["_w_" + c] = df[c] * w
        df["_d_" + c] = w.where(df[c].notna())
    for c in accumulating:
        df["_d_" + c] = w.where(df[c].notna())
    # TMAX_MAX aggregates by max, so it is neither a mean nor a sum, but it
    # still needs coverage tracked or COVERAGE_MIN silently ignores it.
    df["_d_TMAX_MAX"] = w.where(df["TMAX_MAX"].notna())
    covered = accumulating + MEAN_BANDS + ["TMAX_MAX"]

    agg = {"N_DAYS": ("N_DAYS", "sum"), "n_months": ("month", "count")}
    agg.update({"_w_" + c: ("_w_" + c, "sum") for c in MEAN_BANDS})
    agg.update({"_d_" + c: ("_d_" + c, "sum") for c in covered})
    agg.update({c: (c, lambda x: x.sum(min_count=1)) for c in accumulating})
    agg["TMAX_MAX"] = ("TMAX_MAX", "max")

    out = df.groupby(keys, as_index=False).agg(**agg)

    days = out["N_DAYS"].replace(0, pd.NA)
    for c in MEAN_BANDS:
        obs_days = out["_d_" + c].replace(0, pd.NA)   # per-variable denominator
        out[c] = out["_w_" + c] / obs_days
        # _d_ is kept until the coverage block below has used it.
        out = out.drop(columns=["_w_" + c])
    for c in SUM_BANDS:
        obs = out["_d_" + c].replace(0, pd.NA)
        out[c] = out[c] / obs           # totals -> per-day rates over OBSERVED days

    hot_obs = out["_d_HOT_DAYS"].replace(0, pd.NA)
    out["HOT_DAY_FRAC"] = out["HOT_DAYS"] / hot_obs

    # PER-VARIABLE coverage against the EXPECTED season, plus a conservative
    # minimum across all of them.
    #
    # A single COVERAGE_FRAC taken from precipitation could read 1.0 while ETO
    # or the heat indices had missing months. And dividing by the days actually
    # present would hide an absent month entirely - hence _EXPECTED_DAYS.
    #
    # SEASON_DAYS remains the observed total. To recover a season TOTAL from a
    # rate use  RATE * EXPECTED_DAYS * COVERAGE_<VAR>, or drop rows below a
    # coverage threshold (ml.py enforces MIN_CLIMATE_COVERAGE).
    out = out.merge(expected, on=["state_abbr", "year"], how="left")
    exp_days = out["_EXPECTED_DAYS"].replace(0, pd.NA)
    for c in covered:
        out[f"COVERAGE_{c}"] = (out["_d_" + c] / exp_days).astype(float)
    cov_cols = [f"COVERAGE_{c}" for c in covered]
    out["COVERAGE_MIN"] = out[cov_cols].min(axis=1)
    out["EXPECTED_DAYS"] = out["_EXPECTED_DAYS"]
    out = out.drop(columns=[f"_d_{c}" for c in covered] + ["_EXPECTED_DAYS"])
    out = out.rename(columns={"N_DAYS": "SEASON_DAYS"})

    logging.info(f"Aggregated to {len(out):,} county-year rows")
    return out


# -----------------------------------------------------------------------------
# Retry wrapper (helps with transient EE failures)
# -----------------------------------------------------------------------------
def run_with_retries(fn, max_retries: int = 3, base_sleep: float = 4.0):
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_err = e
            sleep_s = base_sleep * (2 ** (attempt - 1))
            logging.warning(f"Attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                logging.info(f"Retrying in {sleep_s:.1f}s...")
                time.sleep(sleep_s)
    raise last_err


# -----------------------------------------------------------------------------
# Chunking helper
# -----------------------------------------------------------------------------
def chunk_years(all_years: List[int], chunk_size: int) -> List[List[int]]:
    return [all_years[i:i + chunk_size] for i in range(0, len(all_years), chunk_size)]


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
def run(
    states: List[str],
    years: Tuple[int, int],
    year_chunk: int = 6,
    resume: bool = True,
    force: bool = False,
) -> None:
    init_gee()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # --force deletes the output files for EVERY state, so any previously
    # completed entry now points at data that no longer exists. Loading the old
    # set here and then processing only the requested states left stale entries
    # behind, and a later plain resume skipped exactly the states whose rows
    # had been deleted. Start empty whenever the outputs are being cleared.
    completed = load_checkpoint(CHECKPOINT_FILE) if (resume and not force) else set()

    # Starting fresh (or forcing) resets the CHECKPOINT but previously reset
    # neither the output file nor the append mode below, so every re-run
    # appended a second copy of every row. ml.py's drop_duplicates() hid this,
    # but which duplicate survived depended on run order, making results
    # silently irreproducible. Clear the output when we are not resuming.
    # Both files must go. MONTHLY_OUTPUT_FILE is the one that accumulates via
    # append across chunks, so leaving it behind means a "fresh" run keeps
    # extending the previous run's file - including its rows and its column
    # header, which is how a column-order change silently corrupted 258k rows.
    if not resume or force:
        CHECKPOINT_FILE.unlink(missing_ok=True)
        for f in (MONTHLY_OUTPUT_FILE, OUTPUT_FILE):
            if f.exists():
                logging.warning(f"Not resuming: removing existing {f}")
                f.unlink()

    y0, y1 = int(years[0]), int(years[1])
    all_years = list(range(y0, y1 + 1))
    states = [s.upper() for s in states]
    year_chunks = chunk_years(all_years, year_chunk)

    logging.info("=" * 70)
    logging.info("CLIMATE DOWNLOAD (gridMET)")
    logging.info(f"Features: {CLIMATE_FEATURES}")
    logging.info(f"States: {len(states)} | Years: {y0}-{y1} | Year-chunk: {year_chunk}")
    logging.info(f"Months per year: {len(DOWNLOAD_MONTHS)} ({DOWNLOAD_MONTHS[0]}-{DOWNLOAD_MONTHS[-1]})")
    logging.info(f"Monthly source: {MONTHLY_OUTPUT_FILE}")
    logging.info(f"Season output:  {OUTPUT_FILE}")
    logging.info("=" * 70)

    for s_idx, st in enumerate(states, 1):
        logging.info(f"[{s_idx}/{len(states)}] State {st}: {len(all_years)} years in {len(year_chunks)} chunks")

        for yc in year_chunks:
            cy0, cy1 = yc[0], yc[-1]
            k = chunk_key(st, cy0, cy1)

            if resume and not force and k in completed:
                logging.info(f"  Skipping {st} {cy0}-{cy1} (checkpointed)")
                continue

            logging.info(f"  Processing {st} {cy0}-{cy1}...")
            t0 = time.time()

            try:
                df = run_with_retries(lambda: process_state_years_chunk(st, yc), max_retries=3, base_sleep=4.0)

                mode = "a" if MONTHLY_OUTPUT_FILE.exists() else "w"
                save_df(df, MONTHLY_OUTPUT_FILE, mode=mode)

                completed.add(k)
                save_checkpoint(CHECKPOINT_FILE, completed)

                logging.info(f"  ✓ {st} {cy0}-{cy1}: {len(df)} rows in {time.time() - t0:.1f}s")

            except Exception as e:
                logging.error(f"  ✗ FAILED {st} {cy0}-{cy1}: {e}")
                logging.error("  Continuing to next chunk...")

    # Build the season table from whatever monthly data now exists
    build_season_table()

    logging.info("DONE")
    logging.info(f"Monthly source: {MONTHLY_OUTPUT_FILE}")
    logging.info(f"Season output:  {OUTPUT_FILE}")


def build_season_table() -> None:
    """
    Rebuild processed/climate.csv from the monthly source.

    Safe to call on its own (--aggregate-only) whenever the growing-season
    definition in config.py changes. No Earth Engine access required.
    """
    if not MONTHLY_OUTPUT_FILE.exists():
        logging.warning(f"No monthly climate data at {MONTHLY_OUTPUT_FILE}; nothing to aggregate")
        return

    monthly = pd.read_csv(MONTHLY_OUTPUT_FILE, dtype={"county_fips": str, "state_fips": str})
    # A resumed/re-run download can re-append the same state-year-month.
    before = len(monthly)
    monthly = monthly.drop_duplicates(subset=["county_fips", "year", "month"], keep="last")
    if before != len(monthly):
        logging.warning(f"Dropped {before - len(monthly):,} duplicate county-year-month rows")

    season = aggregate_to_season(monthly)
    if season.empty:
        logging.warning("Season aggregation produced no rows")
        return
    save_df(season, OUTPUT_FILE)
    logging.info(f"Wrote season table: {OUTPUT_FILE} ({len(season):,} rows)")


def main():
    parser = argparse.ArgumentParser(
        description="Download county-level growing-season climate from gridMET via GEE "
    )
    parser.add_argument("--states", nargs="+", default=DEFAULT_STATES)
    parser.add_argument("--years_start", type=int, default=ANALYSIS_YEARS[0])
    parser.add_argument("--years_end", type=int, default=ANALYSIS_YEARS[1])
    # Sized for MONTHLY output: each chunk now returns counties x years x 12
    # months in one getInfo(), where the old seasonal version returned
    # counties x years. A chunk of 6 was fine at 1 row/county/year but is ~12x
    # the payload here (18k features for TX) and will time out on large states.
    parser.add_argument("--year-chunk", type=int, default=2,
                        help="Years per API call per state (default 2; drop to 1 for "
                             "large states if getInfo times out)")
    parser.add_argument("--force", action="store_true", help="Re-run even if checkpointed")
    parser.add_argument("--no-resume", action="store_true", help="Ignore checkpoint and start fresh")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Skip Earth Engine; rebuild the season table from existing "
                             "monthly data (use after changing the growing season in config.py)")

    parser.add_argument("--study", action="store_true",
                        help="Use config.STUDY_STATES (the modelling scope) instead of "
                             "DEFAULT_STATES. Avoids retyping the list.")

    args = parser.parse_args()
    states = resolve_states(args.states, args.study)

    if args.aggregate_only:
        build_season_table()
        return

    run(
        states=states,
        years=(args.years_start, args.years_end),
        year_chunk=args.year_chunk,
        resume=not args.no_resume,
        force=args.force,
    )


if __name__ == "__main__":
    main()
