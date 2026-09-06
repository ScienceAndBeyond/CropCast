"""
Vegetation Index Data Pipeline using Google Earth Engine + MODIS
=================================================================

Downloads NDVI and EVI data at county level with:
- CDL crop masking (only crop pixels, not forests/urban/water)
- State-specific growing seasons
- Efficient batch processing (ONE GEE reduceRegions call per state-year)

Data Source:
    MODIS MOD13A3 (Monthly Vegetation Indices, 1km resolution)
    https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD13A3

Outputs:
    Monthly (raw/vegetation_monthly.csv):  (preserved detail)
        - state_fips, state_abbr, county_fips, county_name, year, month, ndvi_mean, evi_mean

    Growing season aggregated (processed/vegetation.csv): (default modeling table)
        - state_fips, state_abbr, county_fips, county_name, year
        - ndvi_mean_year, ndvi_min_year, ndvi_max_year
        - evi_mean_year,  evi_min_year,  evi_max_year
        - n_months

Usage:
    python download_vegetation.py
    python download_vegetation.py --states TX CA --start_year 2015 --end_year 2020
    python download_vegetation.py --no-resume
"""

import argparse
import json
import time
from typing import Dict, List, Optional

import ee
import pandas as pd

from config import (
    DATA_PATH,
    DATA_PATH_RAW as RAW_DIR,
    GEE_PROJECT_ID,
    DEFAULT_STATES,
    ANALYSIS_YEARS,
    get_growing_season,
    get_required_months,
    resolve_states,
)
from utils import (
    native_scale,
    fetch_state_fips,
    get_crop_mask,
    get_tiger_counties_fc,
    save_df,
    logging,
)

# ---------------------------------------------------------------------------
# FILE PATHS
# ---------------------------------------------------------------------------
PROCESSED_DIR = DATA_PATH / "processed"

MONTHLY_OUTPUT_FILE = RAW_DIR / "vegetation_monthly.csv"
YEARLY_OUTPUT_FILE = PROCESSED_DIR / "vegetation.csv"
CHECKPOINT_FILE = RAW_DIR / "vegetation_checkpoint.json"


# ---------------------------------------------------------------------------
# GEE INITIALIZATION
# ---------------------------------------------------------------------------

def authenticate_gee(project_id: str = GEE_PROJECT_ID) -> None:
    """Initialize Google Earth Engine."""
    try:
        ee.Initialize(project=project_id)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project_id)
    logging.info("Google Earth Engine authenticated.")


# ---------------------------------------------------------------------------
# CHECKPOINT MANAGEMENT
# ---------------------------------------------------------------------------

def load_checkpoint() -> Dict:
    """Load checkpoint file tracking completed state-years."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return {"completed": []}


def save_checkpoint(checkpoint: Dict) -> None:
    """Save checkpoint to file."""
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=2)


def _key(state_fips: str, year: int) -> str:
    return f"{state_fips}_{year}"


def is_completed(checkpoint: Dict, state_fips: str, year: int) -> bool:
    """Check if a state-year is already completed."""
    return _key(state_fips, year) in checkpoint.get("completed", [])


def mark_completed(checkpoint: Dict, state_fips: str, year: int) -> None:
    """Mark a state-year as completed."""
    k = _key(state_fips, year)
    if k not in checkpoint["completed"]:
        checkpoint["completed"].append(k)
    save_checkpoint(checkpoint)


# ---------------------------------------------------------------------------
# SMALL I/O HELPERS
# ---------------------------------------------------------------------------

def append_csv(df: pd.DataFrame, path) -> None:
    """Append dataframe to CSV (create with header if not exists)."""
    if df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# CACHES (avoid rebuilding same GEE objects repeatedly)
# ---------------------------------------------------------------------------

_CROP_MASK_CACHE: Dict[int, ee.Image] = {}
_COUNTIES_CACHE: Dict[str, ee.FeatureCollection] = {}

def crop_mask_cached(year: int) -> ee.Image:
    if year not in _CROP_MASK_CACHE:
        _CROP_MASK_CACHE[year] = get_crop_mask(year)
    return _CROP_MASK_CACHE[year]

def counties_cached(state_fips: str) -> ee.FeatureCollection:
    if state_fips not in _COUNTIES_CACHE:
        _COUNTIES_CACHE[state_fips] = get_tiger_counties_fc(state_fips)
    return _COUNTIES_CACHE[state_fips]


# ---------------------------------------------------------------------------
# CORE GEE PROCESSING (FAST: ONE reduceRegions PER STATE-YEAR)
# ---------------------------------------------------------------------------

def get_monthly_vegetation_stats(
    state_fips: str,
    state_abbr: str,
    year: int,
) -> pd.DataFrame:
    """
    Download monthly vegetation stats for an entire state-year in ONE GEE call.

    Returns long monthly rows for growing-season months only.

    Columns:
      state_fips, county_fips, county_name, year, month, ndvi_mean, evi_mean
    """
    # Fetch a FIXED month window (config.get_required_months), not this state's
    # growing season. Fetching only the season would bake the season definition
    # into the downloaded data, so changing it later would mean re-downloading
    # every state-year from Earth Engine. The season is applied offline instead,
    # in aggregate_to_yearly() - matching how climate is handled, so both
    # sources stay re-windowable via --aggregate-only.
    months_to_fetch = get_required_months(slack=1)

    # MODIS monthly vegetation indices
    # QA is selected alongside the indices so poor pixels can be masked.
    # MOD13A3 is already a monthly composite, so measured contamination is tiny
    # (8 of 188,019 in-season county-months below NDVI 0.10), but masking is
    # cheap and makes the product defensible rather than merely lucky.
    modis = ee.ImageCollection("MODIS/061/MOD13A3").select(["NDVI", "EVI", "SummaryQA"])

    crop_mask = crop_mask_cached(year)
    counties_fc = counties_cached(state_fips)

    def month_to_img(m):
        m = ee.Number(m).toInt()
        start = ee.Date.fromYMD(year, m, 1)
        end = start.advance(1, "month")

        src = modis.filterDate(start, end).first()
        # SummaryQA: 0 good, 1 marginal, 2 snow/ice, 3 cloud. Keep 0-1 only.
        qa_ok = src.select("SummaryQA").lte(1)
        img = (
            src.select(["NDVI", "EVI"])
            .multiply(0.0001)   # MODIS scale factor
            .updateMask(qa_ok)
            .updateMask(crop_mask)
        )

        mm = m.format("%02d")
        return img.rename([ee.String("NDVI_").cat(mm), ee.String("EVI_").cat(mm)])

    months = ee.List(months_to_fetch)

    # Stack monthly images into one multiband image
    stacked = ee.ImageCollection(months.map(month_to_img)).toBands()

    # toBands prefixes band names like "0_NDVI_04". Clean them to "NDVI_04".
    # toBands prefixes band names like "0_NDVI_04". Remove the numeric prefix.
    band_names = stacked.bandNames()
    cleaned = band_names.map(lambda b: ee.String(b).replace('^[0-9]+_', ''))
    stacked = stacked.rename(cleaned)

    # Reduce once across counties
    fc = stacked.reduceRegions(
        collection=counties_fc,
        reducer=ee.Reducer.mean(),
        scale=native_scale("MODIS/061/MOD13A3"),
        tileScale=4,
    )

    result = fc.getInfo()

    rows = []
    for f in result.get("features", []):
        props = f.get("properties", {})

        base = {
            "state_fips": str(props.get("state_fips", "")),
            "county_fips": str(props.get("county_fips", "")),
            "county_name": props.get("county_name", ""),
            "year": int(year),
        }

        for month in months_to_fetch:
            mm = f"{month:02d}"
            rows.append({
                **base,
                "month": int(month),
                "ndvi_mean": props.get(f"NDVI_{mm}"),
                "evi_mean": props.get(f"EVI_{mm}"),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# AGGREGATION: MONTHLY → GROWING-SEASON (COUNTY-YEAR)
# ---------------------------------------------------------------------------

def aggregate_to_yearly(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate monthly vegetation rows to county-year growing-season statistics.

    The season filter is applied HERE, not at download time, so the growing
    season can be changed in config.py and rebuilt with --aggregate-only
    instead of re-downloading from Earth Engine.
    """
    if monthly_df.empty:
        return pd.DataFrame()

    df = monthly_df
    if "state_abbr" in df.columns:
        bounds = df["state_abbr"].map(get_growing_season)
        gs_start = pd.Series([b[0] for b in bounds], index=df.index)
        gs_end = pd.Series([b[1] for b in bounds], index=df.index)
        n_before = len(df)
        df = df[(df["month"] >= gs_start) & (df["month"] < gs_end)]
        if n_before != len(df):
            logging.info(
                f"  season filter: kept {len(df):,} of {n_before:,} monthly rows"
            )
    else:
        logging.warning(
            "aggregate_to_yearly: no state_abbr column, cannot apply the "
            "state-specific growing season; aggregating all fetched months."
        )
    monthly_df = df

    grouping_cols = ["state_fips", "county_fips", "county_name", "year"]

    monthly_df = monthly_df.copy()
    monthly_df["_both_present"] = (
        monthly_df["ndvi_mean"].notna() & monthly_df["evi_mean"].notna()
    ).astype(int)

    yearly = monthly_df.groupby(grouping_cols, as_index=False).agg(
        ndvi_mean_year=("ndvi_mean", "mean"),
        ndvi_min_year=("ndvi_mean", "min"),
        ndvi_max_year=("ndvi_mean", "max"),
        evi_mean_year=("evi_mean", "mean"),
        evi_min_year=("evi_mean", "min"),
        evi_max_year=("evi_mean", "max"),
        # count months where BOTH indices are present. Counting ndvi alone
        # could certify a season complete while evi had gaps, and the model
        # uses both.
        n_months=("_both_present", "sum"),
        ndvi_std_year=("ndvi_mean", "std"),
        evi_std_year=("evi_mean", "std"),
        ndvi_sum_year=("ndvi_mean", "sum"),
        evi_sum_year=("evi_mean", "sum"),
    )

    return yearly


# ---------------------------------------------------------------------------
# MAIN DOWNLOAD FUNCTION
# ---------------------------------------------------------------------------

def download_vegetation(
    states: List[str],
    start_year: int,
    end_year: int,
    resume: bool = True,
) -> pd.DataFrame:
    """
    Download vegetation indices for all specified states and years.

    Writes:
      - monthly growing-season rows (raw/vegetation_monthly.csv)
      - growing-season county-year aggregates (processed/vegetation.csv)

    Returns:
      A DataFrame of the last batch's yearly aggregates (not the full dataset).
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Map state_abbr -> state_fips
    state_fips_map: Dict[str, str] = {}
    for abbr in states:
        fips_list = fetch_state_fips([abbr])
        if fips_list:
            state_fips_map[abbr] = str(fips_list[0])

    checkpoint = load_checkpoint() if resume else {"completed": []}

    # Not resuming resets the checkpoint but used to leave the output files in
    # place, while append_csv() below appends unconditionally -> a second copy
    # of every row on each fresh run. Clear them so the run is reproducible.
    if not resume:
        for f in (MONTHLY_OUTPUT_FILE, YEARLY_OUTPUT_FILE):
            if f.exists():
                logging.warning(f"Not resuming: removing existing {f} to avoid duplicate rows")
                f.unlink()

    total_tasks = len(states) * (end_year - start_year + 1)
    already_completed = 0
    # only count completed keys that match our requested states/years
    requested_keys = set()
    for y in range(start_year, end_year + 1):
        for abbr in states:
            fips = state_fips_map.get(abbr)
            if fips:
                requested_keys.add(_key(fips, y))
    already_completed = sum(1 for k in checkpoint.get("completed", []) if k in requested_keys)
    remaining = total_tasks - already_completed

    logging.info("=" * 60)
    logging.info("VEGETATION DATA DOWNLOAD")
    logging.info("=" * 60)
    logging.info(f"States: {states}")
    logging.info(f"Years: {start_year}-{end_year}")
    logging.info(f"Total state-years: {total_tasks}")
    logging.info(f"Already completed (within request): {already_completed}")
    logging.info(f"Remaining: {remaining}")
    logging.info("=" * 60)

    processed_idx = 0
    last_yearly_df: Optional[pd.DataFrame] = None

    for year in range(start_year, end_year + 1):
        for state_abbr in states:
            state_fips = state_fips_map.get(state_abbr)
            if not state_fips:
                logging.warning(f"Could not find FIPS for {state_abbr}, skipping")
                continue

            if resume and is_completed(checkpoint, state_fips, year):
                continue

            processed_idx += 1
            start_month, end_month = get_growing_season(state_abbr)
            logging.info(
                f"[{processed_idx}/{remaining if remaining > 0 else total_tasks}] "
                f"{state_abbr} {year} (season: months {start_month}-{end_month-1})"
            )

            t0 = time.time()

            try:
                monthly_df = get_monthly_vegetation_stats(
                    state_fips=state_fips,
                    state_abbr=state_abbr,
                    year=year,
                )

                if monthly_df.empty:
                    logging.warning(f"  No data returned for {state_abbr} {year} (NOT marking completed).")
                    continue

                # Add state_abbr, then append monthly
                monthly_df["state_abbr"] = state_abbr
                append_csv(monthly_df, MONTHLY_OUTPUT_FILE)

                # Compute yearly aggregate for this chunk and append
                yearly_chunk = aggregate_to_yearly(monthly_df)
                yearly_chunk["state_abbr"] = state_abbr

                # Reorder columns for processed output consistency
                cols = [
                    "state_fips", "state_abbr", "county_fips", "county_name", "year",
                    "ndvi_mean_year", "ndvi_min_year", "ndvi_max_year",
                    "evi_mean_year", "evi_min_year", "evi_max_year",
                    "ndvi_std_year","evi_std_year","ndvi_sum_year","evi_sum_year", "n_months"
                ]
                yearly_chunk = yearly_chunk[[c for c in cols if c in yearly_chunk.columns]]
                append_csv(yearly_chunk, YEARLY_OUTPUT_FILE)

                last_yearly_df = yearly_chunk

                elapsed = time.time() - t0
                logging.info(f"  ✓ {len(monthly_df)} monthly rows; {len(yearly_chunk)} county-year rows in {elapsed:.1f}s")

                # Mark completed only after both writes succeeded
                mark_completed(checkpoint, state_fips, year)

            except (NameError, AttributeError, ImportError, TypeError, KeyError) as e:
                # A bug in this file, not a transient Earth Engine problem.
                # These used to be caught by the broad handler below, so a run
                # with a missing import would log a warning per state-year and
                # then "complete" having downloaded nothing at all. Fail fast.
                logging.error(f"  FATAL (code error, not transient): {type(e).__name__}: {e}")
                raise
            except Exception as e:
                # Transient: GEE timeout, quota, network. Skip and let the
                # checkpoint pick this state-year up on the next run.
                logging.error(f"  Failed (will retry on re-run): {e}")
                continue

    if last_yearly_df is None:
        logging.warning("No data downloaded in this run.")
        return pd.DataFrame()

    logging.info("\n✓ Vegetation download complete!")
    logging.info(f"Monthly file:  {MONTHLY_OUTPUT_FILE}")
    logging.info(f"Yearly file:   {YEARLY_OUTPUT_FILE}")
    logging.info(f"Checkpoint:    {CHECKPOINT_FILE}")

    return last_yearly_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download NDVI/EVI data with crop masking and state-specific seasons"
    )
    parser.add_argument(
        "--states",
        nargs="+",
        default=DEFAULT_STATES,
        help=f"State abbreviations (default: {DEFAULT_STATES})"
    )
    parser.add_argument(
        "--start_year",
        type=int,
        default=ANALYSIS_YEARS[0],
        help=f"Start year (default: {ANALYSIS_YEARS[0]}; CDL crop mask needs >=2008)"
    )
    parser.add_argument(
        "--end_year",
        type=int,
        default=ANALYSIS_YEARS[1],
        help=f"End year (default: {ANALYSIS_YEARS[1]})"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from checkpoint (default: True)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, ignore checkpoint"
    )

    parser.add_argument("--study", action="store_true",
                        help="Use config.STUDY_STATES (the modelling scope) instead of "
                             "DEFAULT_STATES. Avoids retyping the list.")

    args = parser.parse_args()
    states = resolve_states(args.states, args.study)
    resume = args.resume and not args.no_resume

    authenticate_gee()

    download_vegetation(
        states=states,
        start_year=args.start_year,
        end_year=args.end_year,
        resume=resume,
    )


if __name__ == "__main__":
    main()