"""
Download crop yield data from USDA NASS QuickStats API
=======================================================

Downloads county-level crop YIELD data for FIELD CROPS.

Filters applied at API level:
- Program:       SURVEY
- Sector:        CROPS
- Group:         FIELD CROPS
- Stat category: YIELD
- Geo Level:     COUNTY
- Period Type:   ANNUAL
- Period:        YEAR

Usage:
    python download_yield.py
    python download_yield.py --states TX CA --start_year 2015 --end_year 2020
"""

import argparse
import random
import time
from pathlib import Path

import pandas as pd
import requests
from requests.exceptions import RequestException, HTTPError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

import utils
from config import (
    NASS_BASE_URL,
    NASS_API_KEY,
    DATA_PATH,
    DATA_PATH_RAW as RAW_DIR,
    DEFAULT_STATES,
    ANALYSIS_YEARS,
    NASS_SKIP_COMMODITIES,
    NASS_SKIP_CLASS_DESC,
    NASS_PRACTICE_MODE,
    NASS_PRACTICE_LABELS,
    IRRIGATION_COLUMN,
    get_practice_filter,
    yield_filename,
    logging,
    resolve_states,
)

HEADERS = {
    "User-Agent": "CropCast (Research project)",
    "Accept": "application/json",
}


# NASS QuickStats is frequently slow: a whole-state FIELD CROPS query can take
# minutes to come back, and the service has stretches where it simply times out.
# A 60s cap with 4 tries and a 16s ceiling was not enough headroom - it burned
# through all attempts inside ~90 seconds and killed the run.
@retry(
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=2, min=5, max=120),
    retry=retry_if_exception_type((RequestException, HTTPError)),
    before_sleep=before_sleep_log(logging.getLogger(), logging.WARNING),
    reraise=True,
)
def make_request_with_retry(params: dict) -> list:
    """Quick Stats /api_GET request with retry logic."""
    # (connect, read): fail fast on a dead host, but be patient once connected.
    resp = requests.get(NASS_BASE_URL, params=params, headers=HEADERS, timeout=(15, 300))
    logging.debug(f"URL: {resp.url}")
    status = resp.status_code

    if status in (400, 404):
        logging.warning(f"{status} — permanent failure for {resp.url}")
        return []

    try:
        resp.raise_for_status()
    except HTTPError:
        logging.warning(f"Request failed with {status}. Retrying...")
        raise

    try:
        return resp.json().get("data", [])
    except requests.exceptions.JSONDecodeError:
        logging.error(f"Invalid JSON from {resp.url} — skipping")
        return []


def fetch_all_pages(params: dict) -> list:
    """Fetch all pages for a query with the API's limit applied."""
    all_data = []
    offset = 0
    limit = 50000

    while True:
        paginated = {**params, "limit": limit, "offset": offset, "format": "JSON"}
        chunk = make_request_with_retry(paginated)

        if not chunk:
            break

        all_data.extend(chunk)
        logging.info(f"Fetched {len(chunk)} rows (offset={offset})")
        offset += len(chunk)

        if len(chunk) < limit:
            break

        time.sleep(random.uniform(2.0, 4.0))

    return all_data


def get_usda_crop_yield(
    api_key: str,
    states: list[str],
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """
    Download county-level YIELD data for ALL "FIELD CROPS" for specified states.
    Most filters are applied at the API level to minimize data downloaded.
    """
    if not api_key:
        raise ValueError("NASS_API_KEY is not set in .env file")

    all_records = []
    failed_states = []

    for state in states:
        params = {
            "key": api_key,
            "source_desc": "SURVEY",
            "sector_desc": "CROPS",
            "group_desc": "FIELD CROPS",
            "statisticcat_desc": "YIELD",
            "agg_level_desc": "COUNTY",
            "state_alpha": state,
            "year__GE": str(start_year),
            "year__LE": str(end_year),
            "freq_desc": "ANNUAL",
            "reference_period_desc": "YEAR",
        }

        logging.info(f"Fetching YIELD stats for {state} ({start_year}-{end_year})...")

        # One state timing out used to abort the whole run: the retry decorator
        # reraises, and nothing between here and main() caught it, so 20 states
        # of successful downloads were discarded because the 21st was slow.
        try:
            data = fetch_all_pages(params)
        except Exception as e:
            failed_states.append(state)
            logging.error(f"  FAILED {state}: {type(e).__name__}: {e}")
            logging.error(f"  Continuing; re-run with --states {state} to fill this gap "
                          f"(merged into the existing table, nothing else is touched).")
            continue

        if data:
            all_records.extend(data)
            logging.info(f"Retrieved {len(data)} rows for {state}")
        else:
            logging.warning(f"No data for {state}")

        time.sleep(random.uniform(1.0, 2.0))

    if failed_states:
        # Loud, and at the end where it will actually be seen. Partial yield
        # data silently missing whole states would bias every downstream score.
        logging.error("=" * 70)
        logging.error(f"INCOMPLETE: {len(failed_states)} state(s) failed: {', '.join(failed_states)}")
        logging.error(f"Re-run:  python download_yield.py --states {' '.join(failed_states)}")
        logging.error("=" * 70)

    df = pd.DataFrame(all_records).drop_duplicates()
    if df.empty:
        return df

    df["county_fips"] = (
        df["state_fips_code"].astype(str).str.zfill(2)
        + df["county_code"].astype(str).str.zfill(3)
    )

    return df


def clean_yield_data(df: pd.DataFrame, practice_mode: str = None) -> pd.DataFrame:
    """
    Clean raw USDA NASS YIELD data.

    Output columns:
      - county_fips
      - state_alpha
      - county_name
      - year
      - commodity_desc
      - class_desc
      - prodn_practice_desc
      - util_practice_desc
      - yield_value
      - yield_unit
      - cv_mean
    """
    if df.empty:
        return df.copy()

    df = df.copy()

    if "county_fips" not in df.columns:
        df["county_fips"] = (
            df["state_fips_code"].astype(str).str.zfill(2)
            + df["county_code"].astype(str).str.zfill(3)
        )

    df["Value"] = df["Value"].astype(str).str.replace(",", "", regex=False)
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    # Production practice: driven by config.NASS_PRACTICE_MODE, not hardcoded.
    # NASS reports the same county-year-crop as an aggregate AND as irrigated /
    # non-irrigated components, so keeping more than one of those duplicates
    # rows. See the mode documentation in config.py.
    keep_practices = get_practice_filter(practice_mode)
    n_before = len(df)
    df = df[df["prodn_practice_desc"].isin(keep_practices)]
    logging.info(
        f"Practice mode '{practice_mode or NASS_PRACTICE_MODE}' keeping {keep_practices}: "
        f"{len(df):,} of {n_before:,} rows"
    )

    df = df[~df["util_practice_desc"].isin(["SILAGE", "SEED"])]
    df = df[df["domain_desc"] == "TOTAL"]
    df = df[df["Value"].notna() & (df["Value"] > 0)]

    # Drop NASS pseudo-counties. county_code 998 ("OTHER (COMBINED) COUNTIES")
    # is not a place: it is a residual bucket of individually-suppressed
    # counties, reported once per agricultural statistics district. Every ASD in
    # a state emits its own 998 row, so they all collapse onto one fake FIPS
    # (e.g. 31998 carried 13 conflicting values for the same crop-year) and
    # drop_duplicates then picked between them arbitrarily. They have no
    # geometry, so the inner join later discards them anyway - but only after
    # they have inflated every record count and log line along the way.
    n_before = len(df)
    county_code = df["county_fips"].astype(str).str[-3:]
    df = df[~county_code.isin(["998", "999"])]
    if n_before != len(df):
        logging.info(
            f"Dropped {n_before - len(df):,} pseudo-county rows "
            f"(county_code 998/999, district-level residuals)"
        )

    # Tidy irrigation label. Anything unmapped (e.g. "NON-IRRIGATED, CONTINUOUS
    # CROP") falls back to a slugged version rather than being silently dropped.
    df[IRRIGATION_COLUMN] = (
        df["prodn_practice_desc"]
        .map(NASS_PRACTICE_LABELS)
        .fillna(df["prodn_practice_desc"].str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True))
    )

    # Skip commodities from config
    df = df[~df["commodity_desc"].isin(NASS_SKIP_COMMODITIES)]
    
    # Skip winter crops (from config)
    for skip_class in NASS_SKIP_CLASS_DESC:
        df = df[~df["class_desc"].str.upper().str.contains(skip_class, na=False)]

    df = df.drop_duplicates(
        subset=[
            "county_fips",
            "year",
            "commodity_desc",
            "class_desc",
            "prodn_practice_desc",
            "util_practice_desc",
        ]
    )

    if "CV (%)" not in df.columns:
        df["CV (%)"] = pd.NA

    out = df[
        [
            "county_fips",
            "state_alpha",
            "county_name",
            "year",
            "commodity_desc",
            "class_desc",
            "prodn_practice_desc",
            IRRIGATION_COLUMN,
            "util_practice_desc",
            "Value",
            "unit_desc",
            "CV (%)",
        ]
    ].rename(
        columns={
            "Value": "yield_value",
            "unit_desc": "yield_unit",
            "CV (%)": "cv_mean",
        }
    )

    return out.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch county-level YIELD data for all FIELD CROPS."
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
        help=f"Start year (default: {ANALYSIS_YEARS[0]})"
    )
    parser.add_argument(
        "--end_year", 
        type=int, 
        default=ANALYSIS_YEARS[1],
        help=f"End year (default: {ANALYSIS_YEARS[1]})"
    )

    parser.add_argument("--reuse-raw", action="store_true",
                        help="Skip the API and re-clean the existing raw file. The "
                             "practice mode only affects CLEANING, so switching "
                             "aggregate<->split needs no new download.")
    parser.add_argument("--practice-mode", default=NASS_PRACTICE_MODE,
                        choices=["aggregate", "split", "both"],
                        help="Override config.NASS_PRACTICE_MODE for this run. "
                             "'split' writes yield_split.csv, keeping the aggregate "
                             "table intact so the two studies cannot clobber each other.")
    parser.add_argument("--study", action="store_true",
                        help="Use config.STUDY_STATES (the modelling scope) instead of "
                             "DEFAULT_STATES. Avoids retyping the list.")

    args = parser.parse_args()
    import sys as _sys
    args.states_given = "--states" in _sys.argv or "--study" in _sys.argv
    states = resolve_states(args.states, args.study)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / f"crop_yield_raw_{args.start_year}_{args.end_year}.csv"

    processed_dir = Path(DATA_PATH) / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    final_path = processed_dir / yield_filename(args.practice_mode)

    if args.reuse_raw:
        if not raw_path.exists():
            raise FileNotFoundError(
                f"--reuse-raw given but {raw_path} does not exist. "
                f"Run without the flag once to fetch it."
            )
        logging.info(f"Reusing existing raw download: {raw_path}")
        raw_df = pd.read_csv(raw_path, low_memory=False)
        # SEMANTICS: --states controls what gets FETCHED, never what gets kept.
        # The cleaned table is always the full union of downloaded states;
        # ml.py applies STUDY_STATES itself. (An earlier version filtered the
        # cache here, which made --reuse-raw --states X silently shrink
        # yield.csv to X - the same destructive behaviour as a subset fetch.)
        if args.states_given:
            logging.warning(
                "--states is ignored with --reuse-raw (nothing is fetched); "
                "cleaning the full cached table."
            )
    else:
        raw_df = get_usda_crop_yield(
            api_key=NASS_API_KEY,
            states=states,
            start_year=args.start_year,
            end_year=args.end_year,
        )
        # MERGE into the existing table by state instead of overwriting it.
        # A recovery run for one failed state used to REPLACE the whole file
        # with that one state, destroying the 20 that had succeeded. Rows for
        # the states just fetched are refreshed; every other state is kept.
        if raw_path.exists() and not raw_df.empty:
            existing = pd.read_csv(raw_path, low_memory=False)
            # Replace only the states that were ACTUALLY refreshed. Dropping
            # every requested state meant a partial failure (request IA+NE,
            # receive only IA) silently deleted NE's cached rows. A failed
            # state keeps its previous data until a replacement succeeds.
            refreshed = set(raw_df["state_alpha"].unique())
            not_refreshed = sorted(set(states) - refreshed)
            if not_refreshed:
                logging.warning(
                    f"Keeping cached rows for {not_refreshed}: requested but no "
                    f"fresh data arrived (fetch failed or empty)."
                )
            kept = existing[~existing["state_alpha"].isin(refreshed)]
            logging.info(
                f"Merging {len(raw_df):,} fresh rows for {len(refreshed)} refreshed "
                f"state(s) into {len(kept):,} existing rows"
            )
            raw_df = pd.concat([kept, raw_df], ignore_index=True)

    if raw_df.empty:
        logging.warning("No raw data retrieved. Exiting.")
        return

    if not args.reuse_raw:
        utils.save_df(raw_df, raw_path)
        logging.info(f"Saved raw data to {raw_path} ({len(raw_df):,} rows)")

    ml_df = clean_yield_data(raw_df, practice_mode=args.practice_mode)
    utils.save_df(ml_df, final_path)
    logging.info(f"Saved cleaned data to {final_path} ({len(ml_df):,} rows)")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Raw rows:     {len(raw_df):,}")
    print(f"Cleaned rows: {len(ml_df):,}")

    if not ml_df.empty:
        print("\n" + "-" * 50)
        print("COMMODITIES:")
        print("-" * 50)

        coverage = ml_df.groupby(["commodity_desc", "class_desc"]).agg({
            "year": ["min", "max", "nunique"],
            "county_fips": "nunique",
            "yield_value": "count"
        })
        coverage.columns = ["min_year", "max_year", "n_years", "n_counties", "n_records"]
        coverage = coverage.reset_index().sort_values(
            ["commodity_desc", "n_records"],
            ascending=[True, False]
        )

        current_commodity = None
        for _, row in coverage.iterrows():
            if row["commodity_desc"] != current_commodity:
                current_commodity = row["commodity_desc"]
                print(f"\n{current_commodity}:")

            print(f"  {row['class_desc']}: "
                  f"{int(row['min_year'])}-{int(row['max_year'])} "
                  f"({int(row['n_years'])} yrs, {int(row['n_counties'])} counties, "
                  f"{int(row['n_records'])} records)")

        print("\n" + "-" * 50)
        print("YIELD UNITS:")
        print("-" * 50)
        print(ml_df["yield_unit"].value_counts().to_string())

    print("\n" + "-" * 50)
    print("OUTPUT FILES:")
    print("-" * 50)
    print(f"  Raw:     {raw_path}")
    print(f"  Cleaned: {final_path}")


if __name__ == "__main__":
    main()
