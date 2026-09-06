"""
USDA Crop Yield Data Downloader
------------------------------------------------------
Downloads county-level crop YIELD data from the USDA NASS Quick Stats API

Inputs:
  - List of states (default: default-states including CA, TX... )
  - start-year (default 2010)
  - end-year (default: 2024)
  
Filters applied at API level:
- Program:      SURVEY
- Sector:       CROPS
- Group:        FIELD CROPS
- Stat category: YIELD
- Geo Level:    COUNTY
- Period Type:  ANNUAL
- Period:       YEAR
"""

import argparse
import random
import time
from pathlib import Path
from typing import Set

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
from config import NASS_BASE_URL, NASS_API_KEY, DATA_PATH, logging

HEADERS = {
    "User-Agent": "ClimateAndAgri (Student research)",
    "Accept": "application/json",
}

# Commodities to skip entirely (not useful for yield prediction)
SKIP_COMMODITIES: Set[str] = {
    "HAY", 
    "HAY & HAYLAGE",
}

# We are skipping this as we only have climate and other data from Apr-Sep
SKIP_CLASS_DESC: Set[str] = {
    "WINTER",
}

@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=4, max=16),
    retry=retry_if_exception_type((RequestException, HTTPError)),
    before_sleep=before_sleep_log(logging.getLogger(), logging.WARNING),
    reraise=True,
)
def make_request_with_retry(params: dict) -> list:
    """ Quick Stats /api_GET request with retry logic."""
    resp = requests.get(NASS_BASE_URL, params=params, headers=HEADERS, timeout=60)
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
    Download county-level YIELD data for ALL "FIELD CROPS" for a specific state.
    Most filters are applied at the API level to minimize the data downloaded
    """
    if not api_key:
        raise ValueError("NASS_API_KEY is not set in config.py")

    all_records = []

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

        logging.info(f"Fetching YIELD stats for {state} for years {start_year}-{end_year}...")

        data = fetch_all_pages(params)
        if data:
            all_records.extend(data)
            logging.info(f"Retrieved {len(data)} rows for {state} {start_year}-{end_year}")
        else:
            logging.warning(f"No data for {state} {start_year}-{end_year}")

        time.sleep(random.uniform(1.0, 2.0))

    df = pd.DataFrame(all_records).drop_duplicates()
    if df.empty:
        return df

    df["county_fips"] = (
        df["state_fips_code"].astype(str).str.zfill(2)
        + df["county_code"].astype(str).str.zfill(3)
    )

    return df


def clean_yield_data(df: pd.DataFrame) -> pd.DataFrame:
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

    # Some basic filters to clean up the data before passing to the ML pipeline
    df = df[df["prodn_practice_desc"] == "ALL PRODUCTION PRACTICES"]
    df = df[~df["util_practice_desc"].isin(["SILAGE", "SEED"])]  # Filter these out
    df = df[df["domain_desc"] == "TOTAL"]
    df = df[~df["class_desc"].str.upper().str.contains("WINTER", na=False)]  # Exclude Winter crops for now
    df = df[df["Value"].notna() & (df["Value"] > 0)]
    
    # Skip commodities that aren't useful for our study
    df = df[~df["commodity_desc"].isin(SKIP_COMMODITIES)]

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
    default_states = [
        "IA", "CA", "IL", "NE", "MN", "TX", "AR", "LA", "WA", "OR", "ID",
      # "KS", "ND", "SD", "IN", "OH", "MO", "WI", "MS", "GA", "NC",
    ]

    parser = argparse.ArgumentParser(
        description="Fetch county-level YIELD data for all FIELD CROPS."
    )

    parser.add_argument(
        "--states", nargs="+", default=default_states,
        help="List of US state abbreviations (e.g., IA CA)",
    )

    parser.add_argument("--start_year", type=int, default=2010,
                        help="Start year (default: 2010)")

    parser.add_argument("--end_year", type=int, default=2024,
                        help="End year (default: 2024 inclusive)")

    args = parser.parse_args()

    raw_dir = Path(DATA_PATH) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"crop_yield_raw_{args.start_year}_{args.end_year}.csv"

    processed_dir = Path(DATA_PATH) / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    final_path = processed_dir / "yield.csv"

    raw_df = get_usda_crop_yield(
        api_key=NASS_API_KEY,
        states=args.states,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    
    if raw_df.empty:
        logging.warning("No raw data retrieved. Exiting.")
        return

    utils.save_df(raw_df, raw_path)
    logging.info(f"Saved raw data to {raw_path} ({len(raw_df):,} rows)")

    ml_df = clean_yield_data(raw_df)
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
        print("COMMODITIES (all class_desc values kept):")
        print("-" * 50)
        
        # Show commodity + class coverage
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
    print("\nNOTE: All class_desc values kept. Use ML script to filter/expand.")


if __name__ == "__main__":
    main()
