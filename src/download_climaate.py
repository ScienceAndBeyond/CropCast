"""
Downloads climate data at county level using GEE's reduceRegions(),
which is much faster than the original NOAA station-based approach.

Usage:
    python download_climate_gee.py --states IA IL NE --start_year 2010 --end_year 2024
    python download_climate_gee.py --resume  # Resume from last checkpoint
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd

from config import DATA_PATH, GEE_PROJECT_ID, logging
from utils import fetch_state_fips, save_df

# Checkpoint and output files
RAW_PATH = Path(DATA_PATH) / "raw"
RAW_PATH.mkdir(parents=True, exist_ok=True)

PARTIAL_OUTPUT_DIR = RAW_PATH / "climate_gee_partial"
PARTIAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_FILE = RAW_PATH / "checkpoint_climate_gee.json"

PROCESSED_PATH = Path(DATA_PATH) / "processed"
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
FINAL_OUTPUT_FILE = PROCESSED_PATH / "climate.csv"

# For this study, we are using a fixed growing season: April - September
GROWING_SEASON_START_MONTH = 4
GROWING_SEASON_END_MONTH = 9

# gridMET resolution in meters
GRIDMET_SCALE = 4000

# Retry settings for GEE operations
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 30


def load_checkpoint() -> Dict[str, Any]:
    """Load checkpoint file to track completed state-year combinations."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {"completed": [], "failed": []}


def save_checkpoint(checkpoint: Dict[str, Any]) -> None:
    """Save checkpoint to file."""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def is_completed(checkpoint: Dict, state: str, year: int) -> bool:
    """Check if a state-year combination is already completed."""
    key = f"{state}_{year}"
    return key in checkpoint.get("completed", [])


def mark_completed(checkpoint: Dict, state: str, year: int) -> None:
    """Mark a state-year combination as completed."""
    key = f"{state}_{year}"
    if key not in checkpoint["completed"]:
        checkpoint["completed"].append(key)
    # Remove from failed if it was there
    if key in checkpoint.get("failed", []):
        checkpoint["failed"].remove(key)
    save_checkpoint(checkpoint)


def mark_failed(checkpoint: Dict, state: str, year: int, error: str) -> None:
    """Mark a state-year combination as failed."""
    key = f"{state}_{year}"
    if key not in checkpoint.get("failed", []):
        if "failed" not in checkpoint:
            checkpoint["failed"] = []
        checkpoint["failed"].append(key)
    save_checkpoint(checkpoint)


# =============================================================================
# GOOGLE EARTH ENGINE INITIALIZATION
# =============================================================================

def initialize_gee() -> bool:
    """Initialize Google Earth Engine with project ID."""
    try:
        import ee
    except ImportError:
        logging.error("earthengine-api not installed. Run: pip install earthengine-api")
        return False
    
    try:
        # Try to initialize with project ID from config
        if GEE_PROJECT_ID:
            ee.Initialize(project=GEE_PROJECT_ID)
            logging.info(f"GEE initialized with project: {GEE_PROJECT_ID}")
        else:
            ee.Initialize()
            logging.info("GEE initialized without project ID")
        return True
    except Exception as e:
        logging.error(f"GEE initialization failed: {e}")
        logging.error("Run 'earthengine authenticate' to set up authentication")
        return False


def get_county_feature_collection(state_abbr: str) -> "ee.FeatureCollection":
    """
    Get counties for a state as a GEE FeatureCollection.
    Uses TIGER/2018/Counties dataset in GEE.
    """
    import ee
    
    # Get state FIPS from utils
    state_fips_list = fetch_state_fips([state_abbr])
    state_fips = state_fips_list[0]
    
    # Load from GEE's TIGER dataset
    counties = ee.FeatureCollection('TIGER/2018/Counties') \
        .filter(ee.Filter.eq('STATEFP', state_fips))
    
    return counties


def compute_growing_season_climate(year: int, counties: "ee.FeatureCollection") -> "ee.FeatureCollection":
    """
    Compute growing season (Apr-Sep) climate statistics for counties.
    
    Calculates:
    - TMIN: Mean of daily minimum temperature (°C)
    - TMAX: Mean of daily maximum temperature (°C)
    - PRCP: Total precipitation (mm)
    - VPD: mean vapor pressure deficit (kPa) over the growing season
    - ETO: daily grass reference ET (mm) - sum over growing season for water demand
    - SRAD: downward shortwave radiation (W/m^2) - mean over growing season
    - WIND: mean wind speed at 10 m (m/s)
    """
    import ee
    
    # Define growing season date range
    start_date = f'{year}-{GROWING_SEASON_START_MONTH:02d}-01'
    end_date = f'{year}-{GROWING_SEASON_END_MONTH:02d}-30'
    
    # Load gridMET daily data
    gridmet = ee.ImageCollection('IDAHO_EPSCOR/GRIDMET') \
        .filterDate(start_date, end_date)
    
    # gridMET temps are in Kelvin, convert to Celsius
    tmin_collection = gridmet.select('tmmn').map(lambda img: img.subtract(273.15))
    tmax_collection = gridmet.select('tmmx').map(lambda img: img.subtract(273.15))
    
    # Mean temperature for the season
    tmin_mean = tmin_collection.mean().rename('TMIN')
    tmax_mean = tmax_collection.mean().rename('TMAX')
    
    # Sum of daily precipitation (mm)
    prcp_sum = gridmet.select('pr').sum().rename('PRCP')

    # VPD: mean vapor pressure deficit (kPa) over the growing season
    vpd_mean = gridmet.select('vpd').mean().rename('VPD')

    # ETO: daily grass reference ET (mm) - sum over growing season for water demand
    eto_sum = gridmet.select('eto').sum().rename('ETO')

    # SRAD: downward shortwave radiation (W/m^2) - mean over growing season
    srad_mean = gridmet.select('srad').mean().rename('SRAD')

    # WIND: mean wind speed at 10 m (m/s)
    wind_mean = gridmet.select('vs').mean().rename('WIND')

    # Calculate daily average temperature
    def calc_daily_temps(img):
        tmin = img.select('tmmn').subtract(273.15)
        tmax = img.select('tmmx').subtract(273.15)
        tavg = tmin.add(tmax).divide(2)
        return img.addBands(tavg.rename('tavg'))
    
    gridmet_with_tavg = gridmet.map(calc_daily_temps)
    
    # --- Combine all bands ---
    combined = tmin_mean \
        .addBands(tmax_mean) \
        .addBands(prcp_sum) \
        .addBands(vpd_mean) \
        .addBands(eto_sum) \
        .addBands(srad_mean) \
        .addBands(wind_mean)
    
    # --- Reduce to county level ---
    county_stats = combined.reduceRegions(
        collection=counties,
        reducer=ee.Reducer.mean(),
        scale=GRIDMET_SCALE
    )
    
    # Add year to each feature
    county_stats = county_stats.map(lambda f: f.set('year', year))
    
    return county_stats


def fetch_county_climate_data(state_abbr: str, year: int) -> Optional[pd.DataFrame]:
    """
    Fetch climate data for one state and one year.
    Returns DataFrame with county_fips, year, TMIN, TMAX, PRCP, VPD, ETO, SRAD, WIND
    """

    logging.info(f"  Processing {state_abbr} {year}...")
    
    try:
        # Get counties for this state
        counties = get_county_feature_collection(state_abbr)
        
        # Compute climate statistics
        county_stats = compute_growing_season_climate(year, counties)
        
        # Fetch results from GEE
        result = county_stats.getInfo()
        
        if not result or 'features' not in result:
            logging.warning(f"No data returned for {state_abbr} {year}")
            return None
        
        # Convert to DataFrame
        records = []
        for feature in result['features']:
            props = feature['properties']
            records.append({
                'county_fips': props.get('GEOID', ''),
                'county_name': props.get('NAME', ''),
                'state_abbr': state_abbr,
                'year': year,
                'TMIN': props.get('TMIN'),
                'TMAX': props.get('TMAX'),
                'PRCP': props.get('PRCP'),
                'VPD': props.get('VPD'),
                'ETO': props.get('ETO'),
                'SRAD': props.get('SRAD'),
                'WIND': props.get('WIND'),
            })
        
        df = pd.DataFrame(records)
        
        # Standardize county_fips to 5-digit string
        df['county_fips'] = df['county_fips'].astype(str).str.zfill(5)
        
        logging.info(f"{state_abbr} {year}: {len(df)} counties")
        return df
        
    except Exception as e:
        logging.error(f"Error for {state_abbr} {year}: {e}")
        return None


def download_climate_data_gee(
    states: List[str],
    start_year: int,
    end_year: int,
    resume: bool = True
) -> pd.DataFrame:
    """
    Main pipeline to download climate data using GEE.
    
    Processes state-by-state, year-by-year with checkpointing for restartability.
    """
    # Initialize GEE
    if not initialize_gee():
        raise RuntimeError("Failed to initialize Google Earth Engine")
    
    # Load checkpoint
    checkpoint = load_checkpoint() if resume else {"completed": [], "failed": []}
    
    total_tasks = len(states) * (end_year - start_year + 1)
    completed_count = len(checkpoint.get("completed", []))
    
    logging.info("=" * 60)
    logging.info("CLIMATE DATA DOWNLOAD - Google Earth Engine + gridMET")
    logging.info("=" * 60)
    logging.info(f"States: {len(states)}")
    logging.info(f"Years: {start_year} - {end_year}")
    logging.info(f"Total tasks: {total_tasks}")
    logging.info(f"Already completed: {completed_count}")
    logging.info(f"Remaining: {total_tasks - completed_count}")
    logging.info("=" * 60)
    
    all_data = []
    
    # Load any existing partial data
    for partial_file in PARTIAL_OUTPUT_DIR.glob("climate_*.csv"):
        df = pd.read_csv(partial_file, dtype={'county_fips': str})
        all_data.append(df)
        logging.info(f"Loaded existing: {partial_file.name}")
    
    # Process each state-year combination
    for state in states:
        logging.info(f"\n{'='*40}")
        logging.info(f"STATE: {state}")
        logging.info(f"{'='*40}")
        
        for year in range(start_year, end_year + 1):
            # Skip if already completed
            if is_completed(checkpoint, state, year):
                logging.info(f"  Skipping {state} {year} (already completed)")
                continue
            
            # Attempt to fetch data with retries
            df = None
            for attempt in range(MAX_RETRIES):
                try:
                    df = fetch_county_climate_data(state, year)
                    break
                except Exception as e:
                    logging.warning(f"  Attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
                    if attempt < MAX_RETRIES - 1:
                        logging.info(f"  Retrying in {RETRY_DELAY_SECONDS} seconds...")
                        time.sleep(RETRY_DELAY_SECONDS)
            
            if df is not None and not df.empty:
                # Save partial result
                partial_file = PARTIAL_OUTPUT_DIR / f"climate_{state}_{year}.csv"
                save_df(df, partial_file)
                
                all_data.append(df)
                mark_completed(checkpoint, state, year)
            else:
                mark_failed(checkpoint, state, year, "No data returned")
            
            # Small delay between requests so GEE is not overwhelmed
            time.sleep(2)
    
    # Combine all data
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['county_fips', 'year'])
        
        # Ensure proper types
        combined_df['county_fips'] = combined_df['county_fips'].astype(str).str.zfill(5)
        combined_df['year'] = combined_df['year'].astype(int)
        
        # Sort for consistency
        combined_df = combined_df.sort_values(['state_abbr', 'county_fips', 'year'])
        
        # Save combined output
        save_df(combined_df, FINAL_OUTPUT_FILE)
        logging.info(f"\nSaved combined data: {FINAL_OUTPUT_FILE}")

        return combined_df
    else:
        logging.warning("\nNo data was collected!")
        return pd.DataFrame()


def main():

    default_states = [
        "IA", "CA", "IL", "NE", "MN", "TX", "AR", "LA", "WA", "OR", "ID",
        #  "KS", "ND", "SD", "IN", "OH", "MO", "WI", "MS", "GA", "NC",
    ]
    parser = argparse.ArgumentParser(
        description="Download climate data using Google Earth Engine + gridMET",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download for specific states and years
  python download_climate_gee.py --states IA IL NE --start_year 2010 --end_year 2024
  
  # Resume from checkpoint (default behavior)
  python download_climate_gee.py --resume
  
  # Start fresh (ignore checkpoint)
  python download_climate_gee.py --no-resume
        """
    )
    
    parser.add_argument("--states", nargs="+", default=default_states,
                        help="List of US state abbreviations")

    parser.add_argument("--start_year", type=int, default=2010,
                        help="Start year (default: 2010)")

    parser.add_argument("--end_year", type=int, default=2024,
                        help="End year (default: 2024)")

    parser.add_argument("--resume", action="store_true", default=True,
                        help="Resume from checkpoint (default: True)")

    parser.add_argument("--no-resume", action="store_true",
                        help="Start fresh, ignore checkpoint")

    parser.add_argument("--drive_folder", type=str, default="crop_yield_climate",
                        help="Google Drive folder name for exports")

    args = parser.parse_args()
        
    # Handle resume logic
    resume = args.resume and not args.no_resume
    
    df = download_climate_data_gee(
        states=args.states,
        start_year=args.start_year,
        end_year=args.end_year,
        resume=resume
    )
        
    checkpoint = load_checkpoint()
        
    if not df.empty:
        logging.info("\n" + "=" * 60)
        logging.info("SUMMARY")
        logging.info("=" * 60)
        logging.info(f"Total records: {len(df):,}")
        logging.info(f"Counties: {df['county_fips'].nunique():,}")
        logging.info(f"Years: {df['year'].min()} - {df['year'].max()}")
        logging.info(f"\nSample data:")
        print(df.head(10).to_string())


if __name__ == "__main__":
    main()
