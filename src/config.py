# config.py
# ===========================================================================
# Central configuration for the CropCast pipeline
# Contains: API settings, paths, crop definitions, growing seasons, CDL codes
# ===========================================================================

import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
DATA_PATH = Path("../data")
DATA_PATH_RAW = Path("../data_raw")

COUNTY_SHAPEFILE = "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_county_500k.zip"

# ---------------------------------------------------------------------------
# API ENDPOINTS
# ---------------------------------------------------------------------------
NASS_BASE_URL = "https://quickstats.nass.usda.gov/api/api_GET/"
NOAA_BASE_URL = "https://www.ncdc.noaa.gov/cdo-web/api/v2/"

# ---------------------------------------------------------------------------
# API KEYS (loaded from .env)
# ---------------------------------------------------------------------------
load_dotenv()
NASS_API_KEY = os.getenv("NASS_API_KEY")
NOAA_TOKEN = os.getenv("NOAA_TOKEN")
SH_CLIENT_ID = os.getenv("SH_CLIENT_ID")
SH_CLIENT_SECRET = os.getenv("SH_CLIENT_SECRET")
GEE_PROJECT_ID = os.getenv("GEE_PROJECT_ID")


# ===========================================================================
# CROP CONFIGURATION - SINGLE SOURCE OF TRUTH
# ===========================================================================

# ---------------------------------------------------------------------------
# DEFAULT PARAMETERS (used by all download scripts)
# ---------------------------------------------------------------------------
DEFAULT_STATES: List[str] = [
    "IA", "CA", "IL", "NE", "MN", "TX", "AR", "LA", "WA", "OR", "ID",
    "KS", "ND", "SD", "IN", "OH", "MO", "WI", "MS", "GA", "NC",
]

# Rainfed Corn Belt + Northern Plains
STUDY_STATES = [
    "IA", "IL", "IN", "OH",  # Core Corn Belt
    "MN", "WI", "MO",        # Extended Corn Belt
    "KS", "ND", "SD",        # Northern Great Plains
    "NE",                    # Irrigation contrast - see note below
]

# NE and KS are the only states with a large, BALANCED within-state split of
# irrigated vs non-irrigated corn (NE 1020/999 across 92 counties; KS 556/578).
# Same state, same climate, broadly similar soils, different management - which
# makes them a natural experiment for the central claim: irrigation decouples
# yield from growing-season weather, so irrigated counties should show high
# conventional R2 with anomaly R2 pushed even closer to zero.
#
# CA is deliberately NOT included. It has the thinnest coverage of all 21
# states for these crops (0 soybean records, 203 corn over 25 counties, 1036
# total) because its agriculture is specialty crops, not field crops.
IRRIGATION_CONTRAST_STATES = ["NE", "KS"]

STUDY_CROPS = [
    "CORN",
    "SOYBEANS",
    "WHEAT",
    "SORGHUM",
    "BARLEY",
    "OATS",
]
# Two study periods for different analyses
# Verified 2026-09-05 against the live sources: gridMET has all 365 days of
# 2025, MOD13A3 has all 12 months, and CDL 2025 is published. NASS county yield
# estimates for the 2025 crop year are normally released the following spring;
# if any are still missing they simply produce no rows and the inner join drops
# them, so extending the window is safe either way.
YEARS_CAUSAL = (2005, 2025)  # 21 years, no CDL needed
YEARS_PREDICTION = (2008, 2025)  # earliest CDL available is 2008

# THE canonical analysis window. ml.py filters every source to ANALYSIS_YEARS[0]
# and all four downloaders default to it, so the download window and the
# modelling window cannot drift apart.
#
# It is YEARS_PREDICTION because CDL — and therefore the crop mask on both
# vegetation and soil — only starts in 2008. Downloading climate or yield back
# to 2005 produced three years that ml.py discarded on the next line: ~14% of
# the climate download wasted.
#
# To deliberately fetch the longer climate-only series (YEARS_CAUSAL, no CDL
# needed), pass it explicitly:  --years_start 2005
ANALYSIS_YEARS = YEARS_PREDICTION

# Kept as an alias so existing callers keep working; both now point at the
# analysis window rather than the wider causal one.
DEFAULT_YEARS = ANALYSIS_YEARS


def describe_years() -> str:
    """One-line summary of the year configuration, for startup logging."""
    return (
        f"analysis={ANALYSIS_YEARS[0]}-{ANALYSIS_YEARS[1]} "
        f"(causal-only option: {YEARS_CAUSAL[0]}-{YEARS_CAUSAL[1]}; "
        f"CDL-limited from {YEARS_PREDICTION[0]})"
    )

# ---------------------------------------------------------------------------
# GROWING SEASONS
# ---------------------------------------------------------------------------
# Format: (start_month, end_month_exclusive). (4, 10) means April 1 through September 30

DEFAULT_GROWING_SEASON: Tuple[int, int] = (4, 10)  # Apr-Sep

# Only define states that DIFFER from default
GROWING_SEASON_EXCEPTIONS: Dict[str, Tuple[int, int]] = {
    # Southern states - earlier growing season
    "TX": (3, 9),
    "LA": (3, 9),
    "AR": (3, 10),

    # Northern states - later start due to frost
    "MN": (5, 10),  # May-Sep
}


def get_growing_season(state_abbr: str) -> Tuple[int, int]:
    """
    Get growing season (start_month, end_month_exclusive) for a state.
    Returns default (4, 10) if state not in exceptions.
    
    Args:
        state_abbr: Two-letter state abbreviation (e.g., "TX", "MN")
    
    Returns:
        Tuple of (start_month, end_month_exclusive)
        
    Examples:
        >>> get_growing_season("TX")
        (3, 9)  # Mar-Aug
        
        >>> get_growing_season("IA")
        (4, 10)  # Apr-Sep (default)
    """
    return GROWING_SEASON_EXCEPTIONS.get(state_abbr.upper(), DEFAULT_GROWING_SEASON)


def get_required_months(slack: int = 1) -> List[int]:
    """
    The month range the downloaders need to fetch, derived from the configured
    growing seasons rather than hardcoded.

    Climate and vegetation are downloaded MONTHLY and aggregated to a season
    offline, so that changing a growing season is a re-aggregation rather than a
    re-download. That only pays off if the fetched window covers every season we
    might use - and only stays cheap if it is not wildly wider than needed.

    Deriving it here means widening a season in GROWING_SEASON_EXCEPTIONS
    automatically widens what gets downloaded, instead of silently producing
    months of missing data.

    NOTE: winter crops are excluded upstream (NASS_SKIP_CLASS_DESC) and their
    season spans two calendar years, which this month-window cannot express.
    Supporting them needs more than a wider range.
    """
    seasons = [DEFAULT_GROWING_SEASON] + list(GROWING_SEASON_EXCEPTIONS.values())
    lo = min(s[0] for s in seasons) - slack
    hi = max(s[1] - 1 for s in seasons) + slack
    return list(range(max(1, lo), min(12, hi) + 1))


def get_growing_season_dates(state_abbr: str, year: int) -> Tuple[str, str]:
    """
    Get growing season as date strings for GEE queries.
    
    Returns:
        Tuple of (start_date, end_date) as "YYYY-MM-DD" strings
    """
    start_month, end_month = get_growing_season(state_abbr)
    start_date = f"{year}-{start_month:02d}-01"
    end_date = f"{year}-{end_month:02d}-01"
    return start_date, end_date


# ---------------------------------------------------------------------------
# CDL (CROPLAND DATA LAYER) CODES
# ---------------------------------------------------------------------------
# Source: USDA NASS Cropland Data Layer
# https://www.nass.usda.gov/Research_and_Science/Cropland/metadata/meta.php

CDL_CODES: Dict[str, List[int]] = {
    "CORN": [1],
    "COTTON": [2],
    "RICE": [3],
    "SORGHUM": [4],
    "SOYBEANS": [5],
    "BARLEY": [21],
    "WHEAT": [22, 23, 24],  # 22=Durum, 23=Spring, 24=Winter
    "OATS": [28],
}

# Human-readable names for reference
CDL_CODE_NAMES: Dict[int, str] = {
    1: "Corn",
    2: "Cotton",
    3: "Rice",
    4: "Sorghum",
    5: "Soybeans",
    21: "Barley",
    22: "Durum Wheat",
    23: "Spring Wheat",
    24: "Winter Wheat",
    28: "Oats",
}


def get_cdl_codes(crop_name: str) -> List[int]:
    """
    Get CDL codes for a specific crop.
    
    Args:
        crop_name: Crop name (case-insensitive)
        
    Returns:
        List of CDL codes for that crop
        
    Raises:
        KeyError: If crop not found
    """
    name_upper = crop_name.upper().strip()
    if name_upper not in CDL_CODES:
        available = ", ".join(CDL_CODES.keys())
        raise KeyError(f"Unknown crop: '{crop_name}'. Available: {available}")
    return CDL_CODES[name_upper]


def get_all_cdl_codes() -> List[int]:
    """
    Get combined CDL codes for ALL configured crops.
    Used for vegetation masking (combined crop mask).
    
    Returns:
        Sorted list of unique CDL codes
        
    Example:
        >>> get_all_cdl_codes()
        [1, 2, 3, 4, 5, 21, 22, 23, 24, 28]
    """
    codes = []
    for crop_codes in CDL_CODES.values():
        codes.extend(crop_codes)
    return sorted(set(codes))


# ---------------------------------------------------------------------------
# NASS QUERY FILTERS (for download_yield.py)
# ---------------------------------------------------------------------------
# Commodities to skip (not useful for yield prediction)
NASS_SKIP_COMMODITIES = {
    "HAY",
    "HAY & HAYLAGE",
}

# Class descriptions to skip (winter crops need different handling)
NASS_SKIP_CLASS_DESC = {
    "WINTER",
}

# ---------------------------------------------------------------------------
# PRODUCTION PRACTICE / IRRIGATION
# ---------------------------------------------------------------------------
# NASS reports the SAME county-year-crop up to three ways:
#     ALL PRODUCTION PRACTICES  the aggregate
#     IRRIGATED                 component
#     NON-IRRIGATED             component
# Keeping more than one of these at once duplicates rows, so the choice is an
# explicit MODE rather than a filter. Changing it changes the unit of analysis.
#
#   "aggregate"  ALL PRODUCTION PRACTICES only. One row per county-year-crop.
#                Irrigation is invisible - this is what produced the spurious
#                "29 C heat threshold", because irrigated southern counties
#                look like hot places with high yields.
#
#   "split"      IRRIGATED + NON-IRRIGATED only, never the aggregate. Fewer
#                counties (most report only the aggregate) but management
#                becomes an observable. ml.py appends the label to the crop
#                name, so CORN evaluates as CORN__IRRIGATED and
#                CORN__NON_IRRIGATED - two strata, scored separately.
#
#   "both"       Everything, flagged. Rows WILL duplicate by design; you must
#                filter on the `irrigation` column yourself. For inspection.
NASS_PRACTICE_MODE = "aggregate"

# Raw NASS string -> tidy label written to the `irrigation` column
NASS_PRACTICE_LABELS = {
    "ALL PRODUCTION PRACTICES": "all",
    "IRRIGATED": "irrigated",
    "NON-IRRIGATED": "non_irrigated",
}

# Which raw practice strings each mode keeps
NASS_PRACTICE_MODES = {
    "aggregate": ["ALL PRODUCTION PRACTICES"],
    "split": ["IRRIGATED", "NON-IRRIGATED"],
    "both": ["ALL PRODUCTION PRACTICES", "IRRIGATED", "NON-IRRIGATED"],
}

IRRIGATION_COLUMN = "irrigation"


# ---------------------------------------------------------------------------
# STATE SCOPE RESOLUTION
# ---------------------------------------------------------------------------
def resolve_states(states=None, study: bool = False) -> list:
    """
    Resolve the state list for a download.

    Every downloader exposes `--study`, which pulls STUDY_STATES from here
    rather than requiring the list to be retyped on the command line. Typing it
    out is how Nebraska gets silently dropped from a run after being added to
    the study - and NE is the state that carries the irrigation contrast.

        --study            -> STUDY_STATES (the modelling scope)
        --states A B C     -> exactly those
        neither            -> DEFAULT_STATES (the full download scope)
    """
    if study:
        return list(STUDY_STATES)
    return list(states) if states else list(DEFAULT_STATES)


def yield_filename(mode: str = None) -> str:
    """
    Yield table name for a practice mode.

    The aggregate and split analyses are DIFFERENT studies on different units
    (CORN vs CORN__IRRIGATED / CORN__NON_IRRIGATED), so they get separate files
    and separate results directories. Sharing one yield.csv would mean each run
    silently invalidated the other's outputs.
    """
    mode = (mode or NASS_PRACTICE_MODE).lower()
    return "yield.csv" if mode == "aggregate" else f"yield_{mode}.csv"


def merged_filename(mode: str = None) -> str:
    """
    Merged table name for a practice mode.

    This MUST be mode-specific. ml.py writes merged.csv and evaluate.py reads
    it, so a shared name meant `ml.py --practice-mode split` silently replaced
    the aggregate study's merged table, and the next `evaluate.py` run scored
    split data while writing into the aggregate results directory.
    """
    mode = (mode or NASS_PRACTICE_MODE).lower()
    return "merged.csv" if mode == "aggregate" else f"merged_{mode}.csv"


def results_dirname(mode: str = None) -> str:
    """Results directory for a practice mode (see yield_filename)."""
    mode = (mode or NASS_PRACTICE_MODE).lower()
    return "../results" if mode == "aggregate" else f"../results_{mode}"


def get_practice_filter(mode: str = None) -> list:
    """Raw NASS prodn_practice_desc values to keep for the given mode."""
    mode = (mode or NASS_PRACTICE_MODE).lower()
    if mode not in NASS_PRACTICE_MODES:
        raise KeyError(
            f"Unknown NASS_PRACTICE_MODE: {mode!r}. "
            f"Available: {', '.join(NASS_PRACTICE_MODES)}"
        )
    return NASS_PRACTICE_MODES[mode]


# ---------------------------------------------------------------------------
# PRINT CONFIG SUMMARY (for debugging)
# ---------------------------------------------------------------------------
def print_config_summary():
    """Print summary of configuration for verification."""
    print("\n" + "=" * 60)
    print("CROPCAST CONFIGURATION")
    print("=" * 60)
    
    print(f"\nDefault States ({len(DEFAULT_STATES)}):")
    print(f"  {', '.join(DEFAULT_STATES)}")
    
    print(f"\nDefault Years: {DEFAULT_YEARS[0]} - {DEFAULT_YEARS[1]}")
    
    print(f"\nDefault Growing Season: months {DEFAULT_GROWING_SEASON[0]}-{DEFAULT_GROWING_SEASON[1]-1}")
    print("Growing Season Exceptions:")
    for state, season in GROWING_SEASON_EXCEPTIONS.items():
        print(f"  {state}: months {season[0]}-{season[1]-1}")
    
    print(f"\nCDL Codes ({len(CDL_CODES)} crops):")
    for crop, codes in CDL_CODES.items():
        code_names = [CDL_CODE_NAMES.get(c, str(c)) for c in codes]
        print(f"  {crop}: {codes} → {code_names}")
    
    print(f"\nAll CDL codes for masking: {get_all_cdl_codes()}")
    print("=" * 60)


if __name__ == "__main__":
    print_config_summary()
