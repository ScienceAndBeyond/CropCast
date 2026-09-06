# utils.py
# ===========================================================================
# Shared utility functions for the CropCast pipeline
# ===========================================================================

import logging
from pathlib import Path
from typing import List, TYPE_CHECKING

import pandas as pd

# geopandas / pygris / earthengine-api are imported lazily inside the functions
# that need them. They are only required by the DOWNLOAD stage; importing them
# at module scope made ml.py (which needs nothing but save_df) impossible to run
# without the entire geospatial stack installed and Earth Engine authenticated.
if TYPE_CHECKING:  # for type checkers only; no runtime cost
    import ee
    import geopandas as gpd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def _require_ee():
    """Import earthengine-api on demand, with an actionable error message."""
    try:
        import ee
    except ImportError as e:
        raise ImportError(
            "earthengine-api is required for this function. "
            "Install with: pip install earthengine-api"
        ) from e
    return ee


# ---------------------------------------------------------------------------
# COUNTY GEOMETRIES
# ---------------------------------------------------------------------------

def fetch_state_fips(state_alpha_list: List[str]) -> List[str]:
    """
    Derive state FIPS codes from geometries.
    
    Args:
        state_alpha_list: List of state abbreviations (e.g., ["IA", "IL"])
        
    Returns:
        Sorted list of state FIPS codes (e.g., ["17", "19"])
    """
    counties = get_county_geometries(state_alpha_list)
    return sorted(counties["state_fips"].unique().tolist())


def get_county_geometries(state_abbr_list: List[str]) -> "gpd.GeoDataFrame":
    """
    Fetch county geometries and FIPS codes using pygris.
    
    Args:
        state_abbr_list: List of state abbreviations
        
    Returns:
        GeoDataFrame with columns: county_fips, county, state_abbr, state_fips, geometry
    """
    import geopandas as gpd
    import pygris

    counties_list = []

    for state in state_abbr_list:
        gdf = pygris.counties(state=state)
        gdf["state_abbr"] = state
        counties_list.append(gdf)

    counties = gpd.GeoDataFrame(pd.concat(counties_list, ignore_index=True))

    counties['county_fips'] = counties['STATEFP'] + counties['COUNTYFP']
    counties = counties.rename(columns={
        'NAME': 'county',
        'STATEFP': 'state_fips'
    })[['county_fips', 'county', 'state_abbr', 'state_fips', 'geometry']]

    counties = counties.to_crs("EPSG:4326")
    return counties


# ---------------------------------------------------------------------------
# FILE I/O
# ---------------------------------------------------------------------------

def save_df(df: pd.DataFrame, file_name, mode: str = "w") -> None:
    """
    Save DataFrame to CSV or Parquet based on file extension.
    
    Args:
        df: DataFrame to save
        file_name: Output path (with .csv or .parquet extension)
        mode: Write mode for CSV ("w" = overwrite, "a" = append)
    """
    file_path = Path(file_name)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    extn = file_path.suffix.lower()

    if extn == ".csv":
        if mode == "a" and file_path.exists():
            # ALIGN TO THE EXISTING HEADER BEFORE APPENDING.
            #
            # to_csv(header=False) writes columns in the DataFrame's order, not
            # the file's. Appending a frame whose columns are ordered
            # differently therefore shifts every value into the wrong column,
            # silently and without error. This corrupted 258,066 of 305,046
            # climate_monthly.csv rows when a code change altered the column
            # order mid-collection: state codes landed in EDD, county names in
            # N_DAYS. Nothing failed until aggregation tried to call .upper()
            # on a float three steps downstream.
            existing = pd.read_csv(file_path, nrows=0).columns.tolist()
            extra = [c for c in df.columns if c not in existing]
            if extra:
                raise ValueError(
                    f"Cannot append to {file_path.name}: frame has columns absent "
                    f"from the file header: {extra}. Rewrite the file instead of appending."
                )
            missing = [c for c in existing if c not in df.columns]
            if missing:
                logging.warning(
                    f"Appending to {file_path.name} with {len(missing)} column(s) "
                    f"missing from the frame; they will be blank: {missing}"
                )
            df = df.reindex(columns=existing)
            df.to_csv(file_path, mode="a", header=False, index=False)
        else:
            df.to_csv(file_path, index=False)
    elif extn == ".parquet":
        df.to_parquet(file_path, index=False)
    else:
        raise ValueError(f"Unsupported file extension: {extn}. Use .csv or .parquet")


# ---------------------------------------------------------------------------
# GOOGLE EARTH ENGINE HELPERS
# ---------------------------------------------------------------------------

_NATIVE_SCALE_CACHE: dict = {}


def native_scale(asset_id: str, is_collection: bool = True) -> float:
    """
    The asset's own pixel size, cached.

    Reducing at a scale that is not the source's native grid forces GEE to
    resample, which shifts every value slightly and systematically. Measured on
    gridMET: reducing at 4000 m gave 17.97976 C where the native grid gives
    17.96158 C - and the native-grid figure matches a day-by-day recomputation
    exactly. The pipeline was the outlier, not the check.

    Native scales at the time of writing: gridMET 4638.31 m, MOD13A3 926.63 m,
    OpenLandMap 231.92 m, CDL 30 m. Deriving them here means they cannot drift
    out of sync with hardcoded constants.
    """
    if asset_id in _NATIVE_SCALE_CACHE:
        return _NATIVE_SCALE_CACHE[asset_id]
    ee = _require_ee()
    obj = ee.ImageCollection(asset_id).first() if is_collection else ee.Image(asset_id)
    scale = float(obj.projection().nominalScale().getInfo())
    _NATIVE_SCALE_CACHE[asset_id] = scale
    logging.info(f"native scale for {asset_id}: {scale:.2f} m")
    return scale


def get_crop_specific_mask(year: int):
    """
    Create a binary mask for cropland using USDA CDL (Cropland Data Layer).
    Masks to only the crops defined in config.CDL_CODES.

    NOT CURRENTLY USED BY THE PIPELINE. download_vegetation.py and
    download_soil.py both call get_crop_mask(), which is a GENERIC cropland
    mask (all crops). Using this function instead would require one vegetation
    table per crop, since the mask would differ by crop. Kept for that future
    work; do not assume the shipped results are crop-specific.

    Args:
        year: Year for CDL data (CDL is annual)
        
    Returns:
        ee.Image: Binary mask (1 = crop pixel, 0 = non-crop)
        
    Note:
        Must call ee.Initialize() before using this function.
    """
    from config import get_all_cdl_codes
    ee = _require_ee()

    # Load CDL for the specified year
    cdl = (
        ee.ImageCollection("USDA/NASS/CDL")
        .filter(ee.Filter.calendarRange(year, year, 'year'))
        .first()
        .select('cropland')
    )
    
    # Get all crop codes from config
    crop_codes = get_all_cdl_codes()
    
    # Build mask: pixel == any of the crop codes
    mask = cdl.eq(crop_codes[0])
    for code in crop_codes[1:]:
        mask = mask.Or(cdl.eq(code))
    
    return mask


def get_crop_mask(year: int) -> "ee.Image":
    """
    Cropland mask from USDA/NASS CDL. Always returns a single band named 'mask',
    uint8, self-masked (non-cropland is masked out) so `updateMask` behaves cleanly.

    Definition (applied identically for EVERY year):
        cropland codes 1-60, 66-77, 204-254

    Why not the CDL 'cultivated' band:
        1. It only exists for 2013-2023, so using it where available and falling
           back elsewhere applies TWO different definitions of "cropland" to one
           study period, producing a step change in every downstream vegetation
           feature at 2013 and again at 2024.
        2. Its class values are 1 = Non-cultivated, 2 = Cultivated. A previous
           version of this function used `.eq(1)`, which selected everything that
           was NOT farmland for 2013-2023.

        Both issues are avoided by using the 'cropland' band for all years.
    """
    ee = _require_ee()

    cdl = (
        ee.ImageCollection("USDA/NASS/CDL")
        .filter(ee.Filter.calendarRange(year, year, "year"))
        .first()
        .select("cropland")
    )

    mask = (
        cdl.gte(1).And(cdl.lte(60))
        .Or(cdl.gte(66).And(cdl.lte(77)))
        .Or(cdl.gte(204).And(cdl.lte(254)))
    )

    mask = mask.rename("mask").uint8()
    return mask.updateMask(mask)


def get_stable_crop_mask(start_year: int, end_year: int, min_frac: float = 0.5) -> "ee.Image":
    """
    Mask of pixels that are cropland in at least `min_frac` of the years in
    [start_year, end_year].

    Preferred over a plain union ("cropland in >=1 year"), which maximises CDL
    commission error: field edges and single-year misclassifications all get in.
    A stability threshold keeps land that is genuinely and repeatedly farmed.

    Args:
        start_year, end_year: inclusive year range
        min_frac: fraction of years a pixel must be cropland (0.5 = half)

    Returns:
        ee.Image: single band 'mask', uint8, self-masked.
    """
    ee = _require_ee()

    years = list(range(int(start_year), int(end_year) + 1))
    # unmask(0) so masked-out (non-crop) pixels count as 0 in the sum
    masks = [get_crop_mask(y).unmask(0) for y in years]
    n_years_cropland = ee.ImageCollection.fromImages(masks).sum()

    # ceil, not round: "cropland in at least half of 17 years" is 9, but
    # round(8.5) is 8 in Python (banker's rounding). Off by one on odd spans.
    import math
    threshold = max(1, math.ceil(min_frac * len(years)))
    mask = n_years_cropland.gte(threshold).rename("mask").uint8()
    return mask.updateMask(mask)


def get_tiger_counties_fc(state_fips: str):
    """
    Get counties for a state as a GEE FeatureCollection.
    Uses TIGER/2018/Counties dataset in GEE.
    
    Args:
        state_fips: State FIPS code (e.g., "19" for Iowa)
        
    Returns:
        ee.FeatureCollection with county features
    """
    import ee
    
    return (
        ee.FeatureCollection("TIGER/2018/Counties")
        .filter(ee.Filter.eq("STATEFP", state_fips))
        .map(lambda f: f.set({
            "state_fips": f.get("STATEFP"),
            "county_fips": ee.String(f.get("STATEFP")).cat(ee.String(f.get("COUNTYFP"))),
            "county_name": f.get("NAME"),
        }))
    )
