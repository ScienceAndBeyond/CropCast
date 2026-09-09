"""
Download soil properties from SoilGrids API for U.S. counties.

Downloads clay, silt, sand, pH, organic carbon, and bulk density.
Uses checkpointing to resume interrupted downloads.

Usage:
    python download_soil.py --states MN IA IL --workers 4
"""
import argparse
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from requests.exceptions import RequestException, HTTPError
from soilgrids import SoilGrids
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm

from config import DATA_PATH
from utils import get_county_geometries, logging

CHECKPOINT_FILE = DATA_PATH / "raw/soil_checkpoint.csv"
OUTPUT_FILE = DATA_PATH / "processed/soil.csv"
RESOLUTION = 256  # Pixels for width/height of the downloaded tile

SOIL_PROPERTIES = {
    "clay": ("clay", "clay_0-5cm_mean", 10),  # Unit: g/kg (divide by 10 for %)
    "silt": ("silt", "silt_0-5cm_mean", 10),  # Unit: g/kg (divide by 10 for %)
    "sand": ("sand", "sand_0-5cm_mean", 10),  # Unit: g/kg (divide by 10 for %)
    "ph": ("phh2o", "phh2o_0-5cm_mean", 10),  # Unit: pH (divide by 10)
    "soc": ("soc", "soc_0-5cm_mean", 10),  # Unit: g/kg (dg/kg -> divide by 10)
    "bdod": ("bdod", "bdod_0-5cm_mean", 100),  # Unit: kg/dm³ (cg/cm³ -> divide by 100)
}


@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=16),
    retry=retry_if_exception_type((ConnectionResetError, OSError, RequestException, HTTPError))
)
def download_soil_layer(soil_grids, service_id, coverage_id, bbox, crs, resolution, outpath):
    """Downloads a soil data layer for a given bounding box."""
    return soil_grids.get_coverage_data(
        service_id=service_id,
        coverage_id=coverage_id,
        west=bbox[0], south=bbox[1], east=bbox[2], north=bbox[3],
        crs=crs,
        width=resolution,
        height=resolution,
        output=outpath
    )


def load_checkpoint(path: Path) -> set:
    """Loads processed FIPS codes from the checkpoint file."""
    if not path.exists():
        return set()
    df = pd.read_csv(path, dtype=str)
    return set(df["county_fips"])


def update_checkpoint(path: Path, county_fips: str) -> None:
    """Adds a FIPS code to the checkpoint file."""
    df = pd.DataFrame([[county_fips]], columns=["county_fips"])
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def process_county(county: pd.Series, soil_grids: SoilGrids, resolution: int) -> dict:
    """
    Processes a single county: downloads all soil properties, masks, and calculates the mean.
    """
    county_fips = county['county_fips']

    # Ensure geometry is valid and in the correct CRS (EPSG:4326) for SoilGrids
    geometry = county.geometry
    if not geometry.is_valid:
        geometry = geometry.buffer(0)

    gdf_county = gpd.GeoDataFrame([{'geometry': geometry}], crs="EPSG:4326")
    bbox = gdf_county.total_bounds

    county_data = {
        'county_fips': county_fips,
        'county_name': county['county'],
        'state_abbr': county['state_abbr'],
    }

    for prop_name, (service_id, coverage_id, scaling_factor) in SOIL_PROPERTIES.items():
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp.close()
            tmp_path = Path(tmp.name)
            try:
                download_soil_layer(
                    soil_grids, service_id, coverage_id,
                    bbox=bbox,
                    crs="urn:ogc:def:crs:EPSG::4326",
                    resolution=resolution,
                    outpath=str(tmp_path)
                )

                with rasterio.open(tmp_path) as src:
                    clipped_data, _ = mask(src, gdf_county.geometry, crop=True, nodata=src.nodata)

                    clipped_data = clipped_data[0]

                    masked_data = np.ma.masked_equal(clipped_data, src.nodata)

                    if masked_data.count() > 0:  # Ensure there is valid data
                        mean_val = masked_data.mean()
                        county_data[f"{prop_name}_mean"] = float(mean_val / scaling_factor)
                    else:
                        county_data[f"{prop_name}_mean"] = np.nan  # No valid data in county

            except Exception as e:
                logging.error(f"[{county_fips}] Failed on '{prop_name}': {e}")
                county_data[f"{prop_name}_mean"] = np.nan
            finally:
                tmp_path.unlink(missing_ok=True)

    return county_data


def get_soil_data_for_states(state_abbr_list: List[str], resolution=256, max_workers=8) -> None:
    """Main function to orchestrate downloading and processing of soil data."""
    counties = get_county_geometries(state_abbr_list).to_crs("EPSG:4326")
    processed_fips = load_checkpoint(CHECKPOINT_FILE)
    counties_to_process = counties[~counties['county_fips'].isin(processed_fips)]

    if counties_to_process.empty:
        logging.info("All counties have already been processed.")
        return

    soil_grids = SoilGrids()
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor, \
            tqdm(total=len(counties_to_process), desc="Downloading Soil Data") as pbar:

        future_to_fips = {
            executor.submit(process_county, county, soil_grids, resolution): county['county_fips']
            for _, county in counties_to_process.iterrows()
        }

        for future in as_completed(future_to_fips):
            fips = future_to_fips[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
                    # Write results to CSV and checkpoint periodically
                    if len(results) >= max_workers or pbar.n == pbar.total - 1:
                        df_batch = pd.DataFrame(results)
                        df_batch.to_csv(OUTPUT_FILE, mode="a", header=not OUTPUT_FILE.exists(), index=False)
                        for fips_code in df_batch['county_fips']:
                            update_checkpoint(CHECKPOINT_FILE, fips_code)
                        results.clear()
            except Exception as exc:
                logging.error(f"FIPS {fips} generated an exception: {exc}")
            finally:
                pbar.update(1)


def main():
    default_states = [
        "IA", "CA", "IL", "NE", "MN", "TX", "AR", "LA", "WA", "OR", "ID",
        # "KS", "ND", "SD", "IN", "OH", "MO", "WI", "MS", "GA", "NC",
    ]

    parser = argparse.ArgumentParser(description="Download static soil data by county")
    parser.add_argument("--states", nargs="+",
                        default=default_states, help="List of US state abbreviations")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers")
    args = parser.parse_args()

    get_soil_data_for_states(args.states, max_workers=args.workers)
    logging.info("Soil data download complete.")


if __name__ == "__main__":
    main()