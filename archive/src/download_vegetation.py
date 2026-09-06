"""
Vegetation Index Data Pipeline using Google Earth Engine + MODIS

Downloads NDVI and EVI data at county level using GEE's reduceRegions(),
aggregating monthly satellite imagery to yearly growing season statistics as yield is at annual level.

Data Source:
    MODIS MOD13A3 (Monthly Vegetation Indices, 1km resolution)
    https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD13A3

Features:
    - Monthly NDVI and EVI extraction via Google Earth Engine
    - Aggregation to growing season (Apr-Sep) min/max/mean values
    - County-level spatial averaging using reduceRegions()

Output Features (yearly):
    - ndvi_min_year: Minimum NDVI during growing season (captures stress)
    - ndvi_max_year: Maximum NDVI during growing season (captures peak health)
    - evi_min_year: Minimum EVI during growing season
    - evi_max_year: Maximum EVI during growing season

Prerequisites:
    pip install earthengine-api pandas geopandas shapely
    earthengine authenticate  # Run once to authenticate

Usage:
    python download_vegetation.py --states IA IL NE --start_year 2010 --end_year 2024
"""

import argparse
from datetime import datetime
from pathlib import Path

import ee
import pandas as pd
from shapely.geometry import mapping

from config import GEE_PROJECT_ID, DATA_PATH
from utils import get_county_geometries, save_df, logging


def authenticate_gee(project_id=GEE_PROJECT_ID, service_account_path=None):
    if service_account_path:
        credentials = ee.ServiceAccountCredentials(None, service_account_path)
        ee.Initialize(credentials, project=project_id)
    else:
        try:
            ee.Initialize(project=project_id)
        except Exception:
            ee.Authenticate()
            ee.Initialize(project=project_id)
    logging.info("Google Earth Engine authenticated.")


def load_checkpoint(path: Path) -> set:
    if not path.exists():
        return set()
    df = pd.read_csv(path, dtype=str)
    return set((row["county_fips"], int(row["year"]), int(row["month"])) for _, row in df.iterrows())


def update_checkpoint(path: Path, county_fips: str, year: int, month: int) -> None:
    df = pd.DataFrame([[county_fips, year, month]], columns=["county_fips", "year", "month"])
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def aggregate_yearly(input_path: Path, output_path: Path):
    df = pd.read_csv(input_path)
    #yearly = df.groupby(["county_fips", "year"], as_index=False).agg(
    yearly = df.groupby(["state_fips", "county_fips", "county_name", "year"], as_index=False).agg(
        ndvi_mean_year=("ndvi_mean", "mean"),
        ndvi_min_year=("ndvi_mean", "min"),
        ndvi_max_year=("ndvi_mean", "max"),
        evi_mean_year=("evi_mean", "mean"),
        evi_min_year=("evi_mean", "min"),
        evi_max_year=("evi_mean", "max")
    )
    yearly.to_csv(output_path, index=False)
    logging.info(f"Saved yearly NDVI aggregated data to {output_path}")


def aggregate_ndvi_by_period(ndvi_df: pd.DataFrame, grow_months: list) -> pd.DataFrame:
    df = ndvi_df[ndvi_df["month"].isin(grow_months)].copy()
    grouping_cols = ["state_fips", "county_fips", "county_name", "year"]
    grouped = df.groupby(grouping_cols, as_index=False).agg(
        ndvi_mean_year=("ndvi_mean", "mean"),
        ndvi_min_year=("ndvi_mean", "min"),
        ndvi_max_year=("ndvi_mean", "max"),
        evi_mean_year=("evi_mean", "mean"),
        evi_min_year=("evi_mean", "min"),
        evi_max_year=("evi_mean", "max")
    )

    return grouped


def download_ndvi(states, start_year, end_year, checkpoint_path: Path, monthly_data_path: Path, yearly_data_path: Path,
                  batch_size=10):
    counties = get_county_geometries(states)
    collection = ee.ImageCollection('MODIS/061/MOD13A3').select(['NDVI', 'EVI'])
    completed = load_checkpoint(checkpoint_path)

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            logging.info(f"Processing {year}-{month:02d} for all counties in batches")
            start = datetime(year, month, 1)
            end = datetime(year + 1, 1, 1) if month == 12 else datetime(year, month + 1, 1)
            image = collection.filterDate(start, end).first().multiply(0.0001)

            for i in range(0, len(counties), batch_size):
                batch = counties.iloc[i:i + batch_size]
                features = []
                for _, row in batch.iterrows():
                    county_fips = row.county_fips
                    state_fips = row.state_fips
                    county_name = row.county
                    if (county_fips, year, month) in completed:
                        continue
                    ee_geom = ee.Geometry(mapping(row.geometry))
                    features.append(ee.Feature(ee_geom, {
                        'state_fips': state_fips,
                        'county_fips': county_fips,
                        'county_name': county_name,
                        'year': year,
                        'month': month
                    }))

                if not features:
                    continue

                try:
                    fc = ee.FeatureCollection(features)
                    reduced = image.reduceRegions(
                        collection=fc,
                        reducer=ee.Reducer.mean(),
                        scale=1000,
                        tileScale=4
                    ).getInfo()

                    batch_results = []
                    for f in reduced['features']:
                        props = f['properties']
                        batch_results.append({
                            "state_fips": props['state_fips'],
                            "county_fips": props['county_fips'],
                            "county_name": props['county_name'],
                            "year": props['year'],
                            "month": props['month'],
                            "ndvi_mean": props.get("NDVI"),
                            "evi_mean": props.get("EVI")
                        })

                    df_batch = pd.DataFrame(batch_results)
                    if not df_batch.empty:
                        if monthly_data_path.exists():
                            df_batch.to_csv(monthly_data_path, mode="a", header=False, index=False)
                        else:
                            df_batch.to_csv(monthly_data_path, mode="w", header=True, index=False)

                    for f in batch_results:
                        update_checkpoint(checkpoint_path, f['county_fips'], f['year'], f['month'])

                except Exception as e:
                    logging.warning(f"Batch {i}-{i + batch_size} failed for {year}-{month:02d}: {e}")

    ndvi_mnthly_df = pd.read_csv(monthly_data_path)
    ndvi_yearly_df = aggregate_ndvi_by_period(ndvi_mnthly_df, grow_months=list(range(4, 10)))   # aggregate for growing months
    save_df(ndvi_yearly_df, yearly_data_path)


def main():
    default_states = [
        "IA", "CA", "IL", "NE", "MN", "TX", "AR", "LA", "WA", "OR", "ID",
        #  "KS", "ND", "SD", "IN", "OH", "MO", "WI", "MS", "GA", "NC",
    ]
    parser = argparse.ArgumentParser(description="Download monthly NDVI and EVI data using Google Earth Engine")
    parser.add_argument("--states", nargs="+", default=default_states,
                        help="List of US state abbreviations")
    parser.add_argument("--start_year", type=int, default=2010,
                        help="Start year (default: 2010)")
    parser.add_argument("--end_year", type=int, default=2024,
                        help="End year (default: 2024 inclusive)")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Number of counties to process per API call")
    args = parser.parse_args()

    raw_dir = Path(DATA_PATH) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    ndvi_data_path_monthly = raw_dir / f"monthly_vegetation_{args.start_year}_{args.end_year}.csv"
    checkpoint_file = raw_dir / "vegetation_checkpoint.csv"

    processed_dir = Path(DATA_PATH) / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    final_vegetation_file = processed_dir / "vegetation.csv"

    authenticate_gee()
    download_ndvi(
        args.states,
        args.start_year,
        args.end_year,
        checkpoint_path=Path(checkpoint_file),
        monthly_data_path=ndvi_data_path_monthly,
        yearly_data_path=final_vegetation_file,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
