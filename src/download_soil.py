"""
Soil Properties Data Pipeline (CROPLAND-MASKED) using Google Earth Engine
========================================================================

- Output is county-level (one row per county).
- Soil is static, but cropland location moves year to year, so the FOOTPRINT
  we average over is defined from CDL across the study window.

Masking: STABLE cropland, not "ever cropland"
---------------------------------------------
A plain union (cropland in >=1 of 17 years) is almost useless: it maximises CDL
commission error, so field edges and single-year misclassifications all get in
and the mask ends up covering practically the whole county. Verified against the
previous run - masked and unmasked soil agreed to r=0.9999 and a median relative
difference of 0.00%, i.e. the masking did nothing at all.

We instead require a pixel to be cropland in at least MASK_MIN_FRAC of the years
(see utils.get_stable_crop_mask), which keeps genuinely, repeatedly farmed land.

Depth: 0-30 cm, not 0 cm
------------------------
OpenLandMap bands b0/b10/b30/b60/b100/b200 are POINT estimates at 0, 10, 30, 60,
100 and 200 cm - b0 alone is the bare surface, not a layer average. Crops root
through the top ~30 cm, so we average b0/b10/b30 as a 0-30 cm proxy. This also
matches what the AGU poster claimed was being used.

Outputs:
- processed/soil.csv                        (final county table)
- data_raw/soil_cropland_checkpoint.json    (checkpoint for states)

Run:
  python download_soil.py
  python download_soil.py --states IA IL MN
  python download_soil.py --force
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import ee

from config import (
    DATA_PATH,
    DATA_PATH_RAW as RAW_DIR,
    GEE_PROJECT_ID,
    DEFAULT_STATES,
    ANALYSIS_YEARS,
    resolve_states,
)
from utils import (
    native_scale,
    fetch_state_fips,
    get_stable_crop_mask,
    get_tiger_counties_fc,
    save_df,
    logging,
)

# ---------------------------------------------------------------------------
# OUTPUT PATHS
# ---------------------------------------------------------------------------
PROCESSED_DIR = DATA_PATH / "processed"
OUTPUT_FILE = PROCESSED_DIR / "soil.csv"
CHECKPOINT_FILE = RAW_DIR / "soil_cropland_checkpoint.json"

# ---------------------------------------------------------------------------
# GEE PARAMETERS (match your current soil program)
# ---------------------------------------------------------------------------
# OpenLandMap's native grid is 231.92 m; 250 m forced resampling.
GEE_SCALE_METERS = None  # resolved from the asset on first use
GEE_TILE_SCALE = 4

# OpenLandMap depth bands are POINT estimates at 0/10/30/60/100/200 cm.
# Average the top three as a 0-30 cm rooting-zone proxy (see module docstring).
DEPTH_BANDS = ["b0", "b10", "b30"]

# Trapezoidal weights for a 0-30 cm layer mean from POINT estimates at 0/10/30:
#   integral = (b0+b10)/2*10 + (b10+b30)/2*20, divided by 30
#            = b0*(1/6) + b10*(1/2) + b30*(1/3)
# Equal thirds overweights the surface, which biases SOC high because organic
# carbon falls steeply with depth.
DEPTH_WEIGHTS = [1.0 / 6.0, 1.0 / 2.0, 1.0 / 3.0]

# A pixel must be CDL cropland in at least this fraction of the study years to
# be included in the mask. 0.5 = "cropland in at least half the years".
MASK_MIN_FRAC = 0.5

# ---------------------------------------------------------------------------
# Soil datasets (OpenLandMap) + scaling to final units
# Each entry: (asset_id, band, scale_factor, description)
# ---------------------------------------------------------------------------
SOIL_PROPERTIES: Dict[str, Tuple[str, List[str], float, str]] = {
    "clay": (
        "OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02",
        DEPTH_BANDS,
        1.0,   # already in % (kg/kg in catalog; used as % proxy at this level)
        "Clay content (%)",
    ),
    "sand": (
        "OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02",
        DEPTH_BANDS,
        1.0,
        "Sand content (%)",
    ),
    "ph": (
        "OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02",
        DEPTH_BANDS,
        0.1,   # stored as pH * 10
        "Soil pH (H2O)",
    ),
    "soc": (
        "OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02",
        DEPTH_BANDS,
        5.0,   # raw * 5 => g/kg (per GEE catalog)
        "Soil organic carbon (g/kg)",
    ),
    "bdod": (
        "OpenLandMap/SOL/SOL_BULKDENS-FINEEARTH_USDA-4A1H_M/v02",
        DEPTH_BANDS,
        0.01,  # (raw*10)=kg/m^3 ; /1000 => g/cm^3  => net *0.01
        "Bulk density (g/cm³)",
    ),
}


# ---------------------------------------------------------------------------
# GEE INITIALIZATION
# ---------------------------------------------------------------------------
def authenticate_gee(project_id: str | None = GEE_PROJECT_ID) -> None:
    """Initialize Google Earth Engine."""
    try:
        if project_id:
            ee.Initialize(project=project_id)
        else:
            ee.Initialize()
        logging.info("✓ Google Earth Engine initialized")
    except Exception as e:
        logging.error(f"✗ Failed to initialize Earth Engine: {e}")
        raise


# ---------------------------------------------------------------------------
# CHECKPOINT HELPERS
# ---------------------------------------------------------------------------
def load_checkpoint(path: Path) -> set[str]:
    """Return set of completed state abbreviations."""
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text())
        if isinstance(data, list):
            return {str(x).upper() for x in data}
        return set()
    except Exception:
        return set()


def save_checkpoint(path: Path, completed: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(list(completed)), indent=2))


def state_already_in_output(state_abbr: str) -> bool:
    """Quick skip if state already exists in output file."""
    if not OUTPUT_FILE.exists():
        return False
    try:
        df = pd.read_csv(OUTPUT_FILE, usecols=["state_abbr"], dtype=str)
        return state_abbr.upper() in set(df["state_abbr"].dropna().str.upper())
    except Exception:
        return False


# NOTE: build_union_crop_mask() was removed. The "cropland in >=1 year" union it
# produced covered essentially the whole county, so masking had no measurable
# effect on the output (see module docstring). Use
# utils.get_stable_crop_mask(start, end, min_frac) instead.


# ---------------------------------------------------------------------------
# BUILD SOIL IMAGE (masked once)
# ---------------------------------------------------------------------------
def build_soil_image(crop_mask: ee.Image) -> Tuple[ee.Image, Dict[str, Dict[str, Any]]]:
    """
    Create a multi-band soil image with the stable-cropland mask applied.

    Each property is the MEAN of its depth bands (DEPTH_BANDS = b0/b10/b30),
    i.e. a 0-30 cm rooting-zone value rather than the bare surface alone.
    """
    bands = []
    info: Dict[str, Dict[str, Any]] = {}

    for prop, (asset_id, depth_bands, scale, desc) in SOIL_PROPERTIES.items():
        img = (
            ee.Image(asset_id)
            .select(depth_bands)
            # thickness-weighted 0-30 cm mean, not an equal average
            .multiply(ee.Image.constant(DEPTH_WEIGHTS))
            .reduce(ee.Reducer.sum())
            .rename([prop])
            .updateMask(crop_mask)
        )
        bands.append(img)
        info[prop] = {
            "scale": float(scale),
            "desc": desc,
            "asset": asset_id,
            "bands": list(depth_bands),
        }

    return ee.Image.cat(bands), info


# ---------------------------------------------------------------------------
# EE FEATURECOLLECTION -> DataFrame
# ---------------------------------------------------------------------------
def fc_to_df(
    fc_info: Dict[str, Any],
    band_info: Dict[str, Dict[str, Any]],
    state_abbr: str,
    mask_start_year: int,
    mask_end_year: int,
) -> pd.DataFrame:
    feats = fc_info.get("features", [])
    rows: List[Dict[str, Any]] = []

    for f in feats:
        props = f.get("properties", {}) or {}

        row: Dict[str, Any] = {
            "mask_start_year": int(mask_start_year),
            "mask_end_year": int(mask_end_year),
            "state_abbr": state_abbr.upper(),
            "state_fips": str(props.get("state_fips") or ""),
            "county_fips": str(props.get("county_fips") or ""),
            "county_name": str(props.get("county_name") or ""),
        }

        # apply scaling factors
        for prop in band_info.keys():
            v = props.get(prop, None)
            if v is None:
                row[f"{prop}_mean"] = np.nan
            else:
                try:
                    row[f"{prop}_mean"] = float(v) * band_info[prop]["scale"]
                except Exception:
                    row[f"{prop}_mean"] = np.nan

        rows.append(row)

    df = pd.DataFrame(rows)

    ordered_cols = (
        ["mask_start_year", "mask_end_year", "state_abbr", "state_fips", "county_fips", "county_name"]
        + [f"{k}_mean" for k in band_info.keys()]
    )
    for c in ordered_cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[ordered_cols]

    # preserve leading zeros in fips if any (they’re strings already)
    df["state_fips"] = df["state_fips"].astype(str)
    df["county_fips"] = df["county_fips"].astype(str)

    return df


# ---------------------------------------------------------------------------
# PROCESS ONE STATE
# ---------------------------------------------------------------------------
def _resolve_soil_scale() -> float:
    """OpenLandMap's native pixel size, resolved once."""
    global GEE_SCALE_METERS
    if GEE_SCALE_METERS is None:
        asset = SOIL_PROPERTIES["clay"][0]
        GEE_SCALE_METERS = native_scale(asset, is_collection=False)
    return GEE_SCALE_METERS


def process_state(
    state_abbr: str,
    soil_image: ee.Image,
    band_info: Dict[str, Dict[str, Any]],
    mask_start_year: int,
    mask_end_year: int,
) -> pd.DataFrame:
    state_abbr = state_abbr.upper()
    state_fips = fetch_state_fips([state_abbr])[0]
    counties_fc = get_tiger_counties_fc(state_fips)

    reduced_fc = soil_image.reduceRegions(
        collection=counties_fc,
        reducer=ee.Reducer.mean(),
        scale=_resolve_soil_scale(),
        tileScale=GEE_TILE_SCALE,
    )

    # keep only needed props to reduce payload size
    props_to_keep = ["state_fips", "county_fips", "county_name"] + list(band_info.keys())
    reduced_fc = reduced_fc.select(props_to_keep)

    # one request per state
    fc_info = reduced_fc.getInfo()
    return fc_to_df(fc_info, band_info, state_abbr, mask_start_year, mask_end_year)


# ---------------------------------------------------------------------------
# MAIN DRIVER
# ---------------------------------------------------------------------------
def run(states: List[str], resume: bool = True, force: bool = False) -> None:
    authenticate_gee()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    mask_start_year, mask_end_year = ANALYSIS_YEARS
    logging.info(f"Using ANALYSIS_YEARS={ANALYSIS_YEARS} for STABLE cropland mask")

    # build stability mask + masked soil image ONCE
    t0 = time.time()
    logging.info(
        f"Building stable cropland mask for {mask_start_year}-{mask_end_year} "
        f"(pixel must be cropland in >={int(MASK_MIN_FRAC * 100)}% of years)..."
    )
    stable_mask = get_stable_crop_mask(mask_start_year, mask_end_year, min_frac=MASK_MIN_FRAC)
    soil_image, band_info = build_soil_image(stable_mask)
    logging.info(f"Built masked soil image in {time.time() - t0:.1f}s")

    # --force deletes the output files for EVERY state, so any previously
    # completed entry now points at data that no longer exists. Loading the old
    # set here and then processing only the requested states left stale entries
    # behind, and a later plain resume skipped exactly the states whose rows
    # had been deleted. Start empty whenever the outputs are being cleared.
    completed = load_checkpoint(CHECKPOINT_FILE) if (resume and not force) else set()
    total = len(states)

    # Same duplicate-append hazard as the other downloaders: --no-resume/--force
    # reset the checkpoint but left soil.csv in place while the writer below
    # appends. state_already_in_output() only guarded the resume path.
    if not resume or force:
        CHECKPOINT_FILE.unlink(missing_ok=True)
    if (not resume or force) and OUTPUT_FILE.exists():
        logging.warning(f"Not resuming: removing existing {OUTPUT_FILE} to avoid duplicate rows")
        OUTPUT_FILE.unlink()

    for i, st in enumerate(states, start=1):
        st = st.upper()

        if not force:
            if resume and st in completed:
                logging.info(f"[{i}/{total}] Skipping {st} (checkpointed)")
                continue
            if resume and state_already_in_output(st):
                logging.info(f"[{i}/{total}] Skipping {st} (already in output)")
                completed.add(st)
                save_checkpoint(CHECKPOINT_FILE, completed)
                continue

        logging.info(f"[{i}/{total}] Processing {st}...")
        start = time.time()

        # A stability mask over 17 CDL years reduced at 250 m for a whole state
        # is a heavy getInfo(); large states can time out. Previously any failure
        # propagated and aborted the entire run, losing every remaining state.
        try:
            df_state = process_state(st, soil_image, band_info, mask_start_year, mask_end_year)
        except Exception as e:
            logging.error(f"  FAILED {st}: {e}")
            logging.error("  Continuing to next state; re-run to retry this one.")
            continue

        # append to output (state-by-state so you never lose progress)
        if OUTPUT_FILE.exists():
            df_state.to_csv(OUTPUT_FILE, mode="a", header=False, index=False)
        else:
            df_state.to_csv(OUTPUT_FILE, mode="w", header=True, index=False)

        completed.add(st)
        save_checkpoint(CHECKPOINT_FILE, completed)

        logging.info(f"✓ Done {st}: {len(df_state)} counties in {time.time() - start:.1f}s")

    logging.info(f"✅ Wrote: {OUTPUT_FILE}")
    logging.info(f"✅ Checkpoint: {CHECKPOINT_FILE}")


def main():
    parser = argparse.ArgumentParser(
        description="Download cropland-masked county soil properties from OpenLandMap via GEE"
    )
    parser.add_argument(
        "--states",
        nargs="+",
        default=DEFAULT_STATES,
        help="State abbreviations (default: config.DEFAULT_STATES)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from checkpoint (default: True)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start from scratch (ignore checkpoint)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess states even if already present",
    )

    parser.add_argument("--study", action="store_true",
                        help="Use config.STUDY_STATES (the modelling scope) instead of "
                             "DEFAULT_STATES. Avoids retyping the list.")

    args = parser.parse_args()
    states = resolve_states(args.states, args.study)
    resume = False if args.no_resume else args.resume

    run(states=states, resume=resume, force=args.force)


if __name__ == "__main__":
    main()
