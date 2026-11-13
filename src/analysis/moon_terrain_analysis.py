
from pathlib import Path
import json

import numpy as np
import rasterio

from rasterio.transform import from_origin
from rasterio.crs import CRS

from scipy.ndimage import (
    uniform_filter,
    maximum_filter,
    minimum_filter,
)


# ============================================================
# TERRAFORM — GLOBAL LUNAR TERRAIN ANALYSIS
# Source: NASA LRO LOLA LDEM_16 V3.1
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

SOURCE_IMG = ROOT / "data" / "Global" / "Moon" / "ldem_16.img"
SOURCE_LABEL = ROOT / "data" / "Global" / "Moon" / "LDEM_16.LBL"

OUTPUT_DIR = ROOT / "data" / "processed" / "moon"
REPORT_DIR = ROOT / "outputs" / "Global" / "Moon"

ROWS = 2880
COLS = 5760

SCALE_FACTOR = 0.5
REFERENCE_RADIUS_M = 1_737_400.0

PIXEL_SIZE_DEGREES = 0.0625
NEIGHBOURHOOD_SIZE = 3

NODATA = -9999.0

# PDS label extrema recorded for metadata comparison.
LABEL_RAW_MIN = -18150
LABEL_RAW_MAX = 21580

# Lunar spherical geographic coordinate system.
MOON_CRS = CRS.from_proj4(
    "+proj=longlat "
    "+a=1737400 "
    "+b=1737400 "
    "+no_defs "
    "+type=crs"
)

TRANSFORM = from_origin(
    0.0,
    90.0,
    PIXEL_SIZE_DEGREES,
    PIXEL_SIZE_DEGREES,
)

# Approximate physical spacing at the equator.
DLON = np.deg2rad(PIXEL_SIZE_DEGREES)
DLAT = np.deg2rad(PIXEL_SIZE_DEGREES)

DY = REFERENCE_RADIUS_M * DLAT
EQUATOR_DX = REFERENCE_RADIUS_M * DLON


# ============================================================
# DIRECTORIES
# ============================================================

def create_directories():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD AND VALIDATE PDS BINARY
# ============================================================

def load_lunar_elevation():

    print("\n[1/6] READING NASA LOLA BINARY")

    if not SOURCE_IMG.exists():
        raise FileNotFoundError(SOURCE_IMG)

    if not SOURCE_LABEL.exists():
        raise FileNotFoundError(SOURCE_LABEL)

    expected_bytes = ROWS * COLS * 2
    actual_bytes = SOURCE_IMG.stat().st_size

    if actual_bytes != expected_bytes:
        raise ValueError(
            f"Unexpected binary size. "
            f"Expected {expected_bytes:,}, "
            f"received {actual_bytes:,}."
        )

    raw = np.fromfile(
        SOURCE_IMG,
        dtype="<i2",
    )

    if raw.size != ROWS * COLS:
        raise ValueError(
            "LOLA cell count validation failed."
        )

    raw = raw.reshape((ROWS, COLS))

    # Preserve original integer data.
    # Convert elevation values to float64 for calculations.
    elevation = (
        raw.astype(np.float64) * SCALE_FACTOR
    )

    print(f"Dimensions: {COLS} x {ROWS}")
    print("Data type: signed 16-bit little-endian")
    print(f"Valid cells: {elevation.size:,}")
    print(f"Raw minimum: {raw.min():,}")
    print(f"Raw maximum: {raw.max():,}")
    print(f"Elevation minimum: {elevation.min():,.2f} m")
    print(f"Elevation maximum: {elevation.max():,.2f} m")

    return raw, elevation


# ============================================================
# STATISTICS
# ============================================================

def calculate_statistics(array):

    values = np.asarray(array)

    valid = values[np.isfinite(values)]

    if valid.size == 0:
        raise ValueError(
            "No valid observations available for statistics."
        )

    return {
        "valid_pixels": int(valid.size),
        "minimum": float(np.min(valid)),
        "maximum": float(np.max(valid)),
        "mean": float(np.mean(valid)),
        "median": float(np.median(valid)),
        "standard_deviation": float(np.std(valid)),
        "percentiles": {
            "p05": float(np.percentile(valid, 5)),
            "p10": float(np.percentile(valid, 10)),
            "p25": float(np.percentile(valid, 25)),
            "p50": float(np.percentile(valid, 50)),
            "p75": float(np.percentile(valid, 75)),
            "p90": float(np.percentile(valid, 90)),
            "p95": float(np.percentile(valid, 95)),
        },
    }


# ============================================================
# SAVE GEOTIFF
# ============================================================

def save_raster(path, array, description, units):

    output = np.asarray(
        array,
        dtype=np.float32,
    ).copy()

    if output.shape != (ROWS, COLS):
        raise ValueError(
            f"Invalid output dimensions for {path.name}: "
            f"{output.shape}"
        )

    output[~np.isfinite(output)] = NODATA

    profile = {
        "driver": "GTiff",
        "height": ROWS,
        "width": COLS,
        "count": 1,
        "dtype": "float32",
        "crs": MOON_CRS,
        "transform": TRANSFORM,
        "nodata": NODATA,
        "compress": "deflate",
        "predictor": 3,
        "BIGTIFF": "IF_SAFER",
    }

    with rasterio.open(path, "w", **profile) as dst:

        dst.write(output, 1)

        dst.set_band_description(
            1,
            description,
        )

        dst.update_tags(
            PROJECT="TERRAFORM",
            PLANET="Moon",
            SOURCE="NASA LRO LOLA LDEM_16 V3.1",
            REFERENCE_RADIUS_M=str(REFERENCE_RADIUS_M),
            UNITS=units,
            NEIGHBOURHOOD="3 x 3 pixels",
            PIXEL_REGISTRATION="Pixel registered",
            PROCESSING="TERRAFORM Lunar Terrain Analysis v1.1",
        )

    print(f"Saved: {path.name}")


# ============================================================
# SPHERICAL PIXEL SPACING
# ============================================================

def calculate_row_spacing():

    latitudes = (
        90.0
        - (np.arange(ROWS) + 0.5) * PIXEL_SIZE_DEGREES
    )

    dx = (
        REFERENCE_RADIUS_M
        * np.cos(np.deg2rad(latitudes))
        * DLON
    )

    return latitudes, dx


# ============================================================
# SLOPE
# ============================================================

def calculate_slope(elevation):

    print("\n[2/6] CALCULATING LUNAR SLOPE")

    elevation64 = np.asarray(
        elevation,
        dtype=np.float64,
    )

    latitudes, dx = calculate_row_spacing()

    if elevation64.shape != (ROWS, COLS):
        raise ValueError(
            "Unexpected lunar elevation dimensions."
        )

    # Longitude wraps around the global raster.
    east = np.roll(
        elevation64,
        -1,
        axis=1,
    )

    west = np.roll(
        elevation64,
        1,
        axis=1,
    )

    dz_dx = (
        east - west
    ) / (2.0 * dx[:, None])

    # Latitude does not wrap.
    dz_dy = np.full(
        elevation64.shape,
        np.nan,
        dtype=np.float64,
    )

    dz_dy[1:-1, :] = (
        elevation64[2:, :]
        - elevation64[:-2, :]
    ) / (2.0 * DY)

    gradient = np.sqrt(
        dz_dx ** 2 + dz_dy ** 2
    )

    slope = np.degrees(
        np.arctan(gradient)
    )

    # Polar longitude convergence makes east-west
    # derivatives unreliable very close to the poles.
    polar_mask = np.abs(latitudes) >= 85.0

    slope[polar_mask, :] = np.nan

    # Exclude outer latitude rows.
    slope[0, :] = np.nan
    slope[-1, :] = np.nan

    return slope.astype(np.float32)


# ============================================================
# ROUGHNESS
# ============================================================

def calculate_roughness(elevation):

    print("\n[3/6] CALCULATING LUNAR ROUGHNESS")

    # Float64 intermediates reduce cancellation errors
    # when subtracting squared local means.
    elevation64 = np.asarray(
        elevation,
        dtype=np.float64,
    )

    # Latitude edges use constant padding.
    # Longitude wraps around the global grid.
    modes = ("constant", "wrap")

    local_mean = uniform_filter(
        elevation64,
        size=NEIGHBOURHOOD_SIZE,
        mode=modes,
        cval=0.0,
    )

    local_mean_square = uniform_filter(
        elevation64 ** 2,
        size=NEIGHBOURHOOD_SIZE,
        mode=modes,
        cval=0.0,
    )

    variance = np.maximum(
        local_mean_square - local_mean ** 2,
        0.0,
    )

    roughness = np.sqrt(variance)

    # Exclude outer latitude rows.
    roughness[0, :] = np.nan
    roughness[-1, :] = np.nan

    return roughness.astype(np.float32)


# ============================================================
# LOCAL RELIEF
# ============================================================

def calculate_local_relief(elevation):

    print("\n[4/6] CALCULATING LUNAR LOCAL RELIEF")

    elevation64 = np.asarray(
        elevation,
        dtype=np.float64,
    )

    modes = ("constant", "wrap")

    local_maximum = maximum_filter(
        elevation64,
        size=NEIGHBOURHOOD_SIZE,
        mode=modes,
        cval=-np.inf,
    )

    local_minimum = minimum_filter(
        elevation64,
        size=NEIGHBOURHOOD_SIZE,
        mode=modes,
        cval=np.inf,
    )

    relief = (
        local_maximum - local_minimum
    )

    # Exclude outer latitude rows.
    relief[0, :] = np.nan
    relief[-1, :] = np.nan

    return relief.astype(np.float32)


# ============================================================
# SAVE PRODUCTS AND REPORT
# ============================================================

def save_outputs(raw, elevation, slope, roughness, relief):

    print("\n[5/6] SAVING LUNAR TERRAIN PRODUCTS")

    products = {
        "moon_global_dem.tif": (
            elevation,
            "Lunar elevation relative to 1737.4 km reference sphere",
            "metres",
        ),

        "moon_global_slope_degrees.tif": (
            slope,
            "Lunar terrain slope",
            "degrees",
        ),

        "moon_global_roughness_3x3_m.tif": (
            roughness,
            "Local elevation standard deviation, 3x3",
            "metres",
        ),

        "moon_global_local_relief_3x3_m.tif": (
            relief,
            "Local maximum minus minimum elevation, 3x3",
            "metres",
        ),
    }

    for filename, (array, description, units) in products.items():

        save_raster(
            OUTPUT_DIR / filename,
            array,
            description,
            units,
        )

    raw_min = int(raw.min())
    raw_max = int(raw.max())

    report = {
        "project": "TERRAFORM - Planetary Analog Intelligence",
        "analysis": "Moon Global Terrain Analysis",
        "version": "1.1",

        "dataset": {
            "product": "LOLA Global Digital Elevation Model",
            "product_id": "LDEM_16",
            "data_set_id": "LRO-L-LOLA-4-GDR-V1.0",
            "product_version": "V3.1",
            "source": "NASA Planetary Data System - LRO LOLA",
            "filename": SOURCE_IMG.name,
            "dimensions": {
                "columns": COLS,
                "rows": ROWS,
                "total_cells": ROWS * COLS,
            },
            "data_type": "Signed 16-bit integer",
            "byte_order": "Little-endian",
            "scaling_factor": SCALE_FACTOR,
            "reference_radius_m": REFERENCE_RADIUS_M,
            "pixel_resolution_degrees": PIXEL_SIZE_DEGREES,
            "equatorial_pixel_scale_m": EQUATOR_DX,
            "projection": "Simple cylindrical",
            "longitude_range_degrees": [0, 360],
            "latitude_range_degrees": [-90, 90],
            "pixel_registration": "Pixel registered",
        },

        "raw_data_validation": {
            "file_size_validation": "PASS",
            "cell_count_validation": "PASS",
            "observed_raw_minimum": raw_min,
            "observed_raw_maximum": raw_max,
            "label_raw_minimum": LABEL_RAW_MIN,
            "label_raw_maximum": LABEL_RAW_MAX,
            "range_matches_label": (
                raw_min == LABEL_RAW_MIN
                and raw_max == LABEL_RAW_MAX
            ),
            "range_discrepancy": (
                "UNRESOLVED_METADATA_DISCREPANCY"
                if (
                    raw_min != LABEL_RAW_MIN
                    or raw_max != LABEL_RAW_MAX
                )
                else "NONE"
            ),
        },

        "elevation_statistics": calculate_statistics(elevation),

        "terrain_statistics": {
            "slope_degrees": calculate_statistics(slope),
            "roughness_m": calculate_statistics(roughness),
            "local_relief_m": calculate_statistics(relief),
        },

        "processing": {
            "conversion": "height_m = digital_number * 0.5",
            "coordinate_system": "Lunar spherical geographic CRS",
            "reference_radius_m": REFERENCE_RADIUS_M,
            "slope_method": "Spherical central differences",
            "roughness_method": (
                "3x3 local elevation standard deviation "
                "using float64 intermediate calculations"
            ),
            "local_relief_method": "3x3 local maximum minus minimum",
            "longitude_edge_handling": "Periodic wrap",
            "latitude_edge_handling": "Outer rows excluded",
            "polar_slope_exclusion": "Absolute latitude >= 85 degrees",
            "polar_exclusion_reason": (
                "Longitude convergence makes east-west derivatives "
                "unstable near the poles."
            ),
            "numerical_precision": (
                "Float64 intermediate calculations; float32 GeoTIFF outputs."
            ),
            "source_data_modified": False,
        },

        "scientific_limitations": [
            "LOLA elevations are relative to a 1737.4 km reference sphere.",
            "Observed raster extrema do not match the PDS label extrema; discrepancy remains unresolved.",
            "The source grid is approximately 1895.21 m at the equator.",
            "Physical east-west pixel spacing decreases with latitude.",
            "Slope statistics exclude latitudes at or beyond 85 degrees.",
            "The 3x3 neighbourhood has latitude-dependent physical dimensions.",
            "The PDS label describes surface generation using interpolation to compensate for incomplete coverage.",
            "Terrain derivatives are not equivalent to geological or operational suitability.",
            "Resampling cannot create additional source terrain detail.",
        ],

        "outputs": {
            "elevation": "data/processed/moon/moon_global_dem.tif",
            "slope": "data/processed/moon/moon_global_slope_degrees.tif",
            "roughness": "data/processed/moon/moon_global_roughness_3x3_m.tif",
            "local_relief": "data/processed/moon/moon_global_local_relief_3x3_m.tif",
        },
    }

    report_path = (
        REPORT_DIR / "moon_global_terrain_analysis.json"
    )

    report_path.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print(f"\nReport saved: {report_path}")

    print("\nLUNAR TERRAIN STATISTICS")
    print("=" * 55)

    for metric, stats in report["terrain_statistics"].items():

        print(f"\n{metric}")

        for key, value in stats.items():

            if isinstance(value, dict):

                print(f"  {key}:")

                for subkey, subvalue in value.items():
                    print(f"    {subkey}: {subvalue:.4f}")

            elif isinstance(value, float):
                print(f"  {key}: {value:.4f}")

            else:
                print(f"  {key}: {value:,}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("TERRAFORM — GLOBAL LUNAR TERRAIN ANALYSIS")
    print("=" * 65)

    create_directories()

    raw, elevation = load_lunar_elevation()

    slope = calculate_slope(elevation)

    roughness = calculate_roughness(elevation)

    relief = calculate_local_relief(elevation)

    save_outputs(
        raw,
        elevation,
        slope,
        roughness,
        relief,
    )

    print("\n[6/6] ANALYSIS COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()
