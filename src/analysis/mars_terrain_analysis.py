
from pathlib import Path
import json
import math

import numpy as np
import rasterio

from rasterio.windows import from_bounds, Window
from rasterio.warp import reproject, transform_bounds, Resampling
from rasterio.transform import from_origin
from rasterio.crs import CRS

from scipy.ndimage import (
    uniform_filter,
    maximum_filter,
    minimum_filter
)


# ============================================================
# TERRAFORM — CORRECTED MARS TERRAIN ANALYSIS ENGINE
# Study area: Jezero Crater
# Source: NASA MOLA MEGDR 128 pixels/degree
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

SOURCE_DEM = (
    ROOT / "data" / "processed" / "mars"
    / "mars_mola_128ppd_megt44n000hb.tif"
)

OUTPUT_DIR = ROOT / "data" / "processed" / "mars"
REPORT_DIR = ROOT / "outputs" / "reports"

# Jezero reference centre
JEZERO_LAT = 18.4
JEZERO_LON = 77.7

# Geographic study-area bounds
LAT_MIN = 17.5
LAT_MAX = 19.3
LON_MIN = 76.8
LON_MAX = 78.6

# Target pixel resolution
TARGET_RESOLUTION_M = 463.0

# Terrain neighbourhood
NEIGHBOURHOOD_SIZE = 3

NODATA = -9999.0

# Local Mars azimuthal equidistant projection
MARS_LOCAL_CRS = CRS.from_proj4(
    "+proj=aeqd "
    "+lat_0=18.4 "
    "+lon_0=77.7 "
    "+R=3396000 "
    "+units=m "
    "+no_defs "
    "+type=crs"
)


# ============================================================
# DIRECTORIES
# ============================================================

def create_directories():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# STATISTICS
# ============================================================

def calculate_statistics(array):

    valid = array[np.isfinite(array)]

    if valid.size == 0:
        raise ValueError(
            "No valid pixels available for statistics."
        )

    return {
        "valid_pixels": int(valid.size),
        "minimum": float(np.min(valid)),
        "maximum": float(np.max(valid)),
        "mean": float(np.mean(valid)),
        "median": float(np.median(valid)),
        "standard_deviation": float(np.std(valid))
    }


# ============================================================
# SAVE GEOTIFF
# ============================================================

def save_raster(
    path,
    array,
    transform,
    description,
    units
):

    output = np.asarray(
        array,
        dtype=np.float32
    ).copy()

    output[~np.isfinite(output)] = NODATA

    profile = {
        "driver": "GTiff",
        "height": output.shape[0],
        "width": output.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": MARS_LOCAL_CRS,
        "transform": transform,
        "nodata": NODATA,
        "compress": "deflate",
        "predictor": 3,
        "BIGTIFF": "IF_SAFER"
    }

    with rasterio.open(
        path,
        "w",
        **profile
    ) as destination:

        destination.write(output, 1)

        destination.set_band_description(
            1,
            description
        )

        destination.update_tags(
            PROJECT="TERRAFORM",
            STUDY_AREA="Jezero Crater, Mars",
            SOURCE="NASA MOLA MEGDR",
            UNITS=units,
            NEIGHBOURHOOD="3 x 3 pixels",
            PROCESSING_VERSION="2.0 - edge-safe"
        )

    print(f"Saved: {path.name}")


# ============================================================
# EXTRACT JEZERO
# ============================================================

def extract_jezero_region():

    print("\n[1/6] EXTRACTING JEZERO STUDY AREA")

    if not SOURCE_DEM.exists():
        raise FileNotFoundError(
            f"MOLA raster not found:\n{SOURCE_DEM}"
        )

    with rasterio.open(SOURCE_DEM) as source:

        print(
            f"Source dimensions: "
            f"{source.width} x {source.height}"
        )

        window = from_bounds(
            LON_MIN,
            LAT_MIN,
            LON_MAX,
            LAT_MAX,
            source.transform
        )

        window = window.round_offsets().round_lengths()

        window = window.intersection(
            Window(
                0,
                0,
                source.width,
                source.height
            )
        )

        elevation = source.read(
            1,
            window=window,
            masked=True
        ).astype(np.float64).filled(np.nan)

        source_transform = source.window_transform(window)
        source_crs = source.crs

        if elevation.size == 0:
            raise ValueError(
                "Empty Jezero study area."
            )

        print(
            f"Extracted dimensions: "
            f"{elevation.shape[1]} x {elevation.shape[0]}"
        )

        print(
            f"Valid source pixels: "
            f"{np.isfinite(elevation).sum():,}"
        )

    return elevation, source_transform, source_crs


# ============================================================
# REPROJECT TO LOCAL MARS METRIC CRS
# ============================================================

def reproject_jezero(
    elevation,
    source_transform,
    source_crs
):

    print("\n[2/6] REPROJECTING TO LOCAL MARS COORDINATES")

    rows, columns = elevation.shape

    left = source_transform.c
    top = source_transform.f

    right = left + columns * source_transform.a
    bottom = top + rows * source_transform.e

    projected_bounds = transform_bounds(
        source_crs,
        MARS_LOCAL_CRS,
        left,
        bottom,
        right,
        top,
        densify_pts=41
    )

    min_x, min_y, max_x, max_y = projected_bounds

    width = math.ceil(
        (max_x - min_x) / TARGET_RESOLUTION_M
    )

    height = math.ceil(
        (max_y - min_y) / TARGET_RESOLUTION_M
    )

    target_transform = from_origin(
        min_x,
        max_y,
        TARGET_RESOLUTION_M,
        TARGET_RESOLUTION_M
    )

    projected = np.full(
        (height, width),
        np.nan,
        dtype=np.float32
    )

    reproject(
        source=elevation,
        destination=projected,
        src_transform=source_transform,
        src_crs=source_crs,
        dst_transform=target_transform,
        dst_crs=MARS_LOCAL_CRS,
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
        num_threads=2
    )

    valid = np.isfinite(projected)

    if not valid.any():
        raise ValueError(
            "Reprojection produced no valid pixels."
        )

    print(f"Projected dimensions: {width} x {height}")
    print(f"Resolution: {TARGET_RESOLUTION_M} m")
    print(f"Valid pixels: {valid.sum():,}")

    return projected.astype(np.float64), target_transform


# ============================================================
# VALID 3x3 NEIGHBOURHOOD MASK
# ============================================================

def complete_neighbourhood_mask(valid):

    count = uniform_filter(
        valid.astype(np.float64),
        size=NEIGHBOURHOOD_SIZE,
        mode="constant",
        cval=0.0
    )

    # Only retain cells with all nine valid neighbours.
    return count >= (1.0 - 1e-9)


# ============================================================
# SLOPE
# ============================================================

def calculate_slope(elevation):

    print("\n[3/6] CALCULATING SLOPE")

    valid = np.isfinite(elevation)

    result = np.full(
        elevation.shape,
        np.nan,
        dtype=np.float64
    )

    complete = complete_neighbourhood_mask(valid)

    # Central differences: left/right and up/down.
    dz_dx = (
        elevation[1:-1, 2:]
        - elevation[1:-1, :-2]
    ) / (2.0 * TARGET_RESOLUTION_M)

    dz_dy = (
        elevation[2:, 1:-1]
        - elevation[:-2, 1:-1]
    ) / (2.0 * TARGET_RESOLUTION_M)

    gradient = np.sqrt(
        dz_dx ** 2 + dz_dy ** 2
    )

    interior_complete = complete[1:-1, 1:-1]

    interior_result = result[1:-1, 1:-1]

    interior_result[interior_complete] = np.degrees(
        np.arctan(
            gradient[interior_complete]
        )
    )

    return result


# ============================================================
# ROUGHNESS
# ============================================================

def calculate_roughness(elevation):

    print("\n[4/6] CALCULATING ROUGHNESS")

    valid = np.isfinite(elevation)

    complete = complete_neighbourhood_mask(valid)

    filled = np.where(
        valid,
        elevation,
        0.0
    )

    local_mean = uniform_filter(
        filled,
        size=NEIGHBOURHOOD_SIZE,
        mode="constant",
        cval=0.0
    )

    local_mean_square = uniform_filter(
        filled ** 2,
        size=NEIGHBOURHOOD_SIZE,
        mode="constant",
        cval=0.0
    )

    # Complete neighbourhoods contain exactly nine valid cells.
    variance = np.maximum(
        local_mean_square - local_mean ** 2,
        0.0
    )

    roughness = np.sqrt(variance)

    roughness[~complete] = np.nan

    return roughness


# ============================================================
# LOCAL RELIEF
# ============================================================

def calculate_local_relief(elevation):

    print("\n[5/6] CALCULATING LOCAL RELIEF")

    valid = np.isfinite(elevation)

    complete = complete_neighbourhood_mask(valid)

    filled_max = np.where(
        valid,
        elevation,
        -np.inf
    )

    filled_min = np.where(
        valid,
        elevation,
        np.inf
    )

    local_maximum = maximum_filter(
        filled_max,
        size=NEIGHBOURHOOD_SIZE,
        mode="constant",
        cval=-np.inf
    )

    local_minimum = minimum_filter(
        filled_min,
        size=NEIGHBOURHOOD_SIZE,
        mode="constant",
        cval=np.inf
    )

    relief = local_maximum - local_minimum

    relief[~complete] = np.nan

    return relief


# ============================================================
# SAVE PRODUCTS AND REPORT
# ============================================================

def save_outputs(
    elevation,
    slope,
    roughness,
    local_relief,
    transform
):

    print("\n[6/6] SAVING TERRAIN PRODUCTS")

    products = {
        "elevation": (
            "jezero_dem_local_mars.tif",
            elevation,
            "Elevation above Martian areoid",
            "metres"
        ),
        "slope": (
            "jezero_slope_deg.tif",
            slope,
            "Terrain slope",
            "degrees"
        ),
        "roughness": (
            "jezero_roughness_std_3x3_m.tif",
            roughness,
            "Local elevation standard deviation, 3x3",
            "metres"
        ),
        "local_relief": (
            "jezero_local_relief_3x3_m.tif",
            local_relief,
            "Local maximum minus minimum elevation, 3x3",
            "metres"
        )
    }

    for (
        filename,
        array,
        description,
        units
    ) in products.values():

        save_raster(
            OUTPUT_DIR / filename,
            array,
            transform,
            description,
            units
        )

    report = {
        "project": "TERRAFORM - Planetary Analog Intelligence",
        "planet": "Mars",
        "study_area": "Jezero Crater",
        "source": "NASA MOLA MEGDR 128 pixels per degree",
        "source_file": SOURCE_DEM.name,
        "reference_center": {
            "latitude_planetocentric": JEZERO_LAT,
            "longitude_east": JEZERO_LON
        },
        "study_bounds_degrees": {
            "minimum_latitude": LAT_MIN,
            "maximum_latitude": LAT_MAX,
            "minimum_longitude": LON_MIN,
            "maximum_longitude": LON_MAX
        },
        "processing": {
            "projection": "Azimuthal Equidistant",
            "mars_radius_metres": 3396000,
            "target_resolution_metres": TARGET_RESOLUTION_M,
            "resampling": "Bilinear",
            "neighbourhood": "3 x 3 pixels",
            "neighbourhood_approximate_width_metres": (
                NEIGHBOURHOOD_SIZE * TARGET_RESOLUTION_M
            ),
            "slope_method": "Central difference gradient",
            "roughness_method": "Local elevation standard deviation",
            "local_relief_method": "Local maximum minus minimum",
            "edge_handling": (
                "Only complete 3x3 valid neighbourhoods retained"
            )
        },
        "terrain_statistics": {
            "elevation_m": calculate_statistics(elevation),
            "slope_degrees": calculate_statistics(slope),
            "roughness_m": calculate_statistics(roughness),
            "local_relief_m": calculate_statistics(local_relief)
        },
        "scientific_limitations": [
            "MOLA MEGDR represents median observed topography.",
            "Interpolated values are supplied where observations are absent.",
            "MOLA resolution is substantially coarser than Earth GLO-30.",
            "Reprojection does not create additional source detail.",
            "Terrain metrics describe approximately 1.39 km neighbourhoods.",
            "Terrain metrics are regional descriptors, not a complete geological analogue assessment."
        ]
    }

    report_path = REPORT_DIR / "jezero_terrain_analysis.json"

    report_path.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8"
    )

    print(f"\nReport saved: {report_path}")

    print("\nCORRECTED TERRAIN STATISTICS")
    print("=" * 55)

    for metric, stats in report["terrain_statistics"].items():

        print(f"\n{metric}")

        for key, value in stats.items():

            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value:,}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("TERRAFORM — JEZERO TERRAIN ANALYSIS ENGINE v2.0")
    print("=" * 65)

    create_directories()

    elevation, source_transform, source_crs = (
        extract_jezero_region()
    )

    projected, transform = reproject_jezero(
        elevation,
        source_transform,
        source_crs
    )

    slope = calculate_slope(projected)

    roughness = calculate_roughness(projected)

    local_relief = calculate_local_relief(projected)

    save_outputs(
        projected,
        slope,
        roughness,
        local_relief,
        transform
    )

    print("\n" + "=" * 65)
    print("CORRECTED JEZERO TERRAIN ANALYSIS COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()
