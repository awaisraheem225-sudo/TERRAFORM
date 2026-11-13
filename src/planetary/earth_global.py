
"""
TERRAFORM — Global Earth Raster Analysis
Version: 1.3

Dataset:
    NOAA ETOPO 2022 — 60 arc-second global relief

Features:
    1. Raster metadata inspection
    2. Block-wise elevation statistics
    3. Preliminary land/ocean classification
    4. NoData-aware global elevation visualisation
    5. CRS inspection
    6. Machine-readable JSON reporting

The original source raster is never modified.
"""

import json
from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt

from rasterio.enums import Resampling
from pyproj import CRS as PyprojCRS


# ==================================================
# 1. PROJECT CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "Global"
    / "Earth"
    / "earth_etopo_2022_60s.tif"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "Global"
    / "Earth"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "Visualisations"
    / "Global"
)

REPORT_FILE = OUTPUT_DIR / "earth_global_report.json"
FIGURE_FILE = FIGURE_DIR / "earth_global_elevation.png"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. RASTER VALIDITY
# ==================================================

def get_valid_mask(data, nodata):

    valid = np.isfinite(data)

    if nodata is not None:

        if np.isnan(nodata):
            valid &= ~np.isnan(data)

        else:
            valid &= data != nodata

    return valid


# ==================================================
# 3. GLOBAL ELEVATION STATISTICS
# ==================================================

def analyse_elevation(src):

    nodata = src.nodata

    valid_count = 0
    land_count = 0
    ocean_count = 0

    elevation_sum = 0.0
    land_sum = 0.0
    ocean_sum = 0.0

    minimum = float("inf")
    maximum = float("-inf")

    land_minimum = float("inf")
    land_maximum = float("-inf")

    ocean_minimum = float("inf")
    ocean_maximum = float("-inf")

    print("\nPROCESSING GLOBAL ELEVATION BLOCKS")
    print("-" * 45)

    for _, window in src.block_windows(1):

        data = src.read(1, window=window)

        valid = get_valid_mask(data, nodata)

        values = data[valid].astype(
            np.float64,
            copy=False
        )

        if values.size == 0:
            continue

        valid_count += values.size

        elevation_sum += float(
            values.sum(dtype=np.float64)
        )

        minimum = min(
            minimum,
            float(values.min())
        )

        maximum = max(
            maximum,
            float(values.max())
        )

        # Preliminary elevation-based classification
        land_values = values[values > 0]
        ocean_values = values[values <= 0]

        land_count += land_values.size
        ocean_count += ocean_values.size

        if land_values.size:

            land_sum += float(
                land_values.sum(dtype=np.float64)
            )

            land_minimum = min(
                land_minimum,
                float(land_values.min())
            )

            land_maximum = max(
                land_maximum,
                float(land_values.max())
            )

        if ocean_values.size:

            ocean_sum += float(
                ocean_values.sum(dtype=np.float64)
            )

            ocean_minimum = min(
                ocean_minimum,
                float(ocean_values.min())
            )

            ocean_maximum = max(
                ocean_maximum,
                float(ocean_values.max())
            )

    if valid_count == 0:

        raise ValueError(
            "The raster contains no valid elevation cells."
        )

    def safe_range(low, high):

        if low == float("inf"):
            return None, None

        return low, high

    land_minimum, land_maximum = safe_range(
        land_minimum,
        land_maximum
    )

    ocean_minimum, ocean_maximum = safe_range(
        ocean_minimum,
        ocean_maximum
    )

    return {

        "valid_cells": int(valid_count),

        "land_cells": int(land_count),

        "ocean_cells": int(ocean_count),

        "land_percentage_of_valid_cells": round(
            land_count / valid_count * 100,
            4
        ),

        "ocean_percentage_of_valid_cells": round(
            ocean_count / valid_count * 100,
            4
        ),

        "global_elevation_min_m": minimum,

        "global_elevation_max_m": maximum,

        "global_elevation_mean_m": (
            elevation_sum / valid_count
        ),

        "land_elevation_min_m": land_minimum,

        "land_elevation_max_m": land_maximum,

        "land_elevation_mean_m": (
            land_sum / land_count if land_count else None
        ),

        "ocean_elevation_min_m": ocean_minimum,

        "ocean_elevation_max_m": ocean_maximum,

        "ocean_elevation_mean_m": (
            ocean_sum / ocean_count if ocean_count else None
        )
    }


# ==================================================
# 4. CRS INSPECTION
# ==================================================

def inspect_crs(src):

    print("\nCOORDINATE REFERENCE SYSTEM")
    print("-" * 45)

    if src.crs is None:

        print("WARNING: Raster CRS is not defined.")

        return {
            "crs": None,
            "crs_name": None,
            "crs_wkt": None,
            "axis_information": [],
            "sub_crs_information": []
        }

    crs_info = PyprojCRS.from_user_input(src.crs)

    print("CRS:", src.crs)
    print("CRS DESCRIPTION:", crs_info.name)

    axis_information = []

    print("\nAXIS INFORMATION")

    for axis in crs_info.axis_info:

        record = {
            "name": axis.name,
            "direction": axis.direction,
            "unit": axis.unit_name
        }

        axis_information.append(record)

        print(
            f"{axis.name} | "
            f"Direction: {axis.direction} | "
            f"Unit: {axis.unit_name}"
        )

    sub_crs_information = []

    print("\nSUB-CRS INFORMATION")

    if crs_info.is_compound:

        for sub_crs in crs_info.sub_crs_list:

            record = {
                "name": sub_crs.name,
                "type": sub_crs.type_name
            }

            sub_crs_information.append(record)

            print(
                f"{sub_crs.name} | "
                f"Type: {sub_crs.type_name}"
            )

    else:

        print("CRS is not identified as a compound CRS.")

    return {

        "crs": src.crs.to_string(),

        "crs_name": crs_info.name,

        "crs_wkt": src.crs.to_wkt(),

        "axis_information": axis_information,

        "sub_crs_information": sub_crs_information
    }


# ==================================================
# 5. GLOBAL ELEVATION VISUALISATION
# ==================================================

def create_global_map(src):

    print("\nGENERATING GLOBAL EARTH MAP")
    print("-" * 45)

    preview_height = 1080
    preview_width = 2160

    elevation = src.read(
        1,
        out_shape=(
            preview_height,
            preview_width
        ),
        masked=True,
        resampling=Resampling.average
    ).astype(np.float32)

    masked = np.ma.masked_invalid(elevation)

    # Standard scientific atlas appearance.
    fig, ax = plt.subplots(
        figsize=(16, 8),
        facecolor="white"
    )

    ax.set_facecolor("white")

    # Main map.
    image = ax.imshow(

        masked,

        extent=(-180, 180, -90, 90),

        origin="upper",

        cmap="terrain",

        vmin=-8000,

        vmax=6000,

        interpolation="nearest",

        aspect="auto"
    )

    ax.set_title(

        "TERRAFORM | GLOBAL EARTH ELEVATION",

        fontsize=17,

        fontweight="bold",

        color="#202B36",

        pad=14
    )

    ax.set_xlabel(
        "Longitude (degrees)",
        fontsize=10,
        labelpad=7,
        color="#303B46"
    )

    ax.set_ylabel(
        "Latitude (degrees)",
        fontsize=10,
        labelpad=7,
        color="#303B46"
    )

    ax.set_xticks(np.arange(-180, 181, 30))
    ax.set_yticks(np.arange(-90, 91, 15))

    ax.tick_params(
        axis="both",
        labelsize=9,
        colors="#303B46"
    )

    ax.grid(
        color="white",
        alpha=0.45,
        linewidth=0.55
    )

    for spine in ax.spines.values():
        spine.set_color("#AAB4BE")

    # Keep the map large and use only a small
    # amount of space for the colour bar.
    fig.subplots_adjust(
        left=0.075,
        right=0.96,
        top=0.87,
        bottom=0.17
    )

    # Colour bar.
    colorbar = fig.colorbar(

        image,

        ax=ax,

        orientation="horizontal",

        fraction=0.045,

        pad=0.12,

        aspect=50
    )

    colorbar.set_label(
        "Elevation relative to dataset vertical reference (m)",
        fontsize=9,
        labelpad=5,
        color="#303B46"
    )

    colorbar.ax.tick_params(
        labelsize=8,
        colors="#303B46"
    )

    # One compact source attribution.
    fig.text(

        0.5,

        0.035,

        "Source: NOAA ETOPO 2022 | 60 arc-second global relief",

        ha="center",

        va="center",

        fontsize=8.5,

        color="#586675"
    )

    # Save without tight_layout or excessive padding.
    fig.savefig(

        FIGURE_FILE,

        dpi=180,

        facecolor="white",

        bbox_inches="tight",

        pad_inches=0.08
    )

    plt.close(fig)

    print("Global elevation map generated.")

    return {

        "preview_width": preview_width,

        "preview_height": preview_height,

        "figure_path": str(
            FIGURE_FILE.relative_to(PROJECT_ROOT)
        )
    }


# ==================================================
# 6. MAIN ANALYSIS
# ==================================================

def main():

    print("=" * 60)
    print("TERRAFORM — GLOBAL EARTH ANALYSIS")
    print("=" * 60)

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Earth elevation raster not found: {INPUT_FILE}"
        )

    with rasterio.open(INPUT_FILE) as src:

        print("\nDATASET METADATA")
        print("-" * 45)

        print("Driver:", src.driver)
        print("Width:", src.width)
        print("Height:", src.height)
        print("Bands:", src.count)
        print("Resolution:", src.res)
        print("NoData:", src.nodata)
        print("Bounds:", src.bounds)
        print("Data types:", src.dtypes)
        print("Block shape:", src.block_shapes)

        crs_report = inspect_crs(src)

        statistics = analyse_elevation(src)

        visualisation = create_global_map(src)

        report = {

            "project": "TERRAFORM",

            "analysis": "Global Earth Elevation Analysis",

            "version": "1.3",

            "dataset": {

                "name": "NOAA ETOPO 2022",

                "provider": "NOAA",

                "source_file": INPUT_FILE.name,

                "source_resolution": "60 arc-seconds",

                "measurement": "Elevation and bathymetry",

                "units": "Metres"
            },

            "coordinate_reference_system": crs_report,

            "dimensions": {

                "width": src.width,

                "height": src.height,

                "bands": src.count
            },

            "raster_metadata": {

                "driver": src.driver,

                "resolution": list(src.res),

                "bounds": {

                    "left": src.bounds.left,

                    "bottom": src.bounds.bottom,

                    "right": src.bounds.right,

                    "top": src.bounds.top
                },

                "data_types": list(src.dtypes),

                "nodata_value": src.nodata
            },

            "statistics": statistics,

            "visualisation": visualisation,

            "methodology": {

                "processing": "Block-wise raster analysis",

                "land_definition": "Elevation greater than zero metres",

                "ocean_definition": "Elevation less than or equal to zero metres",

                "classification_type": "Preliminary elevation-based classification",

                "nodata_handling": "Invalid and NoData cells excluded",

                "original_raster_modified": False
            },

            "limitations": [

                "ETOPO is a global overview dataset, not a regional high-resolution DEM.",

                "The elevation-zero threshold is a preliminary classification, not an authoritative coastline mask.",

                "Below-sea-level terrestrial regions may be classified as ocean.",

                "Land and ocean percentages represent valid raster-cell counts, not physical surface-area percentages.",

                "Geographic grid cells have latitude-dependent physical dimensions.",

                "Vertical reference information must be interpreted using official dataset metadata.",

                "Global terrain derivatives require geodesically appropriate processing."
            ]
        }

    # Save machine-readable report.
    with REPORT_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
            allow_nan=False
        )

    print("\nGLOBAL ELEVATION STATISTICS")
    print("-" * 45)

    for key, value in statistics.items():
        print(f"{key}: {value}")

    print("\nOUTPUT FILES")
    print("-" * 45)

    print("JSON REPORT:", REPORT_FILE)
    print("GLOBAL MAP:", FIGURE_FILE)

    print("\nEARTH GLOBAL ANALYSIS: COMPLETED")
    print("=" * 60)


# ==================================================
# 7. EXECUTION
# ==================================================

if __name__ == "__main__":
    main()
