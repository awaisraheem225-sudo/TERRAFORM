
from pathlib import Path
import json
import gc

import numpy as np
import rasterio

from rasterio.warp import calculate_default_transform, reproject, Resampling
from scipy.ndimage import (
    minimum_filter,
    maximum_filter,
    uniform_filter
)

# ============================================================
# TERRAFORM — PLANETARY ANALOG INTELLIGENCE
# Earth Terrain Analysis Engine
#
# Pilot: Atacama Desert, Chile
# Source: Copernicus DEM GLO-30
#
# Outputs:
#   1. Projected elevation model
#   2. Slope (degrees)
#   3. Terrain roughness (3x3 standard deviation)
#   4. Local relief (3x3 elevation range)
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = ROOT / "data" / "raw" / "earth"
OUTPUT_DIR = ROOT / "data" / "processed" / "earth"
REPORT_DIR = ROOT / "outputs" / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_CRS = "EPSG:4326"
TARGET_CRS = "EPSG:32719"  # UTM Zone 19 South

TARGET_RESOLUTION = 30.0
NODATA = -9999.0

PROJECTED_FILE = OUTPUT_DIR / "earth_dem_projected_utm19s.tif"
SLOPE_FILE = OUTPUT_DIR / "earth_slope_deg.tif"
ROUGHNESS_FILE = OUTPUT_DIR / "earth_roughness_std_3x3_m.tif"
RELIEF_FILE = OUTPUT_DIR / "earth_local_relief_3x3_m.tif"

REPORT_FILE = REPORT_DIR / "earth_terrain_analysis.json"


def find_source_raster():
    """Locate the original Copernicus DEM."""

    files = list(INPUT_DIR.glob("*.tif"))

    if not files:
        raise FileNotFoundError(
            f"No TIFF files found in {INPUT_DIR}"
        )

    if len(files) > 1:
        print("Multiple TIFF files found. Selecting the first one.")

    return files[0]


def calculate_stats(array, valid_mask):
    """Calculate descriptive statistics without NoData pixels."""

    values = array[valid_mask]

    if values.size == 0:
        return {
            "valid_pixels": 0,
            "minimum": None,
            "maximum": None,
            "mean": None,
            "standard_deviation": None
        }

    return {
        "valid_pixels": int(values.size),
        "minimum": round(float(np.min(values)), 4),
        "maximum": round(float(np.max(values)), 4),
        "mean": round(float(np.mean(values)), 4),
        "standard_deviation": round(float(np.std(values)), 4)
    }


def save_raster(path, array, profile, description, units):
    """Save a single-band GeoTIFF with metadata."""

    output_profile = profile.copy()

    output_profile.update(
        driver="GTiff",
        count=1,
        dtype="float32",
        nodata=NODATA,
        compress="deflate",
        predictor=3,
        tiled=True,
        blockxsize=256,
        blockysize=256
    )

    with rasterio.open(path, "w", **output_profile) as dst:
        dst.write(array.astype(np.float32), 1)

        dst.update_tags(
            project="TERRAFORM",
            product_description=description,
            units=units,
            study_region="Atacama Desert, Chile",
            source_dataset="Copernicus DEM GLO-30",
            processing_crs=TARGET_CRS,
            processing_resolution_m=str(TARGET_RESOLUTION),
            software="TERRAFORM Terrain Analysis Engine"
        )

    print(f"Saved: {path.name}")


def project_elevation(source_file):
    """Reproject geographic DEM into metric UTM coordinates."""

    print("\n[1/5] REPROJECTING ELEVATION DATA")

    with rasterio.open(source_file) as src:

        if src.crs is None:
            raise ValueError("Source raster has no CRS.")

        if src.crs.to_string() != SOURCE_CRS:
            raise ValueError(
                f"Expected {SOURCE_CRS}, received {src.crs}"
            )

        transform, width, height = calculate_default_transform(
            src.crs,
            TARGET_CRS,
            src.width,
            src.height,
            *src.bounds,
            resolution=TARGET_RESOLUTION
        )

        profile = src.profile.copy()

        profile.update(
            crs=TARGET_CRS,
            transform=transform,
            width=width,
            height=height,
            count=1,
            dtype="float32",
            nodata=NODATA,
            compress="deflate",
            predictor=3,
            tiled=True,
            blockxsize=256,
            blockysize=256
        )

        destination = np.full(
            (height, width),
            NODATA,
            dtype=np.float32
        )

        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=transform,
            dst_crs=TARGET_CRS,
            dst_nodata=NODATA,
            resampling=Resampling.bilinear
        )

    with rasterio.open(PROJECTED_FILE, "w", **profile) as dst:
        dst.write(destination, 1)

        dst.update_tags(
            project="TERRAFORM",
            source_dataset="Copernicus DEM GLO-30",
            processing="Reprojected using bilinear resampling",
            target_crs=TARGET_CRS,
            horizontal_units="metres",
            vertical_units="metres"
        )

    del destination
    gc.collect()

    print("Target CRS:", TARGET_CRS)
    print("Target resolution:", TARGET_RESOLUTION, "metres")
    print("Output dimensions:", width, "x", height)

    return width, height


def analyse_terrain():
    """Calculate slope, roughness and local relief."""

    print("\n[2/5] LOADING PROJECTED ELEVATION DATA")

    report = {
        "project": "TERRAFORM",
        "study_region": "Atacama Desert, Chile",
        "source_dataset": "Copernicus DEM GLO-30",
        "source_type": "Digital Surface Model",
        "target_crs": TARGET_CRS,
        "target_resolution_m": TARGET_RESOLUTION,
        "processing_window": "3x3 pixels",
        "products": {}
    }

    with rasterio.open(PROJECTED_FILE) as src:

        profile = src.profile.copy()

        dem = src.read(1).astype(np.float32)

        valid = (
            np.isfinite(dem)
            & (dem != NODATA)
        )

        if not np.any(valid):
            raise ValueError(
                "Projected raster contains no valid elevation data."
            )

        pixel_x = abs(src.transform.a)
        pixel_y = abs(src.transform.e)

        if pixel_x <= 0 or pixel_y <= 0:
            raise ValueError("Invalid projected pixel dimensions.")

        print("Valid elevation pixels:", int(valid.sum()))
        print("Pixel dimensions:", pixel_x, "x", pixel_y, "m")

        report["products"]["elevation"] = calculate_stats(
            dem, valid
        )

        print("\n[3/5] CALCULATING SLOPE")

        # Fill invalid pixels temporarily so gradients can be calculated.
        filled = dem.copy()
        filled[~valid] = 0.0

        gradient_y, gradient_x = np.gradient(
            filled,
            pixel_y,
            pixel_x
        )

        slope = np.degrees(
            np.arctan(
                np.hypot(gradient_x, gradient_y)
            )
        ).astype(np.float32)

        # Require a complete valid 3x3 neighbourhood.
        valid_neighbourhood = minimum_filter(
            valid.astype(np.uint8),
            size=3,
            mode="constant",
            cval=0
        ).astype(bool)

        slope[~valid_neighbourhood] = NODATA

        save_raster(
            SLOPE_FILE,
            slope,
            profile,
            "Terrain slope derived from projected elevation",
            "degrees"
        )

        report["products"]["slope"] = calculate_stats(
            slope,
            valid_neighbourhood
        )

        del gradient_x, gradient_y, slope
        gc.collect()

        print("\n[4/5] CALCULATING TERRAIN ROUGHNESS")

        # Roughness = local standard deviation of elevation.
        # Invalid pixels are excluded from neighbourhood statistics.

        valid_float = valid.astype(np.float32)

        count = uniform_filter(
            valid_float,
            size=3,
            mode="constant",
            cval=0.0
        )

        mean = uniform_filter(
            filled,
            size=3,
            mode="constant",
            cval=0.0
        )

        mean_square_source = np.square(filled)

        mean_square = uniform_filter(
            mean_square_source,
            size=3,
            mode="constant",
            cval=0.0
        )

        np.divide(
            mean,
            count,
            out=mean,
            where=count > 0
        )

        np.divide(
            mean_square,
            count,
            out=mean_square,
            where=count > 0
        )

        variance = mean_square - np.square(mean)

        np.maximum(variance, 0.0, out=variance)
        np.sqrt(variance, out=variance)

        roughness = variance.astype(np.float32)

        roughness[~valid_neighbourhood] = NODATA

        save_raster(
            ROUGHNESS_FILE,
            roughness,
            profile,
            "Local elevation standard deviation using a 3x3 window",
            "metres"
        )

        report["products"]["roughness"] = calculate_stats(
            roughness,
            valid_neighbourhood
        )

        del (
            valid_float,
            count,
            mean,
            mean_square,
            mean_square_source,
            variance,
            roughness,
            filled
        )

        gc.collect()

        print("\n[5/5] CALCULATING LOCAL RELIEF")

        # Local relief = maximum elevation - minimum elevation
        # within a 3x3 pixel neighbourhood.

        max_input = dem.copy()
        min_input = dem.copy()

        max_input[~valid] = -np.inf
        min_input[~valid] = np.inf

        local_max = maximum_filter(
            max_input,
            size=3,
            mode="constant",
            cval=-np.inf
        )

        local_min = minimum_filter(
            min_input,
            size=3,
            mode="constant",
            cval=np.inf
        )

        local_relief = local_max - local_min

        local_relief[~valid_neighbourhood] = NODATA

        save_raster(
            RELIEF_FILE,
            local_relief,
            profile,
            "Local elevation range using a 3x3 window",
            "metres"
        )

        report["products"]["local_relief"] = calculate_stats(
            local_relief,
            valid_neighbourhood
        )

        del (
            dem,
            valid,
            valid_neighbourhood,
            max_input,
            min_input,
            local_max,
            local_min,
            local_relief
        )

    gc.collect()

    return report


def main():

    print("=" * 60)
    print(" TERRAFORM — EARTH TERRAIN ANALYSIS ENGINE")
    print(" Atacama Desert | Copernicus DEM GLO-30")
    print("=" * 60)

    source_file = find_source_raster()

    print("\nSource:", source_file.name)

    project_elevation(source_file)

    report = analyse_terrain()

    with open(REPORT_FILE, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\n" + "=" * 60)
    print(" TERRAIN ANALYSIS SUCCESSFUL")
    print("=" * 60)

    print("\nGenerated products:")

    for path in [
        PROJECTED_FILE,
        SLOPE_FILE,
        ROUGHNESS_FILE,
        RELIEF_FILE,
        REPORT_FILE
    ]:
        print(" -", path)

    print("\nAnalysis statistics:")

    for name, stats in report["products"].items():
        print(f"\n{name.upper()}")

        for key, value in stats.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
