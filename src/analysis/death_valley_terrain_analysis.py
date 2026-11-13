
from pathlib import Path
import json
import gc

import numpy as np
import rasterio

from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from scipy.ndimage import (
    minimum_filter,
    maximum_filter,
    uniform_filter
)


# ============================================================
# TERRAFORM - DEATH VALLEY TERRAIN ANALYSIS
# Source: Copernicus DEM GLO-30 Public
# Region: Death Valley, California, USA
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = ROOT / "data" / "raw" / "earth" / "death_valley"

OUTPUT_DIR = (
    ROOT / "data" / "processed" / "earth" / "death_valley"
)

REPORT_DIR = ROOT / "outputs" / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# Geographic study area: western/eastern longitude,
# southern/northern latitude.
STUDY_BOUNDS = (-117.3, 35.8, -116.7, 36.4)

SOURCE_CRS = "EPSG:4326"
TARGET_CRS = "EPSG:32611"

TARGET_RESOLUTION = 30.0
NODATA = -9999.0

MOSAIC_FILE = OUTPUT_DIR / "death_valley_dem_mosaic.tif"
PROJECTED_FILE = OUTPUT_DIR / "death_valley_dem_projected_utm11n.tif"
SLOPE_FILE = OUTPUT_DIR / "death_valley_slope_deg.tif"
ROUGHNESS_FILE = OUTPUT_DIR / "death_valley_roughness_std_3x3_m.tif"
RELIEF_FILE = OUTPUT_DIR / "death_valley_local_relief_3x3_m.tif"

REPORT_FILE = REPORT_DIR / "death_valley_terrain_analysis.json"


def calculate_stats(array, valid_mask):
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
            study_region="Death Valley, California, USA",
            product_description=description,
            units=units,
            source_dataset="Copernicus DEM GLO-30",
            source_type="Digital Surface Model",
            processing_crs=TARGET_CRS,
            processing_resolution_m=str(TARGET_RESOLUTION),
            software="TERRAFORM Terrain Analysis Engine"
        )

    print("Saved:", path.name)


def create_mosaic():
    print("\n[1/5] MOSAICKING DEATH VALLEY DEM TILES")

    files = sorted(INPUT_DIR.glob("*.tif"))

    if len(files) != 4:
        raise FileNotFoundError(
            f"Expected 4 DEM tiles in {INPUT_DIR}, found {len(files)}"
        )

    with rasterio.open(files[0]) as first:
        source_crs = first.crs
        source_resolution = first.res
        source_nodata = first.nodata

        if source_crs is None or source_crs.to_string() != SOURCE_CRS:
            raise ValueError(f"Unexpected source CRS: {source_crs}")

        sources = []

        try:
            for path in files:
                dataset = rasterio.open(path)

                if dataset.crs != source_crs:
                    dataset.close()
                    raise ValueError(f"CRS mismatch in {path.name}")

                if dataset.count < 1:
                    dataset.close()
                    raise ValueError(f"No raster band in {path.name}")

                sources.append(dataset)

            mosaic, transform = merge(
                sources,
                bounds=STUDY_BOUNDS,
                res=source_resolution,
                nodata=source_nodata,
                dtype="float32"
            )

            profile = first.profile.copy()

        finally:
            for dataset in sources:
                dataset.close()

    profile.update(
        driver="GTiff",
        height=mosaic.shape[1],
        width=mosaic.shape[2],
        transform=transform,
        count=1,
        dtype="float32",
        nodata=source_nodata,
        compress="deflate",
        predictor=3
    )

    with rasterio.open(MOSAIC_FILE, "w", **profile) as dst:
        dst.write(mosaic[0], 1)
        dst.update_tags(
            project="TERRAFORM",
            study_region="Death Valley, California, USA",
            source_dataset="Copernicus DEM GLO-30",
            geographic_bounds=str(STUDY_BOUNDS)
        )

    print("Input tiles:", len(files))
    print("Mosaic dimensions:", mosaic.shape[2], "x", mosaic.shape[1])
    print("Mosaic saved:", MOSAIC_FILE)

    del mosaic
    gc.collect()


def project_elevation():
    print("\n[2/5] REPROJECTING ELEVATION DATA")

    with rasterio.open(MOSAIC_FILE) as src:

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
            study_region="Death Valley, California, USA",
            source_dataset="Copernicus DEM GLO-30",
            processing="Bilinear reprojection",
            target_crs=TARGET_CRS,
            horizontal_units="metres",
            vertical_units="metres"
        )

    print("Target CRS:", TARGET_CRS)
    print("Resolution:", TARGET_RESOLUTION, "metres")
    print("Dimensions:", width, "x", height)

    del destination
    gc.collect()


def analyse_terrain():
    print("\n[3/5] ANALYSING TERRAIN")

    report = {
        "project": "TERRAFORM",
        "study_region": "Death Valley, California, USA",
        "source_dataset": "Copernicus DEM GLO-30",
        "source_type": "Digital Surface Model",
        "target_crs": TARGET_CRS,
        "target_resolution_m": TARGET_RESOLUTION,
        "study_bounds": {
            "west": STUDY_BOUNDS[0],
            "south": STUDY_BOUNDS[1],
            "east": STUDY_BOUNDS[2],
            "north": STUDY_BOUNDS[3]
        },
        "processing_window": "3x3 pixels",
        "products": {}
    }

    with rasterio.open(PROJECTED_FILE) as src:

        profile = src.profile.copy()
        dem = src.read(1).astype(np.float32)

        valid = np.isfinite(dem) & (dem != NODATA)

        if not np.any(valid):
            raise ValueError("No valid elevation pixels found.")

        pixel_x = abs(src.transform.a)
        pixel_y = abs(src.transform.e)

        if pixel_x <= 0 or pixel_y <= 0:
            raise ValueError("Invalid pixel dimensions.")

        print("Valid elevation pixels:", int(valid.sum()))

        report["products"]["elevation"] = calculate_stats(dem, valid)

        # Require complete 3x3 neighbourhoods for terrain metrics.
        valid_neighbourhood = minimum_filter(
            valid.astype(np.uint8),
            size=3,
            mode="constant",
            cval=0
        ).astype(bool)

        filled = dem.copy()
        filled[~valid] = 0.0

        print("\n[4/5] CALCULATING SLOPE")

        gradient_y, gradient_x = np.gradient(
            filled,
            pixel_y,
            pixel_x
        )

        slope = np.degrees(
            np.arctan(np.hypot(gradient_x, gradient_y))
        ).astype(np.float32)

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

        print("\nCALCULATING TERRAIN ROUGHNESS")

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

        mean_square = uniform_filter(
            np.square(filled),
            size=3,
            mode="constant",
            cval=0.0
        )

        np.divide(mean, count, out=mean, where=count > 0)
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
            variance,
            roughness,
            filled
        )
        gc.collect()

        print("\nCALCULATING LOCAL RELIEF")

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
    print(" TERRAFORM - DEATH VALLEY TERRAIN ANALYSIS")
    print(" Copernicus DEM GLO-30")
    print("=" * 60)

    create_mosaic()
    project_elevation()

    report = analyse_terrain()

    with open(REPORT_FILE, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\n" + "=" * 60)
    print(" DEATH VALLEY ANALYSIS SUCCESSFUL")
    print("=" * 60)

    print("\nGenerated products:")

    for path in [
        MOSAIC_FILE,
        PROJECTED_FILE,
        SLOPE_FILE,
        ROUGHNESS_FILE,
        RELIEF_FILE,
        REPORT_FILE
    ]:
        print(" -", path)

    print("\nTERRAIN STATISTICS")

    for name, stats in report["products"].items():
        print(f"\n{name.upper()}")

        for key, value in stats.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
