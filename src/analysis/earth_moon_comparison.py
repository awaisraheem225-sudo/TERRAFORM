
from pathlib import Path
import json
import math

import numpy as np
import rasterio

from pyproj import Transformer
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling

from scipy.ndimage import (
    uniform_filter,
    maximum_filter,
    minimum_filter,
)


ROOT = Path(__file__).resolve().parents[2]

MOON_DIR = ROOT / "data" / "processed" / "moon"
EARTH_DIR = ROOT / "data" / "processed" / "earth"

OUTPUT_DIR = ROOT / "data" / "processed" / "earth_moon"
REPORT_DIR = ROOT / "outputs" / "Global" / "Comparisons"

MOON_RADIUS = 1_737_400.0
MOON_PIXEL_SCALE = 1895.21
MOON_PIXEL_DEGREES = 0.0625

NEIGHBOURHOOD = 3
LATITUDE_BAND_HALF_WIDTH = 1.5
NODATA = -9999.0

REGIONS = {
    "Atacama": {
        "dem": EARTH_DIR / "earth_dem_projected_utm19s.tif",
        "output": "atacama",
    },
    "Death Valley": {
        "dem": EARTH_DIR / "death_valley"
        / "death_valley_dem_projected_utm11n.tif",
        "output": "death_valley",
    },
}

MOON_PRODUCTS = {
    "slope_degrees": MOON_DIR / "moon_global_slope_degrees.tif",
    "roughness_m": MOON_DIR / "moon_global_roughness_3x3_m.tif",
    "local_relief_m": MOON_DIR / "moon_global_local_relief_3x3_m.tif",
}


def statistics(array):

    values = np.asarray(array)
    values = values[np.isfinite(values)]

    if values.size == 0:
        raise ValueError("No valid values available.")

    return {
        "valid_pixels": int(values.size),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "standard_deviation": float(np.std(values)),
        "percentiles": {
            f"p{p:02d}": float(np.percentile(values, p))
            for p in (5, 25, 50, 75, 95)
        },
    }


def complete_neighbourhood(valid):

    count = uniform_filter(
        valid.astype(np.float64),
        size=NEIGHBOURHOOD,
        mode="constant",
        cval=0.0,
    )

    return count >= (1.0 - 1e-9)


def calculate_earth_metrics(elevation, dx, dy):

    valid = np.isfinite(elevation)
    complete = complete_neighbourhood(valid)

    slope = np.full(elevation.shape, np.nan, dtype=np.float32)
    roughness = np.full_like(slope, np.nan)
    relief = np.full_like(slope, np.nan)

    # Slope: central differences with separate x/y spacing.
    dz_dx = (
        elevation[1:-1, 2:] - elevation[1:-1, :-2]
    ) / (2.0 * dx)

    dz_dy = (
        elevation[2:, 1:-1] - elevation[:-2, 1:-1]
    ) / (2.0 * dy)

    gradient = np.sqrt(dz_dx ** 2 + dz_dy ** 2)

    slope_inner = slope[1:-1, 1:-1]
    mask_inner = complete[1:-1, 1:-1]

    slope_inner[mask_inner] = np.degrees(
        np.arctan(gradient[mask_inner])
    )

    # Roughness: local elevation standard deviation.
    filled = np.where(valid, elevation, 0.0)

    mean = uniform_filter(
        filled, size=NEIGHBOURHOOD, mode="constant"
    )

    mean_square = uniform_filter(
        filled ** 2, size=NEIGHBOURHOOD, mode="constant"
    )

    variance = np.maximum(mean_square - mean ** 2, 0.0)

    roughness = np.sqrt(variance).astype(np.float32)
    roughness[~complete] = np.nan

    # Local relief: local maximum minus local minimum.
    local_max = maximum_filter(
        np.where(valid, elevation, -np.inf),
        size=NEIGHBOURHOOD,
        mode="constant",
        cval=-np.inf,
    )

    local_min = minimum_filter(
        np.where(valid, elevation, np.inf),
        size=NEIGHBOURHOOD,
        mode="constant",
        cval=np.inf,
    )

    relief = (local_max - local_min).astype(np.float32)
    relief[~complete] = np.nan

    return {
        "slope_degrees": slope,
        "roughness_m": roughness,
        "local_relief_m": relief,
    }


def save_metric(path, array, profile, name):

    output = np.asarray(array, dtype=np.float32).copy()
    output[~np.isfinite(output)] = NODATA

    output_profile = profile.copy()
    output_profile.update(
        driver="GTiff",
        count=1,
        dtype="float32",
        nodata=NODATA,
        compress="deflate",
        predictor=3,
    )

    with rasterio.open(path, "w", **output_profile) as dst:
        dst.write(output, 1)
        dst.set_band_description(1, name)
        dst.update_tags(
            PROJECT="TERRAFORM",
            COMPARISON="Earth-Moon",
            METHOD="Latitude-adjusted LOLA-scale terrain analysis",
            NEIGHBOURHOOD="3x3",
        )

    print(f"Saved: {path}")


def process_earth_region(name, config):

    print("\n" + "=" * 65)
    print(f"PROCESSING EARTH REGION: {name.upper()}")
    print("=" * 65)

    source_path = config["dem"]

    if not source_path.exists():
        raise FileNotFoundError(source_path)

    with rasterio.open(source_path) as src:

        source = src.read(1, masked=True).astype(
            np.float32
        ).filled(np.nan)

        bounds = src.bounds
        crs = src.crs

        centre_x = (bounds.left + bounds.right) / 2
        centre_y = (bounds.bottom + bounds.top) / 2

        transformer = Transformer.from_crs(
            crs,
            "EPSG:4326",
            always_xy=True,
        )

        longitude, latitude = transformer.transform(
            centre_x, centre_y
        )

        profile = src.profile.copy()

    # Match the Moon's approximate physical pixel dimensions
    # at the corresponding Earth-region latitude.
    dx = MOON_PIXEL_SCALE * abs(
        math.cos(math.radians(latitude))
    )
    dy = MOON_PIXEL_SCALE

    if dx < 100:
        raise ValueError("Invalid target east-west resolution.")

    width = math.ceil(
        (bounds.right - bounds.left) / dx
    )

    height = math.ceil(
        (bounds.top - bounds.bottom) / dy
    )

    transform = from_origin(
        bounds.left,
        bounds.top,
        dx,
        dy,
    )

    aggregated = np.full(
        (height, width),
        np.nan,
        dtype=np.float32,
    )

    reproject(
        source=source,
        destination=aggregated,
        src_transform=profile["transform"],
        src_crs=crs,
        src_nodata=np.nan,
        dst_transform=transform,
        dst_crs=crs,
        dst_nodata=np.nan,
        resampling=Resampling.average,
        num_threads=2,
    )

    if not np.isfinite(aggregated).any():
        raise ValueError(f"{name}: aggregation produced no valid data.")

    metrics = calculate_earth_metrics(
        aggregated, dx, dy
    )

    region_output = OUTPUT_DIR / config["output"]
    region_output.mkdir(parents=True, exist_ok=True)

    output_profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": NODATA,
    }

    results = {}

    for metric, array in metrics.items():

        save_metric(
            region_output / f"{metric}.tif",
            array,
            output_profile,
            metric,
        )

        results[metric] = statistics(array)

    return {
        "region": name,
        "centre": {
            "latitude": latitude,
            "longitude": longitude,
        },
        "comparison_scale": {
            "east_west_pixel_size_m": dx,
            "north_south_pixel_size_m": dy,
            "neighbourhood": "3x3",
            "neighbourhood_east_west_m": dx * 3,
            "neighbourhood_north_south_m": dy * 3,
            "aggregation": "Average",
        },
        "earth_statistics": results,
        "lunar_reference": get_moon_band_statistics(latitude),
    }


def get_moon_band_statistics(latitude):

    lower = latitude - LATITUDE_BAND_HALF_WIDTH
    upper = latitude + LATITUDE_BAND_HALF_WIDTH

    # Pixel-centre latitude for each LOLA raster row.
    row_latitudes = (
        90.0
        - (np.arange(2880) + 0.5) * MOON_PIXEL_DEGREES
    )

    selected_rows = (
        (row_latitudes >= lower)
        & (row_latitudes <= upper)
    )

    result = {
        "latitude_band_degrees": [lower, upper],
        "metrics": {},
    }

    for metric, path in MOON_PRODUCTS.items():

        if not path.exists():
            raise FileNotFoundError(path)

        with rasterio.open(path) as src:

            values = src.read(
                1,
                window=((0, 2880), (0, 5760)),
                masked=True,
            ).filled(np.nan)

        band = values[selected_rows, :]

        result["metrics"][metric] = statistics(band)

    return result


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("TERRAFORM — EARTH / MOON COMMON-SCALE COMPARISON")
    print("=" * 65)

    results = {}

    for name, config in REGIONS.items():
        results[name] = process_earth_region(name, config)

    report = {
        "project": "TERRAFORM - Planetary Analog Intelligence",
        "analysis": "Earth-Moon Latitude-Matched Terrain Comparison",
        "version": "1.0",
        "reference_dataset": "NASA LRO LOLA LDEM_16 V3.1",
        "methodology": {
            "earth_source": "Copernicus GLO-30 DSM",
            "earth_aggregation": "Average resampling",
            "reference_equatorial_pixel_scale_m": MOON_PIXEL_SCALE,
            "earth_pixel_spacing": (
                "East-west spacing adjusted by cosine of region latitude; "
                "north-south spacing fixed at reference scale."
            ),
            "terrain_neighbourhood": "3x3",
            "moon_reference": (
                "Global lunar terrain metrics restricted to a "
                "1.5-degree latitude band on either side of each "
                "Earth region's centre latitude."
            ),
            "comparison_type": "Statistical distribution comparison",
        },
        "regions": results,
        "scientific_limitations": [
            "The comparison is statistical, not spatially matched.",
            "The lunar reference uses latitude bands, not a selected lunar site.",
            "The LOLA source grid has latitude-dependent physical spacing.",
            "Earth source elevation is a digital surface model, not a bare-earth terrain model.",
            "LOLA elevations are relative to a lunar reference sphere.",
            "Earth and Moon elevation datums are not directly comparable.",
            "Aggregation reduces Earth detail to the planetary reference scale.",
            "Terrain similarity alone does not establish landing or base suitability.",
            "The lunar PDS label extrema discrepancy remains unresolved.",
        ],
    }

    report_path = (
        REPORT_DIR / "earth_moon_common_scale_report.json"
    )

    report_path.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("EARTH-MOON COMPARISON COMPLETED")
    print("=" * 65)
    print(f"Report: {report_path}")

    for name, result in results.items():

        print(f"\n{name}")

        print("Latitude:", round(
            result["centre"]["latitude"], 4
        ))

        for metric, stats in result["earth_statistics"].items():

            moon_stats = result["lunar_reference"]["metrics"][metric]

            print(f"\n{metric}")
            print(
                "  Earth median:",
                round(stats["median"], 4),
            )
            print(
                "  Moon band median:",
                round(moon_stats["median"], 4),
            )


if __name__ == "__main__":
    main()
