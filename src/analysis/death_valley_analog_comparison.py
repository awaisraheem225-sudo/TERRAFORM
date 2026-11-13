from pathlib import Path
import json
import math

import numpy as np
import rasterio

from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling

from scipy.ndimage import (
    uniform_filter,
    maximum_filter,
    minimum_filter,
)


# ============================================================
# TERRAFORM — DEATH VALLEY / JEZERO COMPARISON
# Same methodology as the Atacama–Jezero comparison engine
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

EARTH_SOURCE = (
    ROOT / "data" / "processed" / "earth" / "death_valley"
    / "death_valley_dem_projected_utm11n.tif"
)

MARS_SOURCE = (
    ROOT / "data" / "processed" / "mars"
    / "jezero_dem_local_mars.tif"
)

EARTH_OUTPUT = (
    ROOT / "data" / "processed" / "earth" / "death_valley"
    / "comparison_scale"
)

REPORT_DIR = ROOT / "outputs" / "reports"

REPORT_PATH = (
    REPORT_DIR / "death_valley_jezero_terrain_comparison.json"
)

TARGET_RESOLUTION = 463.0
NEIGHBOURHOOD_SIZE = 3
NODATA = -9999.0
HISTOGRAM_BINS = 20


def create_directories():
    EARTH_OUTPUT.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def read_mars_dem():
    print("\n[1/6] LOADING JEZERO ELEVATION MODEL")

    if not MARS_SOURCE.exists():
        raise FileNotFoundError(f"Mars DEM not found: {MARS_SOURCE}")

    with rasterio.open(MARS_SOURCE) as source:
        elevation = (
            source.read(1, masked=True)
            .astype(np.float64)
            .filled(np.nan)
        )

        transform = source.transform
        crs = source.crs
        resolution = source.res

    if not np.isclose(abs(resolution[0]), TARGET_RESOLUTION, atol=1.0):
        raise ValueError(
            f"Jezero resolution is {resolution}, "
            f"not approximately {TARGET_RESOLUTION} m."
        )

    print(f"Mars dimensions: {elevation.shape[1]} x {elevation.shape[0]}")
    print(f"Mars resolution: {resolution}")
    print(f"Valid Mars pixels: {np.isfinite(elevation).sum():,}")

    return elevation


def aggregate_death_valley_dem():
    print("\n[2/6] AGGREGATING DEATH VALLEY TO 463 METRES")

    if not EARTH_SOURCE.exists():
        raise FileNotFoundError(
            f"Death Valley DEM not found: {EARTH_SOURCE}"
        )

    with rasterio.open(EARTH_SOURCE) as source:
        bounds = source.bounds
        source_crs = source.crs
        source_nodata = source.nodata

        width = math.ceil(
            (bounds.right - bounds.left) / TARGET_RESOLUTION
        )

        height = math.ceil(
            (bounds.top - bounds.bottom) / TARGET_RESOLUTION
        )

        target_transform = from_origin(
            bounds.left,
            bounds.top,
            TARGET_RESOLUTION,
            TARGET_RESOLUTION,
        )

        aggregated = np.full(
            (height, width),
            np.nan,
            dtype=np.float32,
        )

        reproject(
            source=rasterio.band(source, 1),
            destination=aggregated,
            src_transform=source.transform,
            src_crs=source_crs,
            src_nodata=source_nodata,
            dst_transform=target_transform,
            dst_crs=source_crs,
            dst_nodata=np.nan,
            resampling=Resampling.average,
            num_threads=2,
        )

    if not np.isfinite(aggregated).any():
        raise ValueError("Death Valley aggregation produced no valid pixels.")

    print(f"Aggregated dimensions: {width} x {height}")
    print(f"Target resolution: {TARGET_RESOLUTION} m")
    print(f"Valid Earth pixels: {np.isfinite(aggregated).sum():,}")

    return aggregated.astype(np.float64), target_transform, source_crs


def complete_neighbourhood_mask(valid):
    count = uniform_filter(
        valid.astype(np.float64),
        size=NEIGHBOURHOOD_SIZE,
        mode="constant",
        cval=0.0,
    )

    return count >= (1.0 - 1e-9)


def calculate_slope(elevation):
    valid = np.isfinite(elevation)
    complete = complete_neighbourhood_mask(valid)

    slope = np.full(elevation.shape, np.nan, dtype=np.float64)

    dz_dx = (
        elevation[1:-1, 2:] - elevation[1:-1, :-2]
    ) / (2.0 * TARGET_RESOLUTION)

    dz_dy = (
        elevation[2:, 1:-1] - elevation[:-2, 1:-1]
    ) / (2.0 * TARGET_RESOLUTION)

    gradient = np.sqrt(dz_dx ** 2 + dz_dy ** 2)

    interior_complete = complete[1:-1, 1:-1]
    interior = slope[1:-1, 1:-1]

    interior[interior_complete] = np.degrees(
        np.arctan(gradient[interior_complete])
    )

    return slope


def calculate_roughness(elevation):
    valid = np.isfinite(elevation)
    complete = complete_neighbourhood_mask(valid)

    filled = np.where(valid, elevation, 0.0)

    mean = uniform_filter(
        filled,
        size=NEIGHBOURHOOD_SIZE,
        mode="constant",
        cval=0.0,
    )

    mean_square = uniform_filter(
        filled ** 2,
        size=NEIGHBOURHOOD_SIZE,
        mode="constant",
        cval=0.0,
    )

    variance = np.maximum(mean_square - mean ** 2, 0.0)
    roughness = np.sqrt(variance)
    roughness[~complete] = np.nan

    return roughness


def calculate_local_relief(elevation):
    valid = np.isfinite(elevation)
    complete = complete_neighbourhood_mask(valid)

    filled_max = np.where(valid, elevation, -np.inf)
    filled_min = np.where(valid, elevation, np.inf)

    local_max = maximum_filter(
        filled_max,
        size=NEIGHBOURHOOD_SIZE,
        mode="constant",
        cval=-np.inf,
    )

    local_min = minimum_filter(
        filled_min,
        size=NEIGHBOURHOOD_SIZE,
        mode="constant",
        cval=np.inf,
    )

    relief = local_max - local_min
    relief[~complete] = np.nan

    return relief


def analyse_terrain(elevation):
    return {
        "slope_degrees": calculate_slope(elevation),
        "roughness_m": calculate_roughness(elevation),
        "local_relief_m": calculate_local_relief(elevation),
    }


def distribution_statistics(array):
    values = array[np.isfinite(array)]

    if values.size == 0:
        raise ValueError("No valid values for distribution statistics.")

    percentiles = np.percentile(values, [10, 25, 50, 75, 90])

    return {
        "valid_pixels": int(values.size),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "standard_deviation": float(np.std(values)),
        "percentiles": {
            "p10": float(percentiles[0]),
            "p25": float(percentiles[1]),
            "p50": float(percentiles[2]),
            "p75": float(percentiles[3]),
            "p90": float(percentiles[4]),
        },
    }


def save_raster(path, array, transform, crs, description):
    output = np.asarray(array, dtype=np.float32).copy()
    output[~np.isfinite(output)] = NODATA

    profile = {
        "driver": "GTiff",
        "height": output.shape[0],
        "width": output.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": NODATA,
        "compress": "deflate",
        "predictor": 3,
        "BIGTIFF": "IF_SAFER",
    }

    with rasterio.open(path, "w", **profile) as destination:
        destination.write(output, 1)
        destination.set_band_description(1, description)
        destination.update_tags(
            PROJECT="TERRAFORM",
            STUDY_AREA="Death Valley, California, USA",
            RESOLUTION_METRES=str(TARGET_RESOLUTION),
            NEIGHBOURHOOD="3 x 3 pixels",
        )


def save_metric_products(metrics, transform, crs):
    print("\n[4/6] SAVING DEATH VALLEY TERRAIN PRODUCTS")

    for name, array in metrics.items():
        filename = f"death_valley_{name}.tif"

        save_raster(
            EARTH_OUTPUT / filename,
            array,
            transform,
            crs,
            name,
        )

        print(f"Saved: {filename}")


def create_histograms(earth_metrics, mars_metrics):
    results = {}

    for metric in earth_metrics:
        earth_values = earth_metrics[metric]
        mars_values = mars_metrics[metric]

        earth_values = earth_values[np.isfinite(earth_values)]
        mars_values = mars_values[np.isfinite(mars_values)]

        combined_min = float(
            min(earth_values.min(), mars_values.min())
        )

        combined_max = float(
            max(earth_values.max(), mars_values.max())
        )

        edges = np.linspace(
            combined_min,
            combined_max,
            HISTOGRAM_BINS + 1,
        )

        earth_counts, _ = np.histogram(earth_values, bins=edges)
        mars_counts, _ = np.histogram(mars_values, bins=edges)

        results[metric] = {
            "bin_edges": edges.tolist(),
            "earth_counts": earth_counts.tolist(),
            "mars_counts": mars_counts.tolist(),
        }

    return results


def build_report(earth_metrics, mars_metrics, earth_elevation, mars_elevation):
    print("\n[5/6] BUILDING DEATH VALLEY–JEZERO REPORT")

    report = {
        "project": "TERRAFORM - Planetary Analog Intelligence",
        "comparison": {
            "earth_study_area": "Death Valley, California, USA",
            "mars_study_area": "Jezero Crater, Mars",
            "target_resolution_metres": TARGET_RESOLUTION,
            "neighbourhood": "3 x 3 pixels",
            "neighbourhood_approximate_width_metres": (
                NEIGHBOURHOOD_SIZE * TARGET_RESOLUTION
            ),
        },
        "processing_methods": {
            "earth_resampling": "Average aggregation from 30 m Copernicus DEM GLO-30",
            "mars_resampling": (
                "Previously reprojected to local Mars metric grid "
                "using bilinear resampling"
            ),
            "slope": "Central difference elevation gradient",
            "roughness": "3 x 3 local elevation standard deviation",
            "local_relief": "3 x 3 local maximum minus minimum",
            "edge_handling": "Only complete valid neighbourhoods retained",
        },
        "elevation_note": (
            "Absolute elevation is reported separately and is not used "
            "as a direct similarity metric. Earth and Mars elevations "
            "use different vertical reference surfaces."
        ),
        "elevation_context": {
            "earth": distribution_statistics(earth_elevation),
            "mars": distribution_statistics(mars_elevation),
        },
        "terrain_metrics": {},
        "histograms": create_histograms(earth_metrics, mars_metrics),
        "scientific_limitations": [
            "The Earth Copernicus DEM GLO-30 is a digital surface model.",
            "MOLA MEGDR represents median observed topography and includes interpolated cells.",
            "The two regions have different geological histories and surface processes.",
            "Matching spatial resolution does not make the underlying sensors or data products identical.",
            "Terrain similarity alone does not establish geological or operational equivalence.",
            "These results are a regional screening comparison, not proof of a planetary analogue.",
        ],
    }

    for metric in earth_metrics:
        earth_stats = distribution_statistics(earth_metrics[metric])
        mars_stats = distribution_statistics(mars_metrics[metric])

        earth_percentiles = earth_stats["percentiles"]
        mars_percentiles = mars_stats["percentiles"]

        differences = {
            key: mars_percentiles[key] - earth_percentiles[key]
            for key in earth_percentiles
        }

        report["terrain_metrics"][metric] = {
            "earth": earth_stats,
            "mars": mars_stats,
            "mars_minus_earth_percentile_difference": differences,
        }

    REPORT_PATH.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print(f"Report saved: {REPORT_PATH}")
    return report


def print_summary(report):
    print("\n" + "=" * 65)
    print("DEATH VALLEY, EARTH  |  JEZERO CRATER, MARS")
    print("=" * 65)

    for metric, comparison in report["terrain_metrics"].items():
        earth = comparison["earth"]
        mars = comparison["mars"]

        print(f"\n{metric.upper()}")
        print(f"Earth mean:   {earth['mean']:.4f}")
        print(f"Mars mean:    {mars['mean']:.4f}")
        print(f"Earth median: {earth['median']:.4f}")
        print(f"Mars median:  {mars['median']:.4f}")
        print(f"Earth pixels: {earth['valid_pixels']:,}")
        print(f"Mars pixels:  {mars['valid_pixels']:,}")

    print("\nComparison processing completed.")


def main():
    print("=" * 65)
    print("TERRAFORM — DEATH VALLEY / JEZERO COMPARISON ENGINE")
    print("=" * 65)

    create_directories()

    mars_elevation = read_mars_dem()

    earth_elevation, earth_transform, earth_crs = (
        aggregate_death_valley_dem()
    )

    print("\n[3/6] CALCULATING MATCHED TERRAIN METRICS")

    earth_metrics = analyse_terrain(earth_elevation)
    mars_metrics = analyse_terrain(mars_elevation)

    save_metric_products(
        earth_metrics,
        earth_transform,
        earth_crs,
    )

    report = build_report(
        earth_metrics,
        mars_metrics,
        earth_elevation,
        mars_elevation,
    )

    print_summary(report)


if __name__ == "__main__":
    main()