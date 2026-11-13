
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
    minimum_filter
)


# ============================================================
# TERRAFORM — EARTH / MARS ANALOG COMPARISON ENGINE
# Earth study area: Atacama, Chile
# Mars study area: Jezero Crater
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

EARTH_SOURCE = (
    ROOT / "data" / "processed" / "earth"
    / "earth_dem_projected_utm19s.tif"
)

MARS_SOURCE = (
    ROOT / "data" / "processed" / "mars"
    / "jezero_dem_local_mars.tif"
)

EARTH_OUTPUT = ROOT / "data" / "processed" / "earth" / "comparison_scale"
MARS_OUTPUT = ROOT / "data" / "processed" / "mars" / "comparison_scale"

REPORT_DIR = ROOT / "outputs" / "reports"

REPORT_PATH = REPORT_DIR / "earth_mars_terrain_comparison.json"

TARGET_RESOLUTION = 463.0
NEIGHBOURHOOD_SIZE = 3
NODATA = -9999.0
HISTOGRAM_BINS = 20


# ============================================================
# DIRECTORY SETUP
# ============================================================

def create_directories():

    for directory in [
        EARTH_OUTPUT,
        MARS_OUTPUT,
        REPORT_DIR
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# READ MARS ELEVATION
# ============================================================

def read_mars_dem():

    print("\n[1/7] LOADING JEZERO ELEVATION MODEL")

    if not MARS_SOURCE.exists():
        raise FileNotFoundError(
            f"Mars DEM not found:\n{MARS_SOURCE}"
        )

    with rasterio.open(MARS_SOURCE) as source:

        elevation = source.read(
            1,
            masked=True
        ).astype(np.float64).filled(np.nan)

        transform = source.transform
        crs = source.crs
        resolution = source.res

        print(f"Dimensions: {source.width} x {source.height}")
        print(f"Resolution: {resolution}")
        print(f"CRS: {crs}")

        if not np.isclose(
            abs(resolution[0]),
            TARGET_RESOLUTION,
            atol=1.0
        ):
            raise ValueError(
                "Mars DEM does not have the expected "
                "approximately 463 m pixel resolution."
            )

    print(
        f"Valid Mars pixels: "
        f"{np.isfinite(elevation).sum():,}"
    )

    return elevation, transform, crs


# ============================================================
# AGGREGATE EARTH DEM TO 463 METRES
# ============================================================

def aggregate_earth_dem():

    print("\n[2/7] AGGREGATING ATACAMA DEM TO 463 METRES")

    if not EARTH_SOURCE.exists():
        raise FileNotFoundError(
            f"Earth DEM not found:\n{EARTH_SOURCE}"
        )

    with rasterio.open(EARTH_SOURCE) as source:

        bounds = source.bounds
        source_crs = source.crs
        source_nodata = source.nodata

        print(f"Original dimensions: {source.width} x {source.height}")
        print(f"Original resolution: {source.res}")
        print(f"Source CRS: {source_crs}")

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
            TARGET_RESOLUTION
        )

        aggregated = np.full(
            (height, width),
            np.nan,
            dtype=np.float32
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
            num_threads=2
        )

    valid = np.isfinite(aggregated)

    if not valid.any():
        raise ValueError(
            "Earth DEM aggregation produced no valid pixels."
        )

    print(f"Aggregated dimensions: {width} x {height}")
    print(f"Target resolution: {TARGET_RESOLUTION} m")
    print(f"Valid Earth pixels: {valid.sum():,}")

    return (
        aggregated.astype(np.float64),
        target_transform,
        source_crs
    )


# ============================================================
# COMMON TERRAIN NEIGHBOURHOOD
# ============================================================

def complete_neighbourhood_mask(valid):

    count = uniform_filter(
        valid.astype(np.float64),
        size=NEIGHBOURHOOD_SIZE,
        mode="constant",
        cval=0.0
    )

    return count >= (1.0 - 1e-9)


# ============================================================
# SLOPE
# ============================================================

def calculate_slope(elevation):

    valid = np.isfinite(elevation)

    complete = complete_neighbourhood_mask(valid)

    slope = np.full(
        elevation.shape,
        np.nan,
        dtype=np.float64
    )

    dz_dx = (
        elevation[1:-1, 2:]
        - elevation[1:-1, :-2]
    ) / (2.0 * TARGET_RESOLUTION)

    dz_dy = (
        elevation[2:, 1:-1]
        - elevation[:-2, 1:-1]
    ) / (2.0 * TARGET_RESOLUTION)

    gradient = np.sqrt(
        dz_dx ** 2 + dz_dy ** 2
    )

    interior_complete = complete[1:-1, 1:-1]

    interior = slope[1:-1, 1:-1]

    interior[interior_complete] = np.degrees(
        np.arctan(
            gradient[interior_complete]
        )
    )

    return slope


# ============================================================
# ROUGHNESS
# ============================================================

def calculate_roughness(elevation):

    valid = np.isfinite(elevation)

    complete = complete_neighbourhood_mask(valid)

    filled = np.where(
        valid,
        elevation,
        0.0
    )

    mean = uniform_filter(
        filled,
        size=NEIGHBOURHOOD_SIZE,
        mode="constant",
        cval=0.0
    )

    mean_square = uniform_filter(
        filled ** 2,
        size=NEIGHBOURHOOD_SIZE,
        mode="constant",
        cval=0.0
    )

    variance = np.maximum(
        mean_square - mean ** 2,
        0.0
    )

    roughness = np.sqrt(variance)

    roughness[~complete] = np.nan

    return roughness


# ============================================================
# LOCAL RELIEF
# ============================================================

def calculate_local_relief(elevation):

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

    local_max = maximum_filter(
        filled_max,
        size=NEIGHBOURHOOD_SIZE,
        mode="constant",
        cval=-np.inf
    )

    local_min = minimum_filter(
        filled_min,
        size=NEIGHBOURHOOD_SIZE,
        mode="constant",
        cval=np.inf
    )

    relief = local_max - local_min

    relief[~complete] = np.nan

    return relief


# ============================================================
# CALCULATE ALL TERRAIN METRICS
# ============================================================

def analyse_terrain(elevation, planet):

    print(f"\nCalculating terrain metrics: {planet}")

    return {
        "slope_degrees": calculate_slope(elevation),
        "roughness_m": calculate_roughness(elevation),
        "local_relief_m": calculate_local_relief(elevation)
    }


# ============================================================
# SAVE RASTER
# ============================================================

def save_raster(
    path,
    array,
    transform,
    crs,
    description,
    units,
    planet
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
        "crs": crs,
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
            PLANET=planet,
            RESOLUTION_METRES=str(TARGET_RESOLUTION),
            NEIGHBOURHOOD="3 x 3 pixels"
        )


def save_metric_products(
    metrics,
    output_dir,
    transform,
    crs,
    planet
):

    for name, array in metrics.items():

        filename = f"{planet.lower()}_{name}.tif"

        save_raster(
            output_dir / filename,
            array,
            transform,
            crs,
            name,
            "degrees" if name == "slope_degrees" else "metres",
            planet
        )

        print(f"Saved: {filename}")


# ============================================================
# DISTRIBUTION STATISTICS
# ============================================================

def distribution_statistics(array):

    values = array[np.isfinite(array)]

    if values.size == 0:
        raise ValueError(
            "No valid values for distribution statistics."
        )

    percentiles = np.percentile(
        values,
        [10, 25, 50, 75, 90]
    )

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
            "p90": float(percentiles[4])
        }
    }


# ============================================================
# COMPARATIVE HISTOGRAMS
# ============================================================

def create_histogram_comparison(earth, mars):

    results = {}

    for metric in earth:

        earth_values = earth[metric]
        mars_values = mars[metric]

        earth_values = earth_values[
            np.isfinite(earth_values)
        ]

        mars_values = mars_values[
            np.isfinite(mars_values)
        ]

        combined_min = float(
            min(
                earth_values.min(),
                mars_values.min()
            )
        )

        combined_max = float(
            max(
                earth_values.max(),
                mars_values.max()
            )
        )

        edges = np.linspace(
            combined_min,
            combined_max,
            HISTOGRAM_BINS + 1
        )

        earth_counts, _ = np.histogram(
            earth_values,
            bins=edges
        )

        mars_counts, _ = np.histogram(
            mars_values,
            bins=edges
        )

        results[metric] = {
            "bin_edges": edges.tolist(),
            "earth_counts": earth_counts.tolist(),
            "mars_counts": mars_counts.tolist()
        }

    return results


# ============================================================
# COMPARISON REPORT
# ============================================================

def build_report(
    earth_metrics,
    mars_metrics,
    earth_elevation,
    mars_elevation
):

    print("\n[7/7] BUILDING EARTH-MARS COMPARISON REPORT")

    report = {
        "project": "TERRAFORM - Planetary Analog Intelligence",
        "comparison": {
            "earth_study_area": "San Pedro de Atacama region, Chile",
            "mars_study_area": "Jezero Crater, Mars",
            "target_resolution_metres": TARGET_RESOLUTION,
            "neighbourhood": "3 x 3 pixels",
            "neighbourhood_approximate_width_metres": (
                NEIGHBOURHOOD_SIZE * TARGET_RESOLUTION
            )
        },
        "processing_methods": {
            "earth_resampling": "Average aggregation from 30 m DEM",
            "mars_resampling": (
                "Previously reprojected to local Mars metric grid "
                "using bilinear resampling"
            ),
            "slope": "Central difference elevation gradient",
            "roughness": "3 x 3 local elevation standard deviation",
            "local_relief": "3 x 3 local maximum minus minimum",
            "edge_handling": "Only complete valid neighbourhoods retained"
        },
        "elevation_note": (
            "Absolute elevation is reported separately and is not "
            "used as a direct similarity metric. Earth elevation "
            "and Mars elevation use different vertical reference surfaces."
        ),
        "elevation_context": {
            "earth": distribution_statistics(earth_elevation),
            "mars": distribution_statistics(mars_elevation)
        },
        "terrain_metrics": {},
        "histograms": create_histogram_comparison(
            earth_metrics,
            mars_metrics
        ),
        "scientific_limitations": [
            "The Earth Copernicus DEM is a digital surface model.",
            "MOLA MEGDR represents median observed topography and includes interpolated cells.",
            "The two regions have different geological histories and surface processes.",
            "Matching spatial resolution does not make the underlying sensors or data products identical.",
            "Terrain similarity alone does not establish geological or operational equivalence.",
            "These results are a regional screening comparison, not proof of a planetary analogue."
        ]
    }

    for metric in earth_metrics:

        earth_stats = distribution_statistics(
            earth_metrics[metric]
        )

        mars_stats = distribution_statistics(
            mars_metrics[metric]
        )

        earth_percentiles = earth_stats["percentiles"]
        mars_percentiles = mars_stats["percentiles"]

        percentile_differences = {
            key: (
                mars_percentiles[key]
                - earth_percentiles[key]
            )
            for key in earth_percentiles
        }

        report["terrain_metrics"][metric] = {
            "earth": earth_stats,
            "mars": mars_stats,
            "mars_minus_earth_percentile_difference": (
                percentile_differences
            )
        }

    REPORT_PATH.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8"
    )

    print(f"\nComparison report saved:\n{REPORT_PATH}")

    return report


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(report):

    print("\n")
    print("=" * 68)
    print("TERRAFORM — EARTH / MARS TERRAIN COMPARISON")
    print("=" * 68)

    for metric, comparison in report["terrain_metrics"].items():

        earth = comparison["earth"]
        mars = comparison["mars"]

        print(f"\n{metric.upper()}")
        print("-" * 55)

        print(
            f"Earth mean:   {earth['mean']:.4f}"
        )

        print(
            f"Mars mean:    {mars['mean']:.4f}"
        )

        print(
            f"Earth median: {earth['median']:.4f}"
        )

        print(
            f"Mars median:  {mars['median']:.4f}"
        )

        print(
            f"Earth pixels: {earth['valid_pixels']:,}"
        )

        print(
            f"Mars pixels:  {mars['valid_pixels']:,}"
        )

        print("\nPercentile differences (Mars - Earth):")

        for key, value in comparison[
            "mars_minus_earth_percentile_difference"
        ].items():

            print(f"  {key}: {value:.4f}")

    print("\n" + "=" * 68)
    print("EARTH-MARS COMPARISON COMPLETED")
    print("=" * 68)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 68)
    print("TERRAFORM — EARTH / MARS COMPARISON ENGINE")
    print("=" * 68)

    create_directories()

    # Load the corrected Jezero elevation raster.
    mars_elevation, mars_transform, mars_crs = (
        read_mars_dem()
    )

    # Aggregate Atacama elevation to the same working scale.
    (
        earth_elevation,
        earth_transform,
        earth_crs
    ) = aggregate_earth_dem()

    # Calculate comparable terrain metrics.
    earth_metrics = analyse_terrain(
        earth_elevation,
        "EARTH"
    )

    mars_metrics = analyse_terrain(
        mars_elevation,
        "MARS"
    )

    # Save Earth products.
    save_metric_products(
        earth_metrics,
        EARTH_OUTPUT,
        earth_transform,
        earth_crs,
        "EARTH"
    )

    # Save Mars products.
    save_metric_products(
        mars_metrics,
        MARS_OUTPUT,
        mars_transform,
        mars_crs,
        "MARS"
    )

    # Build and save the scientific comparison report.
    report = build_report(
        earth_metrics,
        mars_metrics,
        earth_elevation,
        mars_elevation
    )

    print_summary(report)


if __name__ == "__main__":
    main()
