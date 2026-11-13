
from pathlib import Path
import json
import csv

import numpy as np
import rasterio
import matplotlib.pyplot as plt

from scipy import ndimage


# ============================================================
# TERRAFORM — TERRAIN SCREENING SENSITIVITY ANALYSIS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

EARTH_DIR = (
    ROOT / "data" / "processed" / "earth" / "comparison_scale"
)

MARS_DIR = (
    ROOT / "data" / "processed" / "mars" / "comparison_scale"
)

OUTPUT_DIR = (
    ROOT / "outputs" / "maps" / "sensitivity_analysis"
)

REPORT_DIR = ROOT / "outputs" / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


METRICS = {
    "slope": {
        "earth": "earth_slope_degrees.tif",
        "mars": "mars_slope_degrees.tif",
        "label": "Slope",
        "unit": "degrees",
    },
    "roughness": {
        "earth": "earth_roughness_m.tif",
        "mars": "mars_roughness_m.tif",
        "label": "Roughness",
        "unit": "metres",
    },
    "local_relief": {
        "earth": "earth_local_relief_m.tif",
        "mars": "mars_local_relief_m.tif",
        "label": "Local relief",
        "unit": "metres",
    },
}


SCENARIOS = {
    "broad": {
        "lower": 10,
        "upper": 90,
        "label": "Broad screening (P10-P90)",
    },
    "moderate": {
        "lower": 25,
        "upper": 75,
        "label": "Moderate screening (P25-P75)",
    },
    "strict": {
        "lower": 40,
        "upper": 60,
        "label": "Strict screening (P40-P60)",
    },
}


def load_raster(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Required raster not found: {path}"
        )

    with rasterio.open(path) as src:

        values = src.read(1).astype(np.float64)
        valid = np.isfinite(values)

        if src.nodata is not None:
            valid &= values != src.nodata

        return {
            "values": values,
            "valid": valid,
            "profile": src.profile.copy(),
            "transform": src.transform,
            "crs": src.crs,
        }


def load_datasets():

    earth = {}
    mars = {}

    for name, settings in METRICS.items():

        earth[name] = load_raster(
            EARTH_DIR / settings["earth"]
        )

        mars[name] = load_raster(
            MARS_DIR / settings["mars"]
        )

    first = earth["slope"]

    for name in METRICS:

        current = earth[name]

        if current["values"].shape != first["values"].shape:
            raise ValueError(
                "Earth metric raster dimensions do not match."
            )

        if current["transform"] != first["transform"]:
            raise ValueError(
                "Earth metric raster transforms do not match."
            )

        if current["crs"] != first["crs"]:
            raise ValueError(
                "Earth metric raster coordinate systems do not match."
            )

    common_valid = np.logical_and.reduce([
        earth[name]["valid"] for name in METRICS
    ])

    if not all(
        np.array_equal(earth[name]["valid"], common_valid)
        for name in METRICS
    ):
        raise ValueError(
            "Earth metrics must share identical valid-pixel masks."
        )

    return earth, mars, common_valid


def get_connectivity_structure(connectivity):

    if connectivity == 4:
        return np.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ])

    return np.ones((3, 3), dtype=np.uint8)


def analyse_clusters(mask, connectivity, pixel_area_km2):

    structure = get_connectivity_structure(connectivity)

    labels, cluster_count = ndimage.label(
        mask,
        structure=structure,
    )

    if cluster_count == 0:

        return {
            "cluster_count": 0,
            "largest_cluster_pixels": 0,
            "largest_cluster_area_km2": 0.0,
            "largest_cluster_share_percent": 0.0,
            "largest_10_share_percent": 0.0,
        }

    sizes = np.bincount(labels.ravel())
    sizes[0] = 0

    cluster_sizes = np.sort(sizes[1:])[::-1]

    matching_pixels = int(mask.sum())

    largest = int(cluster_sizes[0])
    largest_10 = int(cluster_sizes[:10].sum())

    return {
        "cluster_count": int(cluster_count),
        "largest_cluster_pixels": largest,
        "largest_cluster_area_km2": float(
            largest * pixel_area_km2
        ),
        "largest_cluster_share_percent": float(
            largest / matching_pixels * 100
        ),
        "largest_10_share_percent": float(
            largest_10 / matching_pixels * 100
        ),
    }


def save_mask(mask, valid, profile, output_path):

    output = np.full(
        mask.shape,
        255,
        dtype=np.uint8,
    )

    output[valid] = mask[valid].astype(np.uint8)

    output_profile = profile.copy()

    output_profile.update(
        dtype="uint8",
        count=1,
        nodata=255,
        compress="deflate",
    )

    with rasterio.open(
        output_path,
        "w",
        **output_profile,
    ) as dst:

        dst.write(output, 1)


def create_scenario_map(
    masks,
    common_valid,
    transform,
    scenario_name,
    scenario_label,
):

    fig, ax = plt.subplots(figsize=(10, 8))

    mask = masks[scenario_name]

    display = np.ma.masked_where(
        ~common_valid,
        mask.astype(np.uint8),
    )

    from matplotlib.colors import ListedColormap

    cmap = ListedColormap([
        "#d95f59",
        "#35a66f",
    ])

    height, width = mask.shape

    x_min = transform.c
    y_max = transform.f

    x_max = x_min + width * transform.a
    y_min = y_max + height * transform.e

    ax.imshow(
        display,
        extent=[
            x_min / 1000,
            x_max / 1000,
            y_min / 1000,
            y_max / 1000,
        ],
        origin="upper",
        cmap=cmap,
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )

    ax.set_title(
        f"TERRAFORM | {scenario_label}",
        fontsize=14,
        pad=14,
    )

    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    ax.grid(alpha=0.15)

    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(
                facecolor="#35a66f",
                label="Within all three reference ranges",
            ),
            Patch(
                facecolor="#d95f59",
                label="Outside at least one reference range",
            ),
        ],
        loc="best",
    )

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR / f"{scenario_name}_screening_map.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved map: {output_path.name}")


def create_comparison_chart(results):

    names = list(SCENARIOS.keys())

    percentages = [
        results[name]["matching_percentage"]
        for name in names
    ]

    areas = [
        results[name]["matching_area_km2"]
        for name in names
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5),
    )

    labels = [
        SCENARIOS[name]["label"]
        for name in names
    ]

    axes[0].bar(labels, percentages)

    axes[0].set_title(
        "Atacama Pixels Passing All Three Conditions"
    )

    axes[0].set_ylabel("Matching pixels (%)")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].bar(labels, areas)

    axes[1].set_title(
        "Total Terrain-Matching Area"
    )

    axes[1].set_ylabel("Area (km²)")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].grid(axis="y", alpha=0.2)

    fig.suptitle(
        "TERRAFORM | Screening Sensitivity Analysis",
        fontsize=15,
    )

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR / "sensitivity_comparison.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved chart: {output_path.name}")


def main():

    print("\nTERRAFORM SENSITIVITY ANALYSIS")
    print("=" * 65)

    earth, mars, common_valid = load_datasets()

    valid_pixels = int(common_valid.sum())

    transform = earth["slope"]["transform"]
    profile = earth["slope"]["profile"]

    pixel_area_km2 = abs(
        transform.a * transform.e
        - transform.b * transform.d
    ) / 1_000_000

    print(f"Common valid Earth pixels: {valid_pixels:,}")
    print(f"Pixel area: {pixel_area_km2:.6f} km²")

    # Calculate reference thresholds once per metric.
    reference_thresholds = {}

    for name in METRICS:

        mars_values = mars[name]["values"][
            mars[name]["valid"]
        ]

        reference_thresholds[name] = {
            "values": mars_values,
            "valid_pixels": int(mars_values.size),
        }

    all_results = {}

    for scenario_name, scenario in SCENARIOS.items():

        print("\n" + "-" * 65)
        print(scenario["label"].upper())
        print("-" * 65)

        lower_percentile = scenario["lower"]
        upper_percentile = scenario["upper"]

        individual_masks = {}
        thresholds = {}

        for name, settings in METRICS.items():

            mars_values = reference_thresholds[name]["values"]

            lower = float(np.percentile(
                mars_values,
                lower_percentile,
            ))

            upper = float(np.percentile(
                mars_values,
                upper_percentile,
            ))

            thresholds[name] = {
                "lower": lower,
                "upper": upper,
                "unit": settings["unit"],
            }

            earth_values = earth[name]["values"]

            mask = (
                common_valid
                & (earth_values >= lower)
                & (earth_values <= upper)
            )

            individual_masks[name] = mask

            print(
                f"{settings['label']}: "
                f"{lower:.4f} - {upper:.4f} "
                f"{settings['unit']}"
            )

        combined_mask = np.logical_and.reduce([
            individual_masks[name]
            for name in METRICS
        ])

        matching_pixels = int(combined_mask.sum())

        percentage = (
            matching_pixels / valid_pixels * 100
            if valid_pixels else 0
        )

        matching_area = (
            matching_pixels * pixel_area_km2
        )

        print(
            f"\nMatching pixels: "
            f"{matching_pixels:,}/{valid_pixels:,}"
        )

        print(f"Matching percentage: {percentage:.2f}%")
        print(f"Matching area: {matching_area:.2f} km²")

        scenario_result = {
            "label": scenario["label"],
            "reference_percentiles": {
                "lower": lower_percentile,
                "upper": upper_percentile,
            },
            "reference_thresholds": thresholds,
            "matching_pixels": matching_pixels,
            "valid_pixels": valid_pixels,
            "matching_percentage": percentage,
            "matching_area_km2": matching_area,
            "connectivity": {},
        }

        # Analyse both connectivity definitions.
        for connectivity in [4, 8]:

            cluster_stats = analyse_clusters(
                combined_mask,
                connectivity,
                pixel_area_km2,
            )

            scenario_result["connectivity"][
                f"{connectivity}_neighbour"
            ] = cluster_stats

            print(
                f"\n{connectivity}-neighbour clusters: "
                f"{cluster_stats['cluster_count']:,}"
            )

            print(
                "Largest cluster: "
                f"{cluster_stats['largest_cluster_area_km2']:.2f} km²"
            )

            print(
                "Largest cluster share: "
                f"{cluster_stats['largest_cluster_share_percent']:.2f}%"
            )

            print(
                "Largest 10 share: "
                f"{cluster_stats['largest_10_share_percent']:.2f}%"
            )

        # Save the three-metric mask for this scenario.
        mask_path = (
            OUTPUT_DIR
            / f"{scenario_name}_three_metric_mask.tif"
        )

        save_mask(
            combined_mask,
            common_valid,
            profile,
            mask_path,
        )

        create_scenario_map(
            {scenario_name: combined_mask},
            common_valid,
            transform,
            scenario_name,
            scenario["label"],
        )

        all_results[scenario_name] = scenario_result

    # Save comparison chart.
    create_comparison_chart(all_results)

    # Save machine-readable report.
    report = {
        "project": "TERRAFORM",
        "analysis": "Terrain screening sensitivity analysis",
        "earth_study_region": "Atacama pilot area, Chile",
        "mars_reference_region": "Jezero Crater study area",
        "working_resolution_m": 463,
        "earth_valid_pixels": valid_pixels,
        "pixel_area_km2": pixel_area_km2,
        "scenarios": all_results,
        "methodological_limitations": [
            "Percentile intervals are derived independently for each terrain metric.",
            "The three-metric intersection is not a probability of planetary analogy.",
            "Terrain variables may be correlated and are not independent evidence.",
            "Results depend on selected reference region and percentile thresholds.",
            "Cluster sizes depend on connectivity and raster resolution.",
            "Terrain screening does not establish geological or operational equivalence.",
            "Earth and Mars datasets have different measurement characteristics.",
        ],
    }

    report_path = (
        REPORT_DIR / "sensitivity_analysis_report.json"
    )

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    # Save a compact CSV for future dashboard integration.
    csv_path = OUTPUT_DIR / "sensitivity_summary.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "scenario",
            "matching_pixels",
            "matching_percentage",
            "matching_area_km2",
            "clusters_4_neighbour",
            "largest_cluster_4_km2",
            "clusters_8_neighbour",
            "largest_cluster_8_km2",
        ])

        for name, result in all_results.items():

            c4 = result["connectivity"]["4_neighbour"]
            c8 = result["connectivity"]["8_neighbour"]

            writer.writerow([
                name,
                result["matching_pixels"],
                round(result["matching_percentage"], 4),
                round(result["matching_area_km2"], 4),
                c4["cluster_count"],
                round(c4["largest_cluster_area_km2"], 4),
                c8["cluster_count"],
                round(c8["largest_cluster_area_km2"], 4),
            ])

    print("\n" + "=" * 65)
    print("TERRAFORM SENSITIVITY ANALYSIS COMPLETED")
    print(f"Report: {report_path}")
    print(f"CSV: {csv_path}")
    print(f"Maps: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
