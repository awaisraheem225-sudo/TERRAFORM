
from pathlib import Path
import json

import numpy as np
import rasterio
import matplotlib.pyplot as plt


# ============================================================
# TERRAFORM — EARTH / MARS TERRAIN VISUALISATION ENGINE
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

EARTH_DIR = ROOT / "data" / "processed" / "earth" / "comparison_scale"
MARS_DIR = ROOT / "data" / "processed" / "mars" / "comparison_scale"

OUTPUT_DIR = ROOT / "outputs" / "maps" / "terrain_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = {
    "slope": {
        "earth": "earth_slope_degrees.tif",
        "mars": "mars_slope_degrees.tif",
        "title": "Terrain Slope Distribution",
        "xlabel": "Slope (degrees)",
        "filename": "slope_distribution.png",
    },
    "roughness": {
        "earth": "earth_roughness_m.tif",
        "mars": "mars_roughness_m.tif",
        "title": "Terrain Surface Roughness",
        "xlabel": "Local elevation standard deviation (m)",
        "filename": "roughness_distribution.png",
    },
    "local_relief": {
        "earth": "earth_local_relief_m.tif",
        "mars": "mars_local_relief_m.tif",
        "title": "Local Terrain Relief",
        "xlabel": "Local elevation range (m)",
        "filename": "local_relief_distribution.png",
    },
}


def load_raster(path):
    """Load valid raster values, excluding NoData and non-finite cells."""

    if not path.exists():
        raise FileNotFoundError(f"Required raster not found: {path}")

    with rasterio.open(path) as dataset:
        values = dataset.read(1).astype(np.float64)
        nodata = dataset.nodata

    valid = np.isfinite(values)

    if nodata is not None:
        valid &= values != nodata

    result = values[valid]

    if result.size == 0:
        raise ValueError(f"No valid data found in: {path}")

    return result


def load_metric(metric_name, settings):
    earth_path = EARTH_DIR / settings["earth"]
    mars_path = MARS_DIR / settings["mars"]

    earth = load_raster(earth_path)
    mars = load_raster(mars_path)

    return earth, mars


def calculate_statistics(values):
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def plot_metric(metric_name, settings, earth, mars):
    """Create an individual Earth–Mars distribution chart."""

    combined = np.concatenate([earth, mars])

    # Shared bin edges make the two distributions comparable.
    bins = np.linspace(
        float(np.min(combined)),
        float(np.max(combined)),
        31
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        earth,
        bins=bins,
        density=True,
        alpha=0.55,
        label="Atacama, Earth",
        edgecolor="black",
        linewidth=0.35,
    )

    ax.hist(
        mars,
        bins=bins,
        density=True,
        alpha=0.55,
        label="Jezero, Mars",
        edgecolor="black",
        linewidth=0.35,
    )

    earth_median = np.median(earth)
    mars_median = np.median(mars)

    ax.axvline(
        earth_median,
        linestyle="--",
        linewidth=1.8,
        label=f"Earth median: {earth_median:.2f}",
    )

    ax.axvline(
        mars_median,
        linestyle=":",
        linewidth=2,
        label=f"Mars median: {mars_median:.2f}",
    )

    ax.set_title(settings["title"], fontsize=15, pad=14)
    ax.set_xlabel(settings["xlabel"], fontsize=11)
    ax.set_ylabel("Probability density", fontsize=11)

    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.text(
        0.5,
        0.015,
        "Atacama: Copernicus DEM GLO-30 | "
        "Jezero: MOLA MEGDR | "
        "Working resolution: 463 m",
        ha="center",
        fontsize=8,
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])

    output_path = OUTPUT_DIR / settings["filename"]
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved chart: {output_path}")


def create_combined_figure(results):
    """Create one figure containing all three comparisons."""

    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(19, 6),
    )

    for ax, (metric_name, result) in zip(axes, results.items()):

        earth = result["earth"]
        mars = result["mars"]
        settings = result["settings"]

        combined = np.concatenate([earth, mars])

        bins = np.linspace(
            float(np.min(combined)),
            float(np.max(combined)),
            26
        )

        ax.hist(
            earth,
            bins=bins,
            density=True,
            alpha=0.55,
            label="Atacama",
            edgecolor="black",
            linewidth=0.3,
        )

        ax.hist(
            mars,
            bins=bins,
            density=True,
            alpha=0.55,
            label="Jezero",
            edgecolor="black",
            linewidth=0.3,
        )

        ax.axvline(
            np.median(earth),
            linestyle="--",
            linewidth=1.5,
        )

        ax.axvline(
            np.median(mars),
            linestyle=":",
            linewidth=1.8,
        )

        ax.set_title(settings["title"], fontsize=12)
        ax.set_xlabel(settings["xlabel"], fontsize=9)
        ax.set_ylabel("Probability density", fontsize=9)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=8)

    fig.suptitle(
        "TERRAFORM | Earth–Mars Terrain Comparison",
        fontsize=17,
        y=1.02,
    )

    fig.text(
        0.5,
        -0.015,
        "Atacama, Chile vs Jezero Crater, Mars | "
        "463 m working resolution | Descriptive comparison only",
        ha="center",
        fontsize=9,
    )

    fig.tight_layout()

    output_path = OUTPUT_DIR / "earth_mars_terrain_overview.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved combined figure: {output_path}")


def main():

    print("\nTERRAFORM TERRAIN VISUALISATION")
    print("=" * 55)

    results = {}
    report = {
        "project": "TERRAFORM",
        "comparison": "Atacama, Earth vs Jezero, Mars",
        "working_resolution_m": 463,
        "metrics": {},
        "limitations": [
            "Distribution comparison does not establish spatial correspondence.",
            "Earth and Mars elevation products have different measurement characteristics.",
            "Terrain similarity alone does not establish geological or operational equivalence.",
            "MOLA contains interpolated elevation values in areas with insufficient observations.",
        ],
    }

    for metric_name, settings in METRICS.items():

        print(f"\nProcessing: {metric_name.upper()}")

        earth, mars = load_metric(metric_name, settings)

        earth_stats = calculate_statistics(earth)
        mars_stats = calculate_statistics(mars)

        results[metric_name] = {
            "earth": earth,
            "mars": mars,
            "settings": settings,
        }

        report["metrics"][metric_name] = {
            "earth": earth_stats,
            "mars": mars_stats,
            "median_difference_mars_minus_earth": (
                mars_stats["median"] - earth_stats["median"]
            ),
        }

        print(
            f"Earth: {earth.size:,} valid pixels | "
            f"median = {earth_stats['median']:.3f}"
        )

        print(
            f"Mars:  {mars.size:,} valid pixels | "
            f"median = {mars_stats['median']:.3f}"
        )

        plot_metric(metric_name, settings, earth, mars)

    create_combined_figure(results)

    report_path = OUTPUT_DIR / "terrain_visualisation_statistics.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\n" + "=" * 55)
    print("TERRAFORM VISUALISATION COMPLETED")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Statistics report: {report_path}")


if __name__ == "__main__":
    main()
