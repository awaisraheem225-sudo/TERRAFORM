
from pathlib import Path
import json

import numpy as np
import rasterio
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


# ============================================================
# TERRAFORM — TERRAIN-RANGE ANALOGUE SCREENING
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

EARTH_DIR = (
    ROOT / "data" / "processed" / "earth" / "comparison_scale"
)

MARS_DIR = (
    ROOT / "data" / "processed" / "mars" / "comparison_scale"
)

OUTPUT_DIR = (
    ROOT / "outputs" / "maps" / "terrain_range_screening"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_DIR = ROOT / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Central 80% of the Jezero distribution.
LOWER_PERCENTILE = 10
UPPER_PERCENTILE = 90

METRICS = {
    "slope": {
        "earth": "earth_slope_degrees.tif",
        "mars": "mars_slope_degrees.tif",
        "label": "Slope",
        "unit": "degrees",
        "output": "atacama_slope_range_mask.tif",
    },
    "roughness": {
        "earth": "earth_roughness_m.tif",
        "mars": "mars_roughness_m.tif",
        "label": "Terrain roughness",
        "unit": "metres",
        "output": "atacama_roughness_range_mask.tif",
    },
    "local_relief": {
        "earth": "earth_local_relief_m.tif",
        "mars": "mars_local_relief_m.tif",
        "label": "Local relief",
        "unit": "metres",
        "output": "atacama_local_relief_range_mask.tif",
    },
}


def load_raster(path):
    """Read raster and identify valid pixels."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required raster is missing: {path}"
        )

    with rasterio.open(path) as src:
        values = src.read(1).astype(np.float64)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

    valid = np.isfinite(values)

    if nodata is not None:
        valid &= values != nodata

    return values, valid, profile, transform, crs


def validate_earth_grids(earth_grids):
    """Ensure all Earth metric rasters share the same grid."""

    first = earth_grids[0]

    for current in earth_grids[1:]:
        if current["values"].shape != first["values"].shape:
            raise ValueError(
                "Earth metric rasters have different dimensions."
            )

        if current["transform"] != first["transform"]:
            raise ValueError(
                "Earth metric rasters have different transforms."
            )

        if current["crs"] != first["crs"]:
            raise ValueError(
                "Earth metric rasters have different coordinate systems."
            )


def save_mask(mask, valid, profile, output_path):
    """
    Save a binary mask.

    1   = within the Jezero reference range
    0   = outside the Jezero reference range
    255 = NoData
    """

    output = np.full(mask.shape, 255, dtype=np.uint8)

    output[valid] = mask[valid].astype(np.uint8)

    output_profile = profile.copy()
    output_profile.update(
        dtype="uint8",
        count=1,
        nodata=255,
        compress="deflate",
    )

    with rasterio.open(output_path, "w", **output_profile) as dst:
        dst.write(output, 1)


def calculate_area_km2(pixel_count, transform):
    pixel_area = abs(transform.a * transform.e)
    return float(pixel_count * pixel_area / 1_000_000)


def create_map(
    ax,
    mask,
    valid,
    transform,
    title,
    subtitle=None,
):
    """Plot a binary terrain-range screening result."""

    height, width = mask.shape

    x_min = transform.c
    y_max = transform.f
    x_max = x_min + width * transform.a
    y_min = y_max + height * transform.e

    display = np.ma.masked_where(~valid, mask)

    cmap = ListedColormap([
        "#d95f59",  # Outside reference range
        "#35a66f",  # Within reference range
    ])

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

    ax.set_title(title, fontsize=12, pad=12)
    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    ax.grid(alpha=0.15)

    if subtitle:
        ax.text(
            0.5,
            -0.18,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            fontsize=8,
        )


def main():

    print("\nTERRAFORM TERRAIN-RANGE SCREENING")
    print("=" * 60)

    earth_grids = []
    mars_grids = []

    # --------------------------------------------------------
    # 1. LOAD ALL METRICS
    # --------------------------------------------------------

    print("\n[1/5] LOADING EARTH AND MARS TERRAIN DATA")

    for name, settings in METRICS.items():

        earth_values, earth_valid, earth_profile, earth_transform, earth_crs = (
            load_raster(EARTH_DIR / settings["earth"])
        )

        mars_values, mars_valid, _, _, _ = (
            load_raster(MARS_DIR / settings["mars"])
        )

        earth_grids.append({
            "name": name,
            "values": earth_values,
            "valid": earth_valid,
            "profile": earth_profile,
            "transform": earth_transform,
            "crs": earth_crs,
        })

        mars_grids.append({
            "name": name,
            "values": mars_values,
            "valid": mars_valid,
        })

    validate_earth_grids(earth_grids)

    earth_shape = earth_grids[0]["values"].shape

    # All metrics must have the same valid Earth pixels
    # because they were calculated from complete 3x3 windows.
    common_earth_valid = np.logical_and.reduce([
        grid["valid"] for grid in earth_grids
    ])

    if not np.array_equal(
        common_earth_valid,
        earth_grids[0]["valid"]
    ):
        raise ValueError(
            "Earth metrics do not share identical valid-pixel masks."
        )

    print(f"Earth grid: {earth_shape[1]} x {earth_shape[0]}")
    print(f"Common valid Earth pixels: {common_earth_valid.sum():,}")

    # --------------------------------------------------------
    # 2. CALCULATE JEZERO REFERENCE RANGES
    # --------------------------------------------------------

    print("\n[2/5] CALCULATING JEZERO REFERENCE RANGES")

    thresholds = {}

    for grid, mars_grid in zip(earth_grids, mars_grids):

        name = grid["name"]
        settings = METRICS[name]

        mars_values = mars_grid["values"][mars_grid["valid"]]

        lower = float(np.percentile(
            mars_values, LOWER_PERCENTILE
        ))

        upper = float(np.percentile(
            mars_values, UPPER_PERCENTILE
        ))

        thresholds[name] = {
            "lower": lower,
            "upper": upper,
            "unit": settings["unit"],
            "mars_valid_pixels": int(mars_values.size),
        }

        print(
            f"{settings['label']}: "
            f"{lower:.4f} to {upper:.4f} "
            f"{settings['unit']}"
        )

    # --------------------------------------------------------
    # 3. SCREEN ATACAMA PIXELS
    # --------------------------------------------------------

    print("\n[3/5] SCREENING ATACAMA TERRAIN")

    individual_masks = {}
    report_metrics = {}

    for grid in earth_grids:

        name = grid["name"]
        settings = METRICS[name]

        earth_values = grid["values"]
        earth_valid = grid["valid"]

        lower = thresholds[name]["lower"]
        upper = thresholds[name]["upper"]

        within_range = (
            earth_valid
            & (earth_values >= lower)
            & (earth_values <= upper)
        )

        individual_masks[name] = within_range

        match_count = int(within_range.sum())
        valid_count = int(earth_valid.sum())

        percentage = (
            match_count / valid_count * 100
            if valid_count else 0
        )

        report_metrics[name] = {
            "reference_range_lower": lower,
            "reference_range_upper": upper,
            "unit": settings["unit"],
            "matching_earth_pixels": match_count,
            "valid_earth_pixels": valid_count,
            "matching_percentage": percentage,
            "matching_area_km2": calculate_area_km2(
                match_count, grid["transform"]
            ),
        }

        output_path = OUTPUT_DIR / settings["output"]

        save_mask(
            within_range,
            earth_valid,
            grid["profile"],
            output_path,
        )

        print(
            f"{settings['label']}: "
            f"{match_count:,}/{valid_count:,} pixels "
            f"({percentage:.2f}%)"
        )

        print(f"Saved: {output_path.name}")

    # --------------------------------------------------------
    # 4. THREE-METRIC INTERSECTION
    # --------------------------------------------------------

    print("\n[4/5] CALCULATING THREE-METRIC INTERSECTION")

    all_three = np.logical_and.reduce([
        individual_masks[name]
        for name in METRICS
    ])

    valid_count = int(common_earth_valid.sum())
    match_count = int(all_three.sum())

    intersection_percentage = (
        match_count / valid_count * 100
        if valid_count else 0
    )

    intersection_area = calculate_area_km2(
        match_count,
        earth_grids[0]["transform"],
    )

    intersection_path = (
        OUTPUT_DIR / "atacama_three_metric_intersection.tif"
    )

    save_mask(
        all_three,
        common_earth_valid,
        earth_grids[0]["profile"],
        intersection_path,
    )

    print(f"Three-metric matches: {match_count:,}")
    print(f"Percentage: {intersection_percentage:.2f}%")
    print(f"Area: {intersection_area:.2f} km²")
    print(f"Saved: {intersection_path.name}")

    # --------------------------------------------------------
    # 5. CREATE MAPS AND REPORT
    # --------------------------------------------------------

    print("\n[5/5] GENERATING SCREENING MAPS AND REPORT")

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(15, 12),
    )

    axes = axes.flatten()

    for index, name in enumerate(METRICS):

        settings = METRICS[name]
        mask = individual_masks[name]

        matching_count = int(mask.sum())
        percentage = (
            matching_count / valid_count * 100
            if valid_count else 0
        )

        create_map(
            axes[index],
            mask.astype(np.uint8),
            common_earth_valid,
            earth_grids[0]["transform"],
            settings["label"],
            f"{percentage:.2f}% of valid Atacama pixels within range",
        )

    create_map(
        axes[3],
        all_three.astype(np.uint8),
        common_earth_valid,
        earth_grids[0]["transform"],
        "Three-Metric Intersection",
        f"{intersection_percentage:.2f}% satisfy all three conditions",
    )

    legend_items = [
        Patch(
            facecolor="#35a66f",
            label="Within Jezero reference range",
        ),
        Patch(
            facecolor="#d95f59",
            label="Outside Jezero reference range",
        ),
    ]

    fig.legend(
        handles=legend_items,
        loc="lower center",
        ncol=2,
        frameon=False,
    )

    fig.suptitle(
        "TERRAFORM | Atacama Terrain-Range Screening",
        fontsize=17,
        y=0.98,
    )

    fig.text(
        0.5,
        0.025,
        "Reference: Jezero Crater, Mars | "
        "Jezero central 80% percentile ranges | "
        "Working resolution: 463 m",
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(rect=[0, 0.07, 1, 0.95])

    map_path = OUTPUT_DIR / "atacama_terrain_screening_overview.png"

    fig.savefig(
        map_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    report = {
        "project": "TERRAFORM",
        "analysis": "Terrain-Range Analogue Screening",
        "earth_study_region": "Atacama Desert pilot area, Chile",
        "mars_reference_region": "Jezero Crater study area, Mars",
        "working_resolution_m": 463,
        "reference_percentiles": {
            "lower": LOWER_PERCENTILE,
            "upper": UPPER_PERCENTILE,
        },
        "earth_valid_pixels": valid_count,
        "metrics": report_metrics,
        "three_metric_intersection": {
            "matching_pixels": match_count,
            "valid_pixels": valid_count,
            "matching_percentage": intersection_percentage,
            "matching_area_km2": intersection_area,
        },
        "interpretation_limitations": [
            "This is a preliminary terrain-range screening method.",
            "The reference intervals are based on Jezero's empirical P10-P90 distributions.",
            "The intersection identifies Earth pixels satisfying all three independent range conditions.",
            "It does not establish geological, mineralogical, climatic, biological, or operational equivalence.",
            "The three conditions are not statistically independent.",
            "Earth and Mars elevation products have different measurement characteristics.",
            "MOLA contains interpolated values in areas with insufficient observations.",
            "Results apply only to the selected Atacama and Jezero study areas.",
        ],
    }

    report_path = (
        REPORT_DIR / "terrain_range_screening_report.json"
    )

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print(f"Saved overview map: {map_path}")
    print(f"Saved report: {report_path}")

    print("\n" + "=" * 60)
    print("TERRAFORM TERRAIN-RANGE SCREENING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
