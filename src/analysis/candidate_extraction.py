
from pathlib import Path
import csv
import json

import numpy as np
import rasterio
import matplotlib.pyplot as plt

from scipy import ndimage
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from rasterio.warp import transform as reproject_coordinates


# ============================================================
# TERRAFORM — CANDIDATE REGION EXTRACTION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MAP_DIR = ROOT / "outputs" / "maps"

SENSITIVITY_DIR = MAP_DIR / "sensitivity_analysis"
OUTPUT_DIR = MAP_DIR / "candidate_regions"

EARTH_DIR = (
    ROOT / "data" / "processed" / "earth" / "comparison_scale"
)

REPORT_DIR = ROOT / "outputs" / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# Conservative candidate definition:
# strict P40-P60 intersection, using 4-neighbour connectivity.

MASK_FILES = {
    "broad": SENSITIVITY_DIR / "broad_three_metric_mask.tif",
    "moderate": SENSITIVITY_DIR / "moderate_three_metric_mask.tif",
    "strict": SENSITIVITY_DIR / "strict_three_metric_mask.tif",
}

METRIC_FILES = {
    "slope": "earth_slope_degrees.tif",
    "roughness": "earth_roughness_m.tif",
    "local_relief": "earth_local_relief_m.tif",
}

MAP_TOP_N = 20


def read_raster(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    with rasterio.open(path) as src:

        values = src.read(1)

        return {
            "values": values,
            "profile": src.profile.copy(),
            "transform": src.transform,
            "crs": src.crs,
            "nodata": src.nodata,
        }


def read_screening_mask(path):

    raster = read_raster(path)

    values = raster["values"]

    valid = np.isfinite(values)

    if raster["nodata"] is not None:
        valid &= values != raster["nodata"]

    valid &= values != 255

    matching = valid & (values == 1)

    return raster, matching, valid


def check_grids(reference, current, label):

    if reference["values"].shape != current["values"].shape:
        raise ValueError(
            f"{label}: raster dimensions do not match."
        )

    if reference["transform"] != current["transform"]:
        raise ValueError(
            f"{label}: raster transforms do not match."
        )

    if reference["crs"] != current["crs"]:
        raise ValueError(
            f"{label}: coordinate systems do not match."
        )


def calculate_connectivity_counts(mask):

    structure_4 = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ])

    structure_8 = np.ones((3, 3), dtype=np.uint8)

    _, count_4 = ndimage.label(
        mask,
        structure=structure_4,
    )

    _, count_8 = ndimage.label(
        mask,
        structure=structure_8,
    )

    return int(count_4), int(count_8)


def save_candidate_raster(labels, valid, profile):

    output = np.full(
        labels.shape,
        -1,
        dtype=np.int32,
    )

    output[valid] = labels[valid].astype(np.int32)

    output_profile = profile.copy()

    output_profile.update(
        dtype="int32",
        count=1,
        nodata=-1,
        compress="deflate",
    )

    output_path = (
        OUTPUT_DIR / "atacama_candidate_region_ids.tif"
    )

    with rasterio.open(
        output_path,
        "w",
        **output_profile,
    ) as dst:

        dst.write(output, 1)

    print(f"Saved candidate raster: {output_path.name}")


def create_candidate_map(
    labels,
    valid,
    ordered_ids,
    transform,
):

    display = np.full(
        labels.shape,
        np.nan,
        dtype=np.float32,
    )

    # Non-candidate pixels.
    display[valid & (labels == 0)] = 0

    top_ids = ordered_ids[:MAP_TOP_N]

    for rank, cluster_id in enumerate(top_ids, start=1):

        display[labels == cluster_id] = rank

    other_candidates = (
        valid
        & (labels > 0)
        & ~np.isin(labels, top_ids)
    )

    display[other_candidates] = MAP_TOP_N + 1

    colors = [
        "#d9d9d9",
        "#1b9e77",
        "#d95f02",
        "#7570b3",
        "#e7298a",
        "#66a61e",
        "#e6ab02",
        "#a6761d",
        "#1f78b4",
        "#b2df8a",
        "#fb9a99",
        "#8dd3c7",
        "#bebada",
        "#80b1d3",
        "#fdb462",
        "#b3de69",
        "#fccde5",
        "#bc80bd",
        "#ccebc5",
        "#ffed6f",
        "#80b1d3",
        "#777777",
    ]

    cmap = ListedColormap(colors)

    height, width = labels.shape

    x_min = transform.c
    y_max = transform.f

    x_max = x_min + width * transform.a
    y_min = y_max + height * transform.e

    fig, ax = plt.subplots(figsize=(12, 9))

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
        vmax=MAP_TOP_N + 1,
        interpolation="nearest",
    )

    ax.set_title(
        "TERRAFORM | Preliminary Atacama Candidate Regions",
        fontsize=15,
        pad=15,
    )

    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    ax.grid(alpha=0.15)

    legend_items = [
        Patch(
            facecolor="#d9d9d9",
            label="Outside strict screening",
        ),
        Patch(
            facecolor="#777777",
            label="Other strict-screening regions",
        ),
    ]

    for rank, cluster_id in enumerate(top_ids, start=1):

        legend_items.append(
            Patch(
                facecolor=colors[rank],
                label=f"Candidate {rank:02d}",
            )
        )

    ax.legend(
        handles=legend_items,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        fontsize=8,
    )

    fig.text(
        0.5,
        0.02,
        "Strict P40-P60 screening | "
        "4-neighbour connected components | "
        "463 m working resolution",
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(rect=[0, 0.05, 0.82, 1])

    output_path = (
        OUTPUT_DIR / "atacama_candidate_regions_map.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved candidate map: {output_path.name}")


def main():

    print("\nTERRAFORM CANDIDATE REGION EXTRACTION")
    print("=" * 65)

    # --------------------------------------------------------
    # 1. LOAD SCREENING MASKS
    # --------------------------------------------------------

    print("\n[1/5] LOADING SCREENING MASKS")

    masks = {}
    reference = None
    common_valid = None

    for name, path in MASK_FILES.items():

        raster, matching, valid = read_screening_mask(path)

        if reference is None:

            reference = raster
            common_valid = valid

        else:

            check_grids(
                reference,
                raster,
                f"{name} screening mask",
            )

            if not np.array_equal(valid, common_valid):
                raise ValueError(
                    "Screening masks do not share the same valid pixels."
                )

        masks[name] = matching

        print(
            f"{name.title()} matching pixels: "
            f"{matching.sum():,}"
        )

    # Strict should be a subset of moderate and broad.
    if np.any(masks["strict"] & ~masks["moderate"]):
        raise ValueError(
            "Strict mask is not a subset of moderate mask."
        )

    if np.any(masks["moderate"] & ~masks["broad"]):
        raise ValueError(
            "Moderate mask is not a subset of broad mask."
        )

    strict_mask = masks["strict"]

    # --------------------------------------------------------
    # 2. LOAD EARTH TERRAIN METRICS
    # --------------------------------------------------------

    print("\n[2/5] LOADING EARTH TERRAIN METRICS")

    metrics = {}

    for name, filename in METRIC_FILES.items():

        raster = read_raster(EARTH_DIR / filename)

        check_grids(
            reference,
            raster,
            f"{name} metric raster",
        )

        values = raster["values"].astype(np.float64)

        valid = np.isfinite(values)

        if raster["nodata"] is not None:
            valid &= values != raster["nodata"]

        if not np.array_equal(valid, common_valid):
            raise ValueError(
                f"{name} metric valid pixels do not match screening grid."
            )

        metrics[name] = values

    # --------------------------------------------------------
    # 3. LABEL STRICT-SCREENING CANDIDATES
    # --------------------------------------------------------

    print("\n[3/5] EXTRACTING CONNECTED CANDIDATE REGIONS")

    structure_4 = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ])

    labels, cluster_count = ndimage.label(
        strict_mask,
        structure=structure_4,
    )

    sizes = np.bincount(labels.ravel())
    sizes[0] = 0

    ordered_ids = np.argsort(
        sizes[1:]
    )[::-1] + 1

    ordered_ids = ordered_ids[
        sizes[ordered_ids] > 0
    ]

    print(f"Candidate regions identified: {cluster_count:,}")

    transform = reference["transform"]

    pixel_area_km2 = abs(
        transform.a * transform.e
        - transform.b * transform.d
    ) / 1_000_000

    print(f"Pixel area: {pixel_area_km2:.6f} km²")

    # --------------------------------------------------------
    # 4. CHARACTERISE EACH CANDIDATE
    # --------------------------------------------------------

    print("\n[4/5] CALCULATING CANDIDATE CHARACTERISTICS")

    rows = []

    candidate_report = []

    for rank, cluster_id in enumerate(ordered_ids, start=1):

        component = labels == cluster_id

        row_indices, col_indices = np.where(component)

        pixel_count = int(len(row_indices))

        area_km2 = pixel_count * pixel_area_km2

        mean_row = float(np.mean(row_indices))
        mean_col = float(np.mean(col_indices))

        center_x, center_y = rasterio.transform.xy(
            transform,
            mean_row,
            mean_col,
            offset="center",
        )

        longitude, latitude = reproject_coordinates(
            reference["crs"],
            "EPSG:4326",
            [center_x],
            [center_y],
        )

        candidate_metrics = {}

        for metric_name, values in metrics.items():

            selected = values[component]

            candidate_metrics[metric_name] = {
                "mean": float(np.mean(selected)),
                "median": float(np.median(selected)),
            }

        candidate_id = f"TF-{rank:03d}"

        # The candidate must be contained in both broader masks.
        broad_percentage = (
            np.count_nonzero(component & masks["broad"])
            / pixel_count * 100
        )

        moderate_percentage = (
            np.count_nonzero(component & masks["moderate"])
            / pixel_count * 100
        )

        record = {
            "candidate_id": candidate_id,
            "rank_by_area": rank,
            "pixel_count": pixel_count,
            "area_km2": float(area_km2),
            "centroid_latitude": float(latitude[0]),
            "centroid_longitude": float(longitude[0]),
            "centroid_easting_m": float(center_x),
            "centroid_northing_m": float(center_y),
            "broad_screening_overlap_percent": float(
                broad_percentage
            ),
            "moderate_screening_overlap_percent": float(
                moderate_percentage
            ),
            "slope_mean_deg": candidate_metrics["slope"]["mean"],
            "slope_median_deg": candidate_metrics["slope"]["median"],
            "roughness_mean_m": candidate_metrics["roughness"]["mean"],
            "roughness_median_m": candidate_metrics["roughness"]["median"],
            "local_relief_mean_m": candidate_metrics["local_relief"]["mean"],
            "local_relief_median_m": candidate_metrics["local_relief"]["median"],
            "minimum_row": int(row_indices.min()),
            "maximum_row": int(row_indices.max()),
            "minimum_column": int(col_indices.min()),
            "maximum_column": int(col_indices.max()),
        }

        rows.append(record)

        if rank <= MAP_TOP_N:
            candidate_report.append(record)

    # --------------------------------------------------------
    # 5. SAVE RESULTS
    # --------------------------------------------------------

    print("\n[5/5] SAVING CANDIDATE INVENTORY")

    csv_path = OUTPUT_DIR / "atacama_candidate_inventory.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()) if rows else [],
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved inventory: {csv_path.name}")

    save_candidate_raster(
        labels,
        common_valid,
        reference["profile"],
    )

    count_4, count_8 = calculate_connectivity_counts(
        strict_mask
    )

    create_candidate_map(
        labels,
        common_valid,
        ordered_ids,
        transform,
    )

    report = {
        "project": "TERRAFORM",
        "analysis": "Preliminary candidate region extraction",
        "study_area": "Atacama pilot region, Chile",
        "reference_region": "Jezero Crater study area, Mars",
        "screening_definition": "Strict P40-P60 intersection across three terrain metrics",
        "primary_connectivity": "4-neighbour",
        "working_resolution_m": 463,
        "pixel_area_km2": float(pixel_area_km2),
        "valid_earth_pixels": int(common_valid.sum()),
        "strict_matching_pixels": int(strict_mask.sum()),
        "candidate_region_count_4_neighbour": int(count_4),
        "candidate_region_count_8_neighbour": int(count_8),
        "candidate_regions": candidate_report,
        "inventory_file": csv_path.name,
        "limitations": [
            "Candidate IDs indicate connected strict-screening patches, not validated analogue sites.",
            "Candidates are ranked by area only, not scientific suitability.",
            "Centroids are representative coordinates and not necessarily suitable field access points.",
            "Terrain metrics do not establish geological or mineralogical equivalence.",
            "No field accessibility, safety, land ownership, or operational suitability assessment is included.",
            "Results are limited to the selected Atacama study area and the selected Jezero reference region.",
        ],
    }

    report_path = (
        REPORT_DIR / "candidate_region_extraction_report.json"
    )

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print(f"Saved report: {report_path}")

    print("\nTOP 10 PRELIMINARY CANDIDATES")
    print("-" * 65)

    for record in rows[:10]:

        print(
            f"{record['candidate_id']} | "
            f"{record['area_km2']:.2f} km² | "
            f"Lat: {record['centroid_latitude']:.5f} | "
            f"Lon: {record['centroid_longitude']:.5f}"
        )

    print("\n" + "=" * 65)
    print("TERRAFORM CANDIDATE EXTRACTION COMPLETED")
    print(f"Total candidate regions: {len(rows):,}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
