
from pathlib import Path
import json

import numpy as np
import rasterio
import matplotlib.pyplot as plt

from scipy import ndimage
from matplotlib.colors import ListedColormap


# ============================================================
# TERRAFORM — SPATIAL CLUSTER ANALYSIS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_MASK = (
    ROOT
    / "outputs"
    / "maps"
    / "terrain_range_screening"
    / "atacama_three_metric_intersection.tif"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "maps"
    / "spatial_clusters"
)

REPORT_DIR = ROOT / "outputs" / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_intersection():
    if not INPUT_MASK.exists():
        raise FileNotFoundError(
            f"Intersection raster not found: {INPUT_MASK}"
        )

    with rasterio.open(INPUT_MASK) as src:
        data = src.read(1)
        profile = src.profile.copy()
        transform = src.transform
        nodata = src.nodata

    valid = np.ones(data.shape, dtype=bool)

    if nodata is not None:
        valid &= data != nodata

    valid &= data != 255

    matching = valid & (data == 1)

    return matching, valid, profile, transform


def analyse_connectivity(matching, connectivity):
    if connectivity == 4:
        structure = np.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ])
    else:
        structure = np.ones((3, 3), dtype=np.uint8)

    labels, number_of_clusters = ndimage.label(
        matching,
        structure=structure,
    )

    sizes = np.bincount(labels.ravel())
    sizes[0] = 0

    cluster_ids = np.arange(1, number_of_clusters + 1)

    ordered_ids = cluster_ids[
        np.argsort(sizes[cluster_ids])[::-1]
    ]

    return labels, sizes, ordered_ids


def summarise_clusters(
    sizes,
    ordered_ids,
    pixel_area_km2,
    matching_pixels,
):
    if len(ordered_ids) == 0:
        return {
            "cluster_count": 0,
            "largest_cluster_area_km2": 0,
            "largest_10_share_percent": 0,
            "clusters": [],
        }

    cluster_sizes = sizes[ordered_ids]

    areas = cluster_sizes * pixel_area_km2

    largest_10_pixels = int(
        np.sum(cluster_sizes[:10])
    )

    top_clusters = []

    for rank, (cluster_id, pixel_count, area) in enumerate(
        zip(
            ordered_ids[:20],
            cluster_sizes[:20],
            areas[:20],
        ),
        start=1,
    ):
        top_clusters.append({
            "rank": rank,
            "cluster_id": int(cluster_id),
            "pixel_count": int(pixel_count),
            "area_km2": float(area),
        })

    return {
        "cluster_count": int(len(ordered_ids)),
        "largest_cluster_pixels": int(cluster_sizes[0]),
        "largest_cluster_area_km2": float(areas[0]),
        "median_cluster_area_km2": float(np.median(areas)),
        "mean_cluster_area_km2": float(np.mean(areas)),
        "largest_10_share_percent": (
            float(largest_10_pixels / matching_pixels * 100)
            if matching_pixels else 0
        ),
        "clusters": top_clusters,
    }


def save_cluster_raster(labels, valid, profile, connectivity):
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
        OUTPUT_DIR
        / f"atacama_clusters_{connectivity}_connectivity.tif"
    )

    with rasterio.open(
        output_path,
        "w",
        **output_profile,
    ) as dst:
        dst.write(output, 1)

    print(f"Saved: {output_path.name}")


def create_cluster_map(
    labels,
    ordered_ids,
    valid,
    transform,
):
    """
    Show the 10 largest clusters separately.
    Other matching pixels appear grey.
    """

    display = np.full(
        labels.shape,
        np.nan,
        dtype=np.float32,
    )

    display[valid & (labels == 0)] = 0

    largest_ids = ordered_ids[:10]

    for rank, cluster_id in enumerate(largest_ids, start=1):
        display[labels == cluster_id] = rank

    # Other matching clusters are assigned class 11.
    other_matching = (
        valid
        & (labels > 0)
        & ~np.isin(labels, largest_ids)
    )

    display[other_matching] = 11

    cmap = ListedColormap([
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
        "#777777",
    ])

    fig, ax = plt.subplots(figsize=(11, 9))

    height, width = labels.shape

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
        vmax=11,
        interpolation="nearest",
    )

    ax.set_title(
        "TERRAFORM | Largest Terrain-Matching Regions",
        fontsize=15,
        pad=15,
    )

    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    ax.grid(alpha=0.15)

    ax.text(
        0.5,
        -0.11,
        "Largest 10 connected regions highlighted | "
        "Grey: other matching pixels | "
        "Light grey: outside terrain-range intersection",
        transform=ax.transAxes,
        ha="center",
        fontsize=8,
    )

    output_path = (
        OUTPUT_DIR / "atacama_largest_matching_clusters.png"
    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Saved: {output_path.name}")


def main():

    print("\nTERRAFORM SPATIAL CLUSTER ANALYSIS")
    print("=" * 60)

    matching, valid, profile, transform = load_intersection()

    matching_pixels = int(matching.sum())
    valid_pixels = int(valid.sum())

    pixel_area_km2 = abs(
        transform.a * transform.e
    ) / 1_000_000

    print(f"Valid pixels: {valid_pixels:,}")
    print(f"Matching pixels: {matching_pixels:,}")
    print(f"Pixel area: {pixel_area_km2:.6f} km²")

    report = {
        "project": "TERRAFORM",
        "analysis": "Spatial cluster analysis",
        "input": INPUT_MASK.name,
        "pixel_area_km2": float(pixel_area_km2),
        "matching_pixels": matching_pixels,
        "valid_pixels": valid_pixels,
        "connectivity_comparison": {},
    }

    eight_connectivity_labels = None
    eight_connectivity_ordered_ids = None

    for connectivity in [4, 8]:

        print(
            f"\nANALYSING {connectivity}-NEIGHBOUR CONNECTIVITY"
        )

        labels, sizes, ordered_ids = analyse_connectivity(
            matching,
            connectivity,
        )

        summary = summarise_clusters(
            sizes,
            ordered_ids,
            pixel_area_km2,
            matching_pixels,
        )

        report["connectivity_comparison"][
            f"{connectivity}_neighbour"
        ] = summary

        print(f"Number of clusters: {summary['cluster_count']:,}")

        if summary["cluster_count"] > 0:

            print(
                "Largest cluster area: "
                f"{summary['largest_cluster_area_km2']:.2f} km²"
            )

            print(
                "Largest 10 clusters contain: "
                f"{summary['largest_10_share_percent']:.2f}% "
                "of all matching pixels"
            )

            print("\nTOP 10 MATCHING REGIONS")

            for cluster in summary["clusters"][:10]:

                print(
                    f"#{cluster['rank']} | "
                    f"{cluster['area_km2']:.2f} km² | "
                    f"{cluster['pixel_count']:,} pixels"
                )

        save_cluster_raster(
            labels,
            valid,
            profile,
            connectivity,
        )

        if connectivity == 8:
            eight_connectivity_labels = labels
            eight_connectivity_ordered_ids = ordered_ids

    if eight_connectivity_labels is not None:

        create_cluster_map(
            eight_connectivity_labels,
            eight_connectivity_ordered_ids,
            valid,
            transform,
        )

    report["limitations"] = [
        "Connected components are spatial groupings, not independent observations.",
        "Eight-neighbour connectivity includes diagonal contact.",
        "Cluster sizes depend on raster resolution and connectivity definition.",
        "Spatial clustering does not establish geological or environmental equivalence.",
        "The analysis is limited to the selected Atacama study area.",
    ]

    report_path = (
        REPORT_DIR / "spatial_cluster_analysis.json"
    )

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\n" + "=" * 60)
    print("SPATIAL CLUSTER ANALYSIS COMPLETED")
    print(f"Report: {report_path}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
