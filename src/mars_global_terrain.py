
from pathlib import Path
import json

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap, BoundaryNorm

from scipy.ndimage import (
    uniform_filter,
    minimum_filter,
    maximum_filter,
)

# ============================================================
# TERRAFORM | GLOBAL MARS TERRAIN INTELLIGENCE
# ============================================================

ROOT = Path(r"D:\TERRAFORM")
DATA = ROOT / "data" / "Global" / "Mars"
OUT = ROOT / "outputs" / "global_mars" / "terrain_metrics"

OUT.mkdir(parents=True, exist_ok=True)

ELEVATION_FILE = DATA / "megt90n000eb.img"
COUNT_FILE = DATA / "megc90n000eb.img"

ROWS = 2880
COLS = 5760
STEP_DEG = 0.0625

RADIUS_KM = 3396.0
RADIUS_M = RADIUS_KM * 1000.0

POLAR_LIMIT_DEG = 89.5

CHUNK_ROWS = 128
HISTOGRAM_BINS = 4096
MAP_STRIDE = 8

DLON = np.deg2rad(STEP_DEG)
DY_M = RADIUS_M * DLON

METRIC_NAMES = [
    "slope_degrees",
    "roughness_3x3_m",
    "local_relief_3x3_m",
]

METRIC_FILES = {
    name: OUT / f"mars_global_{name}.npy"
    for name in METRIC_NAMES
}


# ------------------------------------------------------------
# 1. INPUT VALIDATION
# ------------------------------------------------------------

for file in [ELEVATION_FILE, COUNT_FILE]:

    if not file.exists():
        raise FileNotFoundError(
            f"Missing input file: {file}"
        )

expected_bytes = ROWS * COLS * 2

for file in [ELEVATION_FILE, COUNT_FILE]:

    actual_bytes = file.stat().st_size

    if actual_bytes != expected_bytes:
        raise ValueError(
            f"Unexpected file size for {file.name}: "
            f"{actual_bytes:,} bytes; "
            f"expected {expected_bytes:,}"
        )

print("=" * 65)
print("TERRAFORM | GLOBAL MARS TERRAIN INTELLIGENCE")
print("=" * 65)

print("\nLoading MOLA elevation and observation-count rasters...")

elevation = np.memmap(
    ELEVATION_FILE,
    dtype=">i2",
    mode="r",
    shape=(ROWS, COLS),
)

observation_count = np.memmap(
    COUNT_FILE,
    dtype=">i2",
    mode="r",
    shape=(ROWS, COLS),
)

# ------------------------------------------------------------
# 2. CREATE OUTPUT RASTERS
# ------------------------------------------------------------

metrics = {
    name: np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float32,
        shape=(ROWS, COLS),
    )
    for name, path in METRIC_FILES.items()
}

# ------------------------------------------------------------
# 3. GLOBAL TERRAIN PROCESSING
# ------------------------------------------------------------

print("\nProcessing terrain in latitude chunks...")

for y0 in range(0, ROWS, CHUNK_ROWS):

    y1 = min(y0 + CHUNK_ROWS, ROWS)

    # One-row halo on each side preserves neighbourhood
    # calculations across chunk boundaries.

    block_start = max(0, y0 - 1)
    block_end = min(ROWS, y1 + 1)

    # Float64 is used for intermediate calculations.
    # Original elevation values are integer metres.

    z = np.asarray(
        elevation[block_start:block_end],
        dtype=np.float64,
    )

    block_rows = np.arange(block_start, block_end)

    latitudes = (
        90.0
        - (block_rows + 0.5) * STEP_DEG
    )

    # --------------------------------------------------------
    # 3A. EAST-WEST GRADIENT
    # --------------------------------------------------------

    # Periodic longitude boundary:
    # The first and last columns are neighbours.

    east = np.roll(z, -1, axis=1)
    west = np.roll(z, 1, axis=1)

    # East-west physical spacing changes with latitude.

    dx = (
        RADIUS_M
        * np.cos(np.deg2rad(latitudes))
        * DLON
    )

    dz_dx = (
        (east - west)
        / (2.0 * dx[:, None])
    )

    # --------------------------------------------------------
    # 3B. NORTH-SOUTH GRADIENT
    # --------------------------------------------------------

    # Raster rows progress from north to south.
    # The sign is immaterial for slope magnitude.

    dz_dy = np.gradient(
        z,
        DY_M,
        axis=0,
    )

    # --------------------------------------------------------
    # 3C. SLOPE
    # --------------------------------------------------------

    gradient_magnitude = np.hypot(
        dz_dx,
        dz_dy,
    )

    slope = np.degrees(
        np.arctan(gradient_magnitude)
    )

    # --------------------------------------------------------
    # 3D. TERRAIN ROUGHNESS
    # --------------------------------------------------------

    # Roughness = population standard deviation of elevation
    # within a 3 x 3 grid-cell neighbourhood.
    #
    # Elevations are centred before calculating variance.
    # This reduces numerical cancellation when subtracting
    # two similar large values.

    z_offset = np.mean(z, dtype=np.float64)

    z_centered = z - z_offset

    local_mean = uniform_filter(
        z_centered,
        size=(3, 3),
        mode=("nearest", "wrap"),
    )

    local_mean_square = uniform_filter(
        z_centered * z_centered,
        size=(3, 3),
        mode=("nearest", "wrap"),
    )

    local_variance = (
        local_mean_square
        - local_mean * local_mean
    )

    roughness = np.sqrt(
        np.maximum(local_variance, 0.0)
    )

    # --------------------------------------------------------
    # 3E. LOCAL RELIEF
    # --------------------------------------------------------

    # Local relief = maximum elevation minus minimum elevation
    # in a 3 x 3 grid-cell neighbourhood.

    local_min = minimum_filter(
        z,
        size=(3, 3),
        mode=("nearest", "wrap"),
    )

    local_max = maximum_filter(
        z,
        size=(3, 3),
        mode=("nearest", "wrap"),
    )

    relief = local_max - local_min

    # --------------------------------------------------------
    # 3F. EXTRACT TARGET ROWS
    # --------------------------------------------------------

    a = y0 - block_start
    b = a + (y1 - y0)

    target_latitudes = (
        90.0
        - (np.arange(y0, y1) + 0.5) * STEP_DEG
    )

    # Exclude the near-polar region where east-west
    # spacing becomes too small for stable finite differences.

    polar_mask = (
        np.abs(target_latitudes)
        >= POLAR_LIMIT_DEG
    )

    chunk_results = {
        "slope_degrees": slope[a:b],
        "roughness_3x3_m": roughness[a:b],
        "local_relief_3x3_m": relief[a:b],
    }

    for name, result in chunk_results.items():

        result = np.asarray(
            result,
            dtype=np.float32,
        ).copy()

        result[polar_mask, :] = np.nan

        metrics[name][y0:y1, :] = result

    progress = 100.0 * y1 / ROWS

    print(
        f"  Completed rows {y0:4d}-{y1:4d} "
        f"({progress:5.1f}%)"
    )

# Flush output arrays to disk.

for array in metrics.values():
    array.flush()

del metrics

print("\nTerrain calculations completed.")

# ------------------------------------------------------------
# 4. SPHERICAL CELL AREA WEIGHTS
# ------------------------------------------------------------

row_indices = np.arange(ROWS)

north = np.deg2rad(
    90.0 - row_indices * STEP_DEG
)

south = np.deg2rad(
    90.0 - (row_indices + 1) * STEP_DEG
)

row_area_km2 = (
    RADIUS_KM ** 2
    * DLON
    * (np.sin(north) - np.sin(south))
)

valid_latitude = (
    np.abs(
        90.0
        - (row_indices + 0.5) * STEP_DEG
    )
    < POLAR_LIMIT_DEG
)

# ------------------------------------------------------------
# 5. OBSERVATION MASK
# ------------------------------------------------------------

print("\nPreparing observation mask...")

observed_mask = (
    np.asarray(observation_count) > 0
)

# ------------------------------------------------------------
# 6. AREA-WEIGHTED STATISTICS
# ------------------------------------------------------------

def first_pass_stats(array, observed):

    groups = {
        "all_analysed": {
            "area": 0.0,
            "weighted_sum": 0.0,
            "count": 0,
            "minimum": float("inf"),
            "maximum": float("-inf"),
        },

        "observed_only": {
            "area": 0.0,
            "weighted_sum": 0.0,
            "count": 0,
            "minimum": float("inf"),
            "maximum": float("-inf"),
        },
    }

    for y0 in range(0, ROWS, CHUNK_ROWS):

        y1 = min(y0 + CHUNK_ROWS, ROWS)

        values = np.asarray(
            array[y0:y1]
        )

        observed_chunk = observed[y0:y1]

        for local_row, global_row in enumerate(
            range(y0, y1)
        ):

            if not valid_latitude[global_row]:
                continue

            row_values = values[local_row]

            finite = np.isfinite(row_values)

            masks = {
                "all_analysed": finite,

                "observed_only": (
                    finite
                    & observed_chunk[local_row]
                ),
            }

            cell_area = row_area_km2[global_row]

            for group_name, mask in masks.items():

                selected = row_values[mask]

                if selected.size == 0:
                    continue

                record = groups[group_name]

                record["count"] += int(
                    selected.size
                )

                record["area"] += float(
                    selected.size * cell_area
                )

                record["weighted_sum"] += (
                    float(
                        np.sum(
                            selected,
                            dtype=np.float64,
                        )
                    )
                    * cell_area
                )

                record["minimum"] = min(
                    record["minimum"],
                    float(np.min(selected)),
                )

                record["maximum"] = max(
                    record["maximum"],
                    float(np.max(selected)),
                )

    return groups


def build_histograms(array, observed, value_range):

    edges = np.linspace(
        value_range[0],
        value_range[1],
        HISTOGRAM_BINS + 1,
    )

    histograms = {
        "all_analysed": np.zeros(
            HISTOGRAM_BINS,
            dtype=np.float64,
        ),

        "observed_only": np.zeros(
            HISTOGRAM_BINS,
            dtype=np.float64,
        ),
    }

    for y0 in range(0, ROWS, CHUNK_ROWS):

        y1 = min(y0 + CHUNK_ROWS, ROWS)

        values = np.asarray(
            array[y0:y1]
        )

        observed_chunk = observed[y0:y1]

        row_areas = row_area_km2[y0:y1]

        weights = np.broadcast_to(
            row_areas[:, None],
            values.shape,
        )

        latitude_mask = (
            valid_latitude[y0:y1, None]
        )

        finite = np.isfinite(values)

        full_mask = (
            finite
            & latitude_mask
        )

        observed_valid = (
            full_mask
            & observed_chunk
        )

        masks = {
            "all_analysed": full_mask,
            "observed_only": observed_valid,
        }

        for group_name, mask in masks.items():

            if np.any(mask):

                histograms[group_name] += np.histogram(
                    values[mask],
                    bins=edges,
                    weights=weights[mask],
                )[0]

    return edges, histograms


def weighted_percentile(
    edges,
    histogram,
    percentile,
):

    total = histogram.sum()

    if total <= 0:
        return None

    cumulative = np.cumsum(histogram)

    target = (
        total * percentile / 100.0
    )

    index = int(
        np.searchsorted(
            cumulative,
            target,
            side="left",
        )
    )

    index = min(
        index,
        len(edges) - 2,
    )

    return float(
        (
            edges[index]
            + edges[index + 1]
        ) / 2.0
    )


def calculate_statistics(array, observed):

    first = first_pass_stats(
        array,
        observed,
    )

    global_min = first["all_analysed"]["minimum"]
    global_max = first["all_analysed"]["maximum"]

    if global_min == global_max:
        global_max = global_min + 1.0

    edges, histograms = build_histograms(
        array,
        observed,
        (global_min, global_max),
    )

    output = {}

    for group_name, record in first.items():

        histogram = histograms[group_name]

        if record["area"] <= 0:
            continue

        output[group_name] = {
            "cell_count": record["count"],

            "reference_area_km2": record["area"],

            "minimum": record["minimum"],

            "maximum": record["maximum"],

            "area_weighted_mean": (
                record["weighted_sum"]
                / record["area"]
            ),

            "area_weighted_percentiles_approx": {
                "P01": weighted_percentile(
                    edges,
                    histogram,
                    1,
                ),

                "P25": weighted_percentile(
                    edges,
                    histogram,
                    25,
                ),

                "P50": weighted_percentile(
                    edges,
                    histogram,
                    50,
                ),

                "P75": weighted_percentile(
                    edges,
                    histogram,
                    75,
                ),

                "P99": weighted_percentile(
                    edges,
                    histogram,
                    99,
                ),
            },
        }

    return output


# ------------------------------------------------------------
# 7. METRIC REPORTS
# ------------------------------------------------------------

statistics = {}

for name, path in METRIC_FILES.items():

    print(
        f"\nCalculating statistics: {name}"
    )

    array = np.load(
        path,
        mmap_mode="r",
    )

    statistics[name] = calculate_statistics(
        array,
        observed_mask,
    )

    del array

# ------------------------------------------------------------
# 8. GLOBAL TERRAIN VISUALISATIONS
# ------------------------------------------------------------

print("\nGenerating global terrain maps...")

map_specs = {

    "slope_degrees": {
        "title": "GLOBAL MARS SLOPE",
        "unit": "Slope (degrees)",
        "cmap": "magma",
    },

    "roughness_3x3_m": {
        "title": "GLOBAL MARS TERRAIN ROUGHNESS",
        "unit": "3x3 elevation standard deviation (m)",
        "cmap": "viridis",
    },

    "local_relief_3x3_m": {
        "title": "GLOBAL MARS LOCAL RELIEF",
        "unit": "3x3 elevation range (m)",
        "cmap": "cividis",
    },
}

for name, spec in map_specs.items():

    array = np.load(
        METRIC_FILES[name],
        mmap_mode="r",
    )

    sample = np.asarray(
        array[::MAP_STRIDE, ::MAP_STRIDE]
    )

    upper = float(
        np.nanpercentile(
            sample,
            99,
        )
    )

    if not np.isfinite(upper) or upper <= 0:
        upper = 1.0

    fig, ax = plt.subplots(
        figsize=(16, 7)
    )

    image = ax.imshow(
        sample,
        origin="upper",
        extent=[0, 360, -90, 90],
        aspect="auto",
        cmap=spec["cmap"],
        vmin=0,
        vmax=upper,
    )

    ax.set_title(
        spec["title"],
        fontsize=16,
        fontweight="bold",
        pad=14,
    )

    ax.set_xlabel(
        "East longitude (degrees)"
    )

    ax.set_ylabel(
        "Planetocentric latitude (degrees)"
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.025,
        pad=0.02,
    )

    colorbar.set_label(
        spec["unit"]
    )

    ax.set_xticks(
        np.arange(0, 361, 30)
    )

    ax.set_yticks(
        np.arange(-90, 91, 15)
    )

    ax.grid(
        alpha=0.15,
        linewidth=0.5,
    )

    fig.tight_layout()

    destination = (
        OUT / f"mars_global_{name}.png"
    )

    fig.savefig(
        destination,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    del array

    print(
        f"  Saved: {destination.name}"
    )


# ------------------------------------------------------------
# 9. OBSERVATION QUALITY MAP
# ------------------------------------------------------------

# Classification:
# 0 = Zero recorded observations
# 1 = One or more recorded observations
# 2 = Polar terrain excluded from metric analysis

print("\nGenerating observation quality map...")

quality_sample = np.zeros(
    (
        ROWS // MAP_STRIDE,
        COLS // MAP_STRIDE,
    ),
    dtype=np.uint8,
)

sampled_observations = observed_mask[
    ::MAP_STRIDE,
    ::MAP_STRIDE,
]

quality_sample[
    sampled_observations
] = 1

sampled_latitudes = (
    90.0
    - (
        np.arange(
            0,
            ROWS,
            MAP_STRIDE,
        )
        + 0.5
    ) * STEP_DEG
)

sampled_polar_rows = (
    np.abs(sampled_latitudes)
    >= POLAR_LIMIT_DEG
)

quality_sample[
    sampled_polar_rows,
    :
] = 2

quality_cmap = ListedColormap([
    "#9B7653",
    "#25A18E",
    "#B8B8B8",
])

quality_norm = BoundaryNorm(
    [-0.5, 0.5, 1.5, 2.5],
    quality_cmap.N,
)

fig, ax = plt.subplots(
    figsize=(16, 7)
)

im = ax.imshow(
    quality_sample,
    origin="upper",
    extent=[0, 360, -90, 90],
    aspect="auto",
    cmap=quality_cmap,
    norm=quality_norm,
)

ax.set_title(
    "MOLA OBSERVATION COVERAGE AND TERRAIN MASK",
    fontsize=16,
    fontweight="bold",
)

ax.set_xlabel(
    "East longitude (degrees)"
)

ax.set_ylabel(
    "Planetocentric latitude (degrees)"
)

cbar = fig.colorbar(
    im,
    ax=ax,
    ticks=[0, 1, 2],
    fraction=0.025,
    pad=0.02,
)

cbar.ax.set_yticklabels([
    "Zero recorded observations",
    "Recorded observations",
    "Polar analysis excluded",
])

ax.set_xticks(
    np.arange(0, 361, 30)
)

ax.set_yticks(
    np.arange(-90, 91, 15)
)

ax.grid(
    alpha=0.15,
    linewidth=0.5,
)

fig.tight_layout()

fig.savefig(
    OUT / "mars_global_terrain_quality.png",
    dpi=180,
    bbox_inches="tight",
)

plt.close(fig)

del quality_sample

# ------------------------------------------------------------
# 10. REFERENCE AREA
# ------------------------------------------------------------

reference_area = float(
    row_area_km2[valid_latitude].sum()
    * COLS
)

# ------------------------------------------------------------
# 11. JSON REPORT
# ------------------------------------------------------------

report = {

    "project": (
        "TERRAFORM | Planetary Analog Intelligence"
    ),

    "analysis": (
        "Global Mars terrain intelligence"
    ),

    "source": {

        "elevation": ELEVATION_FILE.name,

        "observation_count": COUNT_FILE.name,

        "dataset_id": (
            "MGS-M-MOLA-5-MEGDR-L3-V1.0"
        ),
    },

    "grid": {

        "rows": ROWS,

        "columns": COLS,

        "angular_spacing_degrees": STEP_DEG,

        "longitude_range_degrees_east": [
            0,
            360,
        ],

        "latitude_range_degrees_planetocentric": [
            -90,
            90,
        ],
    },

    "geometry": {

        "reference_radius_km": RADIUS_KM,

        "model": "Spherical reference surface",

        "analysed_reference_area_km2": (
            reference_area
        ),
    },

    "processing": {

        "intermediate_precision": "float64",

        "output_raster_precision": "float32",

        "slope_method": (
            "Central differences with latitude-adjusted "
            "east-west distance"
        ),

        "longitude_boundary": "Periodic wrap",

        "north_south_spacing_m": DY_M,

        "roughness_method": (
            "3x3 grid-cell population standard deviation"
        ),

        "roughness_calculation": (
            "Centred float64 elevation moments"
        ),

        "local_relief_method": (
            "3x3 grid-cell maximum minus minimum elevation"
        ),

        "polar_exclusion": (
            f"Absolute planetocentric latitude >= "
            f"{POLAR_LIMIT_DEG} degrees"
        ),

        "chunk_rows": CHUNK_ROWS,

        "percentile_method": (
            "Area-weighted histogram approximation"
        ),

        "histogram_bins": HISTOGRAM_BINS,
    },

    "statistics": statistics,

    "limitations": [

        (
            "Slope uses finite differences on a spherical "
            "reference geometry."
        ),

        (
            "East-west pixel distance varies with latitude."
        ),

        (
            "The cylindrical grid becomes singular at the poles; "
            "cells at and beyond the polar cutoff are excluded."
        ),

        (
            "Roughness and local relief use angular 3x3 "
            "grid-cell neighbourhoods, not constant "
            "physical-distance windows."
        ),

        (
            "The global MOLA product includes interpolated "
            "elevation values in zero-observation cells."
        ),

        (
            "Observed-only statistics select centre cells "
            "with one or more recorded observations."
        ),

        (
            "Neighbourhood calculations may still include "
            "interpolated neighbouring elevations, even when "
            "the centre cell has a recorded observation."
        ),

        (
            "The observation mask identifies cells with "
            "recorded observations, not the quality or accuracy "
            "of each individual elevation."
        ),

        (
            "Area-weighted means use spherical reference "
            "cell areas."
        ),

        (
            "Weighted percentiles are approximate "
            "histogram-bin estimates."
        ),

        (
            "These metrics do not independently validate "
            "the source elevation data or interpolation."
        ),
    ],

    "outputs": {

        "rasters": {
            name: path.name
            for name, path in METRIC_FILES.items()
        },

        "maps": [
            f"mars_global_{name}.png"
            for name in METRIC_NAMES
        ] + [
            "mars_global_terrain_quality.png"
        ],
    },
}

json_path = (
    OUT / "mars_global_terrain_report.json"
)

with open(
    json_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        report,
        f,
        indent=4,
    )

# ------------------------------------------------------------
# 12. TEXT REPORT
# ------------------------------------------------------------

txt_path = (
    OUT / "mars_global_terrain_report.txt"
)

with open(
    txt_path,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "TERRAFORM | GLOBAL MARS TERRAIN INTELLIGENCE\n"
    )

    f.write("=" * 65 + "\n\n")

    f.write("REFERENCE GEOMETRY\n")
    f.write("-" * 65 + "\n")

    f.write(
        f"Reference radius: {RADIUS_KM} km\n"
    )

    f.write(
        f"Grid: {ROWS} x {COLS}\n"
    )

    f.write(
        f"Angular spacing: {STEP_DEG} degrees\n"
    )

    f.write(
        f"Analysed reference area: "
        f"{reference_area:,.3f} km2\n"
    )

    f.write(
        f"Polar exclusion: "
        f"|latitude| >= {POLAR_LIMIT_DEG} degrees\n"
    )

    f.write(
        "Intermediate precision: float64\n"
    )

    f.write(
        "Output raster precision: float32\n\n"
    )

    for metric_name, groups in statistics.items():

        f.write("\n" + "=" * 65 + "\n")

        f.write(
            metric_name.upper() + "\n"
        )

        f.write("=" * 65 + "\n")

        for group_name, values in groups.items():

            f.write(
                "\n" + group_name.upper() + "\n"
            )

            f.write("-" * 45 + "\n")

            f.write(
                f"Cells: "
                f"{values['cell_count']:,}\n"
            )

            f.write(
                f"Reference area: "
                f"{values['reference_area_km2']:,.3f} km2\n"
            )

            f.write(
                f"Minimum: "
                f"{values['minimum']:.3f}\n"
            )

            f.write(
                f"Maximum: "
                f"{values['maximum']:.3f}\n"
            )

            f.write(
                f"Area-weighted mean: "
                f"{values['area_weighted_mean']:.3f}\n"
            )

            for percentile, value in (
                values[
                    "area_weighted_percentiles_approx"
                ].items()
            ):

                f.write(
                    f"Area-weighted {percentile}: "
                    f"{value:.3f}\n"
                )

    f.write("\n\nSCIENTIFIC LIMITATIONS\n")
    f.write("-" * 65 + "\n")

    for i, limitation in enumerate(
        report["limitations"],
        1,
    ):

        f.write(
            f"{i}. {limitation}\n"
        )

    f.write("\nOUTPUT FILES\n")
    f.write("-" * 65 + "\n")

    for path in METRIC_FILES.values():
        f.write(path.name + "\n")

    for name in report["outputs"]["maps"]:
        f.write(name + "\n")

    f.write(json_path.name + "\n")

# ------------------------------------------------------------
# 13. FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 65)

print(
    "TERRAFORM | PROCESSING COMPLETE"
)

print("=" * 65)

for metric_name, groups in statistics.items():

    print(
        f"\n{metric_name.upper()}"
    )

    for group_name, values in groups.items():

        print(
            f"  {group_name}:"
        )

        print(
            f"    Area-weighted mean: "
            f"{values['area_weighted_mean']:.3f}"
        )

        print(
            f"    Area-weighted median (approx): "
            f"{values['area_weighted_percentiles_approx']['P50']:.3f}"
        )

print("\nOutput directory:")
print(OUT)

print("\nText report:")
print(txt_path)

print("\nJSON report:")
print(json_path)

print("\n" + "=" * 65)

print(
    "TERRAFORM | PLANETARY ANALOG INTELLIGENCE"
)

print("=" * 65)
