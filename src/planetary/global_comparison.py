
"""
TERRAFORM — Global Planetary Terrain Comparison
Version: 1.1

Earth, Mars and Moon global elevation-distribution comparison.

Scientific scope:
- Earth: elevation-positive land proxy.
- Mars and Moon: full gridded surfaces.
- Approximate spherical latitude weighting.
- No pixel-to-pixel planetary matching.
- Native vertical references are retained.
- Statistics are calculated from 10 m histogram bins.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.windows import Window


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

EARTH_FILE = ROOT / "data" / "Global" / "Earth" / "earth_etopo_2022_60s.tif"
MARS_FILE = ROOT / "data" / "Global" / "Mars" / "megt90n000eb.img"
MOON_FILE = ROOT / "data" / "Global" / "Moon" / "ldem_16.img"

OUTPUT_DIR = ROOT / "outputs" / "Global" / "Comparisons"
VISUAL_DIR = ROOT / "outputs" / "Visualisations" / "Global"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VISUAL_DIR.mkdir(parents=True, exist_ok=True)

VERSION = "1.1"

ELEVATION_MIN = -11000.0
ELEVATION_MAX = 22000.0
ELEVATION_BIN_SIZE = 10.0

ELEVATION_EDGES = np.arange(
    ELEVATION_MIN,
    ELEVATION_MAX + ELEVATION_BIN_SIZE,
    ELEVATION_BIN_SIZE,
    dtype=np.float64,
)

ELEVATION_CENTRES = (
    ELEVATION_EDGES[:-1] + ELEVATION_EDGES[1:]
) / 2.0

EARTH_BLOCK_ROWS = 128
PLANET_BLOCK_ROWS = 128

EARTH_AGGREGATION = 2
MARS_RESOLUTION = 0.0625
MOON_RESOLUTION = 0.0625
MOON_AGGREGATION = 2

MARS_LABEL_MIN = -8177
MARS_LABEL_MAX = 21171

MOON_LABEL_MIN = -18150
MOON_LABEL_MAX = 21580


# ============================================================
# 2. DIAGNOSTIC HELPERS
# ============================================================

def new_diagnostics():
    return {
        "input_cells": 0,
        "nonfinite_values": 0,
        "nonfinite_weights": 0,
        "nonpositive_weights": 0,
        "below_range_cells": 0,
        "above_range_cells": 0,
        "accepted_cells": 0,
        "accepted_weight": 0.0,
        "below_range_weight": 0.0,
        "above_range_weight": 0.0,
    }


def merge_diagnostics(total, block):

    for key in (
        "input_cells",
        "nonfinite_values",
        "nonfinite_weights",
        "nonpositive_weights",
        "below_range_cells",
        "above_range_cells",
        "accepted_cells",
    ):
        total[key] += int(block[key])

    for key in (
        "accepted_weight",
        "below_range_weight",
        "above_range_weight",
    ):
        total[key] += float(block[key])


def finalize_diagnostics(d):

    eligible_cells = (
        d["accepted_cells"]
        + d["below_range_cells"]
        + d["above_range_cells"]
    )

    eligible_weight = (
        d["accepted_weight"]
        + d["below_range_weight"]
        + d["above_range_weight"]
    )

    excluded_weight = (
        d["below_range_weight"]
        + d["above_range_weight"]
    )

    result = dict(d)

    result["eligible_cells"] = int(eligible_cells)
    result["eligible_weight"] = float(eligible_weight)
    result["excluded_histogram_weight"] = float(excluded_weight)

    result["cell_coverage_percent"] = (
        100.0 * d["accepted_cells"] / eligible_cells
        if eligible_cells else None
    )

    result["weighted_coverage_percent"] = (
        100.0 * d["accepted_weight"] / eligible_weight
        if eligible_weight else None
    )

    return result


def new_extrema():
    return {
        "valid_source_cells": 0,
        "minimum": None,
        "maximum": None,
        "zero_cells": 0,
    }


def record_extrema(total, values, valid_mask=None):

    array = np.asarray(values)
    valid = np.isfinite(array)

    if valid_mask is not None:
        valid &= np.asarray(valid_mask, dtype=bool)

    selected = array[valid]

    if selected.size == 0:
        return

    block_min = float(selected.min())
    block_max = float(selected.max())

    total["valid_source_cells"] += int(selected.size)
    total["zero_cells"] += int(np.count_nonzero(selected == 0))

    if total["minimum"] is None:
        total["minimum"] = block_min
        total["maximum"] = block_max
    else:
        total["minimum"] = min(total["minimum"], block_min)
        total["maximum"] = max(total["maximum"], block_max)


# ============================================================
# 3. LATITUDE WEIGHTING
# ============================================================

def latitude_weights(first_row, number_of_rows, resolution_degrees):

    row_indices = np.arange(
        first_row,
        first_row + number_of_rows,
        dtype=np.float64,
    )

    latitude_centres = (
        90.0 - (row_indices + 0.5) * resolution_degrees
    )

    return np.clip(
        np.cos(np.deg2rad(latitude_centres)),
        0.0,
        None,
    )


# ============================================================
# 4. HISTOGRAM ENGINE
# ============================================================

def add_to_histogram(histogram, values, weights):

    values = np.asarray(values, dtype=np.float64).ravel()
    weights = np.asarray(weights, dtype=np.float64).ravel()

    if values.size != weights.size:
        raise ValueError(
            "Values and weights have different sizes."
        )

    finite_values = np.isfinite(values)
    finite_weights = np.isfinite(weights)

    usable = (
        finite_values
        & finite_weights
        & (weights > 0)
    )

    below = usable & (values < ELEVATION_MIN)

    above = usable & (values >= ELEVATION_MAX)

    accepted = (
        usable
        & (values >= ELEVATION_MIN)
        & (values < ELEVATION_MAX)
    )

    if accepted.any():

        block_histogram, _ = np.histogram(
            values[accepted],
            bins=ELEVATION_EDGES,
            weights=weights[accepted],
        )

        histogram += block_histogram

    return {
        "input_cells": int(values.size),
        "nonfinite_values": int((~finite_values).sum()),
        "nonfinite_weights": int((~finite_weights).sum()),
        "nonpositive_weights": int(
            (finite_weights & (weights <= 0)).sum()
        ),
        "below_range_cells": int(below.sum()),
        "above_range_cells": int(above.sum()),
        "accepted_cells": int(accepted.sum()),
        "accepted_weight": float(weights[accepted].sum()),
        "below_range_weight": float(weights[below].sum()),
        "above_range_weight": float(weights[above].sum()),
    }


# ============================================================
# 5. WEIGHTED STATISTICS
# ============================================================

def calculate_weighted_statistics(histogram):

    histogram = np.asarray(histogram, dtype=np.float64)

    if histogram.size != len(ELEVATION_CENTRES):
        raise ValueError("Invalid histogram dimensions.")

    if np.any(~np.isfinite(histogram)) or np.any(histogram < 0):
        raise ValueError("Invalid histogram weights.")

    total = float(histogram.sum())

    if total <= 0:
        raise ValueError("No valid histogram observations.")

    probabilities = histogram / total

    mean = float(
        np.sum(probabilities * ELEVATION_CENTRES)
    )

    variance = float(
        np.sum(
            probabilities
            * (ELEVATION_CENTRES - mean) ** 2
        )
    )

    standard_deviation = float(
        np.sqrt(max(variance, 0.0))
    )

    cumulative = np.cumsum(histogram)

    def percentile(p):

        target = total * p / 100.0

        index = int(
            np.searchsorted(cumulative, target, side="left")
        )

        index = min(index, len(histogram) - 1)

        previous = (
            float(cumulative[index - 1])
            if index > 0 else 0.0
        )

        bin_weight = float(histogram[index])

        fraction = (
            (target - previous) / bin_weight
            if bin_weight > 0 else 0.0
        )

        fraction = float(np.clip(fraction, 0.0, 1.0))

        return float(
            ELEVATION_EDGES[index]
            + fraction * ELEVATION_BIN_SIZE
        )

    percentiles = {
        f"P{p:02d}": percentile(p)
        for p in (5, 10, 25, 50, 75, 90, 95)
    }

    return {
        "weighted_mean_m": mean,
        "weighted_standard_deviation_m": standard_deviation,
        "weighted_percentiles_m": percentiles,
        "weighted_histogram_total": total,
        "histogram_bin_width_m": ELEVATION_BIN_SIZE,
        "percentile_method": (
            "Linear interpolation within 10 m histogram bins, "
            "assuming uniform within-bin weight distribution."
        ),
        "statistics_scope": (
            "Finite positive-weight observations inside histogram bounds."
        ),
    }


# ============================================================
# 6. NORMALIZED DISTRIBUTIONS
# ============================================================

def make_normalized_histogram(histogram, mean, standard_deviation):

    if (
        standard_deviation <= 0
        or not np.isfinite(standard_deviation)
    ):
        raise ValueError("Invalid standard deviation.")

    regular_edges = np.linspace(-6.0, 6.0, 241)

    # Infinite edges retain all values outside +/-6.
    edges = np.concatenate((
        [-np.inf],
        regular_edges,
        [np.inf],
    ))

    regular_centres = (
        regular_edges[:-1] + regular_edges[1:]
    ) / 2.0

    centres = np.concatenate((
        [-6.025],
        regular_centres,
        [6.025],
    ))

    z_values = (
        ELEVATION_CENTRES - mean
    ) / standard_deviation

    normalized, _ = np.histogram(
        z_values,
        bins=edges,
        weights=histogram,
    )

    total = float(histogram.sum())

    if total <= 0:
        raise ValueError("Cannot normalize empty histogram.")

    normalized = normalized.astype(np.float64) / total

    return centres, normalized


# ============================================================
# 7. EARTH PROCESSOR
# ============================================================

def process_earth():

    print("\n[1/3] Processing Earth...")

    if not EARTH_FILE.exists():
        raise FileNotFoundError(EARTH_FILE)

    histogram = np.zeros(
        len(ELEVATION_EDGES) - 1,
        dtype=np.float64,
    )

    diagnostics = new_diagnostics()
    extrema = new_extrema()

    accepted_land_cells = 0
    total_coarse_cells = 0
    positive_source_cells = 0

    with rasterio.open(EARTH_FILE) as src:

        if src.width != 21600 or src.height != 10800:
            raise ValueError(
                f"Unexpected ETOPO dimensions: "
                f"{src.width} x {src.height}"
            )

        output_width = src.width // EARTH_AGGREGATION
        output_height = src.height // EARTH_AGGREGATION

        if (
            src.width % EARTH_AGGREGATION
            or src.height % EARTH_AGGREGATION
        ):
            raise ValueError("Earth dimensions cannot be aggregated evenly.")

        for output_start in range(
            0,
            output_height,
            EARTH_BLOCK_ROWS,
        ):

            output_count = min(
                EARTH_BLOCK_ROWS,
                output_height - output_start,
            )

            source_start = output_start * EARTH_AGGREGATION
            source_count = output_count * EARTH_AGGREGATION

            masked = src.read(
                1,
                window=Window(
                    0,
                    source_start,
                    src.width,
                    source_count,
                ),
                masked=True,
            )

            raw = np.asarray(masked.data, dtype=np.float32)

            valid = (
                np.isfinite(raw)
                & (~np.ma.getmaskarray(masked))
            )

            record_extrema(extrema, raw, valid)

            positive = valid & (raw > 0)

            positive_source_cells += int(positive.sum())

            grouped_values = raw.reshape(
                output_count,
                2,
                output_width,
                2,
            )

            grouped_positive = positive.reshape(
                output_count,
                2,
                output_width,
                2,
            )

            positive_count = grouped_positive.sum(axis=(1, 3))

            positive_sum = np.where(
                grouped_positive,
                grouped_values,
                0.0,
            ).sum(axis=(1, 3), dtype=np.float64)

            keep = positive_count >= 2

            coarse = np.zeros(
                positive_count.shape,
                dtype=np.float32,
            )

            np.divide(
                positive_sum,
                positive_count,
                out=coarse,
                where=keep,
            )

            row_weights = latitude_weights(
                output_start,
                output_count,
                resolution_degrees=2.0 / 60.0,
            )

            weights = np.broadcast_to(
                row_weights[:, None],
                coarse.shape,
            )

            block_diag = add_to_histogram(
                histogram,
                coarse[keep],
                weights[keep],
            )

            merge_diagnostics(diagnostics, block_diag)

            accepted_land_cells += int(keep.sum())
            total_coarse_cells += int(keep.size)

            if output_start % 512 == 0:
                print(
                    f"  Earth: {output_start:,}/{output_height:,} rows"
                )

    statistics = calculate_weighted_statistics(histogram)

    diagnostics = finalize_diagnostics(diagnostics)

    print("  Earth complete.")

    return {
        "planet": "Earth",
        "dataset": "NOAA ETOPO 2022, 60 arc-second",
        "input_file": str(EARTH_FILE.relative_to(ROOT)),
        "native_resolution_degrees": 1 / 60,
        "comparison_resolution_degrees": 2 / 60,
        "approximate_equatorial_spacing_km": 3.7,
        "surface_scope": "Elevation-positive land proxy",
        "land_proxy_rule": (
            "At least 2 of 4 native cells must have valid elevation "
            "> 0 m. Mean uses positive cells only."
        ),
        "accepted_land_proxy_cells": accepted_land_cells,
        "total_coarse_grid_cells": total_coarse_cells,
        "land_proxy_cell_fraction": (
            accepted_land_cells / total_coarse_cells
            if total_coarse_cells else None
        ),
        "positive_valid_native_source_cells": positive_source_cells,
        "source_raster_diagnostics": extrema,
        "histogram_diagnostics": diagnostics,
        "vertical_reference": "WGS84 + EGM2008 geoid",
        "statistics": statistics,
        "limitations": [
            "Elevation > 0 m is not an authoritative land mask.",
            "Below-sea-level terrestrial regions may be excluded.",
            "Earth represents an elevation-positive land proxy, not the full surface.",
            "Histogram coverage and exclusions are reported separately.",
            "Latitude weighting uses cos(latitude).",
        ],
        "histogram": histogram,
    }


# ============================================================
# 8. MARS PROCESSOR
# ============================================================

def process_mars():

    print("\n[2/3] Processing Mars...")

    if not MARS_FILE.exists():
        raise FileNotFoundError(MARS_FILE)

    rows, columns = 2880, 5760

    raw = np.fromfile(MARS_FILE, dtype=">i2")

    if raw.size != rows * columns:
        raise ValueError(
            f"Mars size mismatch: {raw.size:,} cells."
        )

    raw_grid = raw.reshape(rows, columns)

    elevation = raw_grid.astype(np.float32)

    extrema = new_extrema()
    record_extrema(extrema, raw_grid)

    histogram = np.zeros(
        len(ELEVATION_EDGES) - 1,
        dtype=np.float64,
    )

    diagnostics = new_diagnostics()

    for start in range(0, rows, PLANET_BLOCK_ROWS):

        end = min(start + PLANET_BLOCK_ROWS, rows)

        block = elevation[start:end]

        row_weights = latitude_weights(
            start,
            end - start,
            MARS_RESOLUTION,
        )

        weights = np.broadcast_to(
            row_weights[:, None],
            block.shape,
        )

        block_diag = add_to_histogram(
            histogram,
            block,
            weights,
        )

        merge_diagnostics(diagnostics, block_diag)

        if start % 512 == 0:
            print(f"  Mars: {start:,}/{rows:,} rows")

    statistics = calculate_weighted_statistics(histogram)
    diagnostics = finalize_diagnostics(diagnostics)

    raw_min = int(raw_grid.min())
    raw_max = int(raw_grid.max())

    print("  Mars complete.")

    return {
        "planet": "Mars",
        "dataset": "MOLA MEGDR MEGT90N000EB",
        "input_file": str(MARS_FILE.relative_to(ROOT)),
        "native_resolution_degrees": MARS_RESOLUTION,
        "comparison_resolution_degrees": MARS_RESOLUTION,
        "approximate_equatorial_spacing_km": 3.704,
        "surface_scope": "Full gridded planetary surface",
        "aggregation": "Native grid retained",
        "grid_dimensions": {
            "rows": rows,
            "columns": columns,
        },
        "source_raster_diagnostics": {
            **extrema,
            "raw_integer_minimum": raw_min,
            "raw_integer_maximum": raw_max,
            "pds_label_minimum": MARS_LABEL_MIN,
            "pds_label_maximum": MARS_LABEL_MAX,
            "raw_extrema_match_pds_label": (
                raw_min == MARS_LABEL_MIN
                and raw_max == MARS_LABEL_MAX
            ),
        },
        "histogram_diagnostics": diagnostics,
        "vertical_reference": "MOLA areoid",
        "data_interpretation": (
            "Median observed topography where measurements exist; "
            "interpolated values are also present."
        ),
        "zero_value_handling": (
            "Zero values retained pending reconciliation between "
            "PDS interpolation documentation and ENVI ignore metadata."
        ),
        "statistics": statistics,
        "limitations": [
            "MOLA measurement coverage is spatially uneven.",
            "Interpolated cells are included.",
            "Zero-value metadata requires further reconciliation.",
            "Histogram exclusions are reported in diagnostics.",
            "Latitude weighting uses cos(latitude).",
        ],
        "histogram": histogram,
    }


# ============================================================
# 9. MOON PROCESSOR
# ============================================================

def process_moon():

    print("\n[3/3] Processing Moon...")

    if not MOON_FILE.exists():
        raise FileNotFoundError(MOON_FILE)

    rows, columns = 2880, 5760

    raw = np.fromfile(MOON_FILE, dtype="<i2")

    if raw.size != rows * columns:
        raise ValueError(
            f"Moon size mismatch: {raw.size:,} cells."
        )

    raw_grid = raw.reshape(rows, columns)

    extrema = new_extrema()
    record_extrema(extrema, raw_grid)

    raw_min = int(raw_grid.min())
    raw_max = int(raw_grid.max())

    elevation = raw_grid.astype(np.float32) * 0.5

    coarse = elevation.reshape(
        rows // 2,
        2,
        columns // 2,
        2,
    ).mean(axis=(1, 3))

    coarse_rows, coarse_columns = coarse.shape

    histogram = np.zeros(
        len(ELEVATION_EDGES) - 1,
        dtype=np.float64,
    )

    diagnostics = new_diagnostics()

    comparison_resolution = (
        MOON_RESOLUTION * MOON_AGGREGATION
    )

    for start in range(
        0,
        coarse_rows,
        PLANET_BLOCK_ROWS,
    ):

        end = min(start + PLANET_BLOCK_ROWS, coarse_rows)

        block = coarse[start:end]

        row_weights = latitude_weights(
            start,
            end - start,
            comparison_resolution,
        )

        weights = np.broadcast_to(
            row_weights[:, None],
            block.shape,
        )

        block_diag = add_to_histogram(
            histogram,
            block,
            weights,
        )

        merge_diagnostics(diagnostics, block_diag)

        if start % 256 == 0:
            print(
                f"  Moon: {start:,}/{coarse_rows:,} rows"
            )

    statistics = calculate_weighted_statistics(histogram)
    diagnostics = finalize_diagnostics(diagnostics)

    print("  Moon complete.")

    return {
        "planet": "Moon",
        "dataset": "NASA LOLA LDEM_16",
        "input_file": str(MOON_FILE.relative_to(ROOT)),
        "native_resolution_degrees": MOON_RESOLUTION,
        "comparison_resolution_degrees": comparison_resolution,
        "approximate_equatorial_spacing_km": 3.79,
        "surface_scope": "Full gridded lunar surface",
        "aggregation": "2 x 2 native cells averaged",
        "grid_dimensions": {
            "rows": coarse_rows,
            "columns": coarse_columns,
        },
        "source_raster_diagnostics": {
            **extrema,
            "raw_integer_minimum": raw_min,
            "raw_integer_maximum": raw_max,
            "physical_minimum_m": raw_min * 0.5,
            "physical_maximum_m": raw_max * 0.5,
            "pds_label_minimum": MOON_LABEL_MIN,
            "pds_label_maximum": MOON_LABEL_MAX,
            "raw_extrema_match_pds_label": (
                raw_min == MOON_LABEL_MIN
                and raw_max == MOON_LABEL_MAX
            ),
            "raw_extrema_within_pds_label_range": (
                raw_min >= MOON_LABEL_MIN
                and raw_max <= MOON_LABEL_MAX
            ),
        },
        "histogram_diagnostics": diagnostics,
        "vertical_reference": "Reference sphere, radius 1,737.4 km",
        "height_conversion": "Elevation metres = DN * 0.5",
        "zero_value_handling": "Zero values retained.",
        "statistics": statistics,
        "limitations": [
            "LOLA interpolation compensates for incomplete altimetry coverage.",
            "Documented edge artefacts occur near some 45-degree latitude bands.",
            "Vertical reference is a reference sphere, not a geoid.",
            "Observed raw extrema differ from label extrema but remain within the declared range.",
            "Distribution represents aggregated 2 x 2 cells, not native-resolution cells.",
            "Histogram exclusions are reported in diagnostics.",
            "Latitude weighting uses cos(latitude).",
        ],
        "histogram": histogram,
    }


# ============================================================
# 10. ELEVATION DISTRIBUTION CHART
# ============================================================

def create_elevation_distribution_chart(results):

    fig, ax = plt.subplots(figsize=(13, 7))

    for name in ("Earth", "Mars", "Moon"):

        item = results[name]
        histogram = item["histogram"]

        total = float(histogram.sum())

        if total <= 0:
            continue

        density = (
            histogram
            / total
            / ELEVATION_BIN_SIZE
        )

        coverage = item["histogram_diagnostics"][
            "weighted_coverage_percent"
        ]

        label = (
            f"{name} — coverage {coverage:.3f}%"
            if coverage is not None else name
        )

        ax.plot(
            ELEVATION_CENTRES,
            density,
            linewidth=2,
            label=label,
        )

    ax.set_title(
        "Global Planetary Elevation Distributions",
        fontsize=17,
        pad=18,
    )

    ax.set_xlabel(
        "Elevation relative to native vertical reference (m)"
    )

    ax.set_ylabel(
        "Relative frequency per metre (within histogram range)"
    )

    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    fig.text(
        0.5,
        0.015,
        "Earth: elevation-positive land proxy | "
        "Mars and Moon: full gridded surfaces | "
        "Native vertical references differ",
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])

    output = (
        VISUAL_DIR
        / "global_elevation_distribution_comparison.png"
    )

    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return output


# ============================================================
# 11. STANDARDIZED DISTRIBUTION CHART
# ============================================================

def create_normalized_distribution_chart(results):

    fig, ax = plt.subplots(figsize=(13, 7))

    for name in ("Earth", "Mars", "Moon"):

        item = results[name]
        stats = item["statistics"]

        centres, normalized = make_normalized_histogram(
            item["histogram"],
            stats["weighted_mean_m"],
            stats["weighted_standard_deviation_m"],
        )

        ax.plot(
            centres,
            normalized,
            linewidth=2,
            label=name,
        )

    ax.set_title(
        "Standardized Global Terrain Distributions",
        fontsize=17,
        pad=18,
    )

    ax.set_xlabel(
        "Within-body standardized elevation (z-score)"
    )

    ax.set_ylabel(
        "Latitude-weighted relative frequency per bin"
    )

    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    fig.text(
        0.5,
        0.015,
        "Each planet is standardized using its own weighted mean and "
        "standard deviation. Tail bins retain values beyond +/-6. "
        "Shape similarity does not establish geological equivalence.",
        ha="center",
        fontsize=8,
    )

    fig.tight_layout(rect=[0, 0.06, 1, 1])

    output = (
        VISUAL_DIR
        / "global_normalized_terrain_comparison.png"
    )

    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return output


# ============================================================
# 12. JSON CLEANING
# ============================================================

def clean_for_json(value):

    if isinstance(value, dict):
        return {
            key: clean_for_json(item)
            for key, item in value.items()
            if key != "histogram"
        }

    if isinstance(value, (list, tuple)):
        return [clean_for_json(item) for item in value]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    return value


# ============================================================
# 13. MAIN PIPELINE
# ============================================================

def main():

    print("=" * 68)
    print("TERRAFORM — GLOBAL PLANETARY COMPARISON ENGINE")
    print(f"Version {VERSION}")
    print("=" * 68)

    start_clock = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()

    earth = process_earth()
    mars = process_mars()
    moon = process_moon()

    results = {
        "Earth": earth,
        "Mars": mars,
        "Moon": moon,
    }

    print("\nGenerating visualisations...")

    elevation_chart = create_elevation_distribution_chart(results)

    normalized_chart = create_normalized_distribution_chart(results)

    duration = time.perf_counter() - start_clock

    report = {
        "project": "TERRAFORM — Planetary Analog Intelligence",
        "analysis": "Global Planetary Terrain Distribution Comparison",
        "version": VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "processing_started_utc": started_utc,
        "processing_duration_seconds": float(duration),

        "method": {
            "comparison_type": "Global weighted distribution comparison",
            "pixel_to_pixel_matching": False,

            "area_weighting": (
                "Approximate spherical latitude weighting using cos(latitude)."
            ),

            "histogram_configuration": {
                "minimum_m_inclusive": ELEVATION_MIN,
                "maximum_m_exclusive": ELEVATION_MAX,
                "bin_width_m": ELEVATION_BIN_SIZE,
            },

            "histogram_coverage": (
                "Finite observations with positive latitude weights are "
                "classified as accepted, below range or above range."
            ),

            "earth_resolution_protocol": (
                "Aggregate 2 x 2 native cells from 1 arc-minute "
                "to 2 arc-minutes."
            ),

            "mars_resolution_protocol": (
                "Retain native 0.0625-degree grid."
            ),

            "moon_resolution_protocol": (
                "Aggregate 2 x 2 native cells from 0.0625 "
                "to 0.125 degrees."
            ),

            "resolution_note": (
                "Selected grids have approximately comparable north-south "
                "equatorial spacing of 3.7–3.8 km. East-west spacing varies "
                "with latitude."
            ),

            "normalization": (
                "Within-body z-scores based on each dataset's own weighted "
                "mean and standard deviation. Explicit tail bins retain "
                "values outside +/-6 standard deviations."
            ),

            "statistics_precision": (
                "Moments use histogram bin centres. Percentiles are "
                "interpolated within 10 m bins under a uniform "
                "within-bin assumption."
            ),

            "weighted_total_interpretation": (
                "Weighted histogram totals are sums of relative "
                "cos(latitude) cell weights, not pixel counts or "
                "physical areas in square kilometres."
            ),
        },

        "planetary_results": {
            name: {
                key: value
                for key, value in item.items()
                if key != "histogram"
            }
            for name, item in results.items()
        },

        "visualisations": {
            "elevation_distributions": str(
                elevation_chart.relative_to(ROOT)
            ),
            "normalized_distributions": str(
                normalized_chart.relative_to(ROOT)
            ),
        },

        "scientific_interpretation": [
            "The comparison describes global elevation-distribution patterns.",
            "Earth is represented by a preliminary elevation-positive land proxy.",
            "Mars and Moon represent their full gridded surfaces.",
            "Histogram coverage diagnostics identify excluded observations.",
            "Planetary vertical references are different and not directly interchangeable.",
            "Standardized distributions compare statistical shape, not geological equivalence.",
            "Moon statistics represent aggregated 2 x 2 LOLA cells.",
            "Mars interpolated cells are included.",
            "Mars zero-value metadata requires further reconciliation.",
            "The analysis does not establish habitability, resource potential or mission-site suitability.",
            "Global screening does not replace higher-resolution regional analysis.",
        ],
    }

    report_path = (
        OUTPUT_DIR / "global_planetary_comparison_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            clean_for_json(report),
            file,
            indent=4,
            ensure_ascii=False,
            allow_nan=False,
        )

    print("\n" + "=" * 68)
    print("GLOBAL COMPARISON COMPLETED")
    print("=" * 68)

    for name, item in results.items():

        stats = item["statistics"]
        diag = item["histogram_diagnostics"]

        print(f"\n{name}")

        print(
            f"  Weighted mean: "
            f"{stats['weighted_mean_m']:,.2f} m"
        )

        print(
            f"  Weighted standard deviation: "
            f"{stats['weighted_standard_deviation_m']:,.2f} m"
        )

        print(
            f"  Weighted median: "
            f"{stats['weighted_percentiles_m']['P50']:,.2f} m"
        )

        print(
            f"  Accepted histogram cells: "
            f"{diag['accepted_cells']:,}"
        )

        print(
            f"  Below histogram range: "
            f"{diag['below_range_cells']:,}"
        )

        print(
            f"  Above histogram range: "
            f"{diag['above_range_cells']:,}"
        )

        print(
            f"  Weighted histogram coverage: "
            f"{diag['weighted_coverage_percent']:.6f}%"
        )

        print(
            f"  Weighted histogram total: "
            f"{stats['weighted_histogram_total']:,.4f}"
        )

    print("\nSaved outputs:")
    print(f"  Report: {report_path}")
    print(f"  Elevation chart: {elevation_chart}")
    print(f"  Normalized chart: {normalized_chart}")

    print(f"\nProcessing duration: {duration:.2f} seconds")

    print(
        "\nRemember: distribution similarity is not site suitability."
    )


if __name__ == "__main__":
    main()
