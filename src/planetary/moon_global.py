
"""
TERRAFORM — Moon Global Intelligence
Version 1.1

Dataset: NASA LRO LOLA LDEM_16
Source: NASA Planetary Data System (PDS)
"""

from pathlib import Path
from datetime import datetime, timezone
import json

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# 1. PROJECT PATHS
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = ROOT / "data" / "Global" / "Moon" / "ldem_16.img"

OUTPUT_DIR = ROOT / "outputs" / "Global" / "Moon"
VISUAL_DIR = ROOT / "outputs" / "Visualisations" / "Global"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VISUAL_DIR.mkdir(parents=True, exist_ok=True)

REPORT_FILE = OUTPUT_DIR / "moon_global_report.json"
MAP_FILE = VISUAL_DIR / "moon_global_elevation.png"


# ---------------------------------------------------------
# 2. VERIFIED PDS METADATA
# ---------------------------------------------------------

ROWS = 2880
COLS = 5760

EXPECTED_CELLS = ROWS * COLS
EXPECTED_BYTES = EXPECTED_CELLS * 2

SCALING_FACTOR = 0.5
REFERENCE_RADIUS_M = 1_737_400.0

PIXEL_SIZE_DEGREES = 0.0625
EQUATORIAL_PIXEL_SIZE_M = 1895.21

LABEL_RAW_MIN = -18150
LABEL_RAW_MAX = 21580

SOURCE = (
    "NASA Planetary Data System — "
    "Lunar Reconnaissance Orbiter LOLA LDEM_16"
)


# ---------------------------------------------------------
# 3. VALIDATE AND DECODE DATA
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("TERRAFORM | MOON GLOBAL INTELLIGENCE")
print("=" * 60)

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Moon dataset not found:\n{DATA_FILE}"
    )

actual_bytes = DATA_FILE.stat().st_size

if actual_bytes != EXPECTED_BYTES:
    raise ValueError(
        f"Unexpected file size. Expected {EXPECTED_BYTES:,} bytes, "
        f"received {actual_bytes:,} bytes."
    )

print("\n[1/6] Dataset file validated.")

# PDS label:
# SAMPLE_TYPE = LSB_INTEGER
# SAMPLE_BITS = 16

raw = np.fromfile(DATA_FILE, dtype="<i2")

if raw.size != EXPECTED_CELLS:
    raise ValueError(
        f"Unexpected number of cells: {raw.size:,}"
    )

raw_min = int(raw.min())
raw_max = int(raw.max())

print(f"      Dimensions: {COLS:,} × {ROWS:,}")
print(f"      Total cells: {raw.size:,}")
print(f"      Observed raw minimum: {raw_min:,}")
print(f"      Observed raw maximum: {raw_max:,}")

# Compare observed extrema with label extrema.
# A discrepancy is recorded, not silently ignored.

range_matches_label = (
    raw_min == LABEL_RAW_MIN
    and raw_max == LABEL_RAW_MAX
)

if range_matches_label:
    print("      PDS extrema comparison: PASS")
else:
    print("\n      WARNING: PDS extrema discrepancy detected.")
    print(
        f"      Label range: {LABEL_RAW_MIN:,} to {LABEL_RAW_MAX:,}"
    )
    print(
        f"      Observed range: {raw_min:,} to {raw_max:,}"
    )
    print("      Continuing with the observed raster values.")
    print("      Discrepancy will be documented in the report.")

# Convert digital numbers into height relative to the
# reference sphere using the PDS scaling factor.
height = raw.reshape((ROWS, COLS)).astype(np.float32)
height *= SCALING_FACTOR

print("      Byte order: little-endian")
print("      Data type: signed 16-bit integer")
print("      Scaling factor: 0.5 m per DN")
print("      Reference radius: 1,737,400 m")


# ---------------------------------------------------------
# 4. LONGITUDE ORIENTATION
# ---------------------------------------------------------

# Source longitude range: 0° to 360° east.
# Shift half the image width for a -180° to +180° display.

height = np.roll(
    height,
    shift=-(COLS // 2),
    axis=1
)

print("\n[2/6] Longitude orientation adjusted.")
print("      Display longitude: -180° to +180°")
print("      Latitude range: -90° to +90°")


# ---------------------------------------------------------
# 5. GLOBAL ELEVATION STATISTICS
# ---------------------------------------------------------

# Preserve all cells. The supplied PDS label does not
# define a missing-data sentinel for this raster.

values = height.ravel()

statistics = {
    "minimum_m": float(np.min(values)),
    "maximum_m": float(np.max(values)),
    "mean_m": float(np.mean(values)),
    "median_m": float(np.median(values)),
    "standard_deviation_m": float(np.std(values)),
    "percentiles_m": {
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
    },
    "zero_height_cells": int(np.count_nonzero(values == 0)),
}

print("\n[3/6] Elevation statistics calculated.")

for key, value in statistics.items():
    if isinstance(value, dict):
        continue
    print(f"      {key}: {value:,.2f}")


# ---------------------------------------------------------
# 6. GENERATE GLOBAL ELEVATION MAP
# ---------------------------------------------------------

print("\n[4/6] Generating lunar elevation visualisation...")

# Downsampling is only for display.
# Original decoded data remains unchanged.

display = height[::3, ::3]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
})

fig, ax = plt.subplots(figsize=(16, 8.5))

fig.patch.set_facecolor("white")
ax.set_facecolor("white")

image = ax.imshow(
    display,
    cmap="terrain",
    origin="upper",
    extent=(-180, 180, -90, 90),
    aspect="auto",
    interpolation="nearest",
    vmin=statistics["minimum_m"],
    vmax=statistics["maximum_m"],
)

ax.set_title(
    "MOON | GLOBAL ELEVATION MODEL",
    fontsize=19,
    fontweight="bold",
    loc="left",
    pad=18,
)

ax.set_xlabel("Longitude (°)", labelpad=10)
ax.set_ylabel("Latitude (°)", labelpad=10)

ax.set_xticks(np.arange(-180, 181, 30))
ax.set_yticks(np.arange(-90, 91, 30))

ax.grid(
    True,
    color="#777777",
    linewidth=0.35,
    alpha=0.35,
)

for spine in ax.spines.values():
    spine.set_color("#333333")
    spine.set_linewidth(0.7)

colorbar = fig.colorbar(
    image,
    ax=ax,
    orientation="horizontal",
    pad=0.12,
    fraction=0.045,
    aspect=45,
)

colorbar.set_label(
    "Height relative to 1,737.4 km reference sphere (metres)",
    labelpad=8,
)

fig.text(
    0.5,
    0.025,
    "Source: NASA PDS | LRO LOLA | LDEM_16 | "
    "0.0625° global grid",
    ha="center",
    fontsize=8,
    color="#555555",
)

fig.subplots_adjust(
    left=0.07,
    right=0.97,
    top=0.88,
    bottom=0.17,
)

fig.savefig(
    MAP_FILE,
    dpi=220,
    bbox_inches="tight",
    facecolor="white",
)

plt.close(fig)

print(f"      Map saved: {MAP_FILE}")


# ---------------------------------------------------------
# 7. SAVE MACHINE-READABLE REPORT
# ---------------------------------------------------------

print("\n[5/6] Creating machine-readable report...")

report = {
    "project": "TERRAFORM — Planetary Analog Intelligence",
    "analysis": "Moon Global Elevation Analysis",
    "version": "1.1",
    "generated_utc": datetime.now(timezone.utc).isoformat(),

    "dataset": {
        "product": "LOLA Global Digital Elevation Model",
        "product_id": "LDEM_16",
        "data_set_id": "LRO-L-LOLA-4-GDR-V1.0",
        "product_version": "V3.1",
        "source": SOURCE,
        "filename": DATA_FILE.name,

        "dimensions": {
            "columns": COLS,
            "rows": ROWS,
            "bands": 1,
            "total_cells": EXPECTED_CELLS,
        },

        "data_type": "Signed 16-bit integer",
        "byte_order": "Little-endian",
        "scaling_factor": SCALING_FACTOR,
        "elevation_unit": "metres",
        "reference_radius_m": REFERENCE_RADIUS_M,

        "pixel_resolution_degrees": PIXEL_SIZE_DEGREES,
        "equatorial_pixel_scale_m": EQUATORIAL_PIXEL_SIZE_M,

        "projection": "Simple cylindrical",
        "central_meridian_degrees": 180,
        "longitude_direction": "East-positive",
        "coordinate_system": "Mean Earth / Polar Axis of DE421",
    },

    "raw_data_validation": {
        "observed_raw_minimum": raw_min,
        "observed_raw_maximum": raw_max,
        "label_raw_minimum": LABEL_RAW_MIN,
        "label_raw_maximum": LABEL_RAW_MAX,
        "range_matches_label": range_matches_label,
        "range_discrepancy": None if range_matches_label else {
            "minimum_difference": raw_min - LABEL_RAW_MIN,
            "maximum_difference": raw_max - LABEL_RAW_MAX,
            "status": "UNRESOLVED_METADATA_DISCREPANCY",
        },
        "file_size_validation": "PASS",
        "cell_count_validation": "PASS",
    },

    "elevation_statistics": statistics,

    "processing": {
        "conversion": "height_m = digital_number * 0.5",
        "longitude_display_range": [-180, 180],
        "latitude_display_range": [-90, 90],
        "zero_height_cells_retained": True,
        "nodata_mask_applied": False,
        "display_downsampling": "Every third row and column",
        "source_data_modified": False,
    },

    "scientific_limitations": [
        "Heights are relative to a 1,737.4 km reference sphere.",
        "These heights are not automatically equivalent to "
        "elevation relative to a lunar geoid or equipotential surface.",
        "The product description reports possible edge artifacts "
        "at 45-degree latitude-band boundaries.",
        "The source describes interpolation to compensate for "
        "incomplete altimetry coverage.",
        "Global statistics are unweighted grid-cell statistics "
        "and are not area-weighted.",
        "The observed raw extrema differ from the supplied PDS label "
        "extrema and require further investigation.",
        "Elevation alone does not establish geological, "
        "environmental or habitation suitability.",
    ],

    "outputs": {
        "map": str(MAP_FILE.relative_to(ROOT)),
        "report": str(REPORT_FILE.relative_to(ROOT)),
    },
}

with REPORT_FILE.open("w", encoding="utf-8") as file:
    json.dump(report, file, indent=4)

print(f"      Report saved: {REPORT_FILE}")


# ---------------------------------------------------------
# 8. COMPLETION SUMMARY
# ---------------------------------------------------------

print("\n[6/6] Processing complete.")

print("\n" + "=" * 60)
print("MOON GLOBAL ANALYSIS COMPLETED")
print("=" * 60)

print(
    f"\nHeight range: "
    f"{statistics['minimum_m']:,.1f} m to "
    f"{statistics['maximum_m']:,.1f} m"
)

print(f"Mean height: {statistics['mean_m']:,.2f} m")
print(f"Median height: {statistics['median_m']:,.2f} m")

print("\nRange comparison:", 
      "PASS" if range_matches_label else "WARNING — REVIEW REQUIRED")

print("\nGenerated files:")
print(f"1. {MAP_FILE}")
print(f"2. {REPORT_FILE}")

print("\nTERRAFORM | End of processing")
