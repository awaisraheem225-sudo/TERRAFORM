
"""
TERRAFORM — Mars Global Intelligence
Version 1.0

Dataset: MGS MOLA MEGDR median topography
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

DATA_FILE = ROOT / "data" / "Global" / "Mars" / "megt90n000eb.img"

OUTPUT_DIR = ROOT / "outputs" / "Global" / "Mars"
VISUAL_DIR = ROOT / "outputs" / "Visualisations" / "Global"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VISUAL_DIR.mkdir(parents=True, exist_ok=True)

REPORT_FILE = OUTPUT_DIR / "mars_global_report.json"
MAP_FILE = VISUAL_DIR / "mars_global_elevation.png"


# ---------------------------------------------------------
# 2. VERIFIED DATASET METADATA
# ---------------------------------------------------------

ROWS = 2880
COLS = 5760
EXPECTED_CELLS = ROWS * COLS
EXPECTED_BYTES = EXPECTED_CELLS * 2

PIXEL_SIZE_DEGREES = 0.0625
EQUATORIAL_PIXEL_SIZE_KM = 3.7044613374

SOURCE = (
    "NASA Planetary Data System — "
    "Mars Global Surveyor MOLA MEGDR, MEGT90N000EB"
)


# ---------------------------------------------------------
# 3. VALIDATE AND LOAD RAW DATA
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("TERRAFORM | MARS GLOBAL INTELLIGENCE")
print("=" * 60)

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Mars dataset not found:\n{DATA_FILE}"
    )

actual_bytes = DATA_FILE.stat().st_size

if actual_bytes != EXPECTED_BYTES:
    raise ValueError(
        f"Unexpected file size. Expected {EXPECTED_BYTES:,} bytes, "
        f"received {actual_bytes:,} bytes."
    )

print("\n[1/5] Dataset file validated.")

# PDS label confirms signed 16-bit big-endian integers.
raw = np.fromfile(DATA_FILE, dtype=">i2")

if raw.size != EXPECTED_CELLS:
    raise ValueError(
        f"Unexpected number of cells: {raw.size:,}"
    )

elevation = raw.reshape((ROWS, COLS)).astype(np.float32)

print(f"      Dimensions: {COLS:,} × {ROWS:,}")
print(f"      Total cells: {elevation.size:,}")
print("      Data type: signed 16-bit, big-endian")


# ---------------------------------------------------------
# 4. LONGITUDE ORIENTATION
# ---------------------------------------------------------

# The source grid spans 0° to 360° east.
# Shift the columns by half the image width so that
# the displayed map spans -180° to +180°.

elevation = np.roll(
    elevation,
    shift=-(COLS // 2),
    axis=1
)

print("\n[2/5] Longitude orientation adjusted.")
print("      Display range: -180° to +180°")
print("      Latitude range: -90° to +90°")


# ---------------------------------------------------------
# 5. GLOBAL ELEVATION STATISTICS
# ---------------------------------------------------------

# Retain zero-valued cells: zero metres is a valid elevation.
# No undocumented NoData mask is applied.

values = elevation.ravel()

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
    "zero_valued_cells": int(np.count_nonzero(values == 0)),
}

print("\n[3/5] Elevation statistics calculated.")

for key, value in statistics.items():
    if isinstance(value, dict):
        continue
    print(f"      {key}: {value:,.2f}")


# ---------------------------------------------------------
# 6. GENERATE GLOBAL ELEVATION MAP
# ---------------------------------------------------------

print("\n[4/5] Generating global elevation visualisation...")

# Downsample only for display.
# Original source data remains unchanged.
display = elevation[::3, ::3]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
})

fig, ax = plt.subplots(figsize=(16, 8.5))

fig.patch.set_facecolor("white")
ax.set_facecolor("white")

image = ax.imshow(
    display,
    cmap="inferno",
    origin="upper",
    extent=(-180, 180, -90, 90),
    aspect="auto",
    interpolation="nearest",
    vmin=statistics["minimum_m"],
    vmax=statistics["maximum_m"],
)

ax.set_title(
    "MARS | GLOBAL ELEVATION MODEL",
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
    "Elevation relative to MOLA areoid (metres)",
    labelpad=8,
)

fig.text(
    0.5,
    0.025,
    "Source: NASA PDS | Mars Global Surveyor | MOLA MEGDR | "
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

report = {
    "project": "TERRAFORM — Planetary Analog Intelligence",
    "analysis": "Mars Global Elevation Analysis",
    "version": "1.0",
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "dataset": {
        "product": "MOLA MEGDR Median Topography",
        "product_id": "MEGT90N000EB",
        "source": SOURCE,
        "filename": DATA_FILE.name,
        "dimensions": {
            "columns": COLS,
            "rows": ROWS,
            "bands": 1,
            "total_cells": EXPECTED_CELLS,
        },
        "data_type": "Signed 16-bit integer",
        "byte_order": "Big-endian",
        "pixel_resolution_degrees": PIXEL_SIZE_DEGREES,
        "equatorial_pixel_scale_km": EQUATORIAL_PIXEL_SIZE_KM,
        "projection": "Simple cylindrical / equirectangular",
        "central_meridian_degrees": 180,
        "vertical_reference": "MOLA areoid",
        "elevation_unit": "metres",
    },
    "elevation_statistics": statistics,
    "processing": {
        "longitude_display_range": [-180, 180],
        "latitude_display_range": [-90, 90],
        "zero_values_retained": True,
        "nodata_mask_applied": False,
        "display_downsampling": "Every third row and column",
        "source_data_modified": False,
    },
    "scientific_limitations": [
        "MOLA grid cells represent median observed topography "
        "where observations are available.",
        "Interpolated values are supplied where no observations "
        "fall within a grid cell.",
        "Observation coverage is uneven, with lower density "
        "near the equator.",
        "Global statistics are unweighted grid-cell statistics; "
        "they are not area-weighted.",
        "Elevation is referenced to the MOLA areoid.",
        "Terrain elevation alone does not establish geological, "
        "environmental or habitation suitability.",
    ],
    "outputs": {
        "map": str(MAP_FILE.relative_to(ROOT)),
        "report": str(REPORT_FILE.relative_to(ROOT)),
    },
}

with REPORT_FILE.open("w", encoding="utf-8") as file:
    json.dump(report, file, indent=4)

print("\n[5/5] JSON report saved.")


# ---------------------------------------------------------
# 8. COMPLETION SUMMARY
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("MARS GLOBAL ANALYSIS COMPLETED")
print("=" * 60)

print(f"\nElevation range: "
      f"{statistics['minimum_m']:,.0f} m to "
      f"{statistics['maximum_m']:,.0f} m")

print(f"Mean elevation: "
      f"{statistics['mean_m']:,.2f} m")

print(f"Median elevation: "
      f"{statistics['median_m']:,.2f} m")

print("\nGenerated files:")
print(f"1. {MAP_FILE}")
print(f"2. {REPORT_FILE}")

print("\nTERRAFORM | End of processing")
