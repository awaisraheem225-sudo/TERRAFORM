import csv
import numpy as np
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, ListedColormap
    from matplotlib.patches import Patch
except ImportError as exc:
    raise SystemExit(
        f"Missing visualization dependency: {exc}\n"
        "Install it with: python -m pip install matplotlib"
    )

# ==========================================================
# TERRAFORM | GLOBAL MARS MOLA PROCESSING MODULE
# ==========================================================

ROOT = Path(r"D:\TERRAFORM")

DATA = ROOT / "data" / "Global" / "Mars"
OUTPUT = ROOT / "outputs" / "global_mars"

ELEVATION_FILE = DATA / "megt90n000eb.img"
COUNTS_FILE = DATA / "megc90n000eb.img"

SHAPE = (2880, 5760)
PIXEL_DEGREES = 0.0625

OUTPUT.mkdir(parents=True, exist_ok=True)

# ==========================================================
# 1. LOAD AND VALIDATE DATA
# ==========================================================

print("TERRAFORM GLOBAL MARS PROCESSING")
print("=" * 55)

for path in [ELEVATION_FILE, COUNTS_FILE]:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")

    expected_bytes = np.prod(SHAPE) * 2

    if path.stat().st_size != expected_bytes:
        raise ValueError(f"Unexpected file size: {path}")

print("\nLoading MOLA elevation raster...")

elevation = np.fromfile(
    ELEVATION_FILE, dtype=">i2"
).reshape(SHAPE).astype(np.float32)

print("Loading MOLA observation-count raster...")

counts = np.fromfile(
    COUNTS_FILE, dtype=">i2"
).reshape(SHAPE)

observed = counts > 0
unobserved = counts == 0

total_cells = elevation.size
observed_cells = int(observed.sum())
unobserved_cells = int(unobserved.sum())

observed_pct = observed_cells / total_cells * 100
unobserved_pct = unobserved_cells / total_cells * 100

print("\nDATA VALIDATION")
print("-" * 45)
print(f"Raster dimensions: {SHAPE}")
print(f"Total cells: {total_cells:,}")
print(f"Observed cells: {observed_cells:,} ({observed_pct:.4f}%)")
print(f"Zero-observation cells: {unobserved_cells:,} ({unobserved_pct:.4f}%)")
print(f"Elevation range: {elevation.min():.0f} to {elevation.max():.0f} m")
print(f"Observation count range: {counts.min()} to {counts.max()}")

# PDS grid orientation:
# First image row corresponds to northern latitudes.
# First image column corresponds to 0 degrees east.
# This follows the supplied simple cylindrical PDS label.

extent = (0, 360, -90, 90)

plt.rcParams.update({
    "figure.facecolor": "#080D18",
    "axes.facecolor": "#080D18",
    "axes.edgecolor": "#53677F",
    "axes.labelcolor": "#D8E4F0",
    "xtick.color": "#B7C7D9",
    "ytick.color": "#B7C7D9",
    "text.color": "#D8E4F0",
    "font.family": "DejaVu Sans",
})

def save_map(filename):
    path = OUTPUT / filename
    plt.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
        facecolor="#080D18"
    )
    plt.close()
    print(f"Created: {path}")

# ==========================================================
# 2. GLOBAL ELEVATION MAP
# ==========================================================

print("\nGenerating global elevation map...")

fig, ax = plt.subplots(figsize=(16, 8))

image = ax.imshow(
    elevation,
    extent=extent,
    origin="upper",
    cmap="terrain",
    vmin=-8500,
    vmax=22000,
    interpolation="nearest",
    aspect="auto"
)

ax.set_title(
    "MARS | GLOBAL MOLA MEDIAN TOPOGRAPHY",
    fontsize=17,
    fontweight="bold",
    pad=18
)

ax.set_xlabel("East Longitude (degrees)")
ax.set_ylabel("Planetocentric Latitude (degrees)")

ax.set_xticks(np.arange(0, 361, 30))
ax.set_yticks(np.arange(-90, 91, 15))
ax.grid(alpha=0.18, linestyle="--")

cbar = fig.colorbar(image, ax=ax, shrink=0.82, pad=0.025)
cbar.set_label("Elevation relative to MOLA areoid (m)")

ax.text(
    0.01, -0.16,
    "Source: MGS MOLA MEGDR | 0.0625° angular grid | "
    "Approx. 3.705 km at equator",
    transform=ax.transAxes,
    fontsize=9,
    color="#A8B9CC"
)

fig.tight_layout()
save_map("mars_global_elevation.png")

# ==========================================================
# 3. GLOBAL OBSERVATION COUNT MAP
# ==========================================================

print("\nGenerating observation density map...")

fig, ax = plt.subplots(figsize=(16, 8))

count_display = np.ma.masked_where(counts == 0, counts)

cmap_counts = plt.get_cmap("viridis").copy()
cmap_counts.set_bad("#111D30")

image = ax.imshow(
    count_display,
    extent=extent,
    origin="upper",
    cmap=cmap_counts,
    norm=LogNorm(vmin=1, vmax=max(2, int(counts.max()))),
    interpolation="nearest",
    aspect="auto"
)

ax.set_title(
    "MARS | MOLA OBSERVATION DENSITY",
    fontsize=17,
    fontweight="bold",
    pad=18
)

ax.set_xlabel("East Longitude (degrees)")
ax.set_ylabel("Planetocentric Latitude (degrees)")
ax.set_xticks(np.arange(0, 361, 30))
ax.set_yticks(np.arange(-90, 91, 15))
ax.grid(alpha=0.18, linestyle="--")

cbar = fig.colorbar(image, ax=ax, shrink=0.82, pad=0.025)
cbar.set_label("Recorded observations per grid cell (log scale)")

ax.text(
    0.01, -0.16,
    "Dark background = zero recorded observations",
    transform=ax.transAxes,
    fontsize=9,
    color="#A8B9CC"
)

fig.tight_layout()
save_map("mars_global_observation_density.png")

# ==========================================================
# 4. OBSERVED VS ZERO-OBSERVATION CLASSIFICATION
# ==========================================================

print("\nGenerating observation classification map...")

classification = observed.astype(np.uint8)

classification_cmap = ListedColormap([
    "#D97757",  # Zero recorded observations
    "#25C7A5",  # Recorded observations
])

fig, ax = plt.subplots(figsize=(16, 8))

ax.imshow(
    classification,
    extent=extent,
    origin="upper",
    cmap=classification_cmap,
    vmin=0,
    vmax=1,
    interpolation="nearest",
    aspect="auto"
)

ax.set_title(
    "MARS | MOLA OBSERVATION STATUS",
    fontsize=17,
    fontweight="bold",
    pad=18
)

ax.set_xlabel("East Longitude (degrees)")
ax.set_ylabel("Planetocentric Latitude (degrees)")
ax.set_xticks(np.arange(0, 361, 30))
ax.set_yticks(np.arange(-90, 91, 15))
ax.grid(alpha=0.18, linestyle="--")

legend = [
    Patch(color="#25C7A5", label="Recorded MOLA observations"),
    Patch(color="#D97757", label="Zero recorded observations"),
]

ax.legend(
    handles=legend,
    loc="lower left",
    framealpha=0.9,
    facecolor="#111D30",
    edgecolor="#53677F"
)

fig.tight_layout()
save_map("mars_global_observation_classification.png")

# ==========================================================
# 5. ELEVATION DISTRIBUTION COMPARISON
# ==========================================================

print("\nGenerating elevation distribution comparison...")

observed_elevation = elevation[observed]
unobserved_elevation = elevation[unobserved]

fig, ax = plt.subplots(figsize=(13, 7))

bins = np.linspace(-9000, 22000, 120)

ax.hist(
    observed_elevation,
    bins=bins,
    density=True,
    histtype="step",
    linewidth=2,
    label="Cells with observations"
)

ax.hist(
    unobserved_elevation,
    bins=bins,
    density=True,
    histtype="step",
    linewidth=2,
    label="Zero-observation cells"
)

ax.set_title(
    "MARS | GLOBAL ELEVATION DISTRIBUTION",
    fontsize=16,
    fontweight="bold",
    pad=16
)

ax.set_xlabel("Elevation relative to MOLA areoid (m)")
ax.set_ylabel("Probability density")
ax.legend(facecolor="#111D30", edgecolor="#53677F")
ax.grid(alpha=0.18, linestyle="--")

fig.tight_layout()
save_map("mars_global_elevation_distribution.png")

# ==========================================================
# 6. LATITUDE-BAND COVERAGE ANALYSIS
# ==========================================================

print("\nCalculating latitude-band statistics...")

latitude_edges = np.arange(-90, 91, 15)

csv_path = OUTPUT / "mars_latitude_band_coverage.csv"

with csv_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    writer.writerow([
        "latitude_band",
        "total_cells",
        "observed_cells",
        "zero_observation_cells",
        "observed_coverage_percent",
        "mean_observations_per_cell",
        "mean_observed_elevation_m",
        "mean_zero_observation_elevation_m"
    ])

    for lower, upper in zip(
        latitude_edges[:-1],
        latitude_edges[1:]
    ):
        row_indices = np.where(
            (90 - (np.arange(SHAPE[0]) + 0.5) * PIXEL_DEGREES >= lower) &
            (90 - (np.arange(SHAPE[0]) + 0.5) * PIXEL_DEGREES < upper)
        )[0]

        band_counts = counts[row_indices, :]
        band_elevation = elevation[row_indices, :]

        band_observed = band_counts > 0
        band_unobserved = band_counts == 0

        n_total = band_counts.size
        n_observed = int(band_observed.sum())
        n_unobserved = int(band_unobserved.sum())

        coverage = n_observed / n_total * 100 if n_total else 0

        mean_count = (
            float(band_counts.mean()) if n_total else 0
        )

        mean_observed_elevation = (
            float(band_elevation[band_observed].mean())
            if n_observed else float("nan")
        )

        mean_unobserved_elevation = (
            float(band_elevation[band_unobserved].mean())
            if n_unobserved else float("nan")
        )

        writer.writerow([
            f"{lower} to {upper}",
            n_total,
            n_observed,
            n_unobserved,
            round(coverage, 4),
            round(mean_count, 4),
            round(mean_observed_elevation, 2),
            round(mean_unobserved_elevation, 2)
        ])

print(f"Created: {csv_path}")

# ==========================================================
# 7. TECHNICAL SUMMARY REPORT
# ==========================================================

print("\nWriting technical report...")

report_path = OUTPUT / "mars_global_analysis_report.txt"

report = f"""
TERRAFORM - GLOBAL MARS MOLA ANALYSIS
============================================================

DATA SOURCES
------------------------------------------------------------
Elevation: MEGT90N000EB.IMG
Observation counts: MEGC90N000EB.IMG
Instrument: Mars Orbiter Laser Altimeter (MOLA)
Spacecraft: Mars Global Surveyor
Dataset: MGS-M-MOLA-5-MEGDR-L3-V1.0

GRID CHARACTERISTICS
------------------------------------------------------------
Raster dimensions: {SHAPE}
Total cells: {total_cells:,}
Angular pixel spacing: {PIXEL_DEGREES} degrees
Latitude coverage: 90S to 90N
Longitude coverage: 0E to 360E
Projection: Simple Cylindrical
Coordinate latitude: Planetocentric
Longitude direction: East
Equatorial approximate pixel spacing: 3.705 km

ELEVATION STATISTICS
------------------------------------------------------------
Minimum: {float(elevation.min()):.0f} m
Maximum: {float(elevation.max()):.0f} m
Mean: {float(elevation.mean()):.2f} m
Median: {float(np.median(elevation)):.2f} m

OBSERVATION COVERAGE
------------------------------------------------------------
Observed cells: {observed_cells:,}
Observed coverage: {observed_pct:.4f}%
Zero-observation cells: {unobserved_cells:,}
Zero-observation fraction: {unobserved_pct:.4f}%
Maximum observations in one cell: {int(counts.max())}
Mean count across all cells: {float(counts.mean()):.2f}
Mean count among observed cells: {float(counts[observed].mean()):.2f}

OBSERVATION-GROUP ELEVATION STATISTICS
------------------------------------------------------------
Observed-cell mean elevation: {float(observed_elevation.mean()):.2f} m
Observed-cell median elevation: {float(np.median(observed_elevation)):.2f} m
Zero-observation-cell mean elevation: {float(unobserved_elevation.mean()):.2f} m
Zero-observation-cell median elevation: {float(np.median(unobserved_elevation)):.2f} m

SCIENTIFIC INTERPRETATION
------------------------------------------------------------
The elevation product contains median topography where MOLA
observations are available and interpolated values in gaps,
according to its PDS product description.

The observation-count raster identifies cells with recorded
MOLA observations.

A zero observation count does not imply zero elevation.
The two rasters must therefore be interpreted together.

The global elevation map includes both observed and
interpolated terrain values.

The classification map identifies observation status,
not independent validation of interpolation accuracy.

The angular grid has approximately 3.705 km spacing at
the equator. East-west ground spacing decreases with
latitude. The global map is not a uniform metric grid.

These products are not sufficient by themselves to establish
terrain suitability for a planetary analog site.

OUTPUTS
------------------------------------------------------------
mars_global_elevation.png
mars_global_observation_density.png
mars_global_observation_classification.png
mars_global_elevation_distribution.png
mars_latitude_band_coverage.csv
mars_global_analysis_report.txt

============================================================
TERRAFORM | PLANETARY ANALOG INTELLIGENCE
GLOBAL MARS DATA PROCESSING
"""

report_path.write_text(report.strip(), encoding="utf-8")

print(f"Created: {report_path}")

print("\n" + "=" * 55)
print("GLOBAL MARS PROCESSING COMPLETED")
print(f"OUTPUT DIRECTORY: {OUTPUT}")
print("=" * 55)
