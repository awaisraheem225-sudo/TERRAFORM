import csv
import numpy as np
from pathlib import Path

# ==========================================================
# TERRAFORM | MARS AREA-WEIGHTED STATISTICS
# ==========================================================

ROOT = Path(r"D:\TERRAFORM")
DATA = ROOT / "data" / "Global" / "Mars"
OUTPUT = ROOT / "outputs" / "global_mars"

ELEVATION_FILE = DATA / "megt90n000eb.img"
COUNTS_FILE = DATA / "megc90n000eb.img"

REPORT_FILE = OUTPUT / "mars_area_weighted_report.txt"
CSV_FILE = OUTPUT / "mars_area_weighted_latitude_bands.csv"

ROWS = 2880
COLS = 5760
STEP_DEG = 0.0625
RADIUS_KM = 3396.0

OUTPUT.mkdir(parents=True, exist_ok=True)

print("TERRAFORM | AREA-WEIGHTED MARS ANALYSIS")
print("=" * 55)

# ----------------------------------------------------------
# 1. LOAD AND VALIDATE RASTERS
# ----------------------------------------------------------

for path in [ELEVATION_FILE, COUNTS_FILE]:
    if not path.exists():
        raise FileNotFoundError(path)

    expected_bytes = ROWS * COLS * 2

    if path.stat().st_size != expected_bytes:
        raise ValueError(f"Unexpected file size: {path}")

print("\nLoading elevation and observation rasters...")

elevation = np.fromfile(
    ELEVATION_FILE, dtype=">i2"
).reshape(ROWS, COLS)

counts = np.fromfile(
    COUNTS_FILE, dtype=">i2"
).reshape(ROWS, COLS)

if np.any(counts < 0):
    raise ValueError("Negative observation counts detected.")

observed = counts > 0
unobserved = counts == 0

# ----------------------------------------------------------
# 2. CALCULATE SPHERICAL CELL AREAS
# ----------------------------------------------------------

print("Calculating latitude-dependent cell areas...")

step_rad = np.deg2rad(STEP_DEG)
longitude_width_rad = step_rad

row_indices = np.arange(ROWS)

north_edges_deg = 90.0 - row_indices * STEP_DEG
south_edges_deg = 90.0 - (row_indices + 1) * STEP_DEG

north_edges_rad = np.deg2rad(north_edges_deg)
south_edges_rad = np.deg2rad(south_edges_deg)

row_area_km2 = (
    RADIUS_KM ** 2
    * longitude_width_rad
    * (
        np.sin(north_edges_rad)
        - np.sin(south_edges_rad)
    )
)

# Every cell in one latitude row has the same area.
cell_area_km2 = row_area_km2[:, np.newaxis]

total_reference_area = float(row_area_km2.sum() * COLS)

# ----------------------------------------------------------
# 3. AREA-WEIGHTED QUANTILE FUNCTION
# ----------------------------------------------------------

def weighted_quantiles(values, weights, percentiles):
    """
    Exact weighted quantiles for integer-valued elevation data.
    Uses a weighted histogram instead of sorting millions of cells.
    """

    values = np.asarray(values)
    weights = np.asarray(weights, dtype=np.float64)

    if values.size == 0:
        return [float("nan")] * len(percentiles)

    minimum = int(values.min())
    maximum = int(values.max())

    histogram = np.bincount(
        values.astype(np.int32) - minimum,
        weights=weights,
        minlength=maximum - minimum + 1
    )

    cumulative = np.cumsum(histogram)
    total = cumulative[-1]

    results = []

    for percentile in percentiles:
        target = total * percentile / 100.0

        index = np.searchsorted(
            cumulative,
            target,
            side="left"
        )

        index = min(index, len(histogram) - 1)
        results.append(float(index + minimum))

    return results


def weighted_statistics(values, weights, name):
    values = np.asarray(values)
    weights = np.asarray(weights, dtype=np.float64)

    total_area = float(weights.sum())

    if total_area <= 0:
        raise ValueError(f"No area available for {name}")

    mean = float(
        np.sum(values.astype(np.float64) * weights)
        / total_area
    )

    quantiles = weighted_quantiles(
        values,
        weights,
        [1, 25, 50, 75, 99]
    )

    return {
        "name": name,
        "cell_count": int(values.size),
        "area_km2": total_area,
        "mean_m": mean,
        "minimum_m": int(values.min()),
        "maximum_m": int(values.max()),
        "p01_m": quantiles[0],
        "p25_m": quantiles[1],
        "median_m": quantiles[2],
        "p75_m": quantiles[3],
        "p99_m": quantiles[4],
    }


# ----------------------------------------------------------
# 4. GLOBAL SURFACE-AREA-WEIGHTED COVERAGE
# ----------------------------------------------------------

print("Calculating global coverage...")

observed_area = float(
    np.sum(cell_area_km2 * observed)
)

unobserved_area = float(
    np.sum(cell_area_km2 * unobserved)
)

observed_area_pct = (
    observed_area / total_reference_area * 100
)

unobserved_area_pct = (
    unobserved_area / total_reference_area * 100
)

# ----------------------------------------------------------
# 5. AREA-WEIGHTED ELEVATION STATISTICS
# ----------------------------------------------------------

print("Calculating weighted elevation statistics...")

all_stats = weighted_statistics(
    elevation.ravel(),
    np.broadcast_to(cell_area_km2, elevation.shape).ravel(),
    "Entire global elevation grid"
)

observed_stats = weighted_statistics(
    elevation[observed],
    np.broadcast_to(cell_area_km2, elevation.shape)[observed],
    "Cells with recorded observations"
)

unobserved_stats = weighted_statistics(
    elevation[unobserved],
    np.broadcast_to(cell_area_km2, elevation.shape)[unobserved],
    "Cells with zero recorded observations"
)

# ----------------------------------------------------------
# 6. LATITUDE-BAND ANALYSIS
# ----------------------------------------------------------

print("Processing latitude bands...")

band_edges = np.arange(-90, 91, 15)

with CSV_FILE.open(
    "w", newline="", encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "latitude_band",
        "total_reference_area_km2",
        "observed_area_km2",
        "zero_observation_area_km2",
        "observed_area_coverage_percent",
        "zero_observation_area_percent",
        "area_weighted_mean_elevation_m",
        "area_weighted_mean_observation_count"
    ])

    for lower, upper in zip(
        band_edges[:-1], band_edges[1:]
    ):

        lat_centers = (
            90.0 - (row_indices + 0.5) * STEP_DEG
        )

        rows_in_band = (
            (lat_centers >= lower)
            & (lat_centers < upper)
        )

        band_elevation = elevation[rows_in_band]
        band_counts = counts[rows_in_band]

        band_observed = band_counts > 0
        band_unobserved = band_counts == 0

        band_area = np.broadcast_to(
            cell_area_km2[rows_in_band],
            band_elevation.shape
        )

        total_band_area = float(band_area.sum())

        observed_band_area = float(
            np.sum(band_area * band_observed)
        )

        unobserved_band_area = float(
            np.sum(band_area * band_unobserved)
        )

        coverage = (
            observed_band_area / total_band_area * 100
        )

        mean_elevation = float(
            np.sum(
                band_elevation.astype(np.float64) * band_area
            ) / total_band_area
        )

        mean_observation_count = float(
            np.sum(
                band_counts.astype(np.float64) * band_area
            ) / total_band_area
        )

        writer.writerow([
            f"{lower} to {upper}",
            round(total_band_area, 3),
            round(observed_band_area, 3),
            round(unobserved_band_area, 3),
            round(coverage, 5),
            round(100 - coverage, 5),
            round(mean_elevation, 3),
            round(mean_observation_count, 3)
        ])

print(f"Created: {CSV_FILE}")

# ----------------------------------------------------------
# 7. GENERATE TECHNICAL REPORT
# ----------------------------------------------------------

print("Generating scientific report...")

def format_stats(stats):
    return f"""
{stats['name']}
Cells: {stats['cell_count']:,}
Reference surface area: {stats['area_km2']:,.3f} km²
Minimum: {stats['minimum_m']} m
Maximum: {stats['maximum_m']} m
Area-weighted mean: {stats['mean_m']:.3f} m
Area-weighted P01: {stats['p01_m']:.2f} m
Area-weighted P25: {stats['p25_m']:.2f} m
Area-weighted median: {stats['median_m']:.2f} m
Area-weighted P75: {stats['p75_m']:.2f} m
Area-weighted P99: {stats['p99_m']:.2f} m
"""

report = f"""
TERRAFORM - MARS AREA-WEIGHTED GLOBAL STATISTICS
============================================================

REFERENCE GEOMETRY
------------------------------------------------------------
Reference radius: {RADIUS_KM} km
Geometry: Spherical reference surface
Grid: {ROWS} x {COLS}
Angular spacing: {STEP_DEG} degrees
Longitude coverage: 360 degrees
Latitude coverage: 180 degrees

CELL AREA METHOD
------------------------------------------------------------
A = R^2 * delta_longitude *
    (sin(north_latitude) - sin(south_latitude))

Each latitude row receives its own spherical cell area.
All cells in the same latitude row have equal area.

GLOBAL REFERENCE SURFACE AREA
------------------------------------------------------------
Calculated area: {total_reference_area:,.3f} km²
Theoretical sphere area: {4 * np.pi * RADIUS_KM**2:,.3f} km²
Difference: {total_reference_area - 4 * np.pi * RADIUS_KM**2:.6f} km²

OBSERVATION COVERAGE
------------------------------------------------------------
Observed cells: {int(observed.sum()):,}
Zero-observation cells: {int(unobserved.sum()):,}

Observed reference area: {observed_area:,.3f} km²
Zero-observation reference area: {unobserved_area:,.3f} km²

Area-weighted observed coverage: {observed_area_pct:.5f}%
Area-weighted zero-observation coverage: {unobserved_area_pct:.5f}%

{format_stats(all_stats)}

{format_stats(observed_stats)}

{format_stats(unobserved_stats)}

SCIENTIFIC LIMITATIONS
------------------------------------------------------------
1. Surface areas use a spherical reference radius of 3396 km.
2. These are reference-sphere areas, not exact terrain-surface areas.
3. Elevations are relative to the MOLA areoid.
4. Zero-count cells are classified as having no recorded
   MOLA observations. Their supplied elevations are retained.
5. The PDS product description identifies interpolated values
   in observation gaps.
6. This analysis does not independently validate interpolation.
7. Weighted elevation statistics include interpolated values
   in the full-grid results.
8. Observed-only statistics exclude zero-count cells.
9. Area-weighted quantiles use weighted elevation histograms
   because the elevation data are integer-valued metres.

OUTPUT FILES
------------------------------------------------------------
mars_area_weighted_report.txt
mars_area_weighted_latitude_bands.csv

============================================================
TERRAFORM | PLANETARY ANALOG INTELLIGENCE
============================================================
"""

REPORT_FILE.write_text(report.strip(), encoding="utf-8")

print(f"Created: {REPORT_FILE}")

# ----------------------------------------------------------
# 8. DISPLAY KEY RESULTS
# ----------------------------------------------------------

print("\n" + "=" * 55)
print("AREA-WEIGHTED RESULTS")
print("=" * 55)

print(f"Reference surface area: {total_reference_area:,.2f} km²")
print(f"Observed area: {observed_area:,.2f} km²")
print(f"Observed area coverage: {observed_area_pct:.5f}%")
print(f"Zero-observation area coverage: {unobserved_area_pct:.5f}%")

print("\nAREA-WEIGHTED ELEVATION")
print(f"Global mean: {all_stats['mean_m']:.3f} m")
print(f"Global median: {all_stats['median_m']:.2f} m")
print(f"Observed-only mean: {observed_stats['mean_m']:.3f} m")
print(f"Observed-only median: {observed_stats['median_m']:.2f} m")

print("\nPROCESSING COMPLETED SUCCESSFULLY.")
print(f"Report: {REPORT_FILE}")
print(f"CSV: {CSV_FILE}")
