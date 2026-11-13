import numpy as np
from pathlib import Path

# --------------------------------------------------
# TERRAFORM: MOLA ELEVATION / OBSERVATION CROSS-CHECK
# --------------------------------------------------

mars_dir = Path(r"D:\TERRAFORM\data\Global\Mars")

elevation_path = mars_dir / "megt90n000eb.img"
counts_path = mars_dir / "megc90n000eb.img"

output_path = Path(
    r"D:\TERRAFORM\outputs\mars_observation_elevation_crosscheck.txt"
)

shape = (2880, 5760)
expected_size = shape[0] * shape[1]

report = []

def add(text=""):
    report.append(str(text))

def stats(values, label):
    add(f"\n{label}")
    add("-" * 45)
    add(f"Cell count: {values.size:,}")

    if values.size == 0:
        add("No cells in this category.")
        return

    add(f"Minimum elevation: {int(values.min()):,} m")
    add(f"Maximum elevation: {int(values.max()):,} m")
    add(f"Mean elevation: {float(values.mean()):.2f} m")
    add(f"Median elevation: {float(np.median(values)):.2f} m")

    percentiles = np.percentile(values, [1, 25, 50, 75, 99])
    add("Percentiles (1, 25, 50, 75, 99):")
    add(str([round(float(x), 2) for x in percentiles]))

def main():
    add("TERRAFORM - MOLA GLOBAL DATA CROSS-CHECK")
    add("=" * 60)

    for path in [elevation_path, counts_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

        if path.stat().st_size != expected_size * 2:
            raise ValueError(f"Unexpected file size: {path}")

    elevation = np.fromfile(elevation_path, dtype=">i2")
    counts = np.fromfile(counts_path, dtype=">i2")

    if elevation.size != expected_size or counts.size != expected_size:
        raise ValueError("Raster dimensions do not match expected dimensions.")

    elevation = elevation.reshape(shape)
    counts = counts.reshape(shape)

    # Observation masks
    observed = counts > 0
    unobserved = counts == 0

    total = elevation.size
    observed_n = int(np.count_nonzero(observed))
    unobserved_n = int(np.count_nonzero(unobserved))

    add("\n1. GRID AND COVERAGE")
    add("-" * 45)
    add(f"Raster dimensions: {shape}")
    add(f"Total cells: {total:,}")
    add(f"Cells with observations: {observed_n:,} ({observed_n / total * 100:.4f}%)")
    add(f"Cells with zero observations: {unobserved_n:,} ({unobserved_n / total * 100:.4f}%)")
    add(f"Negative observation counts: {int(np.count_nonzero(counts < 0)):,}")

    # Observation count distribution
    add("\n2. OBSERVATION COUNT DISTRIBUTION")
    add("-" * 45)
    add(f"Minimum count: {int(counts.min())}")
    add(f"Maximum count: {int(counts.max())}")
    add(f"Mean count across all cells: {float(counts.mean()):.2f}")
    add(f"Mean count among observed cells: {float(counts[observed].mean()):.2f}")

    # Elevation statistics by observation status
    stats(elevation[observed], "3. ELEVATION IN CELLS WITH OBSERVATIONS")
    stats(elevation[unobserved], "4. ELEVATION IN CELLS WITH ZERO OBSERVATIONS")

    # Cross-tabulation of zero elevation and observation status
    zero_elevation = elevation == 0

    observed_zero_elevation = int(np.count_nonzero(observed & zero_elevation))
    unobserved_zero_elevation = int(np.count_nonzero(unobserved & zero_elevation))
    observed_nonzero_elevation = int(np.count_nonzero(observed & ~zero_elevation))
    unobserved_nonzero_elevation = int(np.count_nonzero(unobserved & ~zero_elevation))

    add("\n5. ELEVATION ZERO-VALUE CROSS-CHECK")
    add("-" * 45)
    add(f"Observed cells with zero elevation: {observed_zero_elevation:,}")
    add(f"Observed cells with nonzero elevation: {observed_nonzero_elevation:,}")
    add(f"Zero-observation cells with zero elevation: {unobserved_zero_elevation:,}")
    add(f"Zero-observation cells with nonzero elevation: {unobserved_nonzero_elevation:,}")
    add(f"Total zero-elevation cells: {int(np.count_nonzero(zero_elevation)):,}")

    # Elevation distribution in unobserved cells
    if unobserved_n:
        nonzero_unobserved = elevation[unobserved & ~zero_elevation]
        add("\n6. CELLS WITHOUT OBSERVATIONS BUT WITH NONZERO ELEVATION")
        add("-" * 45)
        add(f"Number of cells: {nonzero_unobserved.size:,}")
        add(f"Percentage of zero-observation cells: {nonzero_unobserved.size / unobserved_n * 100:.4f}%")

    # Final checks
    add("\n7. VALIDATION SUMMARY")
    add("-" * 45)
    add(f"Elevation dimensions valid: {elevation.shape == shape}")
    add(f"Counts dimensions valid: {counts.shape == shape}")
    add(f"All observation counts nonnegative: {bool(np.all(counts >= 0))}")
    add(f"Observed + unobserved cells equal total: {observed_n + unobserved_n == total}")
    add("Interpretation: zero-count cells have no recorded MOLA observations.")
    add("The product label states that elevation values in observation gaps are interpolated.")
    add("This cross-check does not independently identify the interpolation algorithm or validate each interpolated value.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(report), encoding="utf-8")

    print("\n".join(report))
    print(f"\nREPORT SAVED: {output_path}")

if __name__ == "__main__":
    main()
