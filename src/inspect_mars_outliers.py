
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

ROWS = 2880
COLS = 5760
SPACING = 0.0625

ELEVATION_FILE = ROOT / "data" / "Global" / "Mars" / "megt90n000eb.img"
COUNT_FILE = ROOT / "data" / "Global" / "Mars" / "megc90n000eb.img"
METRIC_DIR = ROOT / "outputs" / "global_mars" / "terrain_metrics"

METRICS = {
    "Slope": "mars_global_slope_degrees.npy",
    "Roughness": "mars_global_roughness_3x3_m.npy",
    "Local relief": "mars_global_local_relief_3x3_m.npy",
}


def coordinates(row, col):
    latitude = 90.0 - (row + 0.5) * SPACING
    longitude = (col + 0.5) * SPACING
    return latitude, longitude


def inspect_neighbourhood(elevation, counts, row, col):
    print("\nOriginal MOLA elevation neighbourhood (metres):")

    for dr in (-1, 0, 1):
        values = []
        for dc in (-1, 0, 1):
            r = row + dr
            c = (col + dc) % COLS

            if r < 0 or r >= ROWS:
                values.append("OUT")
            else:
                values.append(f"{int(elevation[r, c]):7d}")

        print("  ".join(values))

    print("\nObservation counts in the same neighbourhood:")

    for dr in (-1, 0, 1):
        values = []
        for dc in (-1, 0, 1):
            r = row + dr
            c = (col + dc) % COLS

            if r < 0 or r >= ROWS:
                values.append("OUT")
            else:
                values.append(f"{int(counts[r, c]):5d}")

        print("  ".join(values))


def main():
    if not ELEVATION_FILE.exists() or not COUNT_FILE.exists():
        raise FileNotFoundError("Original MOLA input files were not found.")

    elevation = np.memmap(
        ELEVATION_FILE, dtype=">i2", mode="r",
        shape=(ROWS, COLS)
    )

    counts = np.memmap(
        COUNT_FILE, dtype=">i2", mode="r",
        shape=(ROWS, COLS)
    )

    for name, filename in METRICS.items():
        path = METRIC_DIR / filename

        if not path.exists():
            print(f"\nMISSING: {path}")
            continue

        metric = np.load(path, mmap_mode="r")

        if metric.shape != (ROWS, COLS):
            print(f"\nUnexpected shape for {name}: {metric.shape}")
            continue

        valid = np.isfinite(metric)
        valid &= np.indices(metric.shape)[0] < ROWS

        if not np.any(valid):
            print(f"\nNo valid cells for {name}")
            continue

        flat_index = np.nanargmax(np.where(valid, metric, np.nan))
        row, col = np.unravel_index(flat_index, metric.shape)

        latitude, longitude = coordinates(row, col)

        print("\n" + "=" * 65)
        print(f"METRIC: {name}")
        print(f"Maximum: {float(metric[row, col]):.6f}")
        print(f"Raster row: {row}, column: {col}")
        print(f"Planetocentric latitude: {latitude:.6f}°")
        print(f"East longitude: {longitude:.6f}°")
        print(f"Centre elevation: {int(elevation[row, col])} m")
        print(f"Centre observation count: {int(counts[row, col])}")

        inspect_neighbourhood(elevation, counts, row, col)

    print("\nInspection complete.")


if __name__ == "__main__":
    main()
