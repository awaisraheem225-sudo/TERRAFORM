
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

ROWS, COLS = 2880, 5760
SPACING = 0.0625

ELEVATION_FILE = ROOT / "data/Global/Mars/megt90n000eb.img"
COUNT_FILE = ROOT / "data/Global/Mars/megc90n000eb.img"
METRIC_DIR = ROOT / "outputs/global_mars/terrain_metrics"

METRICS = {
    "SLOPE": "mars_global_slope_degrees.npy",
    "ROUGHNESS": "mars_global_roughness_3x3_m.npy",
    "LOCAL_RELIEF": "mars_global_local_relief_3x3_m.npy",
}

CUTOFFS = [80.0, 85.0, 89.5]


def build_neighbourhood_support(observed):
    """Count observed cells in each 3x3 neighbourhood."""
    support = np.zeros((ROWS, COLS), dtype=np.uint8)

    for dr in (-1, 0, 1):
        target_start = max(0, -dr)
        target_end = min(ROWS, ROWS - dr)

        source_start = target_start + dr
        source_end = target_end + dr

        for dc in (-1, 0, 1):
            shifted = np.roll(
                observed[source_start:source_end],
                shift=-dc,
                axis=1,
            )

            support[target_start:target_end] += shifted

    return support


def top_pixels(metric, mask, count=5):
    flat = np.asarray(metric).ravel()
    indices = np.flatnonzero(mask.ravel())

    if not len(indices):
        return []

    k = min(count, len(indices))
    chosen = np.argpartition(flat[indices], -k)[-k:]
    selected = indices[chosen]
    selected = selected[np.argsort(flat[selected])[::-1]]

    results = []

    for index in selected:
        row, col = np.unravel_index(index, (ROWS, COLS))
        lat = 90.0 - (row + 0.5) * SPACING
        lon = (col + 0.5) * SPACING

        results.append(
            (float(flat[index]), row, col, lat, lon)
        )

    return results


def main():
    print("=" * 75)
    print("TERRAFORM â€” STRICT OBSERVATION SUPPORT VALIDATION")
    print("=" * 75)

    elevation = np.memmap(
        ELEVATION_FILE,
        dtype=">i2",
        mode="r",
        shape=(ROWS, COLS),
    )

    counts = np.memmap(
        COUNT_FILE,
        dtype=">i2",
        mode="r",
        shape=(ROWS, COLS),
    )

    observed = np.asarray(counts) > 0
    support = build_neighbourhood_support(observed)

    strict_mask = support == 9

    latitude = 90.0 - (
        np.arange(ROWS) + 0.5
    ) * SPACING

    abs_latitude = np.abs(latitude)

    print("\nOBSERVATION SUPPORT")
    print("-" * 75)
    print(f"All grid cells: {ROWS * COLS:,}")
    print(f"Cells with 9/9 observed neighbours: "
          f"{np.count_nonzero(strict_mask):,}")
    print(f"Cells with incomplete support: "
          f"{np.count_nonzero(~strict_mask):,}")

    for name, filename in METRICS.items():
        path = METRIC_DIR / filename

        if not path.exists():
            print(f"\nMISSING: {path}")
            continue

        metric = np.load(path, mmap_mode="r")
        finite = np.isfinite(metric)

        print("\n" + "=" * 75)
        print(f"METRIC: {name}")
        print("=" * 75)

        # Strict support maximum across the existing raster.
        strict_valid = finite & strict_mask

        if np.any(strict_valid):
            strict_max = float(np.max(metric[strict_valid]))
            print(f"\nMaximum with 9/9 observed neighbours: "
                  f"{strict_max:.6f}")
        else:
            print("\nNo pixels have complete observation support.")

        # Latitude comparisons using strict neighbourhood support.
        print("\nSTRICT SUPPORT â€” LATITUDE CUTOFFS")
        print("Cutoff | Valid pixels | Maximum")
        print("-" * 55)

        for cutoff in CUTOFFS:
            lat_mask = (abs_latitude < cutoff)[:, None]
            valid = finite & strict_mask & lat_mask

            number = int(np.count_nonzero(valid))

            maximum = (
                float(np.max(metric[valid]))
                if number else float("nan")
            )

            print(f"{cutoff:5.1f}Â° | {number:12,} | {maximum:12.6f}")

        # Slope top five under each latitude cutoff.
        if name == "SLOPE":
            print("\nTOP FIVE SLOPE PIXELS BY LATITUDE CUTOFF")

            for cutoff in CUTOFFS:
                lat_mask = (abs_latitude < cutoff)[:, None]
                valid = finite & strict_mask & lat_mask

                print(f"\nLatitude below {cutoff}Â°:")

                for rank, item in enumerate(
                    top_pixels(metric, valid, 5), start=1
                ):
                    value, row, col, lat, lon = item

                    print(
                        f"#{rank}: {value:.4f}Â° | "
                        f"Lat {lat:.6f}Â° | "
                        f"Lon {lon:.6f}Â° | "
                        f"Elevation {int(elevation[row, col])} m | "
                        f"Centre count {int(counts[row, col])} | "
                        f"3x3 support {int(support[row, col])}/9"
                    )

    print("\n" + "=" * 75)
    print("STRICT OBSERVATION VALIDATION COMPLETE")
    print("Original rasters were not modified.")
    print("=" * 75)


if __name__ == "__main__":
    main()

