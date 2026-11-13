
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

ROWS, COLS = 2880, 5760
STEP = 0.0625

ELEVATION = ROOT / "data/Global/Mars/megt90n000eb.img"
COUNTS = ROOT / "data/Global/Mars/megc90n000eb.img"
METRIC_DIR = ROOT / "outputs/global_mars/terrain_metrics"

METRICS = {
    "SLOPE": "mars_global_slope_degrees.npy",
    "ROUGHNESS": "mars_global_roughness_3x3_m.npy",
    "LOCAL_RELIEF": "mars_global_local_relief_3x3_m.npy",
}

CUTOFFS = [80.0, 85.0, 89.5]


def neighbourhood_support(observed):
    support = np.zeros((ROWS, COLS), dtype=np.uint8)

    for dr in (-1, 0, 1):
        r0 = max(0, -dr)
        r1 = min(ROWS, ROWS - dr)

        for dc in (-1, 0, 1):
            shifted = np.roll(
                observed[r0 + dr:r1 + dr],
                shift=-dc,
                axis=1,
            )
            support[r0:r1] += shifted

    return support


def report_neighbourhood(elevation, counts, row, col):
    print("  Elevation (m):")
    for dr in (-1, 0, 1):
        values = []
        for dc in (-1, 0, 1):
            r = row + dr
            c = (col + dc) % COLS
            values.append(
                f"{int(elevation[r, c]):7d}"
                if 0 <= r < ROWS else "    OUT"
            )
        print("   " + " ".join(values))

    print("  Observation counts:")
    for dr in (-1, 0, 1):
        values = []
        for dc in (-1, 0, 1):
            r = row + dr
            c = (col + dc) % COLS
            values.append(
                f"{int(counts[r, c]):5d}"
                if 0 <= r < ROWS else "  OUT"
            )
        print("   " + " ".join(values))


def report_top(metric, mask, elevation, counts, number=5):
    values = np.asarray(metric)
    indices = np.flatnonzero(mask.ravel())

    if not len(indices):
        print("  No qualifying pixels.")
        return

    k = min(number, len(indices))
    selected = indices[
        np.argpartition(values.ravel()[indices], -k)[-k:]
    ]
    selected = selected[
        np.argsort(values.ravel()[selected])[::-1]
    ]

    for rank, index in enumerate(selected, 1):
        row, col = np.unravel_index(index, (ROWS, COLS))
        lat = 90.0 - (row + 0.5) * STEP
        lon = (col + 0.5) * STEP

        print(
            f"\n  #{rank}: {float(values[row, col]):.6f}"
            f" | Lat {lat:.6f}°"
            f" | Lon {lon:.6f}°"
            f" | Centre count {int(counts[row, col])}"
        )

        report_neighbourhood(elevation, counts, row, col)


def main():
    elevation = np.memmap(
        ELEVATION, dtype=">i2", mode="r",
        shape=(ROWS, COLS)
    )
    counts = np.memmap(
        COUNTS, dtype=">i2", mode="r",
        shape=(ROWS, COLS)
    )

    observed = np.asarray(counts) > 0
    support = neighbourhood_support(observed)
    strict = support == 9

    latitude = 90.0 - (
        np.arange(ROWS) + 0.5
    ) * STEP
    abs_latitude = np.abs(latitude)

    print("=" * 75)
    print("STRICT MAXIMA AUDIT — 9/9 OBSERVATION SUPPORT")
    print(f"Qualifying cells: {np.count_nonzero(strict):,}")
    print("=" * 75)

    for name, filename in METRICS.items():
        path = METRIC_DIR / filename

        if not path.exists():
            print(f"\nMissing metric: {path}")
            continue

        metric = np.load(path, mmap_mode="r")
        finite = np.isfinite(metric)

        print("\n" + "=" * 75)
        print(name)
        print("=" * 75)

        for cutoff in CUTOFFS:
            mask = (
                finite
                & strict
                & (abs_latitude < cutoff)[:, None]
            )

            print(f"\nLatitude below {cutoff}°")
            print(f"Qualifying pixels: {np.count_nonzero(mask):,}")

            if np.any(mask):
                print(f"Maximum: {np.max(metric[mask]):.6f}")

            if name == "SLOPE":
                print("Top five — STRICT 9/9 SUPPORT:")
                report_top(metric, mask, elevation, counts)

        # For roughness and relief, inspect their strict global maxima.
        if name != "SLOPE":
            mask = finite & strict
            print("\nTop five — ALL LATITUDES, STRICT 9/9 SUPPORT:")
            report_top(metric, mask, elevation, counts)

    print("\nAudit complete. No original files modified.")


if __name__ == "__main__":
    main()
