
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

ROWS, COLS = 2880, 5760
STEP = 0.0625

COUNTS_FILE = ROOT / "data/Global/Mars/megc90n000eb.img"
METRIC_DIR = ROOT / "outputs/global_mars/terrain_metrics"

METRICS = {
    "SLOPE": "mars_global_slope_degrees.npy",
    "ROUGHNESS": "mars_global_roughness_3x3_m.npy",
    "LOCAL_RELIEF": "mars_global_local_relief_3x3_m.npy",
}

THRESHOLDS = [1, 5, 10, 20, 30]
LATITUDE_CUTOFFS = [80.0, 85.0, 89.5]


def minimum_neighbourhood_count(counts):
    result = np.full((ROWS, COLS), 32767, dtype=np.int16)

    for dr in (-1, 0, 1):
        r0 = max(0, -dr)
        r1 = min(ROWS, ROWS - dr)

        for dc in (-1, 0, 1):
            shifted = np.roll(
                counts[r0 + dr:r1 + dr],
                shift=-dc,
                axis=1,
            )
            result[r0:r1] = np.minimum(
                result[r0:r1], shifted
            )

    return result


def main():
    counts = np.memmap(
        COUNTS_FILE,
        dtype=">i2",
        mode="r",
        shape=(ROWS, COLS),
    )

    minimum_count = minimum_neighbourhood_count(counts)

    latitude = 90.0 - (
        np.arange(ROWS) + 0.5
    ) * STEP

    abs_latitude = np.abs(latitude)

    print("=" * 85)
    print("TERRAFORM — MOLA OBSERVATION THRESHOLD AUDIT")
    print("=" * 85)
    print("Threshold means every cell in the 3x3 neighbourhood")
    print("must have at least that many MOLA observations.")
    print()

    for name, filename in METRICS.items():
        path = METRIC_DIR / filename

        if not path.exists():
            print(f"Missing: {path}")
            continue

        metric = np.load(path, mmap_mode="r")
        finite = np.isfinite(metric)

        print("\n" + "=" * 85)
        print(name)
        print("=" * 85)

        print(
            f"{'Min count':>10} | {'Latitude cutoff':>16} | "
            f"{'Valid cells':>14} | {'Maximum':>14}"
        )
        print("-" * 85)

        for threshold in THRESHOLDS:
            support = minimum_count >= threshold

            for cutoff in LATITUDE_CUTOFFS:
                mask = (
                    finite
                    & support
                    & (abs_latitude < cutoff)[:, None]
                )

                n = int(np.count_nonzero(mask))

                maximum = (
                    float(np.max(metric[mask]))
                    if n else float("nan")
                )

                print(
                    f"{threshold:10d} | "
                    f"{cutoff:13.1f}° | "
                    f"{n:14,} | "
                    f"{maximum:14.6f}"
                )

    print("\nAudit complete. Original rasters unchanged.")


if __name__ == "__main__":
    main()
