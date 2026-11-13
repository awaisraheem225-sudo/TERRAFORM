
from pathlib import Path
import numpy as np

# ============================================================
# TERRAFORM — GLOBAL MARS OUTLIER VALIDATION
# Independent diagnostic; does not modify source rasters.
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

ROWS = 2880
COLS = 5760
SPACING = 0.0625

ELEVATION_FILE = (
    ROOT / "data" / "Global" / "Mars" / "megt90n000eb.img"
)
COUNT_FILE = (
    ROOT / "data" / "Global" / "Mars" / "megc90n000eb.img"
)
METRIC_DIR = ROOT / "outputs" / "global_mars" / "terrain_metrics"

OUTPUT_DIR = ROOT / "outputs" / "global_mars"

LATITUDE_CUTOFFS = [80.0, 85.0, 89.5]
TOP_N = 20
NEIGHBOURHOODS_TO_PRINT = 5

METRICS = {
    "SLOPE": "mars_global_slope_degrees.npy",
    "ROUGHNESS": "mars_global_roughness_3x3_m.npy",
    "LOCAL_RELIEF": "mars_global_local_relief_3x3_m.npy",
}


def coordinates(row, col):
    latitude = 90.0 - (row + 0.5) * SPACING
    longitude = (col + 0.5) * SPACING
    return latitude, longitude


def get_top_indices(metric, mask, number=TOP_N):
    """Return flat indices of the highest valid metric values."""

    values = np.asarray(metric).reshape(-1)
    valid_indices = np.flatnonzero(mask.reshape(-1))

    if len(valid_indices) == 0:
        return np.array([], dtype=np.int64)

    k = min(number, len(valid_indices))

    valid_values = values[valid_indices]
    selected = np.argpartition(valid_values, -k)[-k:]

    selected_indices = valid_indices[selected]
    selected_indices = selected_indices[
        np.argsort(values[selected_indices])[::-1]
    ]

    return selected_indices


def print_neighbourhood(elevation, counts, row, col):
    print("    Elevation neighbourhood (m):")

    for dr in (-1, 0, 1):
        line = []

        for dc in (-1, 0, 1):
            r = row + dr
            c = (col + dc) % COLS

            if r < 0 or r >= ROWS:
                line.append("OUT")
            else:
                line.append(f"{int(elevation[r, c]):7d}")

        print("      " + " ".join(line))

    print("    Observation-count neighbourhood:")

    for dr in (-1, 0, 1):
        line = []

        for dc in (-1, 0, 1):
            r = row + dr
            c = (col + dc) % COLS

            if r < 0 or r >= ROWS:
                line.append("OUT")
            else:
                line.append(f"{int(counts[r, c]):5d}")

        print("      " + " ".join(line))


def describe_top_pixels(
    metric,
    indices,
    elevation,
    counts,
    metric_name,
    selection_name,
):
    print("\n" + "-" * 75)
    print(f"{metric_name} — TOP {len(indices)}: {selection_name}")
    print("-" * 75)

    if len(indices) == 0:
        print("No valid pixels found.")
        return

    values = np.asarray(metric).reshape(-1)

    for rank, flat_index in enumerate(indices, start=1):
        row, col = np.unravel_index(flat_index, (ROWS, COLS))
        latitude, longitude = coordinates(row, col)

        print(
            f"\n#{rank:02d} | Value: {float(values[flat_index]):.6f}"
        )
        print(
            f"    Row: {row} | Column: {col}"
        )
        print(
            f"    Latitude: {latitude:.6f}°"
            f" | East longitude: {longitude:.6f}°"
        )
        print(
            f"    Elevation: {int(elevation[row, col])} m"
            f" | Centre observations: {int(counts[row, col])}"
        )

        if rank <= NEIGHBOURHOODS_TO_PRINT:
            print_neighbourhood(
                elevation, counts, row, col
            )


def main():
    print("=" * 75)
    print("TERRAFORM — GLOBAL MARS OUTLIER VALIDATION")
    print("=" * 75)

    for path in [ELEVATION_FILE, COUNT_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Missing input: {path}")

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

    observation_mask = np.asarray(counts) > 0

    latitude = (
        90.0 - (np.arange(ROWS) + 0.5) * SPACING
    )
    absolute_latitude = np.abs(latitude)

    for metric_name, filename in METRICS.items():
        metric_path = METRIC_DIR / filename

        if not metric_path.exists():
            print(f"\nMISSING METRIC: {metric_path}")
            continue

        metric = np.load(metric_path, mmap_mode="r")

        if metric.shape != (ROWS, COLS):
            print(f"\nINVALID SHAPE: {metric_name}: {metric.shape}")
            continue

        print("\n\n" + "=" * 75)
        print(f"METRIC: {metric_name}")
        print("=" * 75)

        finite_mask = np.isfinite(metric)

        # ----------------------------------------------------
        # 1. LATITUDE CUTOFF COMPARISON
        # ----------------------------------------------------

        print("\nLATITUDE CUTOFF COMPARISON")
        print("-" * 75)
        print(
            "Cutoff | Valid cells | All-cell maximum | "
            "Observed-centre maximum"
        )

        for cutoff in LATITUDE_CUTOFFS:
            latitude_mask = (
                absolute_latitude < cutoff
            )[:, None]

            all_mask = finite_mask & latitude_mask

            observed_mask = (
                all_mask & observation_mask
            )

            if np.any(all_mask):
                all_max = float(np.max(metric[all_mask]))
                all_cells = int(np.count_nonzero(all_mask))
            else:
                all_max = float("nan")
                all_cells = 0

            if np.any(observed_mask):
                observed_max = float(
                    np.max(metric[observed_mask])
                )
            else:
                observed_max = float("nan")

            print(
                f"{cutoff:5.1f}° | "
                f"{all_cells:12,} | "
                f"{all_max:16.6f} | "
                f"{observed_max:22.6f}"
            )

        # ----------------------------------------------------
        # 2. TOP 20 PIXELS ACROSS FULL VALID RASTER
        # ----------------------------------------------------

        full_latitude_mask = (
            absolute_latitude < 89.5
        )[:, None]

        full_mask = finite_mask & full_latitude_mask

        observed_full_mask = (
            full_mask & observation_mask
        )

        top_all = get_top_indices(
            metric, full_mask, TOP_N
        )

        top_observed = get_top_indices(
            metric, observed_full_mask, TOP_N
        )

        describe_top_pixels(
            metric,
            top_all,
            elevation,
            counts,
            metric_name,
            "ALL VALID CENTRE PIXELS",
        )

        describe_top_pixels(
            metric,
            top_observed,
            elevation,
            counts,
            metric_name,
            "OBSERVED CENTRE PIXELS ONLY",
        )

    print("\n\n" + "=" * 75)
    print("VALIDATION DIAGNOSTIC COMPLETE")
    print("No input data or existing outputs were modified.")
    print("=" * 75)


if __name__ == "__main__":
    main()
