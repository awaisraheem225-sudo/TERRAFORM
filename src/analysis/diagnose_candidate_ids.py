
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

ROOT = Path(__file__).resolve().parents[2]

RASTER = (
    ROOT / "outputs" / "maps" /
    "candidate_regions" / "atacama_candidate_region_ids.tif"
)

CSV = (
    ROOT / "outputs" / "maps" /
    "candidate_regions" / "atacama_candidate_inventory.csv"
)

print("\nTERRAFORM CANDIDATE ID DIAGNOSTIC")
print("=" * 55)

inventory = pd.read_csv(CSV).sort_values("rank_by_area")

with rasterio.open(RASTER) as src:

    data = src.read(1)

    print("\nRASTER INFORMATION")
    print("Shape:", data.shape)
    print("CRS:", src.crs)
    print("Data type:", data.dtype)
    print("NoData:", src.nodata)

    values, counts = np.unique(data, return_counts=True)

    print("\nRASTER LABEL COUNTS")
    print("Unique raster values:", len(values))

    raster_counts = {}

    for value, count in zip(values, counts):
        if value == 0 or value == src.nodata:
            continue

        raster_counts[int(value)] = int(count)

        if len(raster_counts) <= 20:
            print(f"Label {int(value)}: {int(count)} pixels")

print("\nINVENTORY PIXEL COUNTS")
print(
    inventory[
        ["candidate_id", "rank_by_area", "pixel_count", "area_km2"]
    ].head(10).to_string(index=False)
)

print("\nDIRECT RANK-TO-RASTER CHECK")

for _, row in inventory.head(10).iterrows():

    rank = int(row["rank_by_area"])
    expected = int(row["pixel_count"])
    actual = raster_counts.get(rank, 0)

    print(
        f"{row['candidate_id']}: "
        f"expected={expected}, "
        f"raster_label_{rank}={actual}, "
        f"match={expected == actual}"
    )

print("\nPIXEL COUNT MATCH SEARCH")

for _, row in inventory.head(10).iterrows():

    expected = int(row["pixel_count"])

    matches = [
        label
        for label, count in raster_counts.items()
        if count == expected
    ]

    print(
        f"{row['candidate_id']} "
        f"({expected} pixels) -> raster labels {matches}"
    )

print("\nDIAGNOSTIC COMPLETED")
