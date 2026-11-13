
from pathlib import Path
import json
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from rasterio.features import shapes
from shapely.geometry import shape
from shapely.ops import unary_union


ROOT = Path(__file__).resolve().parents[2]

GEOLOGY_DIR = ROOT / "data/raw/earth/geology"

CANDIDATE_RASTER = (
    ROOT / "outputs/maps/candidate_regions/"
    "atacama_candidate_region_ids.tif"
)

INVENTORY_CSV = (
    ROOT / "outputs/maps/candidate_regions/"
    "atacama_candidate_inventory.csv"
)

OUTPUT_DIR = ROOT / "outputs/maps/geological_characterization"
REPORT_DIR = ROOT / "outputs/reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TOP_N = 10

FIELDS = [
    "region",
    "cod_geolog",
    "lithostrat",
    "rock_rank",
    "age_min",
    "age_max",
    "lithology"
]


def repair_geometry(geom):

    if geom is None or geom.is_empty:
        return None

    if geom.is_valid:
        return geom

    try:
        from shapely import make_valid
        geom = make_valid(geom)
    except ImportError:
        geom = geom.buffer(0)

    if geom is None or geom.is_empty:
        return None

    return geom


def main():

    print("\nTERRAFORM GEOLOGICAL CHARACTERIZATION V2")
    print("=" * 60)

    # -------------------------------------------------
    # 1. LOAD INVENTORY
    # -------------------------------------------------

    print("\n[1/8] LOADING INVENTORY")

    inventory = pd.read_csv(INVENTORY_CSV)
    inventory = inventory.sort_values("rank_by_area")

    selected = inventory.head(TOP_N).copy()

    print("Selected candidates:", len(selected))

    # -------------------------------------------------
    # 2. READ RASTER AND DISCOVER LABELS
    # -------------------------------------------------

    print("\n[2/8] DISCOVERING ACTUAL RASTER LABELS")

    with rasterio.open(CANDIDATE_RASTER) as src:

        labels = src.read(1)
        raster_crs = src.crs
        transform = src.transform
        pixel_area_km2 = abs(
            transform.a * transform.e -
            transform.b * transform.d
        ) / 1_000_000

        values, counts = np.unique(
            labels[labels > 0],
            return_counts=True
        )

        # Raster labels are ordered by their pixel counts,
        # not by their numeric label values.
        order = np.argsort(counts)[::-1]

        ranked_labels = values[order]
        ranked_counts = counts[order]

        if len(ranked_labels) != len(inventory):
            raise RuntimeError(
                f"Raster has {len(ranked_labels)} regions, "
                f"inventory has {len(inventory)}."
            )

        label_mapping = {}

        for (_, row), label, count in zip(
            inventory.iterrows(),
            ranked_labels,
            ranked_counts
        ):

            expected = int(row["pixel_count"])

            if expected != int(count):
                raise RuntimeError(
                    f"Pixel mismatch for {row['candidate_id']}: "
                    f"expected {expected}, found {count}."
                )

            label_mapping[int(label)] = {
                "candidate_id": row["candidate_id"],
                "rank": int(row["rank_by_area"]),
                "expected_pixels": expected,
                "expected_area_km2": float(row["area_km2"])
            }

        print("All 873 raster labels matched inventory.")
        print("Top candidate label mapping verified.")

        # -------------------------------------------------
        # 3. POLYGONIZE CORRECT LABELS
        # -------------------------------------------------

        print("\n[3/8] POLYGONIZING CORRECT CANDIDATES")

        candidate_features = []

        for label, info in label_mapping.items():

            if info["rank"] > TOP_N:
                continue

            mask = labels == label

            geometries = [
                shape(geom)
                for geom, value in shapes(
                    labels,
                    mask=mask,
                    transform=transform,
                    connectivity=4
                )
                if int(value) == label
            ]

            if not geometries:
                raise RuntimeError(
                    f"No polygon generated for {info['candidate_id']}"
                )

            geom = unary_union(geometries)

            actual_area = geom.area / 1_000_000
            expected_area = info["expected_area_km2"]

            # Allow a small floating point tolerance.
            if not np.isclose(
                actual_area,
                expected_area,
                rtol=0.001,
                atol=0.001
            ):
                raise RuntimeError(
                    f"Polygon area mismatch for {info['candidate_id']}: "
                    f"{actual_area} vs {expected_area} km2"
                )

            candidate_features.append({
                "candidate_id": info["candidate_id"],
                "rank_by_area": info["rank"],
                "pixel_count": info["expected_pixels"],
                "inventory_area_km2": expected_area,
                "geometry": geom
            })

        candidates = gpd.GeoDataFrame(
            candidate_features,
            geometry="geometry",
            crs=raster_crs
        )

    print("Correct candidate polygons:", len(candidates))
    print(
        candidates[
            ["candidate_id", "pixel_count", "inventory_area_km2"]
        ].to_string(index=False)
    )

    # -------------------------------------------------
    # 4. LOAD GEOLOGY
    # -------------------------------------------------

    print("\n[4/8] LOADING GEOLOGICAL MAP")

    shapefiles = list(GEOLOGY_DIR.glob("*.shp"))

    if len(shapefiles) != 1:
        raise RuntimeError(
            f"Expected one shapefile, found {len(shapefiles)}"
        )

    geology_path = shapefiles[0]
    geology = gpd.read_file(geology_path)

    missing_fields = [
        field for field in FIELDS
        if field not in geology.columns
    ]

    if missing_fields:
        raise RuntimeError(
            f"Missing geology fields: {missing_fields}"
        )

    source_count = len(geology)

    missing_geometry = (
        geology.geometry.isna() |
        geology.geometry.is_empty
    )

    missing_count = int(missing_geometry.sum())

    geology = geology.loc[~missing_geometry].copy()

    # -------------------------------------------------
    # 5. REPAIR GEOMETRIES
    # -------------------------------------------------

    print("\n[5/8] REPAIRING GEOLOGICAL GEOMETRIES")

    invalid_before = int((~geology.geometry.is_valid).sum())

    geology["geometry"] = geology.geometry.apply(
        repair_geometry
    )

    failed = (
        geology.geometry.isna() |
        geology.geometry.is_empty
    )

    failed_count = int(failed.sum())

    geology = geology.loc[~failed].copy()

    invalid_after = int((~geology.geometry.is_valid).sum())

    if invalid_after:
        raise RuntimeError(
            "Invalid geological geometries remain."
        )

    print("Missing geometries excluded:", missing_count)
    print("Invalid geometries repaired:", invalid_before)
    print("Failed repairs:", failed_count)

    # -------------------------------------------------
    # 6. REPROJECT AND FILTER
    # -------------------------------------------------

    print("\n[6/8] REPROJECTING DATASETS")

    if geology.crs is None:
        raise RuntimeError("Geology CRS is missing.")

    candidates = candidates.to_crs("EPSG:32719")
    geology = geology.to_crs("EPSG:32719")

    bounds = candidates.total_bounds

    geology = geology.cx[
        bounds[0]:bounds[2],
        bounds[1]:bounds[3]
    ].copy()

    if geology.empty:
        raise RuntimeError(
            "No geological polygons near the candidates."
        )

    print("Geological polygons retained:", len(geology))

    # -------------------------------------------------
    # 7. INTERSECTION
    # -------------------------------------------------

    print("\n[7/8] INTERSECTING CANDIDATES AND GEOLOGY")

    with warnings.catch_warnings():

        warnings.simplefilter("ignore")

        intersections = gpd.overlay(
            candidates,
            geology,
            how="intersection",
            keep_geom_type=False,
            make_valid=False
        )

    intersections = intersections.loc[
        intersections.geometry.area > 0
    ].copy()

    intersections["intersection_area_km2"] = (
        intersections.geometry.area / 1_000_000
    )

    # Geological unit coverage by candidate
    group_fields = ["candidate_id"] + FIELDS

    unit_summary = (
        intersections
        .groupby(group_fields, dropna=False)
        ["intersection_area_km2"]
        .sum()
        .reset_index()
    )

    area_lookup = candidates.set_index(
        "candidate_id"
    ).geometry.area.to_dict()

    area_lookup = {
        key: value / 1_000_000
        for key, value in area_lookup.items()
    }

    unit_summary["candidate_area_km2"] = (
        unit_summary["candidate_id"].map(area_lookup)
    )

    unit_summary["candidate_coverage_percent"] = (
        unit_summary["intersection_area_km2"] /
        unit_summary["candidate_area_km2"] * 100
    )

    unit_summary = unit_summary.sort_values(
        ["candidate_id", "intersection_area_km2"],
        ascending=[True, False]
    )

    # Unique mapped coverage per candidate
    coverage_records = []

    for candidate_id, group in intersections.groupby("candidate_id"):

        merged = unary_union(list(group.geometry))

        mapped_area = merged.area / 1_000_000
        candidate_area = area_lookup[candidate_id]

        coverage_records.append({
            "candidate_id": candidate_id,
            "candidate_area_km2": candidate_area,
            "mapped_geology_area_km2": mapped_area,
            "mapped_coverage_percent": (
                mapped_area / candidate_area * 100
            ),
            "unmapped_area_km2": max(
                0,
                candidate_area - mapped_area
            )
        })

    coverage = pd.DataFrame(coverage_records)

    # -------------------------------------------------
    # 8. SAVE VERIFIED OUTPUTS
    # -------------------------------------------------

    print("\n[8/8] SAVING VERIFIED RESULTS")

    unit_summary.to_csv(
        OUTPUT_DIR / "candidate_geological_units.csv",
        index=False
    )

    coverage.to_csv(
        OUTPUT_DIR / "candidate_geological_coverage.csv",
        index=False
    )

    intersections.to_file(
        OUTPUT_DIR / "candidate_geological_intersections.geojson",
        driver="GeoJSON"
    )

    candidates.to_file(
        OUTPUT_DIR / "verified_candidate_polygons.geojson",
        driver="GeoJSON"
    )

    report = {
        "version": 2,
        "source_file": geology_path.name,
        "source_feature_count": source_count,
        "missing_geometries_excluded": missing_count,
        "invalid_geometries_before_repair": invalid_before,
        "failed_repairs": failed_count,
        "invalid_geometries_remaining": invalid_after,
        "candidate_label_mapping_verified": True,
        "candidate_polygon_area_verified": True,
        "candidates_analyzed": len(candidates),
        "geological_polygons_near_candidates": len(geology),
        "intersection_records": len(intersections),
        "coverage": coverage.to_dict(orient="records"),
        "methodology": (
            "Candidate raster labels were matched to inventory "
            "records by pixel counts and polygon area verification. "
            "Regional geological polygons were intersected with "
            "the ten largest strict terrain-screening candidates."
        ),
        "limitations": [
            "Regional geological map scale limits site-level detail.",
            "Mapped lithology does not establish exact mineral composition.",
            "Geological resemblance does not establish Mars analogue suitability."
        ]
    }

    report_path = (
        REPORT_DIR / "candidate_geological_characterization_v2.json"
    )

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    print("\nVERIFIED GEOLOGICAL COVERAGE")
    print(coverage.to_string(index=False))

    print("\nTOP GEOLOGICAL UNITS")

    for candidate_id, group in unit_summary.groupby("candidate_id"):

        print(f"\n{candidate_id}")

        print(
            group[
                [
                    "cod_geolog",
                    "lithology",
                    "intersection_area_km2",
                    "candidate_coverage_percent"
                ]
            ].head(5).to_string(index=False)
        )

    print("\nAll verified outputs saved to:")
    print(OUTPUT_DIR)

    print("\nGEOLOGICAL CHARACTERIZATION V2 COMPLETED")


if __name__ == "__main__":
    main()
