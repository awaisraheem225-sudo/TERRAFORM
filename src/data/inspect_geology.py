
from pathlib import Path
import geopandas as gpd
import json

ROOT = Path(__file__).resolve().parents[2]
GEOLOGY_DIR = ROOT / "data" / "raw" / "earth" / "geology"
OUTPUT_DIR = ROOT / "outputs" / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\nTERRAFORM GEOLOGICAL DATA INSPECTION")
print("=" * 50)

# Find the actual shapefile without assuming its exact name
shapefiles = list(GEOLOGY_DIR.glob("*.shp"))

if not shapefiles:
    raise FileNotFoundError(
        f"No .shp file found inside: {GEOLOGY_DIR}"
    )

if len(shapefiles) > 1:
    raise RuntimeError(
        f"Multiple shapefiles found: {shapefiles}"
    )

shp_path = shapefiles[0]

print(f"\nShapefile: {shp_path.name}")

# Load geological data
gdf = gpd.read_file(shp_path)

print("\n[1] DATASET OVERVIEW")
print(f"Features: {len(gdf):,}")
print(f"CRS: {gdf.crs}")
print(f"Geometry types: {gdf.geom_type.value_counts().to_dict()}")

print("\n[2] ATTRIBUTE FIELDS")

for column in gdf.columns:
    print(f"\n{column}")
    print(f"  Data type: {gdf[column].dtype}")
    print(f"  Unique values: {gdf[column].nunique(dropna=True)}")
    print(f"  Missing values: {gdf[column].isna().sum()}")

    if gdf[column].dtype == "object":
        print("  Most frequent values:")
        print(gdf[column].value_counts(dropna=False).head(10).to_string())

print("\n[3] SPATIAL COVERAGE")
print("Bounds:", gdf.total_bounds)

# Check geometries
print("\n[4] GEOMETRY QUALITY")
print("Empty geometries:", gdf.geometry.is_empty.sum())
print("Invalid geometries:", (~gdf.geometry.is_valid).sum())

# Save inspection report
report = {
    "source_file": shp_path.name,
    "feature_count": len(gdf),
    "crs": str(gdf.crs),
    "geometry_types": {
        str(k): int(v)
        for k, v in gdf.geom_type.value_counts().items()
    },
    "bounds": [float(x) for x in gdf.total_bounds],
    "fields": {
        col: {
            "dtype": str(gdf[col].dtype),
            "unique_values": int(gdf[col].nunique(dropna=True)),
            "missing_values": int(gdf[col].isna().sum())
        }
        for col in gdf.columns
        if col != gdf.geometry.name
    },
    "invalid_geometries": int((~gdf.geometry.is_valid).sum()),
    "empty_geometries": int(gdf.geometry.is_empty.sum())
}

report_path = OUTPUT_DIR / "geology_data_inspection.json"

with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=4, ensure_ascii=False)

print("\nInspection report saved:")
print(report_path)

print("\nINSPECTION COMPLETED")
