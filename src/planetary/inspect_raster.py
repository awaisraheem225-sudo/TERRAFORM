
from pathlib import Path
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parents[2]

raster_path = (
    PROJECT_ROOT
    / "data"
    / "Global"
    / "Earth"
    / "earth_etopo_2022_60s.tif"
)

print("\nTERRAFORM — RASTER INSPECTION")
print("=" * 45)

if not raster_path.exists():
    raise FileNotFoundError(
        f"Dataset not found: {raster_path}"
    )

with rasterio.open(raster_path) as src:
    print("FILE:", raster_path.name)
    print("DRIVER:", src.driver)
    print("WIDTH:", src.width)
    print("HEIGHT:", src.height)
    print("BANDS:", src.count)
    print("CRS:", src.crs)
    print("BOUNDS:", src.bounds)
    print("RESOLUTION:", src.res)
    print("NODATA:", src.nodata)
    print("DATA TYPE:", src.dtypes)
    print("FILE SIZE (MB):", round(
        raster_path.stat().st_size / (1024 ** 2), 2
    ))

    print("\nRASTER VALIDATION")

    assert src.count >= 1
    assert src.width > 0
    assert src.height > 0
    assert src.crs is not None

    print("RASTER VALIDATION: PASSED")

print("=" * 45)
