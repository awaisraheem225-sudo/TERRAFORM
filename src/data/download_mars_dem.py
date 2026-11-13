
from pathlib import Path
import re
import json

import numpy as np
import rasterio
import requests

from rasterio.transform import from_origin
from rasterio.crs import CRS


# ============================================================
# TERRAFORM — MARS MOLA DATA ACQUISITION
# NASA Mars Global Surveyor / MOLA MEGDR
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "raw" / "mars"
OUTPUT_DIR = ROOT / "data" / "processed" / "mars"
REPORT_DIR = ROOT / "outputs" / "reports"

BASE_URL = (
    "https://pds-geosciences.wustl.edu/"
    "mgs/mgs-m-mola-5-megdr-l3-v1/"
    "mgsl_300x/meg128/"
)

PRODUCT = "megt44n000hb"

LABEL_PATH = RAW_DIR / f"{PRODUCT}.lbl"
IMAGE_PATH = RAW_DIR / f"{PRODUCT}.img"

OUTPUT_TIF = OUTPUT_DIR / f"mars_mola_128ppd_{PRODUCT}.tif"
REPORT_PATH = REPORT_DIR / "mars_mola_metadata.json"


# Values verified against the downloaded PDS label
ROWS = 5632
COLUMNS = 11520
BITS_PER_SAMPLE = 16
EXPECTED_BYTES = ROWS * COLUMNS * (BITS_PER_SAMPLE // 8)

PIXELS_PER_DEGREE = 128.0
WEST = 0.0
EAST = 90.0
NORTH = 44.0
SOUTH = 0.0

MARS_RADIUS_KM = 3396.0

# Jezero Crater approximate reference position
JEZERO_LAT = 18.4
JEZERO_LON = 77.7


# ============================================================
# DIRECTORY SETUP
# ============================================================

def create_directories():
    for directory in [RAW_DIR, OUTPUT_DIR, REPORT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# DOWNLOAD
# ============================================================

def download_file(filename, destination):
    url = BASE_URL + filename

    print(f"\nDownloading: {filename}")

    response = requests.get(
        url,
        stream=True,
        timeout=(30, 120)
    )

    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(destination, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                file.write(chunk)
                downloaded += len(chunk)

                if total_size:
                    progress = downloaded / total_size * 100
                    print(
                        f"\rProgress: {progress:.1f}%",
                        end=""
                    )

    print("\nDownload completed.")


def acquire_files():
    print("\n[1/5] CHECKING NASA PDS FILES")

    if not LABEL_PATH.exists():
        download_file(
            f"{PRODUCT}.lbl",
            LABEL_PATH
        )
    else:
        print("Existing PDS label found. Reusing it.")

    if not IMAGE_PATH.exists():
        download_file(
            f"{PRODUCT}.img",
            IMAGE_PATH
        )
    else:
        print("Existing binary image found. Reusing it.")


# ============================================================
# PDS LABEL READING
# ============================================================

def get_label_value(text, field):
    """
    Extract a PDS label value.

    Supports both:
        FIELD = VALUE

    and:
        FIELD
        = VALUE
    """

    pattern = (
        r"(?im)^\s*"
        + re.escape(field)
        + r"\s*(?:\r?\n\s*)?=\s*([^\r\n]+)"
    )

    match = re.search(pattern, text)

    if not match:
        return None

    return match.group(1).strip()


def clean_label_value(value):
    if value is None:
        return None

    return value.strip().strip('"').strip("'")


def validate_label():
    print("\n[2/5] VALIDATING NASA PDS LABEL")

    if not LABEL_PATH.exists():
        raise FileNotFoundError(LABEL_PATH)

    text = LABEL_PATH.read_text(
        encoding="utf-8",
        errors="replace"
    )

    required = {
        "LINES": ROWS,
        "LINE_SAMPLES": COLUMNS,
        "SAMPLE_BITS": BITS_PER_SAMPLE
    }

    for field, expected in required.items():
        value = get_label_value(text, field)

        if value is None:
            raise ValueError(
                f"Missing PDS metadata field: {field}"
            )

        actual = int(value.strip())

        if actual != expected:
            raise ValueError(
                f"{field} mismatch: expected {expected}, "
                f"found {actual}"
            )

        print(f"{field}: {actual} — PASS")

    sample_type = clean_label_value(
        get_label_value(text, "SAMPLE_TYPE")
    )

    if sample_type is None:
        raise ValueError("SAMPLE_TYPE is missing.")

    normalized_type = sample_type.upper().replace(" ", "").replace("_", "")

    if normalized_type != "MSBINTEGER":
        raise ValueError(
            f"Unexpected sample type: {sample_type}"
        )

    print(f"SAMPLE_TYPE: {sample_type} — PASS")

    projection = clean_label_value(
        get_label_value(text, "MAP_PROJECTION_TYPE")
    )

    if projection is None or projection.upper() != "SIMPLE CYLINDRICAL":
        raise ValueError(
            f"Unexpected map projection: {projection}"
        )

    print(f"MAP_PROJECTION_TYPE: {projection} — PASS")

    radius = get_label_value(text, "A_AXIS_RADIUS")

    if radius is not None:
        radius_value = float(radius.split()[0])

        if abs(radius_value - MARS_RADIUS_KM) > 0.01:
            raise ValueError(
                f"Unexpected Mars radius: {radius_value} km"
            )

    print("Mars reference radius: 3396 km — PASS")

    resolution = get_label_value(text, "MAP_RESOLUTION")

    if resolution is not None:
        resolution_value = float(resolution.split()[0])

        if abs(resolution_value - PIXELS_PER_DEGREE) > 0.001:
            raise ValueError(
                f"Unexpected map resolution: {resolution_value}"
            )

    print("Map resolution: 128 pixels/degree — PASS")

    longitude_direction = clean_label_value(
        get_label_value(text, "POSITIVE_LONGITUDE_DIRECTION")
    )

    if longitude_direction and longitude_direction.upper() != "EAST":
        raise ValueError(
            f"Unexpected longitude direction: {longitude_direction}"
        )

    print("Longitude direction: EAST — PASS")

    return text


# ============================================================
# BINARY VALIDATION
# ============================================================

def validate_binary():
    print("\n[3/5] VALIDATING MOLA BINARY DATA")

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(IMAGE_PATH)

    actual_bytes = IMAGE_PATH.stat().st_size

    print(f"Expected bytes: {EXPECTED_BYTES:,}")
    print(f"Actual bytes:   {actual_bytes:,}")

    if actual_bytes != EXPECTED_BYTES:
        raise ValueError(
            "Binary file size mismatch. "
            "The file may be incomplete or corrupted."
        )

    print("Binary file size validation — PASS")


# ============================================================
# MOLA IMAGE CONVERSION
# ============================================================

def convert_to_geotiff():
    print("\n[4/5] CONVERTING MOLA DATA TO GEOTIFF")

    # MOLA topography uses signed 16-bit,
    # most-significant-byte-first integers.
    elevation = np.fromfile(
        IMAGE_PATH,
        dtype=">i2"
    )

    if elevation.size != ROWS * COLUMNS:
        raise ValueError(
            "Unexpected number of pixels in binary image."
        )

    elevation = elevation.reshape(
        (ROWS, COLUMNS)
    ).astype(np.int16)

    # Mars-specific geographic CRS.
    # Coordinates are planetocentric, east-positive.
    mars_crs = CRS.from_proj4(
        "+proj=longlat "
        "+a=3396000 "
        "+b=3396000 "
        "+no_defs "
        "+type=crs"
    )

    # Each pixel represents 1/128 degree.
    # Top-left outer corner is 0E, 44N.
    transform = from_origin(
        WEST,
        NORTH,
        1.0 / PIXELS_PER_DEGREE,
        1.0 / PIXELS_PER_DEGREE
    )

    minimum = int(elevation.min())
    maximum = int(elevation.max())
    mean = float(elevation.mean())
    standard_deviation = float(elevation.std())

    print(f"Raster shape: {elevation.shape}")
    print(f"Minimum elevation: {minimum} m")
    print(f"Maximum elevation: {maximum} m")
    print(f"Mean elevation: {mean:.2f} m")
    print(f"Standard deviation: {standard_deviation:.2f} m")

    # The published whole-dataset observation range
    # is -8208 m to 21249 m.
    if minimum < -8208 or maximum > 21249:
        print(
            "WARNING: Values fall outside the published "
            "whole-dataset observation range. "
            "Inspect the product before further analysis."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    profile = {
        "driver": "GTiff",
        "height": ROWS,
        "width": COLUMNS,
        "count": 1,
        "dtype": "int16",
        "crs": mars_crs,
        "transform": transform,
        "compress": "deflate",
        "predictor": 2,
        "BIGTIFF": "IF_SAFER"
    }

    with rasterio.open(
        OUTPUT_TIF,
        "w",
        **profile
    ) as destination:

        destination.write(
            elevation,
            1
        )

        destination.set_band_description(
            1,
            "MOLA median topography above Martian areoid"
        )

        destination.update_tags(
            SOURCE="NASA Mars Global Surveyor MOLA MEGDR",
            PRODUCT=PRODUCT,
            UNITS="metres",
            PROJECTION="Simple Cylindrical",
            COORDINATE_SYSTEM="Planetocentric",
            LONGITUDE_DIRECTION="East",
            RESOLUTION="128 pixels per degree",
            MARS_REFERENCE_RADIUS_KM=str(MARS_RADIUS_KM),
            DATA_DESCRIPTION=(
                "Median observed topography. "
                "Interpolated values are supplied where "
                "no observations lie within a grid cell."
            )
        )

    print(f"\nGeoTIFF created:\n{OUTPUT_TIF}")


# ============================================================
# OUTPUT INSPECTION AND JEZERO CHECK
# ============================================================

def inspect_dataset():
    print("\n[5/5] INSPECTING OUTPUT AND CHECKING JEZERO")

    with rasterio.open(OUTPUT_TIF) as dataset:

        print("\nRASTER METADATA")
        print("------------------------------")
        print(f"Width: {dataset.width}")
        print(f"Height: {dataset.height}")
        print(f"CRS: {dataset.crs}")
        print(f"Bounds: {dataset.bounds}")
        print(f"Resolution: {dataset.res}")

        band = dataset.read(1)

        minimum = int(band.min())
        maximum = int(band.max())
        mean = float(band.mean())

        print("\nELEVATION STATISTICS")
        print("------------------------------")
        print(f"Minimum: {minimum} m")
        print(f"Maximum: {maximum} m")
        print(f"Mean: {mean:.2f} m")

        row, col = dataset.index(
            JEZERO_LON,
            JEZERO_LAT
        )

        print("\nJEZERO CRATER REFERENCE CHECK")
        print("------------------------------")
        print(f"Reference latitude: {JEZERO_LAT}")
        print(f"Reference longitude: {JEZERO_LON}")
        print(f"Raster row: {row}")
        print(f"Raster column: {col}")

        if not (
            0 <= row < dataset.height
            and 0 <= col < dataset.width
        ):
            raise ValueError(
                "Jezero reference coordinate is outside the raster."
            )

        jezero_elevation = int(band[row, col])

        print(
            f"Approximate MOLA elevation at reference pixel: "
            f"{jezero_elevation} m"
        )

        report = {
            "dataset": "NASA MOLA MEGDR",
            "product": PRODUCT,
            "source_file": IMAGE_PATH.name,
            "output_file": str(OUTPUT_TIF.relative_to(ROOT)),
            "dimensions": {
                "rows": ROWS,
                "columns": COLUMNS
            },
            "data_type": "Signed 16-bit big-endian source",
            "units": "metres",
            "projection": "Simple Cylindrical",
            "coordinate_system": "Planetocentric",
            "longitude_direction": "East",
            "resolution_pixels_per_degree": PIXELS_PER_DEGREE,
            "mars_reference_radius_km": MARS_RADIUS_KM,
            "coverage": {
                "west": WEST,
                "east": EAST,
                "south": SOUTH,
                "north": NORTH
            },
            "elevation_statistics_m": {
                "minimum": minimum,
                "maximum": maximum,
                "mean": mean,
                "standard_deviation": float(band.std())
            },
            "jezero_reference": {
                "latitude": JEZERO_LAT,
                "longitude": JEZERO_LON,
                "row": int(row),
                "column": int(col),
                "sample_elevation_m": jezero_elevation
            },
            "scientific_limitation": (
                "MOLA MEGDR cells represent median observed "
                "topography. Some cells contain interpolated "
                "values where observations are absent."
            )
        }

    REPORT_PATH.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8"
    )

    print(f"\nMetadata report saved:\n{REPORT_PATH}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("TERRAFORM — MARS MOLA DATA ACQUISITION")
    print("=" * 60)

    create_directories()
    acquire_files()
    validate_label()
    validate_binary()
    convert_to_geotiff()
    inspect_dataset()

    print("\n" + "=" * 60)
    print("MARS MOLA ACQUISITION COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
