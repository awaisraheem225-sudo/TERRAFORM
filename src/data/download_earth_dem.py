
from pathlib import Path
import requests
import xml.etree.ElementTree as ET
import rasterio

# ============================================================
# TERRAFORM — EARTH DATA ACQUISITION
# Dataset: Copernicus DEM GLO-30 Public
# Pilot: San Pedro de Atacama, Chile
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "earth"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BUCKET_URL = "https://copernicus-dem-30m.s3.amazonaws.com"
CATALOGUE_URL = f"{BUCKET_URL}/tileList.txt"

TILE_NAME = "Copernicus_DSM_COG_10_S23_00_W069_00_DEM"

OUTPUT_FILE = OUTPUT_DIR / f"{TILE_NAME}.tif"

HEADERS = {
    "User-Agent": "TERRAFORM-Planetary-Analog-Intelligence/1.0"
}


def download_catalogue():
    print("\n[1/5] Checking official Copernicus tile catalogue...")

    response = requests.get(
        CATALOGUE_URL,
        headers=HEADERS,
        timeout=60
    )
    response.raise_for_status()

    if TILE_NAME not in response.text:
        raise RuntimeError(
            "The selected Atacama tile was not found "
            "in the official public catalogue."
        )

    print("Catalogue check passed.")
    print("Selected tile:", TILE_NAME)


def locate_geotiff():
    print("\n[2/5] Locating the GeoTIFF in the public dataset...")

    params = {
        "list-type": "2",
        "prefix": TILE_NAME + "/"
    }

    response = requests.get(
        BUCKET_URL,
        params=params,
        headers=HEADERS,
        timeout=60
    )
    response.raise_for_status()

    root = ET.fromstring(response.content)

    namespace = {
        "s3": "http://s3.amazonaws.com/doc/2006-03-01/"
    }

    keys = [
        item.text
        for item in root.findall(".//s3:Contents/s3:Key", namespace)
        if item.text
    ]

    geotiffs = [
        key for key in keys
        if key.lower().endswith((".tif", ".tiff"))
    ]

    if not geotiffs:
        raise RuntimeError(
            "No GeoTIFF was found in the selected tile folder. "
            "Check the AWS catalogue response."
        )

    # Prefer the main tile file, not auxiliary files.
    main_files = [
        key for key in geotiffs
        if key.endswith(TILE_NAME + ".tif")
    ]

    selected_key = main_files[0] if main_files else geotiffs[0]

    print("GeoTIFF located.")
    print("Object:", selected_key)

    return f"{BUCKET_URL}/{selected_key}"


def download_raster(url):
    print("\n[3/5] Downloading Earth elevation data...")

    if OUTPUT_FILE.exists():
        print("Existing file found:", OUTPUT_FILE.name)
        print("Skipping download.")
        return

    temporary_file = OUTPUT_FILE.with_suffix(".download")

    try:
        with requests.get(
            url,
            headers=HEADERS,
            stream=True,
            timeout=(30, 300)
        ) as response:

            response.raise_for_status()

            total_bytes = int(
                response.headers.get("content-length", 0)
            )

            downloaded = 0

            with open(temporary_file, "wb") as file:
                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if chunk:
                        file.write(chunk)
                        downloaded += len(chunk)

                        if total_bytes:
                            percent = downloaded / total_bytes * 100
                            print(
                                f"\rDownloaded: {percent:.1f}%",
                                end=""
                            )

        temporary_file.replace(OUTPUT_FILE)
        print("\nDownload completed.")

    except Exception:
        if temporary_file.exists():
            temporary_file.unlink()
        raise


def inspect_raster():
    print("\n[4/5] Inspecting downloaded raster...")

    with rasterio.open(OUTPUT_FILE) as dataset:

        print("\n========== RASTER INFORMATION ==========")
        print("File:", OUTPUT_FILE.name)
        print("Driver:", dataset.driver)
        print("Width:", dataset.width, "pixels")
        print("Height:", dataset.height, "pixels")
        print("Bands:", dataset.count)
        print("Coordinate system:", dataset.crs)
        print("Bounds:", dataset.bounds)
        print("Pixel size:", dataset.res)
        print("Data type:", dataset.dtypes)
        print("NoData value:", dataset.nodata)

        band = dataset.read(1, masked=True)

        if band.count() > 0:
            print("Minimum elevation:", round(float(band.min()), 2), "m")
            print("Maximum elevation:", round(float(band.max()), 2), "m")
            print("Mean elevation:", round(float(band.mean()), 2), "m")
            print("Valid pixels:", int(band.count()))
        else:
            raise RuntimeError("Raster contains no valid elevation pixels.")

        print("========================================")


def main():
    print("=" * 55)
    print(" TERRAFORM — EARTH DATA ACQUISITION")
    print(" Copernicus DEM GLO-30 | Atacama Pilot")
    print("=" * 55)

    try:
        download_catalogue()

        url = locate_geotiff()

        download_raster(url)

        inspect_raster()

        print("\n[5/5] DATA ACQUISITION SUCCESSFUL")
        print("Saved to:", OUTPUT_FILE)

    except Exception as error:
        print("\nDATA ACQUISITION FAILED")
        print("Reason:", error)
        raise


if __name__ == "__main__":
    main()
