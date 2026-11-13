
"""
TERRAFORM — Planetary Dataset Registry
Version: 1.0

Central metadata registry for global Earth, Mars and Moon datasets.
This module records data provenance; it does not download or process data.
"""

import json
from pathlib import Path
from datetime import datetime, timezone


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "Global"
OUTPUT_FILE = OUTPUT_DIR / "dataset_catalog.json"


DATASETS = [
    {
        "dataset_id": "EARTH_ETOPO_2022",
        "planet": "Earth",
        "dataset_name": "ETOPO 2022 Global Relief Model",
        "provider": "NOAA National Centers for Environmental Information",
        "instrument": "Multi-source elevation and bathymetry compilation",
        "product": "Global relief model",
        "spatial_coverage": "Global",
        "nominal_resolution": "60 arc-seconds",
        "approximate_resolution_at_equator": "1.85 km",
        "measurement": "Elevation",
        "measurement_unit": "metres",
        "coordinate_reference": "Geographic coordinates; inspect source metadata",
        "vertical_reference": "Verify against selected product metadata",
        "format": ["GeoTIFF", "NetCDF"],
        "role": "Global Earth terrain overview and screening",
        "source_url": (
            "https://www.ncei.noaa.gov/products/"
            "etopo-global-relief-model"
        ),
        "citation": (
            "NOAA National Centers for Environmental Information. "
            "2022. ETOPO 2022 Global Relief Model."
        ),
        "processing_status": "catalogued_not_downloaded",
        "scientific_notes": [
            "Includes terrestrial elevation and ocean bathymetry.",
            "Ocean cells must be excluded from terrestrial analog screening.",
            "Global overview resolution is not equivalent to regional DEM resolution."
        ]
    },
    {
        "dataset_id": "MARS_MOLA_MEGDR",
        "planet": "Mars",
        "dataset_name": "MOLA Mission Experiment Gridded Data Record",
        "provider": "NASA Planetary Data System",
        "instrument": "Mars Orbiter Laser Altimeter",
        "product": "MEGDR global topography",
        "spatial_coverage": "Global",
        "nominal_resolution": "16 pixels per degree",
        "approximate_resolution_at_equator": "3.7 km",
        "measurement": "Topographic elevation",
        "measurement_unit": "metres",
        "coordinate_reference": "Planetocentric/planetographic conventions must be verified",
        "vertical_reference": "Martian areoid reference; verify product label",
        "format": ["PDS IMG", "PDS label"],
        "role": "Global Mars terrain overview and regional reference selection",
        "source_url": (
            "https://pds-geosciences.wustl.edu/"
            "missions/mgs/megdr.html"
        ),
        "citation": (
            "NASA Mars Global Surveyor. Mars Orbiter Laser Altimeter "
            "Mission Experiment Gridded Data Record."
        ),
        "processing_status": "catalogued_not_downloaded",
        "scientific_notes": [
            "MEGDR products are available at multiple spatial resolutions.",
            "PDS labels contain essential map projection and data information.",
            "Do not interpret raw IMG pixel values without decoding product metadata."
        ]
    },
    {
        "dataset_id": "MOON_LOLA_GDR",
        "planet": "Moon",
        "dataset_name": "LRO LOLA Gridded Data Record",
        "provider": "NASA Planetary Data System",
        "instrument": "Lunar Orbiter Laser Altimeter",
        "product": "Global cylindrical GDR topography",
        "spatial_coverage": "Global",
        "nominal_resolution": "16 pixels per degree",
        "approximate_resolution_at_equator": "1.9 km",
        "measurement": "Lunar topographic elevation",
        "measurement_unit": "metres after product scaling",
        "coordinate_reference": "Lunar map projection; verify product label",
        "vertical_reference": "Lunar radius reference; decode label scaling and offset",
        "format": ["PDS IMG", "GeoJPEG2000", "PDS label"],
        "role": "Global lunar terrain overview and regional reference selection",
        "source_url": (
            "https://pds-geosciences.wustl.edu/"
            "missions/lro/lola.htm"
        ),
        "citation": (
            "NASA Lunar Reconnaissance Orbiter. Lunar Orbiter Laser "
            "Altimeter Gridded Data Records."
        ),
        "processing_status": "catalogued_not_downloaded",
        "scientific_notes": [
            "LOLA GDR products contain gridded altimetry measurements.",
            "Pixel values may require a scale factor and offset.",
            "Global and polar products use different map projections."
        ]
    }
]


def validate_catalog(datasets):
    required_fields = [
        "dataset_id",
        "planet",
        "dataset_name",
        "provider",
        "spatial_coverage",
        "measurement",
        "measurement_unit",
        "source_url",
        "processing_status"
    ]

    errors = []

    for dataset in datasets:
        for field in required_fields:
            if not dataset.get(field):
                errors.append(
                    f"{dataset.get('dataset_id', 'UNKNOWN')}: "
                    f"missing {field}"
                )

    identifiers = [item["dataset_id"] for item in datasets]

    if len(identifiers) != len(set(identifiers)):
        errors.append("Duplicate dataset identifiers detected.")

    return errors


def build_catalog():
    errors = validate_catalog(DATASETS)

    if errors:
        raise ValueError("\n".join(errors))

    return {
        "project": "TERRAFORM",
        "project_title": "Planetary Analog Intelligence",
        "catalog_version": "1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_count": len(DATASETS),
        "planets": ["Earth", "Mars", "Moon"],
        "datasets": DATASETS,
        "methodological_note": (
            "Dataset resolutions, coordinate systems, vertical references "
            "and measurement conventions must be verified before "
            "cross-planetary comparison. Global screening and regional "
            "analysis are separate processing levels."
        )
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    catalog = build_catalog()

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(catalog, file, indent=4, ensure_ascii=False)

    print("=" * 55)
    print("TERRAFORM — PLANETARY DATASET REGISTRY")
    print("=" * 55)

    for dataset in catalog["datasets"]:
        print(
            f"{dataset['planet']:8} | "
            f"{dataset['dataset_name']}"
        )

    print("-" * 55)
    print(f"Datasets registered: {catalog['dataset_count']}")
    print(f"Output: {OUTPUT_FILE}")
    print("CATALOG VALIDATION: PASSED")
    print("=" * 55)


if __name__ == "__main__":
    main()
