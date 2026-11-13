
from pathlib import Path

# ==========================================
# TERRAFORM - PROJECT CONFIGURATION
# ==========================================

PROJECT_NAME = "TERRAFORM"
PROJECT_SUBTITLE = "Planetary Analog Intelligence"

PROJECT_ROOT = Path(__file__).resolve().parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"

EARTH_RAW_DIR = DATA_DIR / "raw" / "earth"
MARS_RAW_DIR = DATA_DIR / "raw" / "mars"

PROCESSED_DIR = DATA_DIR / "processed"
REFERENCE_DIR = DATA_DIR / "reference"

# Source directories
SRC_DIR = PROJECT_ROOT / "src"

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MAPS_DIR = OUTPUT_DIR / "maps"
REPORTS_DIR = OUTPUT_DIR / "reports"

# Documentation
DOCS_DIR = PROJECT_ROOT / "docs"

# Planetary research configuration
TARGET_PLANET = "Mars"
REFERENCE_REGION = "Jezero Crater"
REFERENCE_ROVER = "Perseverance"

# Initial research dimensions
ANALYSIS_DIMENSIONS = [
    "Terrain",
    "Elevation",
    "Slope",
    "Surface Roughness",
    "Geology",
    "Temperature",
    "Aridity"
]

# Project version
VERSION = "2.0.0"

# Ensure project directories exist
for directory in [
    EARTH_RAW_DIR,
    MARS_RAW_DIR,
    PROCESSED_DIR,
    REFERENCE_DIR,
    MAPS_DIR,
    REPORTS_DIR,
    DOCS_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)


def project_info():
    return {
        "name": PROJECT_NAME,
        "version": VERSION,
        "target_planet": TARGET_PLANET,
        "reference_region": REFERENCE_REGION,
        "dimensions": ANALYSIS_DIMENSIONS
    }


if __name__ == "__main__":
    print("TERRAFORM CONFIGURATION")
    print("-" * 40)

    for key, value in project_info().items():
        print(f"{key}: {value}")

    print("\nProject root:", PROJECT_ROOT)
    print("Configuration loaded successfully.")
