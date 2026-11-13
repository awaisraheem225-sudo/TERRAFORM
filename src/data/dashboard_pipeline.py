
"""
TERRAFORM - Dashboard Data Integration
Connects uploaded CSV files to the validation and processing backend.
"""

from pathlib import Path
import pandas as pd

from src.data.processing import process_dataframe


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EARTH_RAW = PROJECT_ROOT / "data" / "raw" / "earth"
MARS_RAW = PROJECT_ROOT / "data" / "raw" / "mars"

EARTH_RAW.mkdir(parents=True, exist_ok=True)
MARS_RAW.mkdir(parents=True, exist_ok=True)


def load_and_process(uploaded_file, dataset_type):
    """
    Load a CSV from the dashboard or raw data directory,
    validate it, and return cleaned/rejected records and report.
    """

    dataset_type = dataset_type.lower()

    if dataset_type not in ("earth", "mars"):
        raise ValueError("Dataset type must be earth or mars.")

    raw_directory = EARTH_RAW if dataset_type == "earth" else MARS_RAW

    if uploaded_file is not None:

        dataframe = pd.read_csv(uploaded_file)
        source = "Dashboard upload"

    else:

        raw_files = sorted(raw_directory.glob("*.csv"))

        if not raw_files:

            return {
                "cleaned": pd.DataFrame(),
                "rejected": pd.DataFrame(),
                "report": None,
                "source": "No dataset available",
                "saved_paths": {},
            }

        selected_file = raw_files[0]

        dataframe = pd.read_csv(selected_file)
        source = f"Local file: {selected_file.name}"

    result = process_dataframe(
        dataframe=dataframe,
        dataset_type=dataset_type,
        save=True
    )

    return {
        "cleaned": result["cleaned"],
        "rejected": result["rejected"],
        "report": result["report"],
        "source": source,
        "saved_paths": result["saved_paths"],
    }
