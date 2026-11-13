
"""
TERRAFORM - Data Processing Pipeline
Version: 1.0
"""

from pathlib import Path
import json
import pandas as pd

from src.data.validation import validate_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def process_dataframe(dataframe, dataset_type, save=True):
    """
    Validate and process a pandas DataFrame.

    Returns a dictionary containing:
    - cleaned dataset
    - rejected records
    - validation report
    - saved file paths
    """

    result = validate_dataset(
        dataframe,
        dataset_type
    )

    cleaned = result["cleaned"]
    rejected = result["rejected"]
    report = result["report"]

    saved_paths = {}

    if save and report["status"] == "PASSED":

        processed_path = (
            PROCESSED_DIR
            / f"{dataset_type}_processed.csv"
        )

        rejected_path = (
            PROCESSED_DIR
            / f"{dataset_type}_rejected.csv"
        )

        report_path = (
            REPORT_DIR
            / f"{dataset_type}_validation_report.json"
        )

        cleaned.to_csv(
            processed_path,
            index=False
        )

        rejected.to_csv(
            rejected_path,
            index=False
        )

        with open(
            report_path,
            "w",
            encoding="utf-8"
        ) as report_file:

            json.dump(
                report,
                report_file,
                indent=4,
                ensure_ascii=False
            )

        saved_paths = {
            "processed_data": str(processed_path),
            "rejected_data": str(rejected_path),
            "validation_report": str(report_path),
        }

    return {
        "cleaned": cleaned,
        "rejected": rejected,
        "report": report,
        "saved_paths": saved_paths,
    }


def process_csv(csv_path, dataset_type, save=True):
    """Read a CSV file and pass it through the processing pipeline."""

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}"
        )

    dataframe = pd.read_csv(csv_path)

    return process_dataframe(
        dataframe,
        dataset_type,
        save=save
    )


def create_quality_summary(report):
    """Convert a validation report into a dashboard-friendly table."""

    rows = [
        ("Dataset type", report["dataset_type"]),
        ("Validation status", report["status"]),
        ("Original rows", report["original_rows"]),
        ("Valid rows", report["valid_rows"]),
        ("Rejected rows", report["rejected_rows"]),
        ("Exact duplicates excluded", report["duplicate_rows"]),
        ("Invalid Earth coordinates", report["invalid_coordinates"]),
    ]

    return pd.DataFrame(
        rows,
        columns=["Check", "Result"]
    )
