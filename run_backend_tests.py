
"""
TERRAFORM - Backend System Tests

Run from the project root:
python run_backend_tests.py

All records below are temporary software test fixtures,
not scientific observations.
"""

import pandas as pd

from src.data.validation import validate_dataset
from src.data.processing import process_dataframe
from src.analysis.comparison import compare_records


def check(condition, message):

    if not condition:
        raise AssertionError(message)

    print(f"[PASS] {message}")


def test_earth_validation():

    test_data = pd.DataFrame([
        {
            "location": "Test Site A",
            "latitude": 25.0,
            "longitude": 70.0,
            "slope_deg": 15.0,
            "roughness_m": 2.0,
            "source": "SOFTWARE_TEST_ONLY",
        },
        {
            "location": "Test Site B",
            "latitude": 120.0,
            "longitude": 70.0,
            "slope_deg": 20.0,
            "source": "SOFTWARE_TEST_ONLY",
        },
        {
            "location": "Test Site C",
            "latitude": 30.0,
            "longitude": 70.0,
            "slope_deg": 120.0,
            "source": "SOFTWARE_TEST_ONLY",
        },
    ])

    result = validate_dataset(
        test_data,
        "earth"
    )

    check(
        result["report"]["status"] == "PASSED",
        "Earth dataset validation completes"
    )

    check(
        len(result["cleaned"]) == 1,
        "Invalid Earth records are rejected"
    )

    check(
        len(result["rejected"]) == 2,
        "Rejected Earth records are preserved"
    )

    check(
        result["report"]["invalid_coordinates"] == 1,
        "Invalid coordinates are detected"
    )


def test_mars_validation():

    test_data = pd.DataFrame([
        {
            "location": "Test Mars Region",
            "slope_deg": 18.0,
            "roughness_m": 3.0,
            "source": "SOFTWARE_TEST_ONLY",
        }
    ])

    result = validate_dataset(
        test_data,
        "mars"
    )

    check(
        result["report"]["status"] == "PASSED",
        "Mars reference dataset validation completes"
    )

    check(
        len(result["cleaned"]) == 1,
        "Valid Mars record is retained"
    )


def test_bad_mars_dataset():

    test_data = pd.DataFrame([
        {
            "location": "Empty Mars Record",
            "geology": "Test only",
        }
    ])

    result = validate_dataset(
        test_data,
        "mars"
    )

    check(
        result["report"]["status"] == "FAILED",
        "Mars dataset without numeric measurements is rejected"
    )


def test_processing_pipeline():

    test_data = pd.DataFrame([
        {
            "location": "Pipeline Test",
            "latitude": 20.0,
            "longitude": 60.0,
            "slope_deg": 10.0,
            "source": "SOFTWARE_TEST_ONLY",
        }
    ])

    result = process_dataframe(
        test_data,
        "earth",
        save=False
    )

    check(
        result["report"]["status"] == "PASSED",
        "Processing pipeline runs successfully"
    )

    check(
        len(result["cleaned"]) == 1,
        "Processing pipeline returns cleaned records"
    )


def test_comparison_engine():

    earth_record = {
        "slope_deg": 15.0,
        "roughness_m": 2.0,
        "local_relief_m": 30.0,
    }

    mars_record = {
        "slope_deg": 20.0,
        "roughness_m": 4.0,
        "local_relief_m": 40.0,
    }

    unapproved = compare_records(
        earth_record,
        mars_record
    )

    check(
        unapproved["absolute_difference"].isna().all(),
        "Comparison engine does not invent unapproved differences"
    )

    approved = compare_records(
        earth_record,
        mars_record,
        approved_metrics=["slope_deg"]
    )

    slope_result = approved[
        approved["metric"] == "slope_deg"
    ].iloc[0]

    check(
        slope_result["absolute_difference"] == 5.0,
        "Approved metric difference is calculated correctly"
    )

    roughness_result = approved[
        approved["metric"] == "roughness_m"
    ].iloc[0]

    check(
        pd.isna(roughness_result["absolute_difference"]),
        "Unapproved metrics remain unscored"
    )


def main():

    print("\n")
    print("=" * 55)
    print("TERRAFORM BACKEND TEST SUITE")
    print("=" * 55)

    test_earth_validation()
    test_mars_validation()
    test_bad_mars_dataset()
    test_processing_pipeline()
    test_comparison_engine()

    print("\n")
    print("=" * 55)
    print("ALL BACKEND TESTS PASSED")
    print("=" * 55)


if __name__ == "__main__":
    main()
