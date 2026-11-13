
"""
TERRAFORM - Dataset Validation Engine
Version: 1.0

Validates Earth candidate and Mars reference CSV datasets.
Does not invent, estimate, or silently repair scientific observations.
"""

import pandas as pd


COLUMN_ALIASES = {
    "name": "location",
    "site": "location",
    "candidate": "location",
    "lat": "latitude",
    "lon": "longitude",
    "long": "longitude",
    "lng": "longitude",
    "elevation": "elevation_m",
    "slope": "slope_deg",
    "roughness": "roughness_m",
    "relief": "local_relief_m",
    "temperature": "surface_temp_c",
    "mean_temperature": "surface_temp_c",
    "precipitation": "precipitation_mm",
    "annual_precipitation": "precipitation_mm",
    "source_name": "source",
}

NUMERIC_COLUMNS = [
    "latitude",
    "longitude",
    "elevation_m",
    "slope_deg",
    "roughness_m",
    "local_relief_m",
    "surface_temp_c",
    "precipitation_mm",
]

EARTH_REQUIRED = ["location", "latitude", "longitude"]

MARS_REQUIRED = ["location"]

OPTIONAL_COLUMNS = [
    "elevation_m",
    "slope_deg",
    "roughness_m",
    "local_relief_m",
    "surface_temp_c",
    "precipitation_mm",
    "geology",
    "source",
]


def normalize_columns(dataframe):
    """Standardise column names and recognised aliases."""

    df = dataframe.copy()

    df.columns = [
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        for column in df.columns
    ]

    rename_map = {}

    for old_name, new_name in COLUMN_ALIASES.items():
        if old_name in df.columns and new_name not in df.columns:
            rename_map[old_name] = new_name

    df = df.rename(columns=rename_map)

    return df


def validate_dataset(dataframe, dataset_type):
    """
    Validate a dataset.

    Parameters
    ----------
    dataframe : pandas.DataFrame
    dataset_type : str
        Either "earth" or "mars".

    Returns
    -------
    dict
        cleaned, rejected, report
    """

    dataset_type = str(dataset_type).strip().lower()

    if dataset_type not in ("earth", "mars"):
        raise ValueError("dataset_type must be 'earth' or 'mars'.")

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    original_count = len(dataframe)

    df = normalize_columns(dataframe)

    errors = []
    warnings = []
    rejected_reasons = []

    if df.empty:
        errors.append("The uploaded dataset contains no rows.")

    required_columns = (
        EARTH_REQUIRED if dataset_type == "earth"
        else MARS_REQUIRED
    )

    missing_required = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_required:
        errors.append(
            "Missing required columns: "
            + ", ".join(missing_required)
        )

    if errors:
        return {
            "cleaned": pd.DataFrame(),
            "rejected": df.copy(),
            "report": {
                "dataset_type": dataset_type,
                "status": "FAILED",
                "original_rows": original_count,
                "valid_rows": 0,
                "rejected_rows": original_count,
                "errors": errors,
                "warnings": warnings,
                "missing_values": {},
                "duplicate_rows": 0,
                "invalid_coordinates": 0,
                "numeric_conversion_failures": {},
                "columns": list(df.columns),
            },
        }

    # Remove surrounding whitespace from text fields.
    for column in df.select_dtypes(include=["object"]).columns:
        df[column] = df[column].map(
            lambda value: value.strip()
            if isinstance(value, str)
            else value
        )

    # Convert recognised measurement columns to numeric.
    conversion_failures = {}

    for column in NUMERIC_COLUMNS:

        if column not in df.columns:
            continue

        original_values = df[column].copy()

        converted = pd.to_numeric(
            original_values,
            errors="coerce"
        )

        failures = (
            original_values.notna()
            & converted.isna()
        )

        failure_count = int(failures.sum())

        if failure_count:
            conversion_failures[column] = failure_count

        df[column] = converted

    # Reject rows without a usable location name.
    invalid_location = (
        df["location"].isna()
        | (df["location"].astype(str).str.strip() == "")
    )

    # Earth records must have valid coordinates.
    invalid_coordinates = pd.Series(
        False,
        index=df.index
    )

    if dataset_type == "earth":

        invalid_coordinates = (
            df["latitude"].isna()
            | df["longitude"].isna()
            | ~df["latitude"].between(-90, 90)
            | ~df["longitude"].between(-180, 180)
        )

    # A Mars reference row must contain at least one recognised
    # numeric measurement in addition to its location.
    if dataset_type == "mars":

        available_metrics = [
            column for column in OPTIONAL_COLUMNS
            if column in NUMERIC_COLUMNS and column in df.columns
        ]

        if not available_metrics:
            errors.append(
                "Mars dataset needs at least one recognised numeric "
                "measurement column."
            )
        else:
            has_measurement = df[available_metrics].notna().any(axis=1)
            invalid_location = invalid_location | ~has_measurement

    # Identify physically invalid values where the variable has
    # a clear mathematical range.
    physically_invalid = pd.Series(
        False,
        index=df.index
    )

    if "slope_deg" in df.columns:
        physically_invalid |= (
            df["slope_deg"].notna()
            & ~df["slope_deg"].between(0, 90)
        )

    for nonnegative_column in [
        "roughness_m",
        "local_relief_m",
        "precipitation_mm",
    ]:

        if nonnegative_column in df.columns:
            physically_invalid |= (
                df[nonnegative_column].notna()
                & (df[nonnegative_column] < 0)
            )

    # Record why each rejected row failed.
    reject_mask = (
        invalid_location
        | invalid_coordinates
        | physically_invalid
    )

    rejected = df.loc[reject_mask].copy()
    cleaned = df.loc[~reject_mask].copy()

    rejected["validation_reason"] = ""

    for index in rejected.index:

        reasons = []

        if bool(invalid_location.loc[index]):
            reasons.append("Missing location or required measurement")

        if bool(invalid_coordinates.loc[index]):
            reasons.append("Missing or invalid Earth coordinates")

        if bool(physically_invalid.loc[index]):
            reasons.append("Physically invalid measurement value")

        rejected.loc[index, "validation_reason"] = "; ".join(reasons)

    # Exact duplicate rows are reported and removed from the
    # processed dataset. The original upload remains untouched.
    duplicate_mask = cleaned.duplicated(keep="first")
    duplicate_count = int(duplicate_mask.sum())

    if duplicate_count:
        warnings.append(
            f"{duplicate_count} exact duplicate rows were excluded."
        )
        cleaned = cleaned.loc[~duplicate_mask].copy()

    if "source" not in cleaned.columns:
        warnings.append(
            "No source column found. Dataset provenance should be recorded."
        )
    else:
        missing_sources = int(cleaned["source"].isna().sum())

        if missing_sources:
            warnings.append(
                f"{missing_sources} records have no source identifier."
            )

    missing_values = {
        column: int(count)
        for column, count in cleaned.isna().sum().items()
        if count > 0
    }

    if missing_values:
        warnings.append(
            "Some optional measurements contain missing values. "
            "They have not been imputed."
        )

    if conversion_failures:
        warnings.append(
            "Some measurement values could not be converted to numbers."
        )

    if cleaned.empty:
        warnings.append("No usable records remain after validation.")

    status = "PASSED" if not errors and not cleaned.empty else "FAILED"

    report = {
        "dataset_type": dataset_type,
        "status": status,
        "original_rows": original_count,
        "valid_rows": len(cleaned),
        "rejected_rows": len(rejected),
        "errors": errors,
        "warnings": warnings,
        "missing_values": missing_values,
        "duplicate_rows": duplicate_count,
        "invalid_coordinates": int(invalid_coordinates.sum()),
        "numeric_conversion_failures": conversion_failures,
        "columns": list(df.columns),
    }

    return {
        "cleaned": cleaned.reset_index(drop=True),
        "rejected": rejected.reset_index(drop=True),
        "report": report,
    }
