
"""
TERRAFORM - Earth/Mars Comparison Engine
Version: 1.0

No universal similarity score is generated.
Comparisons are only calculated for explicitly approved metrics.
"""

import pandas as pd


METRIC_DEFINITIONS = {
    "slope_deg": {
        "label": "Slope",
        "unit": "degrees",
        "description": "Surface inclination",
    },
    "roughness_m": {
        "label": "Surface roughness",
        "unit": "metres",
        "description": "A terrain roughness measurement",
    },
    "local_relief_m": {
        "label": "Local relief",
        "unit": "metres",
        "description": "Elevation range within a defined neighbourhood",
    },
    "elevation_m": {
        "label": "Elevation",
        "unit": "metres",
        "description": "Elevation relative to a specified vertical datum",
    },
    "surface_temp_c": {
        "label": "Surface temperature",
        "unit": "degrees Celsius",
        "description": "Surface temperature measurement",
    },
    "precipitation_mm": {
        "label": "Precipitation",
        "unit": "millimetres",
        "description": "Precipitation measurement",
    },
}


def compare_records(
    earth_record,
    mars_record,
    approved_metrics=None
):
    """
    Compare one Earth record with one Mars record.

    approved_metrics must contain only variables whose definitions,
    units, spatial scale and measurement methods have been checked.

    Without approval, the function reports values but calculates
    no differences.
    """

    if approved_metrics is None:
        approved_metrics = []

    rows = []

    for metric, definition in METRIC_DEFINITIONS.items():

        earth_value = earth_record.get(metric, None)
        mars_value = mars_record.get(metric, None)

        earth_available = pd.notna(earth_value)
        mars_available = pd.notna(mars_value)

        if not earth_available and not mars_available:
            continue

        if earth_available:
            earth_value = float(earth_value)

        if mars_available:
            mars_value = float(mars_value)

        difference = None
        percent_difference = None

        if earth_available and mars_available and metric in approved_metrics:

            difference = abs(
                earth_value - mars_value
            )

            denominator = abs(mars_value)

            if denominator > 0:
                percent_difference = (
                    difference / denominator
                ) * 100

            status = "Approved measurement comparison"

        elif earth_available and mars_available:

            status = "Values available; comparability not verified"

        else:

            status = "One or both measurements unavailable"

        rows.append({
            "metric": metric,
            "label": definition["label"],
            "unit": definition["unit"],
            "earth_value": earth_value if earth_available else None,
            "mars_value": mars_value if mars_available else None,
            "absolute_difference": difference,
            "difference_percent_of_mars": percent_difference,
            "status": status,
        })

    return pd.DataFrame(rows)


def get_common_metrics(earth_record, mars_record):
    """Return numeric metrics available in both records."""

    common = []

    for metric in METRIC_DEFINITIONS:

        earth_value = earth_record.get(metric, None)
        mars_value = mars_record.get(metric, None)

        if pd.notna(earth_value) and pd.notna(mars_value):
            common.append(metric)

    return common
