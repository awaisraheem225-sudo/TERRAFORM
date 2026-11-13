
from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

INVENTORY_FILE = (
    ROOT / "outputs/maps/candidate_regions/"
    "atacama_candidate_inventory.csv"
)

COVERAGE_FILE = (
    ROOT / "outputs/maps/geological_characterization/"
    "candidate_geological_coverage.csv"
)

GEOLOGY_FILE = (
    ROOT / "outputs/maps/geological_characterization/"
    "candidate_geological_units.csv"
)

OUTPUT_DIR = ROOT / "outputs/reports/candidate_profiles"
DASHBOARD_DIR = ROOT / "outputs/dashboard"

# Jezero empirical terrain reference intervals, in the same units
# as the processed Earth and Mars terrain metrics.
JEZERO_REFERENCE = {
    "broad_p10_p90": {
        "slope_deg": [0.6519, 7.3131],
        "roughness_m": [5.1962, 48.8015],
        "local_relief_m": [16.4610, 151.1901],
    },
    "moderate_p25_p75": {
        "slope_deg": [1.2336, 4.1812],
        "roughness_m": [8.9783, 28.1852],
        "local_relief_m": [28.1552, 88.0981],
    },
    "strict_p40_p60": {
        "slope_deg": [1.8181, 2.8404],
        "roughness_m": [12.6838, 19.5101],
        "local_relief_m": [39.8235, 60.8710],
    },
}

LIMITATIONS = [
    "Candidate ranking is by connected-region area, not scientific suitability.",
    "Terrain screening uses empirical Jezero percentile intervals, not validated analogue thresholds.",
    "Terrain compatibility does not establish geological or mineralogical equivalence.",
    "The Earth elevation source is Copernicus DEM GLO-30, a digital surface model.",
    "Mars terrain reference is derived from MOLA gridded elevation data.",
    "The common working resolution is approximately 463 metres.",
    "Earth and Mars use separate local projected coordinate systems.",
    "The SERNAGEOMIN geological map is regional-scale mapping at approximately 1:1,000,000.",
    "Geological map coverage does not establish field-verified lithology or mineral composition.",
    "Earth cannot reproduce Martian gravity, atmospheric pressure, radiation environment or atmospheric chemistry.",
    "The broad and moderate overlap percentages describe the screening workflow, not independent validation."
]


def clean(value):
    """Convert pandas/numpy values into JSON-safe Python values."""
    if pd.isna(value):
        return None

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    return value


def number(value, digits=4):
    value = clean(value)

    if value is None:
        return None

    return round(float(value), digits)


def interval_check(value, interval):
    if value is None:
        return None

    return interval[0] <= value <= interval[1]


def load_inputs():
    for path in [INVENTORY_FILE, COVERAGE_FILE, GEOLOGY_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    inventory = pd.read_csv(INVENTORY_FILE)
    coverage = pd.read_csv(COVERAGE_FILE)
    geology = pd.read_csv(GEOLOGY_FILE)

    required_inventory = [
        "candidate_id", "rank_by_area", "pixel_count", "area_km2",
        "centroid_latitude", "centroid_longitude",
        "broad_screening_overlap_percent",
        "moderate_screening_overlap_percent",
        "slope_mean_deg", "slope_median_deg",
        "roughness_mean_m", "roughness_median_m",
        "local_relief_mean_m", "local_relief_median_m",
    ]

    for column in required_inventory:
        if column not in inventory.columns:
            raise ValueError(f"Inventory missing column: {column}")

    return inventory, coverage, geology


def build_profile(row, coverage_row, geology_rows):
    candidate_id = row["candidate_id"]

    terrain = {
        "slope": {
            "mean_deg": number(row["slope_mean_deg"]),
            "median_deg": number(row["slope_median_deg"]),
        },
        "roughness": {
            "mean_m": number(row["roughness_mean_m"]),
            "median_m": number(row["roughness_median_m"]),
        },
        "local_relief": {
            "mean_m": number(row["local_relief_mean_m"]),
            "median_m": number(row["local_relief_median_m"]),
        },
    }

    # Test candidate median values against the strict Jezero interval.
    # This is descriptive interval membership, NOT a similarity score.
    interval_membership = {
        "slope_median_within_strict_interval": interval_check(
            terrain["slope"]["median_deg"],
            JEZERO_REFERENCE["strict_p40_p60"]["slope_deg"]
        ),
        "roughness_median_within_strict_interval": interval_check(
            terrain["roughness"]["median_m"],
            JEZERO_REFERENCE["strict_p40_p60"]["roughness_m"]
        ),
        "local_relief_median_within_strict_interval": interval_check(
            terrain["local_relief"]["median_m"],
            JEZERO_REFERENCE["strict_p40_p60"]["local_relief_m"]
        ),
    }

    geological_units = []

    for _, unit in geology_rows.iterrows():
        geological_units.append({
            "code": clean(unit["cod_geolog"]),
            "lithostratigraphy": clean(unit["lithostrat"]),
            "rock_rank": clean(unit["rock_rank"]),
            "minimum_age": clean(unit["age_min"]),
            "maximum_age": clean(unit["age_max"]),
            "lithology": clean(unit["lithology"]),
            "intersection_area_km2": number(
                unit["intersection_area_km2"], 4
            ),
            "candidate_coverage_percent": number(
                unit["candidate_coverage_percent"], 4
            ),
        })

    geological_units.sort(
        key=lambda x: x["intersection_area_km2"] or 0,
        reverse=True
    )

    dominant_units = geological_units[:3]

    coverage_data = {
        "mapped_area_km2": number(
            coverage_row["mapped_geology_area_km2"], 4
        ),
        "coverage_percent": number(
            coverage_row["mapped_coverage_percent"], 4
        ),
        "unmapped_area_km2": number(
            coverage_row["unmapped_area_km2"], 4
        ),
    }

    profile = {
        "candidate_id": candidate_id,
        "area_rank": int(row["rank_by_area"]),
        "ranking_basis": "Connected-region area, descending",
        "geography": {
            "region": (
                clean(geology_rows.iloc[0]["region"])
                if len(geology_rows) else "Not available"
            ),
            "latitude": number(row["centroid_latitude"], 6),
            "longitude": number(row["centroid_longitude"], 6),
            "area_km2": number(row["area_km2"], 4),
            "pixel_count": int(row["pixel_count"]),
        },
        "terrain": terrain,
        "jezero_reference": JEZERO_REFERENCE,
        "strict_interval_membership": interval_membership,
        "screening": {
            "broad_overlap_percent": number(
                row["broad_screening_overlap_percent"], 4
            ),
            "moderate_overlap_percent": number(
                row["moderate_screening_overlap_percent"], 4
            ),
            "interpretation": (
                "Inherited screening overlap statistics. "
                "Not independent validation or probability."
            ),
        },
        "geology": {
            "mapped_coverage": coverage_data,
            "number_of_intersecting_geological_records": len(geological_units),
            "dominant_units": dominant_units,
            "all_intersecting_units": geological_units,
        },
        "data_quality": {
            "geological_coverage_percent": coverage_data["coverage_percent"],
            "geological_unmapped_area_km2": coverage_data["unmapped_area_km2"],
            "profile_status": "Generated from verified candidate and geology outputs",
        },
        "scientific_limitations": LIMITATIONS,
        "map_reference": (
            "outputs/maps/geological_characterization/"
            "verified_candidate_polygons.geojson"
        ),
    }

    return profile


def markdown_profile(profile):
    geo = profile["geography"]
    terrain = profile["terrain"]
    geology = profile["geology"]
    coverage = geology["mapped_coverage"]

    lines = [
        f"# {profile['candidate_id']} — Candidate Intelligence Profile",
        "",
        f"**Area rank:** {profile['area_rank']} (by area, not suitability)",
        "",
        "## Geographic identification",
        f"- Region: {geo['region']}",
        f"- Centroid latitude: {geo['latitude']}",
        f"- Centroid longitude: {geo['longitude']}",
        f"- Area: {geo['area_km2']} km²",
        f"- Raster pixels: {geo['pixel_count']}",
        "",
        "## Terrain characterization",
        "| Metric | Mean | Median |",
        "|---|---:|---:|",
        f"| Slope | {terrain['slope']['mean_deg']}° | {terrain['slope']['median_deg']}° |",
        f"| Roughness | {terrain['roughness']['mean_m']} m | {terrain['roughness']['median_m']} m |",
        f"| Local relief | {terrain['local_relief']['mean_m']} m | {terrain['local_relief']['median_m']} m |",
        "",
        "## Jezero reference interval membership",
    ]

    for metric, result in profile["strict_interval_membership"].items():
        lines.append(f"- {metric}: **{result}**")

    lines.extend([
        "",
        "These are median-value interval checks, not a validated similarity score.",
        "",
        "## Screening statistics",
        f"- Broad overlap: {profile['screening']['broad_overlap_percent']}%",
        f"- Moderate overlap: {profile['screening']['moderate_overlap_percent']}%",
        "- These are inherited from the nested screening workflow, not independent validation.",
        "",
        "## Geological characterization",
        f"- Mapped geology area: {coverage['mapped_area_km2']} km²",
        f"- Geological map coverage: {coverage['coverage_percent']}%",
        f"- Unmapped area: {coverage['unmapped_area_km2']} km²",
        "",
        "### Intersecting geological units",
        "| Code | Lithology | Area (km²) | Candidate coverage |",
        "|---|---|---:|---:|",
    ])

    for unit in geology["all_intersecting_units"]:
        lithology = unit["lithology"] or "Not specified"
        lines.append(
            f"| {unit['code']} | {lithology} | "
            f"{unit['intersection_area_km2']} | "
            f"{unit['candidate_coverage_percent']}% |"
        )

    lines.extend([
        "",
        "## Scientific limitations",
    ])

    for limitation in profile["scientific_limitations"]:
        lines.append(f"- {limitation}")

    lines.extend([
        "",
        "## Data interpretation",
        "This profile characterizes terrain and regional geological mapping. "
        "It does not establish mineralogical equivalence with Mars or prove "
        "that the candidate is an operational analogue.",
        "",
        "---",
        "*TERRAFORM — Planetary Analog Intelligence*",
    ])

    return "\n".join(lines)


def main():
    print("=" * 65)
    print("TERRAFORM | CANDIDATE INTELLIGENCE PROFILE GENERATOR")
    print("=" * 65)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/6] Loading verified datasets...")
    inventory, coverage, geology = load_inputs()

    print("[2/6] Selecting top 10 candidates by area...")
    top_candidates = (
        inventory.sort_values("rank_by_area")
        .head(10)
        .copy()
    )

    if len(top_candidates) != 10:
        raise ValueError("Expected 10 candidates.")

    if top_candidates["candidate_id"].nunique() != 10:
        raise ValueError("Duplicate candidate IDs detected.")

    print("[3/6] Validating candidate relationships...")

    if set(top_candidates["candidate_id"]) != set(coverage["candidate_id"]):
        raise ValueError(
            "Top 10 candidate IDs do not match geological coverage IDs."
        )

    coverage_map = coverage.set_index("candidate_id")

    print("[4/6] Generating individual profiles...")
    profiles = []

    for _, row in top_candidates.iterrows():
        candidate_id = row["candidate_id"]

        candidate_coverage = coverage_map.loc[candidate_id]

        candidate_geology = geology[
            geology["candidate_id"] == candidate_id
        ].copy()

        profile = build_profile(
            row,
            candidate_coverage,
            candidate_geology
        )

        profiles.append(profile)

        with open(
            OUTPUT_DIR / f"{candidate_id}_profile.json",
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(profile, f, indent=2, ensure_ascii=False)

        with open(
            OUTPUT_DIR / f"{candidate_id}_profile.md",
            "w",
            encoding="utf-8"
        ) as f:
            f.write(markdown_profile(profile))

        print(
            f"  {candidate_id}: "
            f"{profile['geography']['area_km2']} km² | "
            f"{profile['geology']['mapped_coverage']['coverage_percent']}% geology coverage"
        )

    print("[5/6] Creating master comparison table...")

    master_rows = []

    for profile in profiles:
        geo = profile["geography"]
        terrain = profile["terrain"]
        geology_data = profile["geology"]

        master_rows.append({
            "candidate_id": profile["candidate_id"],
            "area_rank": profile["area_rank"],
            "latitude": geo["latitude"],
            "longitude": geo["longitude"],
            "area_km2": geo["area_km2"],
            "slope_mean_deg": terrain["slope"]["mean_deg"],
            "slope_median_deg": terrain["slope"]["median_deg"],
            "roughness_mean_m": terrain["roughness"]["mean_m"],
            "roughness_median_m": terrain["roughness"]["median_m"],
            "local_relief_mean_m": terrain["local_relief"]["mean_m"],
            "local_relief_median_m": terrain["local_relief"]["median_m"],
            "geology_coverage_percent": (
                geology_data["mapped_coverage"]["coverage_percent"]
            ),
            "geological_unit_count": (
                geology_data["number_of_intersecting_geological_records"]
            ),
            "dominant_geological_unit": (
                geology_data["dominant_units"][0]["code"]
                if geology_data["dominant_units"] else None
            ),
            "broad_overlap_percent": (
                profile["screening"]["broad_overlap_percent"]
            ),
            "moderate_overlap_percent": (
                profile["screening"]["moderate_overlap_percent"]
            ),
        })

    master = pd.DataFrame(master_rows)

    master.to_csv(
        OUTPUT_DIR / "candidate_master_comparison.csv",
        index=False
    )

    print("[6/6] Exporting dashboard-ready data...")

    dashboard_data = {
        "project": "TERRAFORM",
        "title": "Planetary Analog Intelligence",
        "profile_count": len(profiles),
        "ranking_basis": "Connected-region area, descending",
        "reference_region": "Jezero Crater, Mars",
        "earth_study_region": "Atacama Desert, Chile",
        "reference_intervals": JEZERO_REFERENCE,
        "profiles": profiles,
        "scientific_limitations": LIMITATIONS,
    }

    with open(
        DASHBOARD_DIR / "candidate_profiles.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(dashboard_data, f, indent=2, ensure_ascii=False)

    master.to_csv(
        DASHBOARD_DIR / "candidate_master_comparison.csv",
        index=False
    )

    print("\n" + "=" * 65)
    print("CANDIDATE INTELLIGENCE GENERATION COMPLETE")
    print("=" * 65)

    print("\nGenerated files:")
    print(f"Individual profiles: {OUTPUT_DIR}")
    print(f"Dashboard JSON: {DASHBOARD_DIR / 'candidate_profiles.json'}")
    print(f"Master CSV: {DASHBOARD_DIR / 'candidate_master_comparison.csv'}")

    print("\nCandidates generated:", len(profiles))
    print("Geological records processed:", len(geology))

    print("\nNo universal analogue score was calculated.")
    print("No candidate was ranked by scientific suitability.")

    print("\nSUCCESS")


if __name__ == "__main__":
    main()
