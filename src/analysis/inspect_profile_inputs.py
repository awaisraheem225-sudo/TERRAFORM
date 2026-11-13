from pathlib import Path
import pandas as pd
import json

ROOT = Path(__file__).resolve().parents[2]

FILES = [
    "outputs/maps/candidate_regions/atacama_candidate_inventory.csv",
    "outputs/maps/geological_characterization/candidate_geological_coverage.csv",
    "outputs/maps/geological_characterization/candidate_geological_units.csv",
]

REPORTS = [
    "outputs/reports/earth_mars_terrain_comparison.json",
    "outputs/reports/terrain_range_screening_report.json",
    "outputs/reports/sensitivity_analysis_report.json",
    "outputs/reports/candidate_geological_characterization_v2.json",
]

print("=" * 65)
print("TERRAFORM | CANDIDATE PROFILE INPUT INSPECTION")
print("=" * 65)

for relative_path in FILES:
    path = ROOT / relative_path
    print(f"\n\nCSV: {relative_path}")

    if not path.exists():
        print("FILE NOT FOUND")
        continue

    df = pd.read_csv(path)

    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    print("\nFirst 2 records:")
    print(df.head(2).to_string(index=False))

for relative_path in REPORTS:
    path = ROOT / relative_path
    print(f"\n\nJSON: {relative_path}")

    if not path.exists():
        print("FILE NOT FOUND")
        continue

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("Top-level keys:", list(data.keys()))

print("\n\nINSPECTION COMPLETE")