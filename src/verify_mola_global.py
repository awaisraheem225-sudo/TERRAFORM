import numpy as np
from pathlib import Path

mars_dir = Path(r"D:\TERRAFORM\data\Global\Mars")

files = {
    "Elevation": mars_dir / "megt90n000eb.img",
    "Observation Counts": mars_dir / "megc90n000eb.img",
}

expected_shape = (2880, 5760)
expected_values = expected_shape[0] * expected_shape[1]

report = []

report.append("TERRAFORM - GLOBAL MOLA RASTER VERIFICATION")
report.append("=" * 55)

for name, path in files.items():
    report.append(f"\nDATASET: {name}")
    report.append(f"FILE: {path}")

    if not path.exists():
        report.append("STATUS: FILE NOT FOUND")
        continue

    size = path.stat().st_size
    report.append(f"FILE SIZE: {size:,} bytes")

    data = np.fromfile(path, dtype=">i2")

    report.append(f"DECODED VALUES: {data.size:,}")
    report.append(f"EXPECTED VALUES: {expected_values:,}")

    if data.size != expected_values:
        report.append("STATUS: DIMENSION MISMATCH")
        continue

    raster = data.reshape(expected_shape)

    report.append(f"RASTER SHAPE: {raster.shape}")
    report.append(f"MINIMUM VALUE: {int(raster.min())}")
    report.append(f"MAXIMUM VALUE: {int(raster.max())}")

    percentiles = np.percentile(raster, [0, 1, 25, 50, 75, 99, 100])
    report.append("PERCENTILES (0, 1, 25, 50, 75, 99, 100):")
    report.append(str(percentiles.tolist()))

    report.append(f"ZERO VALUES: {int(np.count_nonzero(raster == 0)):,}")
    report.append(f"NEGATIVE VALUES: {int(np.count_nonzero(raster < 0)):,}")
    report.append(f"POSITIVE VALUES: {int(np.count_nonzero(raster > 0)):,}")
    report.append("STATUS: DIMENSIONS AND BINARY DECODING VERIFIED")

output = "\n".join(report)

output_path = Path(r"D:\TERRAFORM\outputs\mars_raster_verification.txt")
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(output, encoding="utf-8")

print(output)
print(f"\nReport saved to: {output_path}")
