from __future__ import annotations

import csv
import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path

from inertial_benchmark import CanonicalSequence


ROOT = Path(__file__).resolve().parents[1]
CONVERTER_PATH = ROOT / "tools/inertial-positioning-lab/tools/convert_dataset.py"
SPEC = importlib.util.spec_from_file_location("inertial_lab_converter", CONVERTER_PATH)
assert SPEC is not None and SPEC.loader is not None
CONVERTER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CONVERTER
SPEC.loader.exec_module(CONVERTER)


def test_android_archive_is_accepted_by_benchmark_loader(tmp_path):
    archive = tmp_path / "android.iplab"
    manifest = {
        "archive_format": "inertial-lab/1",
        "schema_version": "0.1",
        "dataset": "InertialLab",
        "sequence_id": "android-integration",
        "display_name": "Android integration test",
        "world_frame": "arcore_local_x_right_y_forward_z_up",
        "timestamp_type": "relative",
        "orientation_convention": "body_to_world_wxyz",
        "accelerometer_type": "specific_force",
        "position_source": "ARCore visual-inertial odometry",
        "orientation_source": "ARCore Android sensor pose",
        "subject_id": "synthetic",
        "device_id": "synthetic",
        "source_license": "generated",
        "sample_rate_hz": 200,
        "started_at_utc": "2026-08-19T00:00:00Z",
    }
    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=CONVERTER.CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for index in range(3):
        row = {column: "0" for column in CONVERTER.CSV_COLUMNS}
        row.update(
            timestamp=str(index * 0.005),
            accel_z="9.80665",
            orientation_w="1",
            position_x=str(index * 0.01),
            velocity_x="2",
            valid_imu="1",
            valid_orientation="1",
            valid_position="1",
            latitude="",
            longitude="",
            altitude="",
            horizontal_accuracy="",
        )
        writer.writerow(row)

    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("manifest.json", json.dumps(manifest))
        bundle.writestr("data/sequence.csv", csv_buffer.getvalue())

    [converted] = CONVERTER.convert([archive], tmp_path / "canonical")
    sequence = CanonicalSequence.from_hdf5(converted)

    assert sequence.attributes["world_frame"] == "arcore_local_x_right_y_forward_z_up"
    assert sequence.attributes["position_source"] == "ARCore visual-inertial odometry"
    assert sequence.valid_position is not None and sequence.valid_position.all()
