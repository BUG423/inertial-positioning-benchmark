from __future__ import annotations

import csv
import io
import json
import zipfile

import h5py
import numpy as np
import pytest

from tools.convert_dataset import CSV_COLUMNS, ConversionError, convert


def make_archive(path, timestamps=(0.0, 0.01, 0.02), metadata_overrides=None, row_overrides=None):
    metadata = {
        "archive_format": "inertial-lab/1",
        "schema_version": "0.1",
        "dataset": "synthetic",
        "sequence_id": "walk-01",
        "display_name": "Synthetic walk",
        "world_frame": "gravity_aligned_local_enu",
        "timestamp_type": "relative",
        "orientation_convention": "body_to_world_wxyz",
        "accelerometer_type": "specific_force",
        "position_source": "analytic",
        "orientation_source": "analytic",
        "subject_id": "test",
        "device_id": "test",
        "source_license": "generated",
        "sample_rate_hz": 100,
        "started_at_utc": "2026-08-15T00:00:00Z",
    }
    metadata.update(metadata_overrides or {})
    data = io.StringIO()
    writer = csv.DictWriter(data, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for index, timestamp in enumerate(timestamps):
        row = {column: "0" for column in CSV_COLUMNS}
        row.update({
            "timestamp": str(timestamp),
            "accel_z": "9.81",
            "orientation_w": "1",
            "position_x": str(index * 0.01),
            "velocity_x": "1",
            "valid_imu": "1",
            "valid_orientation": "1",
            "valid_position": "1",
            "latitude": "",
            "longitude": "",
            "altitude": "",
            "horizontal_accuracy": "",
        })
        row.update(row_overrides or {})
        writer.writerow(row)
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr("manifest.json", json.dumps(metadata))
        bundle.writestr("data/sequence.csv", data.getvalue())


def test_convert_writes_canonical_hdf5(tmp_path):
    archive = tmp_path / "sample.iplab"
    make_archive(archive)
    output = tmp_path / "dataset"
    [result] = convert([archive], output)

    with h5py.File(result) as handle:
        assert handle.attrs["schema_version"] == "0.1"
        assert handle.attrs["orientation_convention"] == "body_to_world_wxyz"
        assert handle["timestamp"].shape == (3,)
        assert handle["imu/gyroscope"].shape == (3, 3)
        assert handle["imu/accelerometer"].dtype == np.float32
        assert handle["pose/orientation"].shape == (3, 4)
        assert handle["valid/position"][:].all()
    assert (output / "splits" / "all.txt").read_text() == "walk-01\n"


def test_arcore_local_frame_is_preserved(tmp_path):
    archive = tmp_path / "arcore.iplab"
    make_archive(
        archive,
        metadata_overrides={
            "world_frame": "arcore_local_x_right_y_forward_z_up",
            "position_source": "ARCore visual-inertial odometry",
            "orientation_source": "ARCore Android sensor pose",
        },
    )
    [result] = convert([archive], tmp_path / "dataset")

    with h5py.File(result) as handle:
        assert handle.attrs["world_frame"] == "arcore_local_x_right_y_forward_z_up"
        assert handle.attrs["position_source"] == "ARCore visual-inertial odometry"


def test_duplicate_timestamp_is_rejected(tmp_path):
    archive = tmp_path / "invalid.iplab"
    make_archive(archive, (0.0, 0.01, 0.01))
    with pytest.raises(ConversionError, match="strictly increasing"):
        convert([archive], tmp_path / "dataset")


def test_non_binary_validity_mask_is_rejected(tmp_path):
    archive = tmp_path / "invalid-mask.iplab"
    make_archive(archive, row_overrides={"valid_imu": "2"})
    with pytest.raises(ConversionError, match="valid_imu must contain only 0 or 1"):
        convert([archive], tmp_path / "dataset")


def test_non_finite_optional_location_is_rejected(tmp_path):
    archive = tmp_path / "invalid-location.iplab"
    make_archive(archive, row_overrides={"latitude": "NaN", "longitude": "0"})
    with pytest.raises(ConversionError, match="latitude contains NaN or Inf"):
        convert([archive], tmp_path / "dataset")


def test_batch_is_preflighted_before_writing(tmp_path):
    valid = tmp_path / "valid.iplab"
    invalid = tmp_path / "invalid.iplab"
    make_archive(valid, metadata_overrides={"sequence_id": "valid"})
    make_archive(
        invalid,
        metadata_overrides={"sequence_id": "invalid"},
        row_overrides={"valid_position": "broken"},
    )
    output = tmp_path / "dataset"
    with pytest.raises(ConversionError, match="valid_position"):
        convert([valid, invalid], output)
    assert not (output / "sequences" / "valid.h5").exists()
