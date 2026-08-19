#!/usr/bin/env python3
"""Convert Inertial Lab interchange archives to benchmark canonical HDF5 v0.1."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

REQUIRED_ATTRIBUTES = {
    "archive_format",
    "schema_version",
    "dataset",
    "sequence_id",
    "display_name",
    "world_frame",
    "timestamp_type",
    "orientation_convention",
    "accelerometer_type",
    "position_source",
    "orientation_source",
    "subject_id",
    "device_id",
    "source_license",
    "sample_rate_hz",
    "started_at_utc",
}

STRING_ATTRIBUTES = REQUIRED_ATTRIBUTES - {"sample_rate_hz"}
ALLOWED_ARCHIVE_ENTRIES = {"manifest.json", "data/sequence.csv", "README.txt", "data/"}
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_MANIFEST_BYTES = 1024 * 1024
SUPPORTED_WORLD_FRAMES = {
    "gravity_aligned_local_enu",
    "arcore_local_x_right_y_forward_z_up",
}

CSV_COLUMNS = [
    "timestamp",
    "gyro_x", "gyro_y", "gyro_z",
    "accel_x", "accel_y", "accel_z",
    "orientation_w", "orientation_x", "orientation_y", "orientation_z",
    "position_x", "position_y", "position_z",
    "velocity_x", "velocity_y", "velocity_z",
    "valid_imu", "valid_orientation", "valid_position",
    "latitude", "longitude", "altitude", "horizontal_accuracy",
]


class ConversionError(ValueError):
    """Raised when an archive violates the interchange or canonical contract."""


@dataclass(frozen=True)
class Archive:
    metadata: dict[str, object]
    rows: list[dict[str, str]]


def read_archive(path: Path) -> Archive:
    try:
        with zipfile.ZipFile(path) as bundle:
            infos = bundle.infolist()
            names = [info.filename for info in infos]
            duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
            if duplicates:
                raise ConversionError(f"{path}: duplicate archive entries: {', '.join(duplicates)}")
            unknown = sorted(set(names) - ALLOWED_ARCHIVE_ENTRIES)
            if unknown:
                raise ConversionError(f"{path}: unknown archive entries: {', '.join(unknown)}")
            if not {"manifest.json", "data/sequence.csv"}.issubset(names):
                raise ConversionError(f"{path}: missing manifest.json or data/sequence.csv")
            if sum(info.file_size for info in infos if not info.is_dir()) > MAX_ARCHIVE_BYTES:
                raise ConversionError(f"{path}: archive expands beyond 512 MiB")
            manifest_info = bundle.getinfo("manifest.json")
            if manifest_info.file_size > MAX_MANIFEST_BYTES:
                raise ConversionError(f"{path}: manifest.json exceeds 1 MiB")
            metadata = json.loads(bundle.read("manifest.json"))
            if not isinstance(metadata, dict):
                raise ConversionError(f"{path}: manifest root must be an object")
            text = io.TextIOWrapper(bundle.open("data/sequence.csv"), encoding="utf-8", newline="")
            reader = csv.DictReader(text)
            if reader.fieldnames != CSV_COLUMNS:
                raise ConversionError(f"{path}: incompatible CSV columns")
            rows = list(reader)
            if any(None in row or any(value is None for value in row.values()) for row in rows):
                raise ConversionError(f"{path}: CSV rows have inconsistent field counts")
    except ConversionError:
        raise
    except (zipfile.BadZipFile, zipfile.LargeZipFile, json.JSONDecodeError, UnicodeDecodeError, csv.Error) as error:
        raise ConversionError(f"{path}: invalid archive: {error}") from error
    if len(rows) < 2:
        raise ConversionError(f"{path}: at least two samples are required")
    return Archive(metadata, rows)


def _float(rows: list[dict[str, str]], name: str, dtype: np.dtype = np.float32) -> np.ndarray:
    try:
        result = np.asarray([row[name] for row in rows], dtype=dtype)
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ConversionError(f"invalid numeric field {name}: {error}") from error
    if not np.isfinite(result).all():
        raise ConversionError(f"{name} contains NaN or Inf")
    return result


def _matrix(rows: list[dict[str, str]], names: list[str]) -> np.ndarray:
    return np.column_stack([_float(rows, name) for name in names]).astype(np.float32, copy=False)


def _mask(rows: list[dict[str, str]], name: str) -> np.ndarray:
    values = [row[name] for row in rows]
    if any(value not in {"0", "1"} for value in values):
        raise ConversionError(f"{name} must contain only 0 or 1")
    return np.asarray([value == "1" for value in values], dtype=np.bool_)


def _optional_float(rows: list[dict[str, str]], name: str) -> list[float | None]:
    result: list[float | None] = []
    for row in rows:
        raw = row[name]
        if raw == "":
            result.append(None)
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError, OverflowError) as error:
            raise ConversionError(f"invalid optional numeric field {name}: {error}") from error
        if not math.isfinite(value):
            raise ConversionError(f"{name} contains NaN or Inf")
        result.append(value)
    return result


def _validate_metadata(metadata: dict[str, object]) -> None:
    missing = sorted(REQUIRED_ATTRIBUTES - metadata.keys())
    if missing:
        raise ConversionError("manifest missing attributes: " + ", ".join(missing))
    for name in sorted(STRING_ATTRIBUTES):
        value = metadata[name]
        if not isinstance(value, str) or not value.strip():
            raise ConversionError(f"manifest {name} must be a non-empty string")
    if metadata["archive_format"] != "inertial-lab/1":
        raise ConversionError("archive_format must be inertial-lab/1")
    if metadata["schema_version"] != "0.1":
        raise ConversionError("schema_version must be 0.1")
    if metadata["world_frame"] not in SUPPORTED_WORLD_FRAMES:
        raise ConversionError("unsupported world_frame")
    if metadata["timestamp_type"] != "relative":
        raise ConversionError("timestamp_type must be relative")
    if metadata["orientation_convention"] != "body_to_world_wxyz":
        raise ConversionError("orientation_convention must be body_to_world_wxyz")
    if metadata["accelerometer_type"] != "specific_force":
        raise ConversionError("accelerometer_type must be specific_force")
    sample_rate = metadata["sample_rate_hz"]
    if type(sample_rate) is not int or not 10 <= sample_rate <= 1000:
        raise ConversionError("sample_rate_hz must be an integer in 10..1000")
    try:
        started = datetime.fromisoformat(str(metadata["started_at_utc"]).replace("Z", "+00:00"))
    except ValueError as error:
        raise ConversionError("started_at_utc must be an ISO-8601 timestamp") from error
    if started.tzinfo is None:
        raise ConversionError("started_at_utc must include a timezone")
    finished_value = metadata.get("finished_at_utc")
    if finished_value is not None:
        if not isinstance(finished_value, str):
            raise ConversionError("finished_at_utc must be null or an ISO-8601 timestamp")
        try:
            finished = datetime.fromisoformat(finished_value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ConversionError("finished_at_utc must be null or an ISO-8601 timestamp") from error
        if finished.tzinfo is None:
            raise ConversionError("finished_at_utc must include a timezone")
        if finished < started:
            raise ConversionError("finished_at_utc must not precede started_at_utc")
    if "sample_count" in metadata:
        count = metadata["sample_count"]
        if type(count) is not int or count < 0:
            raise ConversionError("sample_count must be a non-negative integer")
    if "duration_seconds" in metadata:
        duration = metadata["duration_seconds"]
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(float(duration)) or duration < 0:
            raise ConversionError("duration_seconds must be a non-negative finite number")


def validate_and_convert(archive: Archive) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    metadata = archive.metadata
    _validate_metadata(metadata)

    rows = archive.rows
    timestamp = _float(rows, "timestamp", np.float64)
    if np.any(timestamp < 0):
        raise ConversionError("timestamp must not be negative")
    if not np.all(np.diff(timestamp) > 0):
        raise ConversionError("timestamp must be strictly increasing")
    gyroscope = _matrix(rows, ["gyro_x", "gyro_y", "gyro_z"])
    accelerometer = _matrix(rows, ["accel_x", "accel_y", "accel_z"])
    orientation = _matrix(rows, ["orientation_w", "orientation_x", "orientation_y", "orientation_z"])
    position = _matrix(rows, ["position_x", "position_y", "position_z"])
    velocity = _matrix(rows, ["velocity_x", "velocity_y", "velocity_z"])
    valid_imu = _mask(rows, "valid_imu")
    valid_orientation = _mask(rows, "valid_orientation")
    valid_position = _mask(rows, "valid_position")

    latitude = _optional_float(rows, "latitude")
    longitude = _optional_float(rows, "longitude")
    _optional_float(rows, "altitude")
    accuracy = _optional_float(rows, "horizontal_accuracy")
    for index, (lat, lon) in enumerate(zip(latitude, longitude, strict=True)):
        if (lat is None) != (lon is None):
            raise ConversionError(f"latitude/longitude must both be present or empty at sample {index}")
        if lat is not None and not -90 <= lat <= 90:
            raise ConversionError(f"latitude is outside -90..90 at sample {index}")
        if lon is not None and not -180 <= lon <= 180:
            raise ConversionError(f"longitude is outside -180..180 at sample {index}")
    if any(value is not None and value < 0 for value in accuracy):
        raise ConversionError("horizontal_accuracy must not be negative")

    norms = np.linalg.norm(orientation, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3, rtol=0.0):
        raise ConversionError("orientation quaternions must be normalized within 1e-3")
    duration = float(timestamp[-1] - timestamp[0])
    declared_count = metadata.get("sample_count", 0)
    if declared_count not in (0, len(rows)):
        raise ConversionError("manifest sample_count does not match CSV")
    declared_duration = float(metadata.get("duration_seconds", 0.0))
    if declared_duration != 0.0 and not math.isclose(declared_duration, duration, abs_tol=1e-6, rel_tol=0.0):
        raise ConversionError("manifest duration_seconds does not match CSV")
    if valid_position.any() and metadata["position_source"] == "unavailable":
        raise ConversionError("valid positions require a declared position_source")
    arrays = {
        "timestamp": timestamp,
        "imu/gyroscope": gyroscope,
        "imu/accelerometer": accelerometer,
        "pose/orientation": orientation,
        "pose/position": position,
        "pose/velocity": velocity,
        "valid/imu": valid_imu,
        "valid/orientation": valid_orientation,
        "valid/position": valid_position,
    }
    attrs = {key: metadata[key] for key in REQUIRED_ATTRIBUTES}
    attrs["conversion_tool"] = "inertial-lab/convert_dataset.py"
    return attrs, arrays


def write_hdf5(path: Path, attrs: dict[str, object], arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        for key, value in sorted(attrs.items()):
            handle.attrs[key] = value
        for name, value in arrays.items():
            handle.create_dataset(name, data=value, compression="gzip", shuffle=True)


def safe_id(value: object) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-")
    if not cleaned:
        raise ConversionError("sequence_id is empty after sanitization")
    return cleaned[:128]


def convert(inputs: list[Path], output: Path, overwrite: bool = False) -> list[Path]:
    sequence_dir = output / "sequences"
    split_dir = output / "splits"
    prepared: list[tuple[Path, str, Path]] = []
    batch_ids: set[str] = set()
    for source in inputs:
        archive = read_archive(source)
        attrs, _ = validate_and_convert(archive)
        sequence_id = safe_id(attrs["sequence_id"])
        if sequence_id in batch_ids:
            raise ConversionError(f"duplicate sequence_id in input batch: {sequence_id}")
        batch_ids.add(sequence_id)
        destination = sequence_dir / f"{sequence_id}.h5"
        if destination.exists() and not overwrite:
            raise ConversionError(f"{destination} exists; pass --overwrite to replace it")
        prepared.append((source, sequence_id, destination))

    sequence_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    ids: list[str] = []
    reports: list[dict[str, object]] = []
    for source, sequence_id, destination in prepared:
        attrs, arrays = validate_and_convert(read_archive(source))
        current_id = safe_id(attrs["sequence_id"])
        if current_id != sequence_id:
            raise ConversionError(f"{source}: sequence_id changed while converting")
        write_hdf5(destination, attrs, arrays)
        ids.append(sequence_id)
        written.append(destination)
        reports.append({
            "source": str(source),
            "sequence_id": sequence_id,
            "samples": int(arrays["timestamp"].shape[0]),
            "duration_seconds": float(arrays["timestamp"][-1] - arrays["timestamp"][0]),
            "valid_position_fraction": float(arrays["valid/position"].mean()),
        })
    (split_dir / "all.txt").write_text("".join(f"{value}\n" for value in ids), encoding="utf-8")
    (output / "conversion_report.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="+", type=Path, help="one or more .iplab archives")
    parser.add_argument("--output", "-o", type=Path, required=True, help="canonical dataset directory")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        paths = convert(args.input, args.output, args.overwrite)
    except ConversionError as error:
        parser.error(str(error))
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
