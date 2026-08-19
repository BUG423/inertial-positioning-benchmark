from __future__ import annotations

import json
import zipfile

import pytest

from tools.pack_model import pack, validate


def manifest():
    return {
        "schema_version": 1,
        "id": "tiny-model",
        "name": "Tiny Model",
        "version": "1.0.0",
        "runtime": "tflite",
        "model_file": "model.tflite",
        "input": {
            "shape": [1, 200, 6],
            "layout": "NTC",
            "sample_rate_hz": 100,
            "channels": ["gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z"],
            "dtype": "float32",
            "mean": [0] * 6,
            "std": [1] * 6,
        },
        "output": {
            "shape": [1, 2],
            "task": "velocity",
            "coordinate_frame": "body",
            "dimensions": 2,
            "unit": "m/s",
        },
    }


def test_pack_creates_two_file_archive(tmp_path):
    model = tmp_path / "model.tflite"
    model.write_bytes(b"\0\0\0\0TFL3payload")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest()))
    result = pack(model, manifest_path, tmp_path / "packed")
    with zipfile.ZipFile(result) as bundle:
        assert set(bundle.namelist()) == {"manifest.json", "model.tflite"}


def test_bad_channel_order_is_rejected(tmp_path):
    model = tmp_path / "model.tflite"
    model.write_bytes(b"\0\0\0\0TFL3payload")
    value = manifest()
    value["input"]["channels"].reverse()
    assert "input channels/order is incompatible" in validate(value, model)


def test_numeric_contract_rejects_values_android_cannot_load(tmp_path):
    model = tmp_path / "model.tflite"
    model.write_bytes(b"\0\0\0\0TFL3payload")
    value = manifest()
    value["input"]["sample_rate_hz"] = "100"
    value["input"]["std"][0] = -1
    errors = validate(value, model)
    assert "sample_rate_hz must be an integer in 10..1000" in errors
    assert "input mean/std must contain six finite values and std must be positive" in errors


def test_benchmark_and_unknown_keys_are_validated(tmp_path):
    model = tmp_path / "model.tflite"
    model.write_bytes(b"\0\0\0\0TFL3payload")
    value = manifest()
    value["unexpected"] = True
    value["benchmark"] = {"threads": 0, "stride": 201}
    errors = validate(value, model)
    assert "unknown manifest keys: unexpected" in errors
    assert "benchmark.threads must be an integer in 1..8" in errors
    assert "benchmark.stride must be an integer in 1..200" in errors
