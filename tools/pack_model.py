#!/usr/bin/env python3
"""Validate and package a TFLite inertial model as .iplmodel."""

from __future__ import annotations

import argparse
import json
import math
import re
import zipfile
from pathlib import Path

CHANNELS = ["gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z"]
ROOT_KEYS = {"schema_version", "id", "name", "version", "runtime", "model_file", "input", "output", "benchmark", "description"}
INPUT_KEYS = {"shape", "layout", "sample_rate_hz", "channels", "dtype", "mean", "std"}
OUTPUT_KEYS = {"shape", "task", "coordinate_frame", "dimensions", "unit"}
BENCHMARK_KEYS = {"threads", "warmup_runs", "stride", "max_windows"}


def validate(manifest: object, model: Path) -> list[str]:
    errors: list[str] = []
    if not isinstance(manifest, dict):
        return ["manifest root must be an object"]
    required = {"schema_version", "id", "name", "version", "runtime", "model_file", "input", "output"}
    missing = sorted(required - manifest.keys())
    if missing:
        return ["missing keys: " + ", ".join(missing)]
    unknown = sorted(manifest.keys() - ROOT_KEYS)
    if unknown:
        errors.append("unknown manifest keys: " + ", ".join(unknown))
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        errors.append("schema_version must be 1")
    model_id = manifest["id"]
    if not isinstance(model_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{1,63}", model_id):
        errors.append("id must match [a-z0-9][a-z0-9._-]{1,63}")
    for key in ("name", "version"):
        if not isinstance(manifest[key], str) or not manifest[key].strip():
            errors.append(f"{key} must be a non-empty string")
    if "description" in manifest and not isinstance(manifest["description"], str):
        errors.append("description must be a string")
    if manifest["runtime"] != "tflite" or manifest["model_file"] != "model.tflite":
        errors.append("runtime/model_file must be tflite/model.tflite")
    input_spec = manifest.get("input", {})
    output_spec = manifest.get("output", {})
    if not isinstance(input_spec, dict) or not isinstance(output_spec, dict):
        return errors + ["input and output must be objects"]
    unknown_input = sorted(input_spec.keys() - INPUT_KEYS)
    unknown_output = sorted(output_spec.keys() - OUTPUT_KEYS)
    if unknown_input:
        errors.append("unknown input keys: " + ", ".join(unknown_input))
    if unknown_output:
        errors.append("unknown output keys: " + ", ".join(unknown_output))
    shape = input_spec.get("shape")
    valid_shape = (
        isinstance(shape, list)
        and len(shape) == 3
        and all(type(value) is int for value in shape)
        and shape[0] == 1
        and shape[2] == 6
        and 2 <= shape[1] <= 10000
    )
    if not valid_shape:
        errors.append("input.shape must be [1,T,6], 2 <= T <= 10000")
    if input_spec.get("layout") != "NTC" or input_spec.get("dtype") != "float32":
        errors.append("input layout/dtype must be NTC/float32")
    if input_spec.get("channels") != CHANNELS:
        errors.append("input channels/order is incompatible")
    sample_rate = input_spec.get("sample_rate_hz")
    if type(sample_rate) is not int or not 10 <= sample_rate <= 1000:
        errors.append("sample_rate_hz must be an integer in 10..1000")

    def finite_values(value: object, *, positive: bool = False) -> bool:
        return (
            isinstance(value, list)
            and len(value) == 6
            and all(
                not isinstance(item, bool)
                and isinstance(item, (int, float))
                and math.isfinite(float(item))
                and (not positive or item > 0)
                for item in value
            )
        )

    if not finite_values(input_spec.get("mean")) or not finite_values(input_spec.get("std"), positive=True):
        errors.append("input mean/std must contain six finite values and std must be positive")
    dimensions = output_spec.get("dimensions")
    output_shape = output_spec.get("shape")
    valid_output_shape = (
        type(dimensions) is int
        and dimensions in (2, 3)
        and isinstance(output_shape, list)
        and all(type(value) is int for value in output_shape)
        and output_shape == [1, dimensions]
    )
    if not valid_output_shape:
        errors.append("output shape/dimensions must be [1,2] or [1,3]")
    if output_spec.get("task") != "velocity" or output_spec.get("unit") != "m/s":
        errors.append("output task/unit must be velocity/m/s")
    if output_spec.get("coordinate_frame") not in ("body", "world"):
        errors.append("output coordinate_frame must be body or world")

    benchmark = manifest.get("benchmark", {})
    if not isinstance(benchmark, dict):
        errors.append("benchmark must be an object")
    else:
        unknown_benchmark = sorted(benchmark.keys() - BENCHMARK_KEYS)
        if unknown_benchmark:
            errors.append("unknown benchmark keys: " + ", ".join(unknown_benchmark))
        limits = {
            "threads": (benchmark.get("threads", 2), 1, 8),
            "warmup_runs": (benchmark.get("warmup_runs", 10), 0, 100),
            "max_windows": (benchmark.get("max_windows", 5000), 1, 100000),
        }
        for key, (value, lower, upper) in limits.items():
            if type(value) is not int or not lower <= value <= upper:
                errors.append(f"benchmark.{key} must be an integer in {lower}..{upper}")
        stride = benchmark.get("stride", 10)
        window = shape[1] if valid_shape else 1
        if type(stride) is not int or not 1 <= stride <= window:
            errors.append(f"benchmark.stride must be an integer in 1..{window}")

    try:
        with model.open("rb") as stream:
            data = stream.read(8)
    except OSError as error:
        errors.append(f"cannot read model: {error}")
        data = b""
    if len(data) < 8 or data[4:8] != b"TFL3":
        errors.append("model is not a TFLite FlatBuffer")
    return errors


def pack(model: Path, manifest_path: Path, output: Path) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = validate(manifest, model)
    if errors:
        raise ValueError("\n".join(f"- {error}" for error in errors))
    output = output.with_suffix(".iplmodel")
    output.parent.mkdir(parents=True, exist_ok=True)
    canonical = json.dumps(manifest, ensure_ascii=False, indent=2).encode()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("manifest.json", canonical)
        bundle.write(model, "model.tflite")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", "-o", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = pack(args.model, args.manifest, args.output)
    except (ValueError, json.JSONDecodeError, OSError) as error:
        parser.error(str(error))
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
