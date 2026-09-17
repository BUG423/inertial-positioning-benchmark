"""测试用的合成转换器（通过模块注入使用，不属于 converters 包）。

``source/spec.json`` 描述序列：``{"sequences": [{"id", "seed", "group", "split", "kind", ...}]}``，
``kind`` 取 ``normal`` / ``reject`` / ``bad_gravity`` / ``short`` / ``nan`` / ``raise``。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from synthetic import make_raw_sequence  # noqa: E402

from inertial_benchmark.data.converters.base import RawSequence  # noqa: E402

NAME = "fake"
VERSION = "0.1"
LICENSE = "CC0 (synthetic)"


def _spec(source: Path) -> list:
    return json.loads((Path(source) / "spec.json").read_text())["sequences"]


def official_splits(source: Path) -> dict:
    splits: dict = {}
    for entry in _spec(source):
        for name in entry.get("split", "train").split("+"):
            splits.setdefault(name, []).append(entry["id"])
    return splits


def list_sequences(source: Path) -> list:
    return [e["id"] for e in _spec(source)]


def iter_raw_sequences(source: Path, only=None):
    for entry in _spec(source):
        sid = entry["id"]
        if only is not None and sid not in only:
            continue
        kind = entry.get("kind", "normal")
        if kind == "raise":
            raise RuntimeError(f"cannot parse {sid}")
        raw = make_raw_sequence(
            sequence_id=sid,
            duration=1.0 if kind == "short" else entry.get("duration", 12.0),
            imu_rate=entry.get("imu_rate", 100.0),
            pose_rate=entry.get("pose_rate"),
            seed=entry.get("seed", 0),
            group_id=entry.get("group", "g0"),
            jitter=entry.get("jitter", 0.0),
            duplicates=entry.get("duplicates", 0),
            imu_gaps=[tuple(g) for g in entry.get("imu_gaps", [])],
            with_device=entry.get("with_device", False),
        )
        if kind == "reject":
            raw = RawSequence(sequence_id=sid, imu_time=np.zeros(0), gyroscope=np.zeros((0, 3)),
                              accelerometer=np.zeros((0, 3)), pose_time=np.zeros(0),
                              position=np.zeros((0, 3)), orientation=np.zeros((0, 4)),
                              rejected="missing pose file")
        elif kind == "bad_gravity":
            raw.accelerometer = -raw.accelerometer
        elif kind == "nan":
            raw.gyroscope[50:60] = np.nan
            raw.imu_valid = np.ones(len(raw.imu_time), bool)
            raw.notes.append("injected NaN gyroscope samples")
        yield raw
