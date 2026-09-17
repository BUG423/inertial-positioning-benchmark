"""测试用的合成转换器（通过模块注入使用，不属于 converters 包）。

``source/spec.json`` 描述序列：``{"sequences": [{"id", "seed", "group", "split", "kind", ...}]}``，
``kind`` 取 ``normal`` / ``reject`` / ``bad_gravity`` / ``short`` / ``nan`` / ``raise`` /
``duplicate_of:<id>``（复制另一条的原始数据，模拟重复发布）。
条目可带 ``pose_offset``（位姿时钟相对 IMU 的平移，秒）与 ``imu_gaps``/``pose_gaps``。
顶层可选 ``extra``（``extra_splits`` 的返回值）、``grouped``（``grouped_splits`` 的返回值，
此时模块表现为 ``OFFICIAL_SPLITS_LEAK = True``，由测试通过 ``make_module`` 生成）。
"""

from __future__ import annotations

import json
import os
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


def _spec_all(source: Path) -> dict:
    return json.loads((Path(source) / "spec.json").read_text())


def extra_splits(source: Path) -> dict:
    return dict(_spec_all(source).get("extra", {}))


def official_splits(source: Path) -> dict:
    splits: dict = {}
    for entry in _spec(source):
        for name in entry.get("split", "train").split("+"):
            if name:  # split="" 表示不在任何官方划分中
                splits.setdefault(name, []).append(entry["id"])
    return splits


def list_sequences(source: Path) -> list:
    return [e["id"] for e in _spec(source)]


def iter_raw_sequences(source: Path, only=None):
    entries = {e["id"]: e for e in _spec(source)}
    for entry in _spec(source):
        sid = entry["id"]
        if only is not None and sid not in only:
            continue
        kind = entry.get("kind", "normal")
        if kind.startswith("duplicate_of:"):
            entry = {**entries[kind.split(":", 1)[1]], "id": sid}
            kind = "normal"
        if kind == "raise":
            raise RuntimeError(f"cannot parse {sid}")
        if kind == "crash":  # 模拟子进程被杀（OOM 等）：只在多进程测试里使用
            os._exit(9)
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
            pose_gaps=[tuple(g) for g in entry.get("pose_gaps", [])],
            with_device=entry.get("with_device", False),
            pose_offset=entry.get("pose_offset", 0.0),
        )
        if kind == "reject":
            raw = RawSequence(sequence_id=sid, imu_time=np.zeros(0), gyroscope=np.zeros((0, 3)),
                              accelerometer=np.zeros((0, 3)), pose_time=np.zeros(0),
                              position=np.zeros((0, 3)), orientation=np.zeros((0, 4)),
                              rejected="missing pose file")
        elif kind == "bad_gravity":
            raw.accelerometer = -raw.accelerometer
        elif kind == "nan":
            raw.gyroscope[500:510] = np.nan
            raw.imu_valid = np.ones(len(raw.imu_time), bool)
            raw.notes.append("injected NaN gyroscope samples")
        yield raw
