"""RIDI 真实数据检查（原始数据不存在时跳过）。数据根目录可用环境变量 IPB_RAW_ROOT 覆盖。"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from inertial_benchmark.data.converters import _phone_utils as pu
from inertial_benchmark.data.converters import ridi

RAW_ROOT = Path(os.environ.get("IPB_RAW_ROOT", "/workspace/webCodex/datasets/imu_odometry/raw"))
SOURCE = RAW_ROOT / "_staging" / "RIDI" / "data_publish_v2"

pytestmark = pytest.mark.skipif(not SOURCE.is_dir(), reason=f"RIDI raw data not found at {SOURCE}")

NAMES = {
    "dan_bag1": "bag",
    "dan_body1": "body",
    "dan_handheld1": "handheld",
    "dan_leg1": "pocket",
    "ruixuan_leg2": "pocket",
}


@pytest.fixture(scope="module")
def sample():
    return {raw.sequence_id: raw for raw in ridi.iter_raw_sequences(SOURCE, only=list(NAMES))}


@pytest.mark.parametrize("name", sorted(NAMES))
def test_same_device_tango_pose_is_consistent(sample, name):
    raw = sample[name]
    assert raw.rejected is None, raw.rejected
    assert raw.check_shapes() == []
    assert raw.attrs["placement"] == NAMES[name]
    stats = pu.parse_physics_note(raw.notes)
    assert stats["failures"] == []
    # IMU 与 Tango 位姿属于同一刚体：手眼残余旋转 < 1°，时间偏移 ≈ 0
    assert stats["gyro_ref_extrinsic_deg"] < 1.0
    assert abs(stats["time_offset_s"]) < 0.005
    assert 195 < stats["imu_rate_hz"] < 210


def test_heading_check_catches_world_frame_mismatch(sample):
    raw = sample["dan_handheld1"]
    stats = pu.parse_physics_note(raw.notes)
    assert abs(stats["heading_offset_deg"]) < 10 and stats["heading_corr"] > 0.6
    yawed = pu.quat_mul(pu.quat_from_axis_angle([0, 0, 1], np.pi / 2), raw.orientation)
    bad = pu.physics_check(replace(raw, orientation=yawed))
    assert any(f.startswith("heading") for f in bad["failures"])


def test_device_orientation_differs_by_constant_yaw(sample):
    raw = sample["dan_handheld1"]
    rel = pu.quat_mul(
        raw.orientation, pu.quat_conj(raw.device_orientation)
    )  # world_ref ← world_grv
    tilt = np.degrees(pu.tilt_of_quat(rel))
    assert np.median(tilt) < 3.0


def test_official_splits_and_placements():
    splits = ridi.official_splits(SOURCE)
    assert {k: len(v) for k, v in splits.items()} == {"train": 49, "test": 23}
    assert not set(splits["train"]) & set(splits["test"])
    names = ridi.list_sequences(SOURCE)
    assert len(names) == 94
    root = ridi._root(SOURCE)
    listed = {}
    for fn in ridi._SPLIT_FILES.values():
        listed.update(dict(ridi._read_list(root / fn)))
    counts = Counter(ridi.placement_of(n, listed) for n in names)
    assert counts == {"body": 29, "handheld": 27, "bag": 23, "leg": 15}
