"""RoNIN 真实数据检查（原始数据不存在时跳过）。数据根目录可用环境变量 IPB_RAW_ROOT 覆盖。"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest

from inertial_benchmark.data.converters import _phone_utils as pu
from inertial_benchmark.data.converters import ronin

RAW_ROOT = Path(os.environ.get("IPB_RAW_ROOT", "/workspace/webCodex/datasets/imu_odometry/raw"))
SOURCE = RAW_ROOT / "_staging" / "RoNIN"

pytestmark = pytest.mark.skipif(not SOURCE.is_dir(), reason=f"RoNIN raw data not found at {SOURCE}")


@pytest.fixture(scope="module")
def sample():
    names = ["a000_1", "a002_1", "a006_2", "a011_3"]
    return {raw.sequence_id: raw for raw in ronin.iter_raw_sequences(SOURCE, only=names)}


@pytest.mark.parametrize("name", ["a000_1", "a002_1", "a006_2"])
def test_accepted_sequences_pass_physics(sample, name):
    raw = sample[name]
    assert raw.rejected is None, raw.rejected
    assert raw.check_shapes() == []
    stats = pu.parse_physics_note(raw.notes)
    assert stats["failures"] == []
    assert stats["gravity_error"] < pu.CHECK_GRAVITY_TOL
    assert stats["gyro_window_err_deg_median"] < pu.CHECK_GYRO_TOL_DEG
    assert abs(stats["time_offset_s"]) < 0.01
    assert raw.device_orientation is not None
    assert np.all(np.diff(raw.imu_time) > 0)


def test_orientation_source_and_known_rejection(sample):
    assert sample["a000_1"].attrs["orientation_source"] == "game_rv_aligned_to_tango_start"
    assert (
        sample["a002_1"].attrs["orientation_source"] == "ekf_aligned_to_tango_start"
    )  # grv 误差 56.6°
    assert sample["a006_2"].attrs["placement"] == "mixed"
    # a011_3：game_rv 与 ekf 末端误差都 > 20°，官方规则选陀螺积分，倾角漂移导致重力检查失败
    assert sample["a011_3"].attrs["orientation_source"] == "gyro_integration_aligned_to_tango_start"
    assert "gravity" in sample["a011_3"].rejected


def test_tango_orientation_is_a_different_body(sample):
    raw = sample["a000_1"]
    with h5py.File(SOURCE / "train1" / "a000_1" / "data.hdf5", "r") as handle:
        start = len(handle["synced/time"]) - len(raw.imu_time)
        tango_ori = np.asarray(handle["pose/tango_ori"])[start:]
    assert pu.physics_check(replace(raw, orientation=tango_ori))["failures"]


def test_official_splits():
    splits = ronin.official_splits(SOURCE)
    assert {k: len(v) for k, v in splits.items()} == {
        "train": 72,  # 官方 73 条，a007_3 未公开
        "val": 16,
        "test_seen": 32,
        "test_unseen": 32,
        "test": 64,
    }
    listed = set(splits["train"]) | set(splits["val"]) | set(splits["test"])
    assert listed == set(ronin.list_sequences(SOURCE))
    assert not set(splits["train"]) & set(splits["val"])
    assert not (set(splits["train"]) | set(splits["val"])) & set(splits["test"])
