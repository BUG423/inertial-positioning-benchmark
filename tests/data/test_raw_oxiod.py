"""OxIOD 真实数据检查（原始数据不存在时跳过）。数据根目录可用环境变量 IPB_RAW_ROOT 覆盖。"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pytest

from inertial_benchmark.data.converters import _phone_utils as pu
from inertial_benchmark.data.converters import oxiod

RAW_ROOT = Path(os.environ.get("IPB_RAW_ROOT", "/workspace/webCodex/datasets/imu_odometry/raw"))
SOURCE = RAW_ROOT / "_staging" / "OxIOD"

pytestmark = pytest.mark.skipif(
    not (SOURCE / oxiod.ROOT_NAME).is_dir(), reason=f"OxIOD raw data not found at {SOURCE}"
)

ACCEPTED = [
    "handheld_data1_seq1",
    "pocket_data2_seq1",  # 陀螺零偏缓慢漂移，需分段零偏模型
    "running_data1_seq3",  # 时钟偏移约 1.66 s
    "multi_users_user3_seq1",  # 时钟速率差约 −490 ppm
    "multi_devices_nexus5_seq1",  # Android 长表格式
    "trolley_data2_seq1",  # Vicon 文件名为 hand1.csv
]
REJECTED = ["handbag_data2_seq3", "multi_users_user5_seq5"]


@pytest.fixture(scope="module")
def sample():
    return {raw.sequence_id: raw for raw in oxiod.iter_raw_sequences(SOURCE, only=ACCEPTED + REJECTED)}


@pytest.mark.parametrize("name", ACCEPTED)
def test_accepted_sequences_pass_physics(sample, name):
    raw = sample[name]
    assert raw.rejected is None, raw.rejected
    assert raw.check_shapes() == []
    stats = pu.parse_physics_note(raw.notes)
    assert stats["failures"] == []
    assert stats["gravity_error"] < 0.3
    assert stats["gyro_window_err_deg_median"] < pu.CHECK_GYRO_TOL_DEG
    assert stats["pose_valid_fraction"] > 0.9
    # Vicon 刚体与手机之间的外参约为绕 z 轴 170–180°
    q_sb = np.array([float(v) for v in raw.attrs["oxiod_extrinsic_wxyz"].split(",")])
    assert 165 < np.degrees(pu.quat_angle(q_sb)) < 180
    assert np.all(np.diff(raw.imu_time) > 0)


def test_clock_models(sample):
    assert sample["running_data1_seq3"].attrs["oxiod_clock_offset_s"] == pytest.approx(1.66, abs=0.05)
    assert sample["multi_users_user3_seq1"].attrs["oxiod_clock_skew_ppm"] < -300
    assert sample["handheld_data1_seq1"].attrs["oxiod_clock_skew_ppm"] == 0.0


def test_corrupted_files_are_rejected(sample):
    for name in REJECTED:
        assert "scientific notation" in sample[name].rejected


def test_ios_sign_convention(sample):
    raw = sample["handheld_data1_seq1"]
    flipped = oxiod.RawSequence(
        raw.sequence_id, raw.imu_time, raw.gyroscope, -raw.accelerometer, raw.pose_time, raw.position,
        raw.orientation, pose_valid=raw.pose_valid,
    )
    stats = pu.physics_check(flipped)
    assert stats["acc_world_mean"][2] < -9.0
    # 设备姿态（CoreMotion）同样重力对齐
    acc_w = pu.quat_rotate(raw.device_orientation, raw.accelerometer).mean(axis=0)
    assert np.linalg.norm(acc_w - [0, 0, pu.GRAVITY]) < 0.3


def test_official_splits():
    splits = oxiod.official_splits(SOURCE)
    assert {k: len(v) for k, v in splits.items()} == {"train": 62, "test": 9}
    assert not set(splits["train"]) & set(splits["test"])
    assert set(splits["test"]) == {
        "handheld_data5_seq1",
        "handheld_data5_seq2",
        "handheld_data5_seq3",
        "handheld_data5_seq4",
        "handbag_data2_seq4",
        "pocket_data2_seq6",
        "running_data1_seq7",
        "slow_walking_data1_seq8",
        "trolley_data2_seq6",
    }
    assert len(oxiod.list_sequences(SOURCE)) == 128


def test_placement_table_matches_readme():
    root = SOURCE / oxiod.ROOT_NAME
    for user in ("user3", "user5"):
        text = (root / "multi users" / user / "syn" / "Readme.txt").read_text(encoding="utf-8", errors="replace")
        for lo, hi, name in re.findall(r"(\d+)\s*-\s*(\d+)\s*:\s*([a-z]+)", text):
            expected = {"handheld": "handheld", "pocket": "pocket", "handbag": "bag"}[name]
            for k in range(int(lo), int(hi) + 1):
                assert oxiod.placement_of("multi users", user, k) == (expected, "readme")
