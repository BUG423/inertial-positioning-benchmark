"""IMUNet 真实数据检查（原始数据不存在时跳过）。数据根目录可用环境变量 IPB_RAW_ROOT 覆盖。"""

from __future__ import annotations

import hashlib
import os
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from inertial_benchmark.data.converters import _phone_utils as pu
from inertial_benchmark.data.converters import imunet

RAW_ROOT = Path(os.environ.get("IPB_RAW_ROOT", "/workspace/webCodex/datasets/imu_odometry/raw"))
SOURCE = RAW_ROOT / "IMUNet" / "IMUNet_dataset"

pytestmark = pytest.mark.skipif(not SOURCE.is_dir(), reason=f"IMUNet raw data not found at {SOURCE}")

ACCEPTED = ["Indoor_Subject_2_S10_1", "Outdoor_Subject_5_S21_3", "Indoor_Subject_1_Tango_1", "Outdoor_Subjetc_1_S10_16"]
REJECTED = ["Outdoor_Subject_1_Xiaomi_1", "Outdoor_Subjetc_1_S10_13", "Indoor_Subject_1_S10_1"]


@pytest.fixture(scope="module")
def sample():
    return {raw.sequence_id: raw for raw in imunet.iter_raw_sequences(SOURCE, only=ACCEPTED + REJECTED)}


@pytest.mark.parametrize("name", ACCEPTED)
def test_accepted_devices_pass_physics(sample, name):
    raw = sample[name]
    assert raw.rejected is None, raw.rejected
    assert raw.check_shapes() == []
    stats = pu.parse_physics_note(raw.notes)
    assert stats["failures"] == []
    assert stats["gravity_error"] < pu.CHECK_GRAVITY_TOL
    # 约定正确时，陀螺与参考姿态之间只剩 < 2° 的残余旋转
    assert stats["gyro_ref_extrinsic_deg"] < 2.0


def test_known_rejections(sample):
    assert "accelerometer scale" in sample["Outdoor_Subject_1_Xiaomi_1"].rejected
    assert "duplicate of Outdoor_Subjetc_1_S10_16" in sample["Outdoor_Subjetc_1_S10_13"].rejected
    assert "gravity" in sample["Indoor_Subject_1_S10_1"].rejected


def test_conjugate_convention_fails_on_real_arcore_data(sample):
    raw = sample["Outdoor_Subject_5_S21_3"]
    data = pu.read_ridi_processed_csv(SOURCE / raw.sequence_id / "processed" / "data.csv")
    ori = pu.quat_normalize(data["ori"][pu.monotonic_mask(data["time"] / 1e9)])
    # 旧流水线：直接把 ori 当作共轭（world→body）使用 —— 症状是重力落到 −x
    stats = pu.physics_check(replace(raw, orientation=pu.quat_conj(ori)))
    assert stats["failures"]
    assert stats["acc_world_mean"][0] < -5
    # 本转换器的变换套在共轭上同样失败
    assert pu.physics_check(replace(raw, orientation=imunet.reference_orientation("S21", pu.quat_conj(ori))))["failures"]
    # 原样使用 ori（未转 z 向上、未补相机外参）也失败
    assert pu.physics_check(replace(raw, orientation=ori))["failures"]


def test_official_splits():
    splits = imunet.official_splits(SOURCE)
    assert {k: len(v) for k, v in splits.items()} == {"train": 90, "test": 36}
    assert not set(splits["train"]) & set(splits["test"])
    assert set(splits["train"]) | set(splits["test"]) == set(imunet.list_sequences(SOURCE))
    for dup, (keep, _) in imunet.KNOWN_DUPLICATES.items():
        assert dup in splits["train"] and keep in splits["test"]


def _fingerprint(path: Path) -> str:
    with open(path, "rb") as handle:
        head = handle.read(1 << 20)
    return f"{path.stat().st_size}:{hashlib.md5(head).hexdigest()}"


def test_duplicate_table_is_complete():
    """扫描全部 data.csv：内容相同的序列对必须都在 KNOWN_DUPLICATES 中（先按大小+首 1 MB 分组，再比全文 md5）。"""

    groups = defaultdict(list)
    for name in imunet.list_sequences(SOURCE):
        groups[_fingerprint(SOURCE / name / "processed" / "data.csv")].append(name)
    pairs = set()
    for names in groups.values():
        if len(names) < 2:
            continue
        by_md5 = defaultdict(list)
        for name in names:
            by_md5[imunet.file_md5(SOURCE / name / "processed" / "data.csv")].append(name)
        for digest, same in by_md5.items():
            if len(same) > 1:
                pairs.add((tuple(sorted(same)), digest))
    expected = {(tuple(sorted((dup, keep))), digest) for dup, (keep, digest) in imunet.KNOWN_DUPLICATES.items()}
    assert pairs == expected


def test_arcore_extrinsic_is_consistent_across_devices(sample):
    for name in ("Indoor_Subject_2_S10_1", "Outdoor_Subject_5_S21_3"):
        raw = sample[name]
        q, _, _ = pu.estimate_body_extrinsic(raw.imu_time, raw.gyroscope, raw.pose_time, raw.orientation)
        assert np.degrees(pu.quat_angle(q)) < 2.0
