"""RoNIN 转换器单元测试：用合成的 data.hdf5 / info.json 验证单位、标定、姿态来源与起点对齐。"""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from inertial_benchmark.data.converters import _phone_utils as pu
from inertial_benchmark.data.converters import ronin
from inertial_benchmark.data.converters.base import RAW_REQUIRED_ATTRS

GYRO_BIAS = np.array([0.011, -0.004, 0.007])
ACCE_BIAS = np.array([-0.09, 0.02, 0.03])
ACCE_SCALE = np.array([0.998, 1.003, 0.997])
Q_GRV_WORLD = pu.quat_from_axis_angle([0, 0, 1], 0.7)  # game_rv 世界系相对 Tango 世界系的偏航


def _write_sequence(folder, motion, errors=(3.0, 2.0, 4.0), start_frame=200, with_ekf=True):
    """按 RoNIN 发布格式写一条序列；手机真值姿态为 motion['q_wb']（Tango 世界系）。"""

    n = len(motion["t"])
    t = motion["t"] + 21000.0  # 手机开机时钟（秒）
    q_true = motion["q_wb"]
    # 胸前 Tango：只随行进方向偏航，另一个刚体
    q_tango = pu.quat_from_rotvec(np.outer(np.unwrap(pu.yaw_of_quat(q_true)), [0, 0, 1]) * 0.5)
    start_calib = pu.quat_mul(pu.quat_conj(q_tango[0]), q_true[0])
    grv = pu.quat_mul(Q_GRV_WORLD, q_true)
    folder.mkdir(parents=True)
    info = {
        "type": "annotated",
        "device": "asus4",
        "date": "01/08/19",
        "length": float(t[-1] - t[0]),
        "start_frame": start_frame,
        "imu_init_gyro_bias": GYRO_BIAS.tolist(),
        "imu_end_gyro_bias": GYRO_BIAS.tolist(),
        "imu_acce_bias": ACCE_BIAS.tolist(),
        "imu_acce_scale": ACCE_SCALE.tolist(),
        "start_calibration": start_calib.tolist(),
        "end_calibration": start_calib.tolist(),
        "gyro_integration_error": errors[0],
        "grv_ori_error": errors[1],
        "ekf_ori_error": errors[2],
    }
    (folder / "info.json").write_text(json.dumps(info))
    with h5py.File(folder / "data.hdf5", "w") as f:
        f["synced/time"] = t
        f["synced/gyro_uncalib"] = motion["gyro"] + GYRO_BIAS
        f["synced/gyro"] = motion["gyro"]
        f["synced/acce"] = motion["specific_force"] / ACCE_SCALE + ACCE_BIAS
        f["synced/game_rv"] = grv
        f["pose/tango_pos"] = motion["position"]
        f["pose/tango_ori"] = q_tango
        if with_ekf:
            f["pose/ekf_ori"] = q_true
    return n


@pytest.fixture(scope="module")
def motion():
    return pu.simulate_motion(duration=60.0, rate=200.0, seed=1)


def _angle_deg(a, b):
    return np.degrees(pu.quat_angle(pu.quat_mul(pu.quat_conj(a), b)))


def test_parse_units_calibration_and_orientation(tmp_path, motion):
    _write_sequence(tmp_path / "train1" / "a900_1", motion)
    (raw,) = list(ronin.iter_raw_sequences(tmp_path))
    assert raw.rejected is None, raw.rejected
    assert raw.check_shapes() == []
    assert set(RAW_REQUIRED_ATTRS) <= set(raw.attrs)
    s = slice(200, None)
    np.testing.assert_allclose(raw.gyroscope, motion["gyro"][s], atol=1e-9)
    np.testing.assert_allclose(raw.accelerometer, motion["specific_force"][s], atol=1e-9)
    np.testing.assert_allclose(raw.position, motion["position"][s])
    np.testing.assert_allclose(raw.imu_time, motion["t"][s] + 21000.0)
    np.testing.assert_array_equal(raw.imu_time, raw.pose_time)
    # 参考姿态 = 起点对齐后的 game_rv，应恢复手机真值姿态（不是 tango_ori）
    assert _angle_deg(raw.orientation, motion["q_wb"][s]).max() < 1e-6
    # 设备姿态 = 原始 game_rv，与参考世界系差常值偏航
    assert (
        _angle_deg(raw.device_orientation, pu.quat_mul(Q_GRV_WORLD, motion["q_wb"][s])).max() < 1e-6
    )
    assert raw.attrs["orientation_source"] == "game_rv_aligned_to_tango_start"
    assert raw.attrs["subject_id"] == "a900" and raw.attrs["group_id"] == "a900"
    assert raw.attrs["device_id"] == "asus4"
    stats = pu.parse_physics_note(raw.notes)
    assert stats["failures"] == []
    assert stats["gravity_error"] < 0.05


def test_tango_orientation_would_fail_physics_check(tmp_path, motion):
    """负对照：把胸前 Tango 的姿态当作手机姿态，物理自检必须失败。"""

    _write_sequence(tmp_path / "a901_1", motion)
    raw = ronin.load_sequence(tmp_path / "a901_1")
    with h5py.File(tmp_path / "a901_1" / "data.hdf5", "r") as f:
        raw.orientation = np.asarray(f["pose/tango_ori"])[200:]
    stats = pu.physics_check(raw)
    assert stats["failures"]


def test_orientation_source_selection_follows_official_rule(tmp_path, motion):
    info = {"gyro_integration_error": 3.0, "grv_ori_error": 19.9, "ekf_ori_error": 1.0}
    assert ronin.select_orientation_source(info, {"ekf": None})[0] == "game_rv"
    info["grv_ori_error"] = 25.0
    assert ronin.select_orientation_source(info, {"ekf": None})[0] == "ekf"
    assert ronin.select_orientation_source(info, {})[0] == "gyro_integration"

    _write_sequence(tmp_path / "a902_1", motion, errors=(9.0, 30.0, 2.0))
    (raw,) = list(ronin.iter_raw_sequences(tmp_path))
    assert raw.attrs["orientation_source"] == "ekf_aligned_to_tango_start"
    assert _angle_deg(raw.orientation, motion["q_wb"][200:]).max() < 1e-6


def test_gyro_integration_source_matches_truth(tmp_path, motion):
    _write_sequence(tmp_path / "a903_1", motion, errors=(2.0, 30.0, 25.0), with_ekf=False)
    raw = ronin.load_sequence(tmp_path / "a903_1")
    assert raw.attrs["orientation_source"] == "gyro_integration_aligned_to_tango_start"
    assert np.median(_angle_deg(raw.orientation, motion["q_wb"][200:])) < 0.5
    assert any("no gravity correction" in note for note in raw.notes)


def test_official_splits_filter_to_available(tmp_path, motion):
    for name in ("a000_1", "a000_7", "a006_2", "a000_6"):
        _write_sequence(tmp_path / "x" / name, motion)
    splits = ronin.official_splits(tmp_path)
    assert splits["train"] == ["a000_1"]
    assert splits["val"] == ["a000_6"]
    assert splits["test_seen"] == ["a000_7"]
    assert splits["test_unseen"] == ["a006_2"]
    assert splits["test"] == ["a000_7", "a006_2"]
    full = ronin._official_lists()
    assert [len(full[k]) for k in ("train", "val", "test_seen", "test_unseen")] == [73, 16, 32, 32]


def test_missing_and_broken_sequences_are_rejected_not_skipped(tmp_path, motion):
    _write_sequence(tmp_path / "a904_1", motion)
    (tmp_path / "a904_1" / "info.json").write_text("{}")
    raws = list(ronin.iter_raw_sequences(tmp_path, only=["a904_1", "a999_9"]))
    assert [r.sequence_id for r in raws] == ["a904_1", "a999_9"]
    assert all(r.rejected for r in raws)
    assert all(r.check_shapes() == [] for r in raws)
