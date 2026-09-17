"""RNIN（SenseINS）解析器的合成数据单元测试。"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from inertial_benchmark.data.converters import _rig_utils as rig
from inertial_benchmark.data.converters import rnin
from inertial_benchmark.data.converters.base import RAW_REQUIRED_ATTRS

EXTRA = [
    "rv_w",
    "rv_x",
    "rv_y",
    "rv_z",
    "magnet_un_x",
    "magnet_un_y",
    "magnet_un_z",
    "magnet_bias_x",
    "magnet_bias_y",
    "magnet_bias_z",
    "pressure",
]
GT_TILT = Rotation.from_rotvec([np.radians(3.0), 0.0, 0.0])  # gt 世界系相对重力倾斜 3°
VIO_YAW = Rotation.from_euler("z", 40.0, degrees=True)  # VIO 世界系与真实世界系之间的偏航
ACC_BIAS = np.array([0.05, -0.04, 0.03])
GYRO_BIAS = np.array([0.002, -0.001, 0.0015])


def make_frame(motion, *, with_gt, t0=5000.0, gt_delay=0.0, rate=250.0):
    t = motion["time"]
    rot = rig.as_rotation(motion["orientation"])
    frame = {"times": t + t0}
    for i, axis in enumerate("xyz"):
        frame[f"gyro_{axis}"] = motion["gyroscope"][:, i] + GYRO_BIAS[i]
        frame[f"acce_{axis}"] = motion["accelerometer"][:, i] + ACC_BIAS[i]
        frame[f"vio_gyro_bias_{axis}"] = np.full(len(t), GYRO_BIAS[i])
        frame[f"vio_acce_bias_{axis}"] = np.full(len(t), ACC_BIAS[i])
    vio_rot = VIO_YAW * rot
    vio_q = rig.from_rotation(vio_rot)
    vio_p = VIO_YAW.apply(motion["position"])
    vio_v = VIO_YAW.apply(motion["velocity"])
    for i, axis in enumerate("xyz"):
        frame[f"vio_p_{axis}"] = vio_p[:, i]
        frame[f"vio_v_{axis}"] = vio_v[:, i]
    for i, axis in enumerate("wxyz"):
        frame[f"vio_q_{axis}"] = vio_q[:, i]
    if with_gt:
        # 真值时钟晚 gt_delay 秒：在 t 时刻记录的是 t - gt_delay 时刻的位姿
        tt = np.clip(t - gt_delay, t[0], t[-1])
        from scipy.spatial.transform import Slerp

        gt_rot = GT_TILT * Slerp(t, rot)(tt)
        gt_p = GT_TILT.apply(
            np.stack([np.interp(tt, t, motion["position"][:, k]) for k in range(3)], 1)
        )
        gt_q = rig.from_rotation(gt_rot)
    else:
        gt_q = np.tile([1.0, 0.0, 0.0, 0.0], (len(t), 1))
        gt_p = np.zeros((len(t), 3))
    for i, axis in enumerate("xyz"):
        frame[f"gt_p_{axis}"] = gt_p[:, i]
    for i, axis in enumerate("wxyz"):
        frame[f"gt_q_{axis}"] = gt_q[:, i]
    gv = rig.from_rotation(Rotation.from_euler("z", -70.0, degrees=True) * rot)
    gv[:5] = 0.0  # 开头缺失
    for i, axis in enumerate("wxyz"):
        frame[f"gv_{axis}"] = gv[:, i]
    for name in EXTRA:
        frame[name] = np.zeros(len(t))
    return pd.DataFrame(frame)


@pytest.fixture(scope="module")
def rnin_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("rnin") / "RNIN" / "data"
    specs = {
        ("data_train", "0"): dict(with_gt=False, t0=5000.0, seed=1),
        ("data_train", "1"): dict(with_gt=True, t0=5100.0, seed=2),
        ("data_train", "10"): dict(with_gt=False, t0=90000.0, seed=3),
        ("data_val", "0"): dict(with_gt=True, t0=5200.0, seed=4, gt_delay=0.08),
        ("data_test", "0"): dict(with_gt=False, t0=5300.0, seed=5),
    }
    for (split, name), spec in specs.items():
        folder = root / split / name
        folder.mkdir(parents=True)
        motion = rig.simulate_rig_motion(40.0, rate=250.0, seed=spec.pop("seed"))
        make_frame(motion, **spec).to_csv(folder / "SenseINS.csv", index=False)
    return root


def test_official_splits_and_ids(rnin_root):
    splits = rnin.official_splits(rnin_root.parent)
    assert splits == {
        "train": ["train_0", "train_1", "train_10"],
        "val": ["val_0"],
        "test": ["test_0"],
    }
    assert rnin.list_sequences(rnin_root) == ["train_0", "train_1", "train_10", "val_0", "test_0"]
    assert set(rnin.list_sequences(rnin_root)) == set().union(*splits.values())


def test_vio_reference_keeps_raw_imu_and_records_bias(rnin_root):
    raw = next(rnin.iter_raw_sequences(rnin_root, only=["train_0"]))
    assert raw.rejected is None, raw.rejected
    assert raw.check_shapes() == [] and set(RAW_REQUIRED_ATTRS) <= set(raw.attrs)
    assert raw.attrs["reference_type"] == "vio" and raw.attrs["position_source"].startswith("VIO")
    motion = rig.simulate_rig_motion(40.0, rate=250.0, seed=1)
    # 零偏未被补偿
    np.testing.assert_allclose(raw.accelerometer, motion["accelerometer"] + ACC_BIAS)
    bias = json.loads(raw.attrs["vio_bias"])
    np.testing.assert_allclose(bias["acce_last"], ACC_BIAS)
    assert raw.velocity is not None
    assert raw.device_orientation is not None and np.isfinite(raw.device_orientation).all()
    assert any("game rotation vector missing in 5 rows" in n for n in raw.notes)
    stats = rig.parse_stats_note(raw.notes)
    assert stats["gravity_error"] < 0.2 and stats["gyro_window_error_median_deg"] < 2.0


def test_gt_reference_is_leveled_and_has_no_velocity(rnin_root):
    raw = next(rnin.iter_raw_sequences(rnin_root, only=["train_1"]))
    assert raw.rejected is None, raw.rejected
    assert raw.attrs["reference_type"] == "gt" and raw.velocity is None
    level_note = [n for n in raw.notes if "leveled" in n][0]
    assert "tilt 3.00 deg" in level_note
    stats = rig.parse_stats_note(raw.notes)
    assert stats["gravity_tilt_deg"] < 0.5
    # 调平后世界系与真实世界系只差一个绕 z 的旋转
    motion = rig.simulate_rig_motion(40.0, rate=250.0, seed=2)
    rel = rig.as_rotation(raw.orientation) * rig.as_rotation(motion["orientation"]).inv()
    tilt = np.degrees(np.linalg.norm(rel.apply([0.0, 0.0, 1.0])[:, :2], axis=1))
    assert tilt.max() < 0.1


def test_gt_time_offset_is_detected_and_corrected(rnin_root):
    raw = next(rnin.iter_raw_sequences(rnin_root, only=["val_0"]))
    assert raw.rejected is None, raw.rejected
    notes = [n for n in raw.notes if n.startswith("time offset corrected")]
    # 真值时间戳比真实时刻晚 0.08 s，修正后参考时间整体提前 0.08 s
    assert notes and "-0.080 s" in notes[0]
    np.testing.assert_allclose(raw.pose_time - raw.imu_time, -0.08, atol=1e-9)


def test_session_grouping_uses_rate_and_clock_continuity(rnin_root):
    table = rnin.session_table(str(rnin_root))
    assert table["train_0"] == table["train_1"] == table["val_0"] == table["test_0"]
    assert table["train_10"] != table["train_0"]
    assert table["train_0"].endswith("_250hz")


def test_official_gt_criterion():
    assert not rnin.official_has_gt(np.ones(200))
    frozen = np.full(200, 0.7)
    assert not rnin.official_has_gt(frozen)
    varying = np.linspace(0.5, 0.9, 200)
    assert rnin.official_has_gt(varying)


def test_missing_columns_and_unsorted_time(tmp_path):
    motion = rig.simulate_rig_motion(30.0, rate=250.0, seed=7)
    frame = make_frame(motion, with_gt=False)
    bad = tmp_path / "bad.csv"
    frame.drop(columns=["gt_q_w"]).to_csv(bad, index=False)
    raw = rnin.parse_csv(bad, "x")
    assert raw.rejected and "missing columns" in raw.rejected
    shuffled = pd.concat([frame.iloc[:100], frame.iloc[[50]], frame.iloc[100:]]).iloc[::-1]
    path = tmp_path / "unsorted.csv"
    shuffled.to_csv(path, index=False)
    raw = rnin.parse_csv(path, "y")
    assert np.all(np.diff(raw.imu_time) > 0) and len(raw.imu_time) == len(frame)
    assert any("dropped 1 duplicate" in n for n in raw.notes)
