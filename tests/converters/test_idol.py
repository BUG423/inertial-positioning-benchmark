"""IDOL 解析器的合成数据单元测试（需要可选依赖 pyarrow）。"""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip("pyarrow")
import pandas as pd  # noqa: E402

from inertial_benchmark.data.converters import _rig_utils as rig  # noqa: E402
from inertial_benchmark.data.converters import idol  # noqa: E402
from inertial_benchmark.data.converters.base import RAW_REQUIRED_ATTRS  # noqa: E402

WORLD_TILT = Rotation.from_rotvec([np.radians(4.0), 0.0, 0.0])  # 真值世界系相对重力倾斜 4°
IOS_YAW = Rotation.from_euler("z", 50.0, degrees=True)
GAP = slice(3000, 3500)  # 5 s 数据缺口


def make_frame(seed, *, extra_column=False, shuffle_columns=False, time_offset=0.0):
    motion = rig.simulate_rig_motion(60.0, rate=100.0, seed=seed)
    r_b = rig.as_rotation(motion["orientation"])  # 真实 iPhone 姿态（水平世界系）
    r_si = idol.R_STENCIL_IPHONE
    r_s = WORLD_TILT * r_b * r_si.inv()  # Stencil 姿态（倾斜的地图世界系）
    t = motion["time"]
    q_s = rig.from_rotation(r_s)
    q_ios = rig.from_rotation(IOS_YAW * r_b)
    pos = WORLD_TILT.apply(motion["position"])
    acc_g = -motion["accelerometer"] / rig.STANDARD_GRAVITY  # iOS：−比力，单位 G
    gyro = motion["gyroscope"]
    if time_offset:
        # iPhone 时间戳偏晚 time_offset 秒：同一时刻的 IMU 被标在更晚的时间
        gyro = np.stack([np.interp(t - time_offset, t, gyro[:, k]) for k in range(3)], 1)
        acc_g = np.stack([np.interp(t - time_offset, t, acc_g[:, k]) for k in range(3)], 1)
    data = {
        "index": np.arange(len(t)),
        "timestamp": 1.58e9 + t,
        "orientW": q_s[:, 0], "orientX": q_s[:, 1], "orientY": q_s[:, 2], "orientZ": q_s[:, 3],
        "processedPosX": pos[:, 0], "processedPosY": pos[:, 1], "processedPosZ": pos[:, 2],
        "iphoneOrientW": q_ios[:, 0], "iphoneOrientX": q_ios[:, 1], "iphoneOrientY": q_ios[:, 2],
        "iphoneOrientZ": q_ios[:, 3],
        "iphoneAccX": acc_g[:, 0], "iphoneAccY": acc_g[:, 1], "iphoneAccZ": acc_g[:, 2],
        "iphoneGyroX": gyro[:, 0], "iphoneGyroY": gyro[:, 1], "iphoneGyroZ": gyro[:, 2],
        "iphoneMagX": np.zeros(len(t)), "iphoneMagY": np.zeros(len(t)), "iphoneMagZ": np.zeros(len(t)),
    }
    f_s = r_si.apply(motion["accelerometer"])
    w_s = r_si.apply(motion["gyroscope"])
    for i, axis in enumerate("XYZ"):
        data[f"stencilAcc{axis}"] = f_s[:, i]
        data[f"stencilGyro{axis}"] = w_s[:, i]
    frame = pd.DataFrame(data)
    keep = np.ones(len(frame), dtype=bool)
    keep[GAP] = False
    frame = frame[keep].reset_index(drop=True)
    if extra_column:
        frame.insert(0, "level_0", np.arange(len(frame)))
    if shuffle_columns:
        frame = frame[list(reversed(frame.columns))]
    return frame, motion, keep


@pytest.fixture(scope="module")
def idol_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("idol") / "IDOL"
    layout = {
        ("building1", "train"): [(0, dict(seed=1), {"subjectID": 0, "calibration": "none"})],
        ("building1", "known"): [(0, dict(seed=2, time_offset=0.08), {"subjectID": 1, "calibration": "start"})],
        ("building1", "unknown"): [(0, dict(seed=3), {"subjectID": 2, "calibration": "end"})],
        ("building2", "known"): [(0, dict(seed=4, extra_column=True, shuffle_columns=True),
                                  {"subjectID": 1, "calibration": "none"})],
        ("building2", "unknown"): [(0, dict(seed=5), {"subjectID": 9, "calibration": "none"})],
    }
    for (building, subset), items in layout.items():
        folder = root / building / subset
        folder.mkdir(parents=True)
        meta = {}
        for index, kwargs, info in items:
            frame, _, _ = make_frame(**kwargs)
            frame.to_feather(folder / f"{index}.feather")
            meta[str(index)] = info
        (folder / "metadata.json").write_text(json.dumps(meta))
    return root


def test_official_splits(idol_root):
    splits = idol.official_splits(idol_root.parent)
    assert splits["train"] == ["building1_train_0"]
    assert splits["test_known"] == ["building1_known_0", "building2_known_0"]
    assert splits["test_unknown"] == ["building1_unknown_0", "building2_unknown_0"]
    assert splits["test"] == splits["test_known"] + splits["test_unknown"]
    assert splits["test_known_building1"] == ["building1_known_0"]
    assert "val" not in splits
    assert set(idol.list_sequences(idol_root)) == set(splits["train"]) | set(splits["test"])


def test_units_signs_frames_and_leveling(idol_root):
    raw = next(idol.iter_raw_sequences(idol_root, only=["building1_train_0"]))
    assert raw.rejected is None, raw.rejected
    assert raw.check_shapes() == [] and set(RAW_REQUIRED_ATTRS) <= set(raw.attrs)
    _, motion, keep = make_frame(seed=1)
    # 加速度：G → m/s²，iOS 符号翻转为比力
    np.testing.assert_allclose(raw.accelerometer, motion["accelerometer"][keep], atol=1e-9)
    np.testing.assert_allclose(raw.gyroscope, motion["gyroscope"][keep], atol=1e-12)
    # 参考姿态 = 调平后的 Stencil 姿态换到 iPhone 机体 ≈ 真实 iPhone 姿态
    err = rig.as_rotation(raw.orientation).inv() * rig.as_rotation(motion["orientation"][keep])
    assert np.degrees(err.magnitude()).max() < 0.3
    assert any("tilt 4.0" in n or "tilt 3.9" in n for n in raw.notes if "leveled" in n)
    # CoreMotion 姿态保留为设备姿态（与参考只差偏航）
    rel = rig.as_rotation(raw.device_orientation) * rig.as_rotation(motion["orientation"][keep]).inv()
    np.testing.assert_allclose(np.degrees(rel.magnitude()), 50.0, atol=1e-6)
    assert raw.attrs["subject_id"] == raw.attrs["group_id"] == "subject00"
    assert raw.attrs["start_time_unix"] == pytest.approx(1.58e9)
    stats = rig.parse_stats_note(raw.notes)
    assert stats["gravity_error"] < 0.1 and stats["gyro_window_error_median_deg"] < 0.5


def test_gap_is_kept_not_interpolated(idol_root):
    raw = next(idol.iter_raw_sequences(idol_root, only=["building1_unknown_0"]))
    assert np.diff(raw.imu_time).max() == pytest.approx(5.01, abs=1e-6)
    assert len(raw.imu_time) == 6000 - 500
    assert any("gap" in n for n in raw.notes)


def test_extra_column_and_shuffled_order(idol_root):
    raw = next(idol.iter_raw_sequences(idol_root, only=["building2_known_0"]))
    assert raw.rejected is None, raw.rejected
    assert any("level_0" in n for n in raw.notes)
    assert raw.attrs["building"] == "building2" and raw.attrs["official_subset"] == "known"


def test_time_offset_confirmed_by_gyro_is_corrected(idol_root):
    raw = next(idol.iter_raw_sequences(idol_root, only=["building1_known_0"]))
    assert raw.rejected is None, raw.rejected
    notes = [n for n in raw.notes if n.startswith("time offset corrected")]
    assert notes, raw.notes
    np.testing.assert_allclose(raw.pose_time - raw.imu_time, 0.08, atol=0.011)


def test_mounting_rotation_is_recoverable_from_gyro_and_reference():
    """R_STENCIL_IPHONE 的估计方法：iPhone 陀螺与 Stencil 姿态角速度的 Wahba 对齐。"""

    frame, _, _ = make_frame(seed=7)
    q_s = frame[["orientW", "orientX", "orientY", "orientZ"]].to_numpy()
    gyro = frame[["iphoneGyroX", "iphoneGyroY", "iphoneGyroZ"]].to_numpy()
    t = frame["timestamp"].to_numpy()
    est = rig.estimate_mounting_rotation(t, gyro, t, q_s, prior=idol.R_STENCIL_IPHONE_PRIOR)
    assert np.degrees((est.rotation.inv() * idol.R_STENCIL_IPHONE).magnitude()) < 0.05
    assert est.angle_from_prior_deg == pytest.approx(1.42, abs=0.05)


def test_missing_columns_rejected(tmp_path):
    frame, _, _ = make_frame(seed=6)
    folder = tmp_path / "building1" / "train"
    folder.mkdir(parents=True)
    frame.drop(columns=["stencilAccX"]).to_feather(folder / "0.feather")
    raw = next(idol.iter_raw_sequences(tmp_path))
    assert raw.rejected and "missing columns" in raw.rejected
    assert raw.attrs["subject_id"] == "unknown"
