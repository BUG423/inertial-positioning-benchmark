import h5py
import numpy as np
import pytest
from synthetic import make_sequence

from inertial_benchmark import CanonicalSequence, WindowDataset  # noqa: F401  旧接口仍可导入
from inertial_benchmark.data.format import (
    REQUIRED_ATTRS,
    Sequence,
    SequenceError,
    check_sequence,
    load_sequence,
    save_sequence,
    validate,
)
from inertial_benchmark.utils.geometry import quat_conjugate


def test_synthetic_sequence_is_valid():
    seq = make_sequence(duration=40.0)
    rep = validate(seq)
    assert rep.ok, str(rep)
    g = rep.info["gravity"]
    assert g["tilt_deg"] < 0.5
    assert abs(g["norm_error"]) < 0.05


@pytest.mark.parametrize("compression", ["lzf", "gzip", "none"])
def test_roundtrip(tmp_path, compression):
    seq = make_sequence(duration=5.0, device_yaw_offset=0.3)
    seq.valid_pose[10:20] = False
    seq.valid_device_orientation = np.ones(len(seq), bool)
    seq.valid_device_orientation[100:200] = False
    path = save_sequence(tmp_path / "a.h5", seq, compression=compression)
    back = load_sequence(path, validate=True)
    for name in ("timestamp", "gyroscope", "accelerometer", "orientation", "position",
                 "valid_imu", "valid_pose", "device_orientation", "velocity",
                 "valid_device_orientation"):
        np.testing.assert_array_equal(getattr(back, name), getattr(seq, name), err_msg=name)
    # 设备姿态的缺口只影响自己的掩码，不影响 valid（参考姿态视图）
    assert back.valid[100:200].all() and not back.valid_device[100:200].any()
    assert back.attrs["source_files"] == seq.attrs["source_files"]
    assert np.isnan(back.attrs["start_time_unix"])
    assert back.sample_rate == 200.0
    with h5py.File(path) as f:
        assert f["imu/gyroscope"].dtype == np.float32
        assert f["pose/position"].dtype == np.float64
        assert f["valid/imu"].dtype == bool
        assert f["valid/device_orientation"].dtype == bool
        assert set(REQUIRED_ATTRS) <= set(f.attrs)


def test_save_rejects_invalid(tmp_path):
    seq = make_sequence(duration=2.0)
    del seq.attrs["group_id"]
    with pytest.raises(SequenceError, match="group_id"):
        save_sequence(tmp_path / "bad.h5", seq)
    assert not (tmp_path / "bad.h5").exists()


def test_detects_structural_errors():
    seq = make_sequence(duration=2.0)
    seq.timestamp = seq.timestamp.copy()
    seq.timestamp[5] += 0.001
    seq.orientation = seq.orientation.copy()
    seq.orientation[7] *= -1
    seq.accelerometer = seq.accelerometer.copy()
    seq.accelerometer[3, 0] = np.nan
    seq.attrs["world_frame"] = "enu"
    rep = validate(seq, check_gravity=False)
    text = "\n".join(rep.errors)
    assert "not uniform" in text
    assert "sign discontinuities" in text
    assert "NaN" in text
    assert "world_frame" in text


def test_detects_non_unit_quaternion_and_rate():
    seq = make_sequence(duration=2.0)
    seq.orientation = seq.orientation * 1.01
    seq.attrs["sample_rate_hz"] = 100.0
    rep = validate(seq, check_gravity=False)
    text = "\n".join(rep.errors)
    assert "not unit" in text
    assert "expected 200" in text


def test_gravity_check_diagnostics():
    seq = make_sequence(duration=30.0)
    assert validate(seq).ok

    inverted = make_sequence(duration=30.0)
    inverted.orientation = quat_conjugate(inverted.orientation)
    rep = validate(inverted)
    assert not rep.ok
    assert "world_to_body" in " ".join(rep.errors)

    in_g = make_sequence(duration=30.0)
    in_g.accelerometer = in_g.accelerometer / 9.81
    rep = validate(in_g)
    assert "in g" in " ".join(rep.errors)

    xyzw = make_sequence(duration=30.0)
    xyzw.orientation = np.concatenate([xyzw.orientation[:, 1:], xyzw.orientation[:, :1]], 1)
    rep = validate(xyzw, check_gravity=True)
    assert "xyzw" in " ".join(rep.errors)

    flipped = make_sequence(duration=30.0)
    flipped.accelerometer = -flipped.accelerometer
    rep = validate(flipped)
    assert "-z" in " ".join(rep.errors)


def test_gravity_check_prefers_static_segment():
    seq = make_sequence(duration=30.0)
    n = len(seq)
    # 前 3 秒完全静止：机体系比力恒为 R^T [0,0,g]
    from inertial_benchmark.utils.geometry import quat_rotate

    q0 = seq.orientation[0].astype(np.float64)
    static = slice(0, 600)
    seq.orientation[static] = q0
    seq.gyroscope[static] = 0.0
    seq.accelerometer[static] = quat_rotate(quat_conjugate(q0), np.array([0, 0, 9.81]))
    rep = validate(seq)
    assert rep.info["gravity"]["used"] == "static"
    assert rep.info["gravity"]["static_samples"] >= 200
    assert n == len(seq)


def test_check_sequence_raises():
    seq = make_sequence(duration=2.0)
    seq.valid_imu[:] = False
    with pytest.raises(SequenceError, match="no sample"):
        check_sequence(seq)


def _write_v01(path, t, seq: Sequence, unix=False):
    attrs = {
        "schema_version": "0.1",
        "dataset": "legacy",
        "sequence_id": "old",
        "world_frame": "gravity_aligned_local",
        "timestamp_type": "unix" if unix else "relative",
        "orientation_convention": "body_to_world_wxyz",
        "accelerometer_type": "specific_force",
        "position_source": "analytic",
        "orientation_source": "analytic",
        "subject_id": "s1",
        "device_id": "d1",
        "source_license": "generated",
    }
    with h5py.File(path, "w") as f:
        f["timestamp"] = t
        f["imu/gyroscope"] = seq.gyroscope
        f["imu/accelerometer"] = seq.accelerometer
        f["pose/orientation"] = seq.orientation
        f["pose/position"] = seq.position
        mask = np.ones(len(t), bool)
        mask[50] = False
        f["valid/position"] = mask
        for k, v in attrs.items():
            f.attrs[k] = v


def test_load_v01_resamples_to_v1(tmp_path):
    from synthetic import Motion

    motion = Motion(3)
    rng = np.random.default_rng(0)
    t = np.arange(0, 20, 0.01) + rng.uniform(-2e-4, 2e-4, 2000)  # 100 Hz，带抖动
    src = Sequence(
        timestamp=t, gyroscope=motion.gyroscope(t), accelerometer=motion.accelerometer(t),
        orientation=motion.orientation(t), position=motion.position(t),
        valid_imu=np.ones(len(t), bool), valid_pose=np.ones(len(t), bool),
    )
    path = tmp_path / "old.h5"
    _write_v01(path, t + 1.6e9, src, unix=True)
    seq = load_sequence(path)
    assert seq.attrs["schema_version"] == "1.0"
    assert seq.attrs["legacy_schema_version"] == "0.1"
    assert seq.attrs["group_id"] == "s1"
    assert seq.attrs["start_time_unix"] == pytest.approx(1.6e9 + t[0], abs=1e-3)
    assert "resample_poly" in seq.attrs["resampling"]
    np.testing.assert_allclose(np.diff(seq.timestamp), 0.005)
    assert abs(seq.attrs["source_sample_rate_hz"] - 100.0) < 0.5
    # 位置有效性掩码被传递到网格上
    assert (~seq.valid_pose).sum() >= 1
    rep = validate(seq)
    # 旧 world_frame 不被静默改写，因此只报告这一项错误
    assert [e for e in rep.errors if "world_frame" not in e] == []
    # 重采样后的 IMU 与解析真值一致
    t_new = seq.timestamp + seq.attrs["start_time_unix"] - 1.6e9
    inner = (t_new > 1) & (t_new < 19)
    np.testing.assert_allclose(seq.accelerometer[inner], motion.accelerometer(t_new[inner]),
                               atol=0.02)
    np.testing.assert_allclose(seq.position[inner], motion.position(t_new[inner]), atol=1e-3)


def test_device_orientation_mask_requires_device_orientation(tmp_path):
    seq = make_sequence(duration=3.0)
    assert seq.valid_device_orientation is None
    np.testing.assert_array_equal(seq.valid_device, seq.valid)  # 无该数据集时等于 valid
    seq.valid_device_orientation = np.ones(len(seq), bool)
    rep = validate(seq)
    assert any("imu/orientation is not" in e for e in rep.errors)
    # 有设备姿态时可选写出；不写出时读回为 None
    ok = make_sequence(duration=3.0, device_yaw_offset=0.2)
    path = save_sequence(tmp_path / "b.h5", ok)
    back = load_sequence(path, validate=True)
    assert back.device_orientation is not None and back.valid_device_orientation is None
    with h5py.File(path) as f:
        assert "valid/device_orientation" not in f


def test_gravity_check_tolerates_an_unrepresentative_static_segment():
    """静止段落在参考姿态尚未收敛的开头（IDOL building2/3 的 SLAM 初始化）时，只记警告。"""
    from inertial_benchmark.utils.geometry import quat_multiply, quat_rotate

    seq = make_sequence(duration=60.0)
    static = slice(0, 400)  # 开头 2 s 静止
    q0 = seq.orientation[0].astype(np.float64)
    seq.orientation[static] = q0
    seq.gyroscope[static] = 0.0
    seq.accelerometer[static] = quat_rotate(quat_conjugate(q0), np.array([0.0, 0.0, 9.81]))
    # 只在静止段把参考姿态整体绕 x 轴倾斜 7°（世界系尚未调平）
    tilt = np.array([np.cos(np.radians(3.5)), np.sin(np.radians(3.5)), 0.0, 0.0])
    seq.orientation[static] = quat_multiply(np.repeat(tilt[None], 400, 0),
                                            seq.orientation[static].astype(np.float64))
    rep = validate(seq)
    assert rep.ok, rep.errors
    g = rep.info["gravity"]
    assert g["used"] == "all_valid" and g["passed"] == ["all_valid"]
    assert g["static_span"]["samples"] >= 200 and g["static_span"]["last_s"] < 3.0
    assert any("locally inconsistent with gravity" in w for w in rep.warnings)
    # 约定错误会让两个统计量同时失败
    bad = make_sequence(duration=60.0)
    bad.orientation = quat_conjugate(bad.orientation)
    assert not validate(bad).ok
