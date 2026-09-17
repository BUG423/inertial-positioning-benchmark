"""IMUNet 转换器单元测试：Tango / ARCore 两类设备的四元数约定、重复序列与加速度计比例检查。"""

from __future__ import annotations

import numpy as np
import pytest

from inertial_benchmark.data.converters import _phone_utils as pu
from inertial_benchmark.data.converters import imunet

COLUMNS = (
    "time,gyro_x,gyro_y,gyro_z,acce_x,acce_y,acce_z,linacce_x,linacce_y,linacce_z,grav_x,grav_y,grav_z,"
    "magnet_x,magnet_y,magnet_z,pos_x,pos_y,pos_z,ori_w,ori_x,ori_y,ori_z,rv_w,rv_x,rv_y,rv_z"
).split(",")


def write_csv(path, motion, ori, position, accel_scale=1.0):
    n = len(motion["t"])
    grav = pu.quat_rotate(pu.quat_conj(motion["q_wb"]), [0.0, 0.0, pu.GRAVITY])
    linacce = motion["specific_force"] - grav
    rv = pu.quat_mul(pu.quat_from_axis_angle([0, 0, 1], 2.0), motion["q_wb"])
    body = np.column_stack(
        [
            (motion["t"] + 1104.0) * 1e9,
            motion["gyro"],
            motion["specific_force"] * accel_scale,
            linacce,
            grav,
            np.zeros((n, 3)),
            position,
            ori,
            rv,
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("," + ",".join(COLUMNS) + "\n")
        for i, row in enumerate(body):
            handle.write(f"{i}," + ",".join(repr(float(v)) for v in row) + "\n")


def arcore_pose(motion):
    """真值 → ARCore camera.getPose()（y 向上世界系、相机系）与官方预处理后的 z 向上位置。"""

    q_yup_from_zup = pu.quat_conj(pu.Q_ZUP_FROM_YUP)
    ori = pu.quat_mul(
        pu.quat_mul(q_yup_from_zup, motion["q_wb"]), pu.quat_conj(imunet.Q_CAMERA_FROM_BODY)
    )
    return ori, motion["position"]  # 官方脚本已把位置转成 z 向上


@pytest.fixture(scope="module")
def motion():
    return pu.simulate_motion(duration=40.0, rate=200.0, seed=3)


def _angle(a, b):
    return np.degrees(pu.quat_angle(pu.quat_mul(pu.quat_conj(a), b)))


def test_name_parsing():
    assert imunet.parse_name("Outdoor_Subjetc_1_S10_9") == {
        "env": "Outdoor",
        "subject": "1",
        "device": "S10",
        "index": "9",
    }
    assert imunet.parse_name("Indoor_Subject_5_Tango_2")["device"] == "Tango"
    with pytest.raises(ValueError):
        imunet.parse_name("Indoor_Subject_5_Pixel_2")
    with pytest.raises(ValueError):
        imunet.parse_name("behnam_1__s10")


def test_arcore_and_tango_conventions(tmp_path, motion):
    root = tmp_path / "IMUNet_dataset"
    ori, pos = arcore_pose(motion)
    write_csv(root / "Indoor_Subject_2_S21_1" / "processed" / "data.csv", motion, ori, pos)
    write_csv(
        root / "Indoor_Subject_2_Tango_1" / "processed" / "data.csv",
        motion,
        motion["q_wb"],
        motion["position"],
    )
    (root / "list_train.txt").write_text("Indoor_Subject_2_S21_1\n")
    (root / "list_test.txt").write_text("Indoor_Subject_2_Tango_1\nIndoor_Subject_9_S10_1\n")
    raws = {r.sequence_id: r for r in imunet.iter_raw_sequences(tmp_path)}
    for name, raw in raws.items():
        assert raw.rejected is None, (name, raw.rejected)
        assert raw.check_shapes() == []
        assert _angle(raw.orientation, motion["q_wb"]).max() < 1e-6, name
        np.testing.assert_allclose(raw.accelerometer, motion["specific_force"], atol=1e-9)
        np.testing.assert_allclose(raw.imu_time, motion["t"] + 1104.0, atol=1e-6)
        assert pu.parse_physics_note(raw.notes)["failures"] == []
    assert raws["Indoor_Subject_2_S21_1"].attrs["device_id"] == "samsung_galaxy_s21"
    assert raws["Indoor_Subject_2_S21_1"].attrs["position_source"] == "arcore_vio_same_device"
    assert raws["Indoor_Subject_2_Tango_1"].attrs["subject_id"] == "subject2"
    assert imunet.official_splits(root) == {
        "train": ["Indoor_Subject_2_S21_1"],
        "test": ["Indoor_Subject_2_Tango_1"],
    }


def test_conjugate_arcore_convention_fails_physics_check(tmp_path, motion):
    """负对照：把 ARCore 四元数当作共轭（旧流水线的错误），物理自检必须失败。"""

    ori, pos = arcore_pose(motion)
    write_csv(tmp_path / "Indoor_Subject_1_S10_1" / "processed" / "data.csv", motion, ori, pos)
    raw = imunet.load_sequence(tmp_path / "Indoor_Subject_1_S10_1")
    raw.orientation = imunet.reference_orientation("S10", pu.quat_conj(ori))
    assert pu.physics_check(raw)["failures"]
    # 只做 y-up→z-up、不补相机外参也必须失败
    raw.orientation = pu.quat_mul(pu.Q_ZUP_FROM_YUP, ori)
    assert pu.physics_check(raw)["failures"]


def test_accelerometer_scale_error_is_rejected(tmp_path, motion):
    ori, pos = arcore_pose(motion)
    write_csv(
        tmp_path / "Outdoor_Subject_1_Xiaomi_1" / "processed" / "data.csv",
        motion,
        ori,
        pos,
        accel_scale=0.91,
    )
    raw = next(imunet.iter_raw_sequences(tmp_path))
    assert raw.rejected and "accelerometer scale" in raw.rejected
    assert any("scale = 0.91" in note for note in raw.notes)


def test_known_duplicate_is_rejected_only_when_identical(tmp_path, motion):
    ori, pos = arcore_pose(motion)
    keep = tmp_path / "Outdoor_Subjetc_1_S10_16" / "processed" / "data.csv"
    dup = tmp_path / "Outdoor_Subjetc_1_S10_13" / "processed" / "data.csv"
    write_csv(keep, motion, ori, pos)
    dup.parent.mkdir(parents=True)
    dup.write_bytes(keep.read_bytes())
    raws = {r.sequence_id: r for r in imunet.iter_raw_sequences(tmp_path)}
    assert raws["Outdoor_Subjetc_1_S10_16"].rejected is None
    assert "duplicate of Outdoor_Subjetc_1_S10_16" in raws["Outdoor_Subjetc_1_S10_13"].rejected
    # 内容不同则不视为重复
    write_csv(dup, motion, ori, pos * 1.0001)
    raw = next(imunet.iter_raw_sequences(tmp_path, only=["Outdoor_Subjetc_1_S10_13"]))
    assert raw.rejected is None
