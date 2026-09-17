"""RIDI 转换器单元测试，以及共享工具（四元数、CSV、物理自检）的基础测试。"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from inertial_benchmark.data.converters import _phone_utils as pu
from inertial_benchmark.data.converters import ridi
from inertial_benchmark.data.converters.base import RawSequence

COLUMNS = (
    "time,gyro_x,gyro_y,gyro_z,acce_x,acce_y,acce_z,linacce_x,linacce_y,linacce_z,grav_x,grav_y,grav_z,"
    "magnet_x,magnet_y,magnet_z,pos_x,pos_y,pos_z,ori_w,ori_x,ori_y,ori_z,rv_w,rv_x,rv_y,rv_z"
).split(",")


def write_ridi_csv(path, motion, ori, rv, time_ns, acce=None, linacce=None, grav=None):
    """按官方 gen_dataset.py（pandas.to_csv，首列为空名索引）的格式写 data.csv。"""

    n = len(time_ns)
    acce = motion["specific_force"] if acce is None else acce
    grav = (
        pu.quat_rotate(pu.quat_conj(motion["q_wb"]), [0.0, 0.0, pu.GRAVITY])
        if grav is None
        else grav
    )
    linacce = motion["specific_force"] - grav if linacce is None else linacce
    body = np.column_stack(
        [
            time_ns,
            motion["gyro"],
            acce,
            linacce,
            grav,
            np.zeros((n, 3)),
            motion["position"],
            ori,
            rv,
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("," + ",".join(COLUMNS) + "\n")
        for i, row in enumerate(body):
            handle.write(f"{i}," + ",".join(repr(float(v)) for v in row) + "\n")


@pytest.fixture(scope="module")
def motion():
    return pu.simulate_motion(duration=40.0, rate=200.0, seed=2)


# ---------------------------------------------------------------------------
# 共享工具
# ---------------------------------------------------------------------------


def test_quaternion_helpers_match_scipy():
    rng = np.random.default_rng(0)
    q = pu.quat_normalize(rng.normal(size=(50, 4)))
    v = rng.normal(size=(50, 3))
    rot = Rotation.from_quat(q[:, [1, 2, 3, 0]])
    np.testing.assert_allclose(pu.quat_rotate(q, v), rot.apply(v), atol=1e-12)
    np.testing.assert_allclose(pu.quat_to_matrix(q), rot.as_matrix(), atol=1e-12)
    prod = pu.quat_mul(q[:-1], q[1:])
    expected = (rot[:-1] * rot[1:]).as_matrix()
    np.testing.assert_allclose(pu.quat_to_matrix(prod), expected, atol=1e-12)
    r = rng.normal(size=(20, 3))
    np.testing.assert_allclose(pu.quat_to_rotvec(pu.quat_from_rotvec(r * 0.5)), r * 0.5, atol=1e-12)
    np.testing.assert_allclose(
        pu.quat_rotate(pu.Q_ZUP_FROM_YUP, [1.0, 2.0, 3.0]), [1.0, -3.0, 2.0], atol=1e-12
    )
    cont = pu.quat_make_continuous(
        np.repeat(q[:1], 6, axis=0) * np.array([1, -1, 1, -1, 1, -1])[:, None]
    )
    assert np.all(np.sum(cont[1:] * cont[:-1], axis=1) > 0)
    np.testing.assert_allclose(pu.xyzw_to_wxyz([[1, 2, 3, 4]]), [[4, 1, 2, 3]])


def test_prefix_and_window_products_are_ordered():
    rng = np.random.default_rng(1)
    steps = pu.quat_from_rotvec(rng.normal(scale=0.3, size=(1001, 3)))
    ref = [pu.Q_IDENTITY]
    for s in steps:
        ref.append(pu.quat_mul(ref[-1], s))
    ref = np.array(ref)
    prefix = pu.prefix_products(steps)
    assert np.degrees(pu.quat_angle(pu.quat_mul(pu.quat_conj(prefix), ref[1:]))).max() < 1e-9
    i0, i1 = np.array([0, 10, 500]), np.array([7, 1001, 501])
    win = pu.window_products(steps, i0, i1)
    exp = pu.quat_mul(pu.quat_conj(ref[i0]), ref[i1])
    assert np.degrees(pu.quat_angle(pu.quat_mul(pu.quat_conj(win), exp))).max() < 1e-9


def test_monotonic_mask_and_csv_reader(tmp_path):
    np.testing.assert_array_equal(
        pu.monotonic_mask([0, 1, 1, 3, 2, 4]), [True, True, False, True, False, True]
    )
    path = tmp_path / "x.csv"
    path.write_text(",a,b\n0,1.5,2\n1,3,4e3\n")
    names, data = pu.read_header_csv(path)
    assert names == ["", "a", "b"]
    np.testing.assert_allclose(pu.columns(names, data, ["b", "a"]), [[2, 1.5], [4000, 3]])
    with pytest.raises(KeyError):
        pu.columns(names, data, ["c"])


def test_physics_check_accepts_truth_and_rejects_wrong_conventions(motion):
    t = motion["t"]
    good = RawSequence(
        "s", t, motion["gyro"], motion["specific_force"], t, motion["position"], motion["q_wb"]
    )
    stats = pu.physics_check(good)
    assert stats["failures"] == []
    assert stats["gravity_error"] < 0.05 and stats["gyro_window_err_deg_median"] < 0.5
    assert abs(stats["time_offset_s"]) < 0.005 and stats["gyro_ref_extrinsic_deg"] < 0.5
    for bad_q, bad_acc in [
        (pu.quat_conj(motion["q_wb"]), motion["specific_force"]),  # 四元数方向反了
        (motion["q_wb"], -motion["specific_force"]),  # 加速度符号反了
        (pu.quat_mul(pu.Q_ZUP_FROM_YUP, motion["q_wb"]), motion["specific_force"]),  # 世界系 y 向上
    ]:
        bad = RawSequence("s", t, motion["gyro"], bad_acc, t, motion["position"], bad_q)
        assert pu.physics_check(bad)["failures"]
    slow = RawSequence(
        "s",
        t,
        motion["gyro"],
        motion["specific_force"],
        t,
        motion["position"] * 1e-3,
        motion["q_wb"],
    )
    assert any("speed" in f or "stationary" in f for f in pu.physics_check(slow)["failures"])
    # 姿态世界系与位置世界系差 90° 偏航：重力与陀螺检查都无法发现，只有航向一致性检查能拦截
    yawed = pu.quat_mul(pu.quat_from_axis_angle([0, 0, 1], np.pi / 2), motion["q_wb"])
    stats = pu.physics_check(
        RawSequence("s", t, motion["gyro"], motion["specific_force"], t, motion["position"], yawed)
    )
    assert [f for f in stats["failures"] if f.startswith("heading")] == stats["failures"] != []
    assert stats["heading_offset_deg"] == pytest.approx(-90.0, abs=2.0)
    assert stats["heading_corr"] > 0.9


# ---------------------------------------------------------------------------
# RIDI
# ---------------------------------------------------------------------------


def _write_dataset(root, motion):
    time_ns = (motion["t"] + 87155.0) * 1e9
    rv = pu.quat_mul(pu.quat_from_axis_angle([0, 0, 1], -1.1), motion["q_wb"])
    for name in ("dan_bag1", "hao_leg2", "huayi_lopata1"):
        write_ridi_csv(root / name / "processed" / "data.csv", motion, motion["q_wb"], rv, time_ns)
    (root / "list_train_publish_v2.txt").write_text("dan_bag1,bag\nhuayi_lopata1,body\n")
    (root / "list_test_publish_v2.txt").write_text("hang_leg_new3,leg\n")
    return rv


def test_ridi_parse(tmp_path, motion):
    root = tmp_path / "data_publish_v2"
    rv = _write_dataset(root, motion)
    raws = {r.sequence_id: r for r in ridi.iter_raw_sequences(tmp_path)}  # 父目录也可
    assert sorted(raws) == ["dan_bag1", "hao_leg2", "huayi_lopata1"]
    raw = raws["dan_bag1"]
    assert raw.rejected is None, raw.rejected
    assert raw.check_shapes() == []
    np.testing.assert_allclose(raw.imu_time, motion["t"] + 87155.0, atol=1e-6)
    np.testing.assert_allclose(raw.gyroscope, motion["gyro"], atol=1e-9)
    np.testing.assert_allclose(raw.accelerometer, motion["specific_force"], atol=1e-9)
    ang = np.degrees(pu.quat_angle(pu.quat_mul(pu.quat_conj(raw.orientation), motion["q_wb"])))
    assert ang.max() < 1e-6
    ang = np.degrees(pu.quat_angle(pu.quat_mul(pu.quat_conj(raw.device_orientation), rv)))
    assert ang.max() < 1e-6
    assert raw.attrs["placement"] == "bag" and raw.attrs["subject_id"] == "dan"
    assert raws["hao_leg2"].attrs["placement"] == "pocket"  # 未列出：从名字解析，leg → pocket
    assert raws["huayi_lopata1"].attrs["placement"] == "body"  # 来自官方列表第二列
    assert pu.parse_physics_note(raw.notes)["failures"] == []


def test_ridi_official_splits_and_duplicates(tmp_path, motion):
    root = tmp_path / "data_publish_v2"
    _write_dataset(root, motion)
    splits = ridi.official_splits(root)
    assert splits == {"train": ["dan_bag1", "huayi_lopata1"], "test": []}
    # 不在官方列表中的序列也必须能被枚举（统一流水线据此决定其归属）
    assert ridi.list_sequences(root) == ["dan_bag1", "hao_leg2", "huayi_lopata1"]
    # 重复/倒序时间戳被剔除并记录
    time_ns = (motion["t"] + 1.0) * 1e9
    time_ns[5] = time_ns[4]
    write_ridi_csv(
        root / "tang_body9" / "processed" / "data.csv",
        motion,
        motion["q_wb"],
        motion["q_wb"],
        time_ns,
    )
    raw = next(ridi.iter_raw_sequences(root, only=["tang_body9"]))
    assert len(raw.imu_time) == len(motion["t"]) - 1
    assert any("dropped 1 duplicate" in note for note in raw.notes)
    missing = next(ridi.iter_raw_sequences(root, only=["nobody1"]))
    assert missing.rejected and missing.check_shapes() == []
