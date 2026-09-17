"""OxIOD 转换器单元测试：iOS 符号约定、CoreMotion 欧拉角、Vicon 外参/时钟估计、野值与损坏文件，
以及共享估计工具（手眼对齐、时间偏移、野值掩码）的测试。"""

from __future__ import annotations

import numpy as np
import pytest

from inertial_benchmark.data.converters import _phone_utils as pu
from inertial_benchmark.data.converters import oxiod

T0 = 1_495_462_820.0  # unix 秒
OFFSET = 0.12  # vicon_time = imu_time + OFFSET + SKEW * (imu_time - T_REF)
SKEW = -400e-6
DURATION = 250.0
T_REF = T0 + DURATION / 2
Q_SB = pu.quat_mul(
    pu.quat_from_axis_angle([0, 0, 1], np.radians(174.0)),
    pu.quat_from_axis_angle([1, 0.5, 0], 0.04),
)
SEED = 4


def ios_rows(motion, t_unix):
    """真值 → OxIOD raw imu 行（iOS 约定：gravity 指向地面，单位 G）。"""

    q = motion["q_wb"]
    gravity = pu.quat_rotate(pu.quat_conj(q), [0.0, 0.0, -1.0])
    user_acc = -pu.quat_rotate(pu.quat_conj(q), motion["acc_world"]) / oxiod.G_IOS
    roll, pitch, yaw = motion["euler"].T
    mag = np.zeros((len(q), 3))
    return np.column_stack([t_unix, roll, pitch, yaw, motion["gyro"], gravity, user_acc, mag])


def vicon_rows(seed, t_imu_rel_start, rate=100.0, glitches=(3000, 3001, 9000)):
    """在 Vicon 时钟上等间隔采样真值：返回 vi.csv 行（ns, frame, xyz, xyzw）。"""

    tau = np.arange(T0 - 3.0, T0 + DURATION + 3.0, 1.0 / rate)
    t_imu = (tau - OFFSET + SKEW * T_REF) / (1.0 + SKEW)
    truth = pu.simulate_motion(seed=seed, times=t_imu - t_imu_rel_start)
    q_vicon = pu.quat_mul(truth["q_wb"], pu.quat_conj(Q_SB))
    for g in glitches:  # 标记识别翻转：单帧绕某轴 180°
        q_vicon[g] = pu.quat_mul(q_vicon[g], pu.quat_from_axis_angle([0, 0, 1], np.pi))
    pos = truth["position"] + np.array([-1.3, 2.4, 0.8])
    xyzw = np.column_stack([q_vicon[:, 1:], q_vicon[:, :1]])
    return (
        np.column_stack([np.round(tau * 1e9), np.arange(len(tau)) + 11896, pos, xyzw]),
        t_imu,
        truth,
    )


def write_csv(path, rows, fmt):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(",".join(f(v) for f, v in zip(fmt, row)) + "\n")


def build_session(root, scene="handheld", session="data1", k=1, swap_rows=True):
    t_rel = np.arange(0.0, DURATION, 0.01)
    motion = pu.simulate_motion(seed=SEED, times=t_rel)
    imu = ios_rows(motion, T0 + t_rel)
    if swap_rows:  # 真实数据中相邻两行顺序颠倒
        imu[[500, 501]] = imu[[501, 500]]
    raw_dir = root / oxiod.ROOT_NAME / scene / session / "raw"
    write_csv(raw_dir / f"imu{k}.csv", imu, [lambda v: f"{v:.2f}"] + [lambda v: f"{v:.9f}"] * 15)
    vi, _, _ = vicon_rows(SEED, T0)
    write_csv(
        raw_dir / f"vi{k}.csv",
        vi,
        [lambda v: f"{int(v)}", lambda v: f"{int(v)}"] + [lambda v: f"{v:.9f}"] * 7,
    )
    return motion


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    root = tmp_path_factory.mktemp("oxiod")
    motion = build_session(root)
    raw = next(oxiod.iter_raw_sequences(root))
    return root, motion, raw


def _angle(a, b):
    return np.degrees(pu.quat_angle(pu.quat_mul(pu.quat_conj(a), b)))


def test_ios_units_signs_and_attitude(converted):
    _, motion, raw = converted
    assert raw.sequence_id == "handheld_data1_seq1"
    assert raw.rejected is None, raw.rejected
    assert raw.check_shapes() == []
    np.testing.assert_allclose(raw.imu_time - T0, motion["t"], atol=1e-6)  # 颠倒的两行已按时间排回
    np.testing.assert_allclose(raw.accelerometer, motion["specific_force"], atol=1e-6)
    np.testing.assert_allclose(raw.gyroscope, motion["gyro"], atol=1e-8)
    assert _angle(raw.device_orientation, motion["q_wb"]).max() < 1e-4
    assert raw.attrs["placement"] == "handheld" and raw.attrs["group_id"] == "handheld_data1"
    assert raw.attrs["start_time_unix"] == pytest.approx(T0)
    assert any("2 out-of-order" in n or "1 out-of-order" in n for n in raw.notes)


def test_vicon_extrinsic_clock_and_glitches(converted):
    _, _, raw = converted
    q_sb = np.array([float(v) for v in raw.attrs["oxiod_extrinsic_wxyz"].split(",")])
    assert _angle(q_sb, Q_SB) < 0.3
    assert raw.attrs["oxiod_clock_offset_s"] == pytest.approx(OFFSET, abs=0.003)
    assert raw.attrs["oxiod_clock_skew_ppm"] == pytest.approx(SKEW * 1e6, abs=30)
    # 映射到 IMU 时钟的位姿时间误差 < 3 ms（100 Hz 下亚采样级）
    _, t_exact, _ = vicon_rows(SEED, T0)
    assert np.abs(raw.pose_time - t_exact).max() < 0.003
    # 参考姿态在 IMU 时钟上应与真值一致
    truth = pu.simulate_motion(seed=SEED, times=raw.pose_time - T0)
    ok = raw.pose_valid
    assert np.median(_angle(raw.orientation[ok], truth["q_wb"][ok])) < 0.3
    offset = np.tile([-1.3, 2.4, 0.8], (ok.sum(), 1))
    np.testing.assert_allclose(raw.position[ok] - truth["position"][ok], offset, atol=4e-3)
    # 翻转帧及其 ±0.1 s 被标为无效
    for g in (3000, 3001, 9000):
        assert not raw.pose_valid[g]
        assert not raw.pose_valid[g - 5] and not raw.pose_valid[g + 5]
    assert raw.pose_valid[:2500].all()
    stats = pu.parse_physics_note(raw.notes)
    assert stats["failures"] == []
    assert stats["gravity_error"] < 0.05


def test_positive_sign_would_fail(converted):
    _, _, raw = converted
    raw_bad = oxiod.RawSequence(
        raw.sequence_id,
        raw.imu_time,
        raw.gyroscope,
        -raw.accelerometer,
        raw.pose_time,
        raw.position,
        raw.orientation,
        pose_valid=raw.pose_valid,
    )
    assert any("gravity" in f for f in pu.physics_check(raw_bad)["failures"])


def test_attitude_convention():
    rng = np.random.default_rng(0)
    roll, pitch, yaw = rng.uniform(-1, 1, size=(3, 20))
    q = oxiod.ios_attitude(roll, pitch, yaw)
    from scipy.spatial.transform import Rotation

    expected = Rotation.from_euler("ZXY", np.column_stack([yaw, pitch, roll])).as_matrix()
    np.testing.assert_allclose(pu.quat_to_matrix(q), expected, atol=1e-12)


def test_official_splits_and_placement_table(tmp_path):
    base = tmp_path / oxiod.ROOT_NAME
    for scene, session, ks in [
        ("handheld", "data1", (1, 2)),
        ("handheld", "data5", (1,)),
        ("pocket", "data2", (1, 6)),
    ]:
        for k in ks:
            d = base / scene / session / "raw"
            d.mkdir(parents=True, exist_ok=True)
            (d / f"imu{k}.csv").write_text("")
            (d / f"vi{k}.csv").write_text("")
    (base / "handheld" / "Train.txt").write_text("data1")
    (base / "handheld" / "Test.txt").write_text("data5")
    (base / "pocket" / "Train.txt").write_text("data2/imu1.csv\n")
    (base / "pocket" / "Test.txt").write_text("data2/imu6.csv")
    splits = oxiod.official_splits(tmp_path)
    assert splits == {
        "train": ["handheld_data1_seq1", "handheld_data1_seq2", "pocket_data2_seq1"],
        "test": ["handheld_data5_seq1", "pocket_data2_seq6"],
    }
    assert oxiod.extra_splits(tmp_path) == {"test_unseen_subject": [], "test_unseen_device": []}
    for scene, session in (("multi users", "user3"), ("multi devices", "iPhone 5")):
        d = base / scene / session / "raw"
        d.mkdir(parents=True)
        (d / "imu2.csv").write_text("")
        (d / "vi2.csv").write_text("")
    assert oxiod.extra_splits(tmp_path) == {
        "test_unseen_subject": ["multi_users_user3_seq2"],
        "test_unseen_device": ["multi_devices_iphone5_seq2"],
    }
    assert oxiod.official_splits(tmp_path) == splits  # 附加子集不改变官方划分
    assert oxiod.placement_of("multi users", "user5", 4) == ("pocket", "readme")
    assert oxiod.placement_of("multi users", "user3", 6) == ("bag", "readme")
    assert oxiod.placement_of("multi users", "user2", 7) == ("unknown", "gravity_direction")
    assert oxiod.placement_of("multi devices", "nexus 5", 8) == ("handheld", "gravity_direction")
    assert oxiod.placement_of("handbag", "data1", 1) == ("bag", "scene_folder")


def test_corrupted_timestamps_are_rejected(tmp_path):
    build_session(tmp_path, scene="handbag", session="data2", k=3, swap_rows=False)
    vi = tmp_path / oxiod.ROOT_NAME / "handbag" / "data2" / "raw" / "vi3.csv"
    lines = vi.read_text().splitlines()
    vi.write_text("\n".join("1.49684E+18," + line.split(",", 1)[1] for line in lines) + "\n")
    raw = next(oxiod.iter_raw_sequences(tmp_path))
    assert raw.sequence_id == "handbag_data2_seq3"
    assert "scientific notation" in raw.rejected
    assert raw.check_shapes() == []


def test_nexus_long_format(tmp_path):
    d = tmp_path / oxiod.ROOT_NAME / "multi devices" / "nexus 5" / "raw"
    t_ms = np.arange(0, 30000, 10) + 1_514_482_705_998
    lines = []
    for i, t in enumerate(t_ms):
        lines.append(f"{t},1,0.1,{3.5 + 0.001 * i},9.1")
        lines.append(f"{t + 3},4,{0.001 * i},0.02,-0.03")
        if i % 3 == 0:
            lines.append(f"{t + 5},2,6.8,-7.4,-6.6")
    d.mkdir(parents=True)
    (d / "imu1.csv").write_text("\n".join(lines) + "\n")
    t, gyro, acc, _ = oxiod.read_android_imu(d / "imu1.csv")
    np.testing.assert_allclose(t, t_ms[1:] / 1e3)  # 首个加速度样本早于首个陀螺样本，被裁掉
    np.testing.assert_allclose(acc[:, 1], 3.5 + 0.001 * np.arange(1, len(t_ms)))
    np.testing.assert_allclose(gyro[:, 0], 0.001 * (np.arange(1, len(t_ms)) - 0.3), atol=1e-7)
    entry = oxiod._entries(tmp_path)["multi_devices_nexus5_seq1"]
    assert entry["android"] and entry["device"] == "nexus5" and entry["vicon"] is None
    raw = next(oxiod.iter_raw_sequences(tmp_path))
    assert raw.rejected.startswith("no Vicon file")


# ---------------------------------------------------------------------------
# 共享估计工具
# ---------------------------------------------------------------------------


def test_hand_eye_and_time_offset_recovery():
    t = np.arange(0.0, 120.0, 0.01)
    truth = pu.simulate_motion(seed=7, times=t)
    q_sb = pu.quat_from_axis_angle([0.2, -0.4, 1.0], 2.0)
    q_ws = pu.quat_mul(truth["q_wb"], pu.quat_conj(q_sb))
    est, resid, count = pu.estimate_body_extrinsic(t, truth["gyro"], t, q_ws)
    assert _angle(est, q_sb) < 0.2 and resid < 0.02 and count > 1000
    # ω_imu(t) ≈ ω_pose(t + 0.37)
    t_pose = t + 0.37
    off, corr = pu.estimate_time_offset(
        t,
        np.linalg.norm(truth["gyro"], axis=1),
        t_pose,
        np.linalg.norm(truth["gyro"], axis=1),
        max_lag=2.0,
    )
    assert off == pytest.approx(0.37, abs=0.005) and corr > 0.95
    fine, corr = pu.refine_time_offset(
        t, truth["gyro"], t_pose, q_ws, q_sb, center=0.3, max_lag=0.2
    )
    assert fine == pytest.approx(0.37, abs=0.002) and corr > 0.9


def test_glitch_mask_and_bias_estimates():
    t = np.arange(0.0, 20.0, 0.01)
    truth = pu.simulate_motion(seed=8, times=t)
    q = truth["q_wb"].copy()
    q[700] = pu.quat_mul(q[700], pu.quat_from_axis_angle([1, 0, 0], np.pi))
    pos = truth["position"].copy()
    pos[1500] += 0.5
    valid = pu.pose_glitch_mask(t, q, pos, pad=0.1)
    assert not valid[690:711].any() and valid[:689].all()
    assert not valid[1490:1511].any() and valid[1512:].all()
    bias = np.array([0.01, -0.02, 0.005])
    est = pu.estimate_gyro_bias(t, truth["gyro"] + bias, t, truth["q_wb"])
    np.testing.assert_allclose(est, bias, atol=2e-3)
    seg = pu.estimate_gyro_bias_segments(t, truth["gyro"] + bias, t, truth["q_wb"], segment=10.0)
    np.testing.assert_allclose(seg, np.tile(bias, (len(t), 1)), atol=3e-3)
