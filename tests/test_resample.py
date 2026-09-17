import numpy as np
import pytest

from inertial_benchmark.data.resample import (
    clean_timestamps,
    poly_ratio,
    resample_signal,
    resample_streams,
    uniform_grid,
    valid_on_grid,
)
from inertial_benchmark.utils.geometry import quat_from_axis_angle, quat_from_yaw, yaw_from_quat


def _fit_sine(t, x, freq):
    """最小二乘拟合 ``a·sin + b·cos``，返回 (幅度, 相位)。"""
    w = 2 * np.pi * freq
    design = np.column_stack([np.sin(w * t), np.cos(w * t)])
    (a, b), *_ = np.linalg.lstsq(design, x, rcond=None)
    return np.hypot(a, b), np.arctan2(b, a)


@pytest.mark.parametrize("src_rate", [100.0, 250.0])
def test_sine_amplitude_and_phase(src_rate):
    freq, phase = 5.0, 0.3
    t_src = np.arange(0, 10, 1 / src_rate)
    x_src = np.sin(2 * np.pi * freq * t_src + phase)
    grid = uniform_grid(t_src[0], t_src[-1], 200.0)
    x, valid, info = resample_signal(t_src, x_src[:, None], grid, rate=200.0)
    assert info["method"] == "resample_poly"
    assert valid.all()
    inner = (grid > 0.5) & (grid < grid[-1] - 0.5)
    amp, ph = _fit_sine(grid[inner], x[inner, 0], freq)
    assert abs(amp - 1.0) < 2e-3
    assert abs(ph - phase) < np.radians(0.2)
    err = x[inner, 0] - np.sin(2 * np.pi * freq * grid[inner] + phase)
    assert np.max(np.abs(err)) < 5e-3
    # 边缘（padtype=line）也不应出现明显振铃
    assert np.max(np.abs(x[:, 0] - np.sin(2 * np.pi * freq * grid + phase))) < 0.05


def test_downsampling_suppresses_alias():
    src_rate = 250.0
    t_src = np.arange(0, 20, 1 / src_rate)
    # 120 Hz 高于 200 Hz 的奈奎斯特频率，朴素抽取会混叠到 80 Hz
    x_src = np.sin(2 * np.pi * 120.0 * t_src)
    grid = uniform_grid(t_src[0], t_src[-1], 200.0)
    x, _, _ = resample_signal(t_src, x_src[:, None], grid, rate=200.0)
    inner = (grid > 1) & (grid < grid[-1] - 1)
    amp80, _ = _fit_sine(grid[inner], x[inner, 0], 80.0)
    naive = np.interp(grid, t_src, x_src)
    amp80_naive, _ = _fit_sine(grid[inner], naive[inner], 80.0)
    assert amp80_naive > 0.3
    assert amp80 < 0.01


def test_near_target_rate_uses_linear_interpolation():
    t_src = np.arange(0, 5, 1 / 198.0)
    x_src = np.sin(2 * np.pi * 2.0 * t_src)
    grid = uniform_grid(0.0, t_src[-1], 200.0)
    x, valid, info = resample_signal(t_src, x_src[:, None], grid, rate=200.0)
    assert info["method"] == "linear"
    np.testing.assert_allclose(x[:, 0], np.sin(2 * np.pi * 2.0 * grid), atol=2e-3)


def test_poly_ratio():
    assert poly_ratio(100.0, 200.0)[:2] == (2, 1)
    assert poly_ratio(250.0, 200.0)[:2] == (4, 5)
    up, down, nominal = poly_ratio(99.7, 200.0)
    assert (up, down, nominal) == (2, 1, 100.0)


def test_clean_timestamps_drops_duplicates_and_reversals():
    t = np.array([0.0, 0.1, 0.1, 0.2, 0.15, 0.3, np.nan, 0.4])
    keep = clean_timestamps(t)
    assert keep.tolist() == [True, True, False, True, False, True, False, True]


def test_gap_mask():
    t_src = np.concatenate([np.arange(0, 1, 0.01), np.arange(1.2, 2, 0.01)])
    grid = uniform_grid(0, t_src[-1], 200.0)
    valid = valid_on_grid(t_src, grid, gap_threshold=0.05)
    in_gap = (grid > 0.99 + 1e-9) & (grid < 1.2 - 1e-9)
    assert not valid[in_gap].any()
    assert valid[~in_gap].all()
    # 源样本无效时，两侧相邻的网格点都无效
    mask = np.ones(len(t_src), bool)
    mask[10] = False  # t = 0.10
    valid2 = valid_on_grid(t_src, grid, mask, 0.05)
    bad = grid[~valid2 & ~in_gap]
    np.testing.assert_allclose(bad, [0.095, 0.1, 0.105], atol=1e-9)


def test_gap_mask_is_dilated_for_polyphase():
    t_src = np.concatenate([np.arange(0, 3, 0.01), np.arange(3.5, 6, 0.01)])
    x_src = np.sin(t_src)[:, None]
    grid = uniform_grid(0, t_src[-1], 200.0)
    _, valid, info = resample_signal(t_src, x_src, grid, rate=200.0)
    support = info["support_s"]
    assert support > 0
    assert not valid[(grid > 3.0 - support + 0.02) & (grid < 3.5 + support - 0.02)].any()
    assert valid[(grid < 2.5) | (grid > 4.0)].all()


def test_resample_streams_quaternion_slerp_and_masks():
    t_imu = np.arange(0, 4, 0.004)  # 250 Hz
    t_pose = np.arange(0.5, 4, 0.01)  # 100 Hz，晚于 IMU 开始
    yaw_rate = 0.8
    q = quat_from_yaw(yaw_rate * t_pose)
    q[::2] *= -1  # 符号交替，必须被连续化
    pos = np.column_stack([t_pose, 2 * t_pose, np.zeros_like(t_pose)])
    pose_valid = np.ones(len(t_pose), bool)
    pose_valid[100] = False  # t = 1.5
    res = resample_streams(
        t_imu, np.zeros((len(t_imu), 3)), np.tile([0, 0, 9.81], (len(t_imu), 1)),
        t_pose, pos, q, pose_valid=pose_valid, rate=200.0,
    )
    assert res.start_time == pytest.approx(0.5)
    assert res.timestamp[0] == 0.0
    np.testing.assert_allclose(np.diff(res.timestamp), 0.005)
    t_abs = res.timestamp + res.start_time
    np.testing.assert_allclose(res.arrays["position"][:, 0], t_abs, atol=1e-9)
    ori = res.arrays["orientation"]
    np.testing.assert_allclose(np.linalg.norm(ori, axis=1), 1.0, atol=1e-12)
    assert np.all(np.sum(ori[1:] * ori[:-1], axis=1) > 0)
    np.testing.assert_allclose(np.unwrap(yaw_from_quat(ori)), yaw_rate * t_abs, atol=1e-9)
    bad = t_abs[~res.valid_pose]
    np.testing.assert_allclose(bad, [1.495, 1.5, 1.505], atol=1e-9)
    assert res.valid_imu.all()
    assert res.info["imu"]["up"] == 4 and res.info["imu"]["down"] == 5
    assert "resample_poly" in res.description


def test_resample_streams_rejects_non_overlap():
    t1 = np.arange(0, 1, 0.005)
    t2 = np.arange(5, 6, 0.005)
    q = quat_from_axis_angle(np.array([0, 0, 1.0]), np.zeros(len(t2)))
    with pytest.raises(ValueError, match="overlap"):
        resample_streams(t1, np.zeros((len(t1), 3)), np.zeros((len(t1), 3)), t2,
                         np.zeros((len(t2), 3)), q)


def test_unix_clock_grid_keeps_last_sample():
    # float64 Unix 秒的分辨率约 2.4e-7 s：t_end - t_start 可能比 (n-1)/rate 略小
    t_src = 1761574803.390954 + np.arange(60000) * 0.005
    t_src[-1] = 1761575103.385954  # PedLocData Demo 的真实首尾时间：差值为 299.99499988...
    grid = uniform_grid(t_src[0], t_src[-1], 200.0)
    assert len(grid) == 60000
    assert valid_on_grid(t_src, grid, gap_threshold=0.05).all()


def test_valid_extent():
    from inertial_benchmark.data.resample import valid_extent

    rate = 200.0
    v = np.ones(4000, bool)
    assert valid_extent(v, rate) == (0, 4000)
    v[:10] = False
    v[-5:] = False
    assert valid_extent(v, rate) == (10, 3995)
    # 开头 0.1 s 的孤立有效片段（后接缺口）被裁掉；中间的短片段保留
    w = np.ones(4000, bool)
    w[20:600] = False
    w[2000:2100] = False
    w[2150:2300] = False
    assert valid_extent(w, rate) == (600, 4000)
    # 末尾 0.5 s 孤立片段
    w2 = np.ones(4000, bool)
    w2[3800:3900] = False
    assert valid_extent(w2, rate) == (0, 3800)
    # 没有足够长的片段：只裁首尾无效样本
    u = np.zeros(1000, bool)
    u[100:150] = True
    u[400:420] = True
    assert valid_extent(u, rate) == (100, 420)
    assert valid_extent(np.zeros(10, bool), rate) == (0, 10)
