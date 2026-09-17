import numpy as np
import pytest

from inertial_benchmark.utils.geometry import (
    heading_from_quat,
    quat_conjugate,
    quat_from_axis_angle,
    quat_from_euler_zyx,
    quat_from_yaw,
    quat_interp,
    quat_make_continuous,
    quat_multiply,
    quat_rotate,
    quat_to_matrix,
    rotate_z,
    slerp,
    wrap_angle,
    yaw_from_quat,
)


def test_rotate_matches_matrix_and_multiply():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(50, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    v = rng.normal(size=(50, 3))
    np.testing.assert_allclose(quat_rotate(q, v), np.einsum("nij,nj->ni", quat_to_matrix(q), v),
                               atol=1e-12)
    p = rng.normal(size=(50, 4))
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    np.testing.assert_allclose(quat_rotate(quat_multiply(q, p), v),
                               quat_rotate(q, quat_rotate(p, v)), atol=1e-12)
    np.testing.assert_allclose(quat_rotate(quat_conjugate(q), quat_rotate(q, v)), v, atol=1e-12)


def test_yaw_roundtrip_and_rotate_z():
    yaw = np.linspace(-3.0, 3.0, 13)
    q = quat_from_euler_zyx(yaw, 0.3 * np.ones_like(yaw), -0.2 * np.ones_like(yaw))
    np.testing.assert_allclose(yaw_from_quat(q), yaw, atol=1e-12)
    v = np.array([1.0, 0.0, 0.5])
    np.testing.assert_allclose(rotate_z(v, np.pi / 2), [0.0, 1.0, 0.5], atol=1e-12)
    np.testing.assert_allclose(quat_rotate(quat_from_yaw(0.4), v), rotate_z(v, 0.4), atol=1e-12)


def test_sign_continuity():
    q = quat_from_yaw(np.linspace(0, 6.0, 100))
    flipped = q * np.where(np.arange(100) % 3 == 0, -1, 1)[:, None]
    out = quat_make_continuous(flipped)
    assert np.all(np.sum(out[1:] * out[:-1], axis=1) > 0)
    assert out[0, 0] >= 0
    # 符号连续化不改变所表示的旋转
    np.testing.assert_allclose(np.abs(np.sum(out * q, axis=1)), 1.0, atol=1e-12)


def test_slerp_constant_angular_rate():
    axis = np.array([0.3, -0.5, 0.8])
    q0 = quat_from_axis_angle(axis, 0.2)
    q1 = quat_from_axis_angle(axis, 1.4)
    tau = np.linspace(0, 1, 7)
    out = slerp(np.repeat(q0[None], 7, 0), np.repeat(q1[None], 7, 0), tau)
    expected = quat_from_axis_angle(axis, 0.2 + 1.2 * tau)
    np.testing.assert_allclose(out, expected, atol=1e-12)
    # 输入符号相反时仍走最短弧
    out_neg = slerp(np.repeat(q0[None], 7, 0), -np.repeat(q1[None], 7, 0), tau)
    np.testing.assert_allclose(np.abs(np.sum(out_neg * expected, axis=1)), 1.0, atol=1e-12)


def test_quat_interp_over_time():
    t = np.array([0.0, 1.0, 2.0])
    angles = np.array([0.0, 1.0, 1.5])
    q = quat_from_yaw(angles)
    q[1] *= -1  # 源数据符号不连续
    tq = np.array([-1.0, 0.25, 1.5, 3.0])
    out = quat_interp(t, q, tq)
    np.testing.assert_allclose(yaw_from_quat(out), [0.0, 0.25, 1.25, 1.5], atol=1e-12)


def _most_horizontal_axis_heading(q):
    """参考实现：用最水平的机体轴求航向（DESIGN §3 的等价定义）。"""
    r = quat_to_matrix(q)
    up_body = r[..., 2, :]  # R^T e_z：世界 z 轴在机体系中的方向
    cross = np.cross(up_body, np.array([0.0, 0.0, 1.0]))
    cos = up_body[..., 2]
    skew = np.zeros(up_body.shape[:-1] + (3, 3))
    skew[..., 0, 1], skew[..., 0, 2] = -cross[..., 2], cross[..., 1]
    skew[..., 1, 0], skew[..., 1, 2] = cross[..., 2], -cross[..., 0]
    skew[..., 2, 0], skew[..., 2, 1] = -cross[..., 1], cross[..., 0]
    eye = np.broadcast_to(np.eye(3), skew.shape).copy()
    tilt = eye + skew + np.einsum("...ij,...jk->...ik", skew, skew) / (1.0 + cos)[..., None, None]
    horizontal = np.linalg.norm(r[..., :2, :], axis=-2)
    axis = np.argmax(horizontal, axis=-1)
    pick = axis[..., None, None] * np.ones((1, 3, 1), dtype=int)
    world = np.take_along_axis(r, pick, axis=-1)[..., 0]
    level = np.take_along_axis(tilt, pick, axis=-1)[..., 0]
    return wrap_angle(np.arctan2(world[..., 1], world[..., 0])
                      - np.arctan2(level[..., 1], level[..., 0]))


def test_heading_matches_most_horizontal_axis_definition():
    rng = np.random.default_rng(3)
    q = rng.normal(size=(500, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q[q[:, 0] < 0] *= -1  # 只排除完全倒置附近，避免参考实现的 1 + cos → 0
    q = q[quat_to_matrix(q)[:, 2, 2] > -0.99]
    np.testing.assert_allclose(wrap_angle(heading_from_quat(q) - _most_horizontal_axis_heading(q)),
                               0.0, atol=1e-12)


def test_heading_is_yaw_equivariant_and_sign_invariant():
    rng = np.random.default_rng(4)
    q = rng.normal(size=(200, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    base = heading_from_quat(q)
    for alpha in (0.3, -2.5, 3.0):
        rotated = quat_multiply(quat_from_yaw(np.full(len(q), alpha)), q)
        np.testing.assert_allclose(wrap_angle(heading_from_quat(rotated) - base - alpha), 0.0,
                                   atol=1e-12)
    np.testing.assert_allclose(wrap_angle(heading_from_quat(-q) - base), 0.0, atol=1e-12)


def test_heading_has_no_jump_near_vertical_body_x_axis():
    pitch = np.linspace(np.pi / 2 - 0.3, np.pi / 2 + 0.3, 601)
    yaw = np.full_like(pitch, 0.5)
    for roll in (0.0, -0.2, 0.9):
        q = quat_from_euler_zyx(yaw, pitch, np.full_like(pitch, roll))
        heading = heading_from_quat(q)
        # 连续：相邻样本的航向变化远小于 ZYX 偏航在 pitch = 90° 处的 180° 跳变
        assert np.abs(np.diff(heading)).max() < 0.01
        if roll == 0.0:
            np.testing.assert_allclose(heading, 0.5, atol=1e-12)  # roll = 0 时就是偏航本身
    # 复现旧定义的缺陷：ZYX 偏航在 pitch 80° → 100° 之间跳 180°
    q80, q100 = (quat_from_euler_zyx(np.array([0.5]), np.array([a]), np.array([0.0]))
                 for a in (np.deg2rad(80.0), np.deg2rad(100.0)))
    assert abs(wrap_angle(yaw_from_quat(q100) - yaw_from_quat(q80))[0]) == pytest.approx(np.pi)
    assert abs(wrap_angle(heading_from_quat(q100) - heading_from_quat(q80))[0]) < 1e-12


def test_heading_matches_zyx_yaw_in_regular_attitudes():
    yaw = np.linspace(-3.0, 3.0, 13)
    zeros = np.zeros_like(yaw)
    np.testing.assert_allclose(heading_from_quat(quat_from_euler_zyx(yaw, 0.7 + zeros, zeros)),
                               yaw, atol=1e-12)
    q = quat_from_euler_zyx(yaw, 0.25 + zeros, -0.2 + zeros)
    # 常规手持姿态（pitch/roll ≤ 0.25 rad）下与 ZYX 偏航相差约 pitch·roll/2 < 2°
    assert np.abs(wrap_angle(heading_from_quat(q) - yaw)).max() < np.deg2rad(2.0)


def test_heading_is_continuous_along_a_smooth_tumbling_trajectory():
    t = np.linspace(0.0, 10.0, 20001)
    q = quat_from_euler_zyx(0.4 * np.sin(0.7 * t), 1.5 * np.sin(1.3 * t), 1.0 * np.cos(0.9 * t))
    steps = np.abs(wrap_angle(np.diff(heading_from_quat(q))))
    assert steps.max() < 0.02  # 平滑轨迹上没有跳变
