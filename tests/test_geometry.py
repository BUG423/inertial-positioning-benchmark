import numpy as np

from inertial_benchmark.utils.geometry import (
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
