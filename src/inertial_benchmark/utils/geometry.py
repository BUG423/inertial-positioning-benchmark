"""四元数与旋转工具（纯 numpy）。

约定：四元数元素顺序 ``wxyz``，Hamilton 乘法；``q`` 表示 body→world 旋转，
即 ``v_world = R(q) · v_body``。所有函数支持前导维度广播。
"""

from __future__ import annotations

import numpy as np

GRAVITY = 9.81  # m/s²，DESIGN 2.2 的重力检查与去重力均使用该常数


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """归一化四元数；零范数输入原样返回（避免除零）。"""
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.where(norm > 0, norm, 1.0)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """共轭（单位四元数的逆）。"""
    q = np.asarray(q)
    return np.concatenate([q[..., :1], -q[..., 1:]], axis=-1)


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton 乘积 ``a ⊗ b``（先 b 后 a 的旋转复合）。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """用单位四元数旋转向量：``R(q) v``。"""
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    w = q[..., :1]
    u = q[..., 1:]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """单位四元数 → 旋转矩阵 ``(..., 3, 3)``。"""
    q = quat_normalize(q)
    w, x, y, z = np.moveaxis(q, -1, 0)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    m = np.stack(
        [
            1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
            2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
            2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy),
        ],
        axis=-1,
    )
    return m.reshape(q.shape[:-1] + (3, 3))


def quat_from_axis_angle(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """轴角 → 四元数；``axis`` 会被归一化。"""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
    half = 0.5 * np.asarray(angle, dtype=np.float64)[..., None]
    return np.concatenate([np.cos(half), np.sin(half) * axis], axis=-1)


def quat_from_yaw(yaw: np.ndarray) -> np.ndarray:
    """绕世界 z 轴旋转 ``yaw`` 弧度的四元数。"""
    half = 0.5 * np.asarray(yaw, dtype=np.float64)
    zeros = np.zeros_like(half)
    return np.stack([np.cos(half), zeros, zeros, np.sin(half)], axis=-1)


def quat_from_euler_zyx(yaw: np.ndarray, pitch: np.ndarray, roll: np.ndarray) -> np.ndarray:
    """ZYX 欧拉角 → 四元数：``R = Rz(yaw) · Ry(pitch) · Rx(roll)``。"""
    qz = quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), yaw)
    qy = quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), pitch)
    qx = quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), roll)
    return quat_multiply(quat_multiply(qz, qy), qx)


def yaw_from_quat(q: np.ndarray) -> np.ndarray:
    """ZYX 分解下的偏航角（弧度，范围 (-π, π]）。

    等价于**机体 x 轴**水平投影的方位角，因此机体 x 轴接近竖直（|pitch| → 90°）时不连续：
    pitch 由 80° 变到 100° 会让该值跳变 180°。任务视图与航向计算请改用
    :func:`heading_from_quat`（本函数只用于与 ZYX 欧拉角互相转换的场合）。
    """
    q = np.asarray(q, dtype=np.float64)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


# 航向定义为 “绕世界 z 轴的扭转分量”：q = q_z(ψ) ⊗ q_tilt，其中 q_tilt 的旋转轴水平
# （倾斜分量取 body-z 与世界 z 之间的最小旋转）。该分解下 ψ = 2·atan2(q_z, q_w)，
# 与“取最水平的机体轴求方位角、再减去该轴在纯倾斜帧中的方位角”逐位等价（见 test_geometry.py）。
HEADING_SINGULAR_TOL = 1e-8  # w² + z² = cos²(倾斜角/2)；小于该值视为姿态完全倒置


def heading_from_quat(q: np.ndarray) -> np.ndarray:
    """连续的航向角（弧度，范围 (-π, π]）：绕世界 z 轴的扭转分量。

    性质（均有测试锁定）：

    * **偏航等变**：``heading(q_z(α) ⊗ q) = heading(q) + α``（模 2π）；
    * **连续**：只在机体 z 轴竖直向下（姿态完全倒置，倾斜角 = 180°）这一处奇异，
      其余任意姿态下沿平滑轨迹连续，尤其在 |pitch| → 90° 附近无跳变；
    * **与 ZYX 偏航一致**：roll = 0 时（任意 pitch）逐位相同；一般姿态下差值约为
      ``pitch·roll/2``（常规手持姿态下 < 3°）；
    * 对 ``q`` 与 ``-q`` 给出同一角度（取值范围内）。

    倾斜角接近 180° 时 ``w² + z² → 0``，航向无定义（数学上无法避免：等变的航向定义是
    S² 上非平凡 S¹ 主丛的截面，必有奇点）。此处退化为 0，并由调用方在文档中说明。
    """
    q = np.asarray(q, dtype=np.float64)
    w, z = q[..., 0], q[..., 3]
    ok = (w * w + z * z) > HEADING_SINGULAR_TOL
    return wrap_angle(np.where(ok, 2.0 * np.arctan2(z, w), 0.0))


def quat_make_continuous(q: np.ndarray) -> np.ndarray:
    """符号连续化：保证相邻四元数点积非负，并令首个样本 ``w >= 0``。"""
    q = np.array(q, dtype=np.float64, copy=True)
    if q.shape[0] == 0:
        return q
    if q[0, 0] < 0:
        q[0] = -q[0]
    if q.shape[0] > 1:
        flips = np.einsum("ij,ij->i", q[1:], q[:-1]) < 0
        sign = np.where(np.cumsum(flips) % 2 == 1, -1.0, 1.0)
        q[1:] *= sign[:, None]
    return q


def slerp(q0: np.ndarray, q1: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """逐元素球面线性插值，``tau ∈ [0, 1]``；沿最短弧插值。"""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)[..., None]
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    small = sin_theta < 1e-6
    safe = np.where(small, 1.0, sin_theta)
    w0 = np.where(small, 1.0 - tau, np.sin((1.0 - tau) * theta) / safe)
    w1 = np.where(small, tau, np.sin(tau * theta) / safe)
    return quat_normalize(w0 * q0 + w1 * q1)


def quat_interp(t_src: np.ndarray, q_src: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    """沿时间做 SLERP；查询点超出范围时取端点值。``t_src`` 必须严格递增。"""
    t_src = np.asarray(t_src, dtype=np.float64)
    t_query = np.asarray(t_query, dtype=np.float64)
    q_src = quat_make_continuous(quat_normalize(q_src))
    if len(t_src) == 1:
        return np.repeat(q_src, len(t_query), axis=0)
    idx = np.clip(np.searchsorted(t_src, t_query, side="right") - 1, 0, len(t_src) - 2)
    t0, t1 = t_src[idx], t_src[idx + 1]
    tau = np.clip((t_query - t0) / (t1 - t0), 0.0, 1.0)
    return quat_make_continuous(slerp(q_src[idx], q_src[idx + 1], tau))


def rotate_z(v: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """绕 z 轴旋转向量的前两个分量（支持 ``(...,2)`` 与 ``(...,3)``）。"""
    v = np.asarray(v)
    yaw = np.asarray(yaw, dtype=np.float64)
    c, s = np.cos(yaw), np.sin(yaw)
    out = np.array(v, dtype=np.result_type(v.dtype, np.float32), copy=True)
    x, y = v[..., 0], v[..., 1]
    out[..., 0] = c * x - s * y
    out[..., 1] = s * x + c * y
    return out


def wrap_angle(a: np.ndarray) -> np.ndarray:
    """把角度包裹到 (-π, π]。"""
    return np.arctan2(np.sin(a), np.cos(a))
