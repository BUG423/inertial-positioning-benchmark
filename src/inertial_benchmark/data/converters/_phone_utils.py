"""手机类数据集转换器共享的小工具：四元数、文本表格读取、外参/时间偏移估计与物理自检。

约定（与 ``docs/DESIGN.md`` 第 2 节一致）：

* 四元数一律 ``wxyz``、Hamilton 乘法；``q_wb`` 表示 body→world，``v_w = q_wb ⊗ v_b ⊗ q_wb*``；
* 旋转矩阵 ``R_wb`` 与 ``q_wb`` 等价，``v_w = R_wb @ v_b``；
* 所有函数只依赖 numpy / scipy，均为纯函数，便于单元测试。

本模块只被 ``ronin`` / ``ridi`` / ``imunet`` / ``oxiod`` 转换器与其测试使用。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

GRAVITY = 9.80665  # 标准重力，m/s²

# 物理自检的默认阈值（文档 docs/datasets/*.md 引用这些数字，修改时同步文档）
CHECK_GRAVITY_TOL = 0.5  # 世界系加速度均值与 [0,0,g] 的最大偏差，m/s²
CHECK_SPEED_RANGE = (0.3, 2.0)  # 1 s 分辨率“运动秒”水平速度中位数的行人范围，m/s
CHECK_STATIONARY_SPEED = 0.1  # 低于此速度的 1 s 片段视为静止，m/s
CHECK_MAX_STATIONARY = 0.8  # 静止比例上限（超过说明参考位置冻结或序列几乎不动）
CHECK_GYRO_WINDOW = 10.0  # 陀螺相对姿态比较的窗口长度，s
CHECK_GYRO_TOL_DEG = 5.0  # 窗口角度误差中位数上限（三种零偏模型中取最优），度
CHECK_MAX_GYRO_BIAS = 0.05  # 估计常值陀螺零偏模长上限，rad/s（约 2.9°/s）
CHECK_MIN_OVERLAP = 0.9  # IMU 与位姿时间重叠占两者较短者的最小比例
CHECK_HEADING_SCALES = (0.2, 1.0)  # 航向一致性检查的三角核半宽（s），取相关系数较高者
CHECK_HEADING_TOL_DEG = 30.0  # 航向偏差上限（只拦截坐标系级别的粗大错误）
CHECK_HEADING_MIN_CORR = 0.3  # 相关系数低于此值时航向偏差不可信，不参与判定


# ---------------------------------------------------------------------------
# 四元数（wxyz, Hamilton）
# ---------------------------------------------------------------------------


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def quat_conj(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton 乘积 ``a ⊗ b``，支持广播。"""

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
    """用 ``q``（body→world）把 ``v``（body）旋到 world，支持广播。"""

    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    w = q[..., :1]
    u = q[..., 1:]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    w, x, y, z = np.moveaxis(q, -1, 0)
    m = np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    )
    return m.reshape(q.shape[:-1] + (3, 3))


def matrix_to_quat(m: np.ndarray) -> np.ndarray:
    """旋转矩阵 → 单位四元数（w ≥ 0）。"""

    from scipy.spatial.transform import Rotation

    m = np.asarray(m, dtype=np.float64)
    xyzw = Rotation.from_matrix(m.reshape(-1, 3, 3)).as_quat()
    q = xyzw_to_wxyz(xyzw)
    q = np.where(q[:, :1] < 0, -q, q)
    return q.reshape(m.shape[:-2] + (4,))


def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.concatenate([q[..., 3:4], q[..., :3]], axis=-1)


def quat_from_rotvec(r: np.ndarray) -> np.ndarray:
    """旋转向量（轴×角，弧度）→ 四元数。"""

    r = np.asarray(r, dtype=np.float64)
    angle = np.linalg.norm(r, axis=-1, keepdims=True)
    half = 0.5 * angle
    # sin(half)/angle 在小角度时用泰勒展开
    small = angle < 1e-8
    scale = np.where(small, 0.5 - angle**2 / 48.0, np.sin(half) / np.where(small, 1.0, angle))
    return np.concatenate([np.cos(half), r * scale], axis=-1)


def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    """四元数 → 旋转向量（取最短路径，角度 ∈ [0, π]）。"""

    q = quat_normalize(q)
    q = np.where(q[..., :1] < 0, -q, q)
    vec = q[..., 1:]
    s = np.linalg.norm(vec, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(s, q[..., :1])
    small = s < 1e-12
    scale = np.where(small, 2.0, angle / np.where(small, 1.0, s))
    return vec * scale


def quat_angle(q: np.ndarray) -> np.ndarray:
    """四元数表示的旋转角（弧度，∈ [0, π]）。"""

    q = quat_normalize(q)
    return 2.0 * np.arctan2(np.linalg.norm(q[..., 1:], axis=-1), np.abs(q[..., 0]))


def quat_make_continuous(q: np.ndarray) -> np.ndarray:
    """逐样本翻转符号，使相邻四元数点积非负。"""

    q = np.array(q, dtype=np.float64, copy=True)
    if len(q) < 2:
        return q
    dots = np.sum(q[1:] * q[:-1], axis=-1)
    flips = np.concatenate([[False], dots < 0])
    sign = np.where(np.cumsum(flips) % 2 == 1, -1.0, 1.0)
    return q * sign[:, None]


def quat_from_axis_angle(axis: Sequence[float], angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    return np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis])


def quat_interp(t_src: np.ndarray, q_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    """在 ``t_dst`` 上对四元数做 SLERP（越界处夹到端点）。仅用于自检与估计，不用于写出。"""

    t_src = np.asarray(t_src, dtype=np.float64)
    t_dst = np.asarray(t_dst, dtype=np.float64)
    q_src = quat_make_continuous(quat_normalize(q_src))
    idx = np.clip(np.searchsorted(t_src, t_dst, side="right") - 1, 0, len(t_src) - 2)
    t0, t1 = t_src[idx], t_src[idx + 1]
    alpha = np.clip((t_dst - t0) / np.where(t1 > t0, t1 - t0, 1.0), 0.0, 1.0)
    q0, q1 = q_src[idx], q_src[idx + 1]
    delta = quat_to_rotvec(quat_mul(quat_conj(q0), q1))
    return quat_mul(q0, quat_from_rotvec(delta * alpha[:, None]))


def yaw_of_quat(q: np.ndarray) -> np.ndarray:
    """z 轴向上的世界系中机体系的偏航角（ZYX 欧拉角的 yaw）。"""

    w, x, y, z = np.moveaxis(quat_normalize(q), -1, 0)
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def tilt_of_quat(q: np.ndarray) -> np.ndarray:
    """旋转把 z 轴偏离竖直方向的角度（弧度）；纯偏航旋转为 0。"""

    zb = quat_rotate(q, np.array([0.0, 0.0, 1.0]))
    return np.arccos(np.clip(zb[..., 2], -1.0, 1.0))


# 常用固定旋转
Q_IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])
# 把 y 轴向上的世界系（OpenGL/ARCore）转成 z 轴向上：(x, y, z) → (x, -z, y)，即绕 x 轴 +90°
Q_ZUP_FROM_YUP = quat_from_axis_angle([1.0, 0.0, 0.0], np.pi / 2)


# ---------------------------------------------------------------------------
# 文本表格读取
# ---------------------------------------------------------------------------


def read_header_csv(path: Path, delimiter: str = ",") -> tuple:
    """读取首行为表头的数值 CSV，返回 ``(列名列表, (N, C) float64 数组)``。

    兼容 pandas ``to_csv`` 写出的首列空名索引。
    """

    path = Path(path)
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        header = handle.readline().strip().split(delimiter)
    data = np.loadtxt(path, delimiter=delimiter, skiprows=1, dtype=np.float64, ndmin=2)
    if data.shape[1] != len(header):
        raise ValueError(f"{path}: header has {len(header)} columns, data has {data.shape[1]}")
    return [name.strip() for name in header], data


def columns(names: Sequence[str], data: np.ndarray, wanted: Sequence[str]) -> np.ndarray:
    """按列名取子数组；缺列时报错并列出缺失项。"""

    index = {name: i for i, name in enumerate(names)}
    missing = [name for name in wanted if name not in index]
    if missing:
        raise KeyError("missing columns: " + ", ".join(missing))
    return data[:, [index[name] for name in wanted]]


RIDI_CSV_COLUMNS = {
    "time": ("time",),
    "gyro": ("gyro_x", "gyro_y", "gyro_z"),
    "acce": ("acce_x", "acce_y", "acce_z"),
    "linacce": ("linacce_x", "linacce_y", "linacce_z"),
    "grav": ("grav_x", "grav_y", "grav_z"),
    "magnet": ("magnet_x", "magnet_y", "magnet_z"),
    "pos": ("pos_x", "pos_y", "pos_z"),
    "ori": ("ori_w", "ori_x", "ori_y", "ori_z"),
    "rv": ("rv_w", "rv_x", "rv_y", "rv_z"),
}


def read_ridi_processed_csv(path: Path) -> dict:
    """读取 RIDI 官方 ``gen_dataset.py`` 写出的 ``processed/data.csv``（IMUNet 沿用同一格式）。

    返回键见 ``RIDI_CSV_COLUMNS``；``time`` 为 (N,) 纳秒浮点，其余为 (N, k)。数值保持文件原样，不做任何约定转换。
    """

    names, data = read_header_csv(path)
    out = {key: columns(names, data, cols) for key, cols in RIDI_CSV_COLUMNS.items()}
    out["time"] = out["time"][:, 0]
    return out


def monotonic_mask(t: np.ndarray) -> np.ndarray:
    """保留严格大于此前所有时间戳的样本（剔除重复与倒序），返回布尔掩码。"""

    t = np.asarray(t, dtype=np.float64)
    if len(t) == 0:
        return np.zeros(0, dtype=bool)
    running = np.maximum.accumulate(t)
    keep = np.ones(len(t), dtype=bool)
    keep[1:] = t[1:] > running[:-1]
    return keep


# ---------------------------------------------------------------------------
# 估计工具
# ---------------------------------------------------------------------------


def gyro_steps(t: np.ndarray, gyro: np.ndarray) -> np.ndarray:
    """零阶保持下每个采样间隔的机体系增量旋转 ``exp(ω_k Δt_k)``，形状 (N-1, 4)。"""

    t = np.asarray(t, dtype=np.float64)
    return quat_from_rotvec(np.asarray(gyro[:-1], dtype=np.float64) * np.diff(t)[:, None])


def window_products(steps: np.ndarray, i0: np.ndarray, i1: np.ndarray) -> np.ndarray:
    """对每个窗口按时间顺序右乘 ``steps[i0:i1]``，返回 (W, 4)。

    采用成对树形归约（保持乘法顺序），不足部分用单位四元数填充，全程向量化。
    """

    i0 = np.asarray(i0, dtype=np.int64)
    i1 = np.asarray(i1, dtype=np.int64)
    length = int(max(1, (i1 - i0).max(initial=1)))
    idx = i0[:, None] + np.arange(length)[None, :]
    mask = idx < i1[:, None]
    block = np.where(mask[..., None], steps[np.minimum(idx, len(steps) - 1)], Q_IDENTITY)
    while block.shape[1] > 1:
        if block.shape[1] % 2:
            pad = np.broadcast_to(Q_IDENTITY, (block.shape[0], 1, 4))
            block = np.concatenate([block, pad], axis=1)
        block = quat_mul(block[:, 0::2], block[:, 1::2])
        block /= np.linalg.norm(block, axis=-1, keepdims=True)
    return block[:, 0]


def gyro_relative_rotations(t: np.ndarray, gyro: np.ndarray) -> np.ndarray:
    """对机体系角速度做零阶保持积分，返回 ``q_0k``（样本 0 到样本 k 的累计相对旋转），形状 (N, 4)。"""

    steps = gyro_steps(t, gyro)
    out = np.empty((len(steps) + 1, 4))
    out[0] = Q_IDENTITY
    if len(steps):
        out[1:] = prefix_products(steps)
    return out


def prefix_products(steps: np.ndarray) -> np.ndarray:
    """有序前缀积 ``P[k] = s_0 ⊗ s_1 ⊗ … ⊗ s_k``（递归并行前缀扫描，O(N log N)，向量化）。"""

    steps = np.asarray(steps, dtype=np.float64)
    n = len(steps)
    if n == 1:
        return steps.copy()
    m = n // 2
    paired = quat_mul(steps[0 : 2 * m : 2], steps[1 : 2 * m : 2])
    paired /= np.linalg.norm(paired, axis=-1, keepdims=True)
    sub = prefix_products(paired)  # sub[j] = s_0 ⊗ … ⊗ s_{2j+1}
    out = np.empty_like(steps)
    out[0] = steps[0]
    out[1 : 2 * m : 2] = sub
    evens = out[2::2]
    out[2::2] = quat_mul(sub[: len(evens)], steps[2::2])
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def body_rates_from_orientation(t: np.ndarray, q_wb: np.ndarray, max_gap_factor: float = 3.0) -> tuple:
    """由相邻姿态差分得到机体系角速度，返回 ``(t_mid, omega_body)``。

    跨越缺口（间隔大于 ``max_gap_factor`` 倍中位采样间隔）的样本对被丢弃。
    """

    t = np.asarray(t, dtype=np.float64)
    q = quat_make_continuous(quat_normalize(q_wb))
    rel = quat_mul(quat_conj(q[:-1]), q[1:])
    dt = np.diff(t)
    ok = dt > 0
    if ok.any():
        ok &= dt <= max_gap_factor * np.median(dt[ok])
    omega = quat_to_rotvec(rel[ok]) / dt[ok, None]
    t_mid = 0.5 * (t[:-1] + t[1:])[ok]
    return t_mid, omega


def estimate_rotation(a: np.ndarray, b: np.ndarray, weights: Optional[np.ndarray] = None) -> tuple:
    """Wahba/Kabsch：求 ``R`` 使 ``Σ w ‖R a_i − b_i‖²`` 最小，返回 ``(R, rms_residual)``。"""

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    w = np.ones(len(a)) if weights is None else np.asarray(weights, dtype=np.float64)
    h = (a * w[:, None]).T @ b
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    resid = b - a @ r.T
    rms = float(np.sqrt(np.sum(w * np.sum(resid**2, axis=1)) / max(np.sum(w), 1e-12)))
    return r, rms


def estimate_time_offset(
    t_a: np.ndarray,
    s_a: np.ndarray,
    t_b: np.ndarray,
    s_b: np.ndarray,
    max_lag: float = 2.0,
    rate: float = 100.0,
) -> tuple:
    """用标量信号互相关估计时钟偏移 ``offset``，使 ``s_a(t) ≈ s_b(t + offset)``。

    返回 ``(offset_seconds, peak_correlation)``。两信号先重采样到公共网格并去均值归一化。
    """

    from scipy.signal import correlate

    lo = max(t_a[0], t_b[0])
    hi = min(t_a[-1], t_b[-1])
    if hi - lo < 4 * max_lag:
        return float("nan"), float("nan")
    grid = np.arange(lo, hi, 1.0 / rate)
    a = np.interp(grid, t_a, s_a)
    b = np.interp(grid, t_b, s_b)
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)
    full = correlate(b, a, mode="full", method="fft") / len(grid)  # full[n-1+L] = Σ a[i] b[i+L]
    center = len(grid) - 1
    max_k = int(max_lag * rate)
    corr = full[center - max_k : center + max_k + 1]
    k = int(np.argmax(corr))
    offset = (k - max_k) / rate
    # 抛物线插值细化到亚采样
    if 0 < k < len(corr) - 1:
        c0, c1, c2 = corr[k - 1], corr[k], corr[k + 1]
        denom = c0 - 2 * c1 + c2
        if abs(denom) > 1e-12:
            offset += 0.5 * (c0 - c2) / denom / rate
    return float(offset), float(corr[k])


def _rate_pairs(t_imu, gyro, t_pose, q_ws, rate):
    """在公共网格上返回 (IMU 角速度, 位姿差分角速度) 的箱形平均对，只保留两边都有样本的格点。"""

    t_mid, omega_s = body_rates_from_orientation(t_pose, q_ws)
    if len(t_mid) < 2:
        return np.zeros((0, 3)), np.zeros((0, 3))
    lo = max(t_imu[0], t_mid[0])
    hi = min(t_imu[-1], t_mid[-1])
    grid = np.arange(lo, hi, 1.0 / rate)
    if len(grid) < 2:
        return np.zeros((0, 3)), np.zeros((0, 3))
    wb = _box_resample(t_imu, gyro, grid, 1.0 / rate)
    ws = _box_resample(t_mid, omega_s, grid, 1.0 / rate)
    ok = np.isfinite(wb).all(1) & np.isfinite(ws).all(1)
    return wb[ok], ws[ok]


def estimate_body_extrinsic(
    t_imu: np.ndarray,
    gyro: np.ndarray,
    t_pose: np.ndarray,
    q_ws: np.ndarray,
    rate: float = 20.0,
    min_rate: float = 0.2,
    trim: float = 3.0,
) -> tuple:
    """手眼式旋转对齐：估计常值旋转 ``q_sb``（IMU 机体系 b → 位姿刚体系 s）。

    若 ``q_ws`` 是刚体 s 的姿态，则 IMU 机体系姿态为 ``q_wb = q_ws ⊗ q_sb``，
    且角速度满足 ``ω_s = R_sb ω_b``。在箱形平均后的公共网格上做 Wahba 最小二乘，
    再剔除残差大于 ``trim`` 倍中位残差的格点重拟合一次（抵御参考姿态野值）。
    返回 ``(q_sb, 残差中位数 rad/s, 使用的格点数)``。
    """

    wb, ws = _rate_pairs(np.asarray(t_imu, float), gyro, np.asarray(t_pose, float), q_ws, rate)
    ok = np.linalg.norm(wb, axis=1) > min_rate
    if ok.sum() < 20:
        return Q_IDENTITY.copy(), float("nan"), int(ok.sum())
    a, b = wb[ok], ws[ok]
    r, _ = estimate_rotation(a, b)
    resid = np.linalg.norm(b - a @ r.T, axis=1)
    inlier = resid <= trim * max(np.median(resid), 1e-6)
    if inlier.sum() >= 20:
        r, _ = estimate_rotation(a[inlier], b[inlier])
        resid = np.linalg.norm(b - a @ r.T, axis=1)
    return matrix_to_quat(r), float(np.median(resid)), int(ok.sum())


def estimate_gyro_bias(
    t_imu: np.ndarray, gyro: np.ndarray, t_pose: np.ndarray, q_wb: np.ndarray, rate: float = 20.0
) -> np.ndarray:
    """常值陀螺零偏估计：``median(ω_gyro − ω_ref)``，``ω_ref`` 为参考姿态差分得到的机体系角速度。

    参考角速度虽经平滑，但对零偏无系统影响；用中位数抵御参考姿态野值。
    """

    a, b = _rate_pairs(np.asarray(t_imu, float), gyro, np.asarray(t_pose, float), q_wb, rate)
    if len(a) < 20:
        return np.zeros(3)
    return np.median(a - b, axis=0)


def estimate_gyro_bias_segments(
    t_imu: np.ndarray,
    gyro: np.ndarray,
    t_pose: np.ndarray,
    q_wb: np.ndarray,
    segment: float = 60.0,
    rate: float = 20.0,
) -> np.ndarray:
    """分段常值（每 ``segment`` 秒）的陀螺零偏，返回与 ``t_imu`` 对齐的 (N, 3)。

    手机陀螺零偏会随温度缓慢漂移；分段常值模型只吸收慢变零偏，无法吸收与旋转量成正比的坐标系/约定错误。
    样本不足的分段退回全段中位数。
    """

    t_imu = np.asarray(t_imu, float)
    out = np.zeros((len(t_imu), 3))
    t_mid, omega = body_rates_from_orientation(np.asarray(t_pose, float), q_wb)
    if len(t_mid) < 2:
        return out
    grid = np.arange(max(t_imu[0], t_mid[0]), min(t_imu[-1], t_mid[-1]), 1.0 / rate)
    if len(grid) < 20:
        return out
    wb = _box_resample(t_imu, gyro, grid, 1.0 / rate)
    ws = _box_resample(t_mid, omega, grid, 1.0 / rate)
    ok = np.isfinite(wb).all(1) & np.isfinite(ws).all(1)
    if ok.sum() < 20:
        return out
    grid, diff = grid[ok], (wb - ws)[ok]
    global_bias = np.median(diff, axis=0)
    edges = np.arange(t_imu[0], t_imu[-1] + segment, segment)
    for start in edges:
        sel = (grid >= start) & (grid < start + segment)
        bias = np.median(diff[sel], axis=0) if sel.sum() >= 0.25 * segment * rate else global_bias
        out[(t_imu >= start) & (t_imu < start + segment)] = bias
    return out


def refine_time_offset(
    t_imu: np.ndarray,
    gyro: np.ndarray,
    t_pose: np.ndarray,
    q_ws: np.ndarray,
    q_sb: np.ndarray,
    center: float = 0.0,
    max_lag: float = 0.3,
    rate: float = 100.0,
) -> tuple:
    """在给定外参下，用三轴角速度互相关之和在 ``center ± max_lag`` 内细化时钟偏移。

    约定同 :func:`estimate_time_offset`：``ω_imu(t) ≈ ω_pose(t + offset)``。返回 ``(offset, 平均相关系数)``。
    """

    from scipy.signal import correlate

    t_mid, omega_s = body_rates_from_orientation(t_pose, q_ws)
    shifted = np.asarray(t_imu, float) + center
    w_imu = np.asarray(gyro, float) @ quat_to_matrix(q_sb).T
    lo, hi = max(shifted[0], t_mid[0]), min(shifted[-1], t_mid[-1])
    if hi - lo < 4 * max_lag or len(t_mid) < 2:
        return float("nan"), float("nan")
    grid = np.arange(lo, hi, 1.0 / rate)
    total = np.zeros(2 * len(grid) - 1)
    for k in range(3):
        a = np.interp(grid, shifted, w_imu[:, k])
        b = np.interp(grid, t_mid, omega_s[:, k])
        a = (a - a.mean()) / (a.std() + 1e-12)
        b = (b - b.mean()) / (b.std() + 1e-12)
        total += correlate(b, a, mode="full", method="fft") / len(grid)
    total /= 3.0
    center_idx = len(grid) - 1
    max_k = int(max_lag * rate)
    seg = total[center_idx - max_k : center_idx + max_k + 1]
    k = int(np.argmax(seg))
    offset = (k - max_k) / rate
    if 0 < k < len(seg) - 1:
        c0, c1, c2 = seg[k - 1], seg[k], seg[k + 1]
        denom = c0 - 2 * c1 + c2
        if abs(denom) > 1e-12:
            offset += 0.5 * (c0 - c2) / denom / rate
    return float(center + offset), float(seg[k])


def pose_glitch_mask(
    t: np.ndarray,
    q_wb: np.ndarray,
    position: np.ndarray,
    max_rate_deg: float = 600.0,
    max_speed: float = 10.0,
    pad: float = 0.1,
) -> np.ndarray:
    """只用参考位姿自身检测野值：相邻帧隐含角速度或速度超限的帧，连同前后 ``pad`` 秒标为无效。

    返回有效掩码（True 为可用）。用于光学动捕的标记点识别翻转（常见 180° 单帧翻转）与短时抖动。
    """

    t = np.asarray(t, dtype=np.float64)
    n = len(t)
    valid = np.ones(n, dtype=bool)
    if n < 2:
        return valid
    q = quat_make_continuous(quat_normalize(q_wb))
    dt = np.maximum(np.diff(t), 1e-6)
    ang = quat_angle(quat_mul(quat_conj(q[:-1]), q[1:])) / dt
    spd = np.linalg.norm(np.diff(np.asarray(position, float), axis=0), axis=1) / dt
    bad_step = (np.degrees(ang) > max_rate_deg) | (spd > max_speed)
    bad = np.zeros(n, dtype=bool)
    bad[1:] |= bad_step
    bad[:-1] |= bad_step
    if bad.any():
        # 以时间为单位向两侧膨胀
        idx = np.where(bad)[0]
        lo = np.searchsorted(t, t[idx] - pad, side="left")
        hi = np.searchsorted(t, t[idx] + pad, side="right")
        cover = np.zeros(n + 1, dtype=np.int64)
        np.add.at(cover, lo, 1)
        np.add.at(cover, hi, -1)
        valid = np.cumsum(cover[:-1]) == 0
    return valid


def _box_resample(t: np.ndarray, x: np.ndarray, grid: np.ndarray, width: float) -> np.ndarray:
    """以 ``grid`` 为中心、宽 ``width`` 的箱形平均（累积和实现），空箱为 NaN。"""

    x = np.asarray(x, dtype=np.float64)
    csum = np.concatenate([np.zeros((1, x.shape[1])), np.cumsum(x, axis=0)])
    lo = np.searchsorted(t, grid - width / 2)
    hi = np.searchsorted(t, grid + width / 2)
    count = (hi - lo)[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, (csum[hi] - csum[lo]) / np.maximum(count, 1), np.nan)


# ---------------------------------------------------------------------------
# 物理自检
# ---------------------------------------------------------------------------


def _triangle_filter(t: np.ndarray, x: np.ndarray, grid: np.ndarray, half_width: float, fine: float = 100.0):
    """在 ``grid`` 上求 ``x(t)`` 与半宽 ``half_width`` 三角核的卷积（两次箱形平均实现）。"""

    tf = np.arange(grid[0] - half_width, grid[-1] + half_width + 0.5 / fine, 1.0 / fine)
    width = max(1, int(round(half_width * fine)))
    out = []
    for k in range(x.shape[1]):
        xf = np.interp(tf, t, x[:, k])
        for _ in range(2):
            c = np.concatenate([[0.0], np.cumsum(xf)])
            xf = (c[width:] - c[:-width]) / width
            xf = np.concatenate([np.full(width // 2, xf[0]), xf, np.full(width - 1 - width // 2, xf[-1])])
        out.append(np.interp(grid, tf, xf))
    return np.stack(out, axis=1)


def heading_consistency(ti, acc_world, tp, pos, scales=CHECK_HEADING_SCALES) -> dict:
    """参考姿态的航向与参考位置坐标系是否一致。

    把 IMU 比力经参考姿态旋到世界系并扣除重力，与位置二阶差分得到的加速度（在同一三角核下）做水平面二维对齐，
    返回旋转角（度，IMU 水平加速度需旋转该角才能对上位置加速度）与相关系数；多个尺度中取相关系数最高者。
    """

    best = {"heading_offset_deg": float("nan"), "heading_corr": float("nan"), "heading_scale_s": float("nan")}
    lin = acc_world - np.array([0.0, 0.0, GRAVITY])
    for dt in scales:
        lo, hi = max(ti[0], tp[0]) + 2 * dt, min(ti[-1], tp[-1]) - 2 * dt
        if hi - lo < 20 * dt:
            continue
        grid = np.arange(lo, hi, dt)
        a_imu = _triangle_filter(ti, lin, grid, dt)
        p = np.stack([np.interp(grid, tp, pos[:, k]) for k in range(3)], axis=1)
        a_pos = np.full_like(p, np.nan)
        a_pos[1:-1] = (p[2:] - 2 * p[1:-1] + p[:-2]) / dt**2
        j = np.clip(np.searchsorted(tp, grid), 1, len(tp) - 1)
        near = np.minimum(np.abs(grid - tp[j - 1]), np.abs(tp[j] - grid)) <= max(0.05, dt / 4)
        good = near & np.roll(near, 1) & np.roll(near, -1) & np.isfinite(a_pos).all(1) & np.isfinite(a_imu).all(1)
        if good.sum() < 20:
            continue
        a, b = a_imu[good, :2], a_pos[good, :2]
        cross = np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
        dot = np.sum(a * b)
        # 最优旋转后的相关系数（旋转不变），衡量航向估计是否可信
        corr = float(np.hypot(dot, cross) / np.sqrt(np.sum(a * a) * np.sum(b * b) + 1e-12))
        if not np.isfinite(best["heading_corr"]) or corr > best["heading_corr"]:
            best = {
                "heading_offset_deg": float(np.degrees(np.arctan2(cross, dot))),
                "heading_corr": corr,
                "heading_scale_s": float(dt),
            }
    return best


def physics_check(
    raw,
    gyro_window: float = CHECK_GYRO_WINDOW,
    estimate_offset: bool = True,
    speed_range: tuple = CHECK_SPEED_RANGE,
) -> dict:
    """对一条 ``RawSequence`` 做物理一致性检查，返回统计量与 ``failures`` 列表。

    检查项：

    0. ``overlap``：IMU 与位姿时间区间的重叠比例；
    1. ``gravity``：用 ``orientation``（插值到 IMU 时间）把机体系比力旋到世界系，全段均值 ≈ ``[0,0,+g]``；
    2. ``speed``：参考位置在 1 s 网格上差分，“运动秒”水平速度中位数位于行人范围，静止比例不过高；
    3. ``gyro``：每个 ``gyro_window`` 窗口内陀螺积分的相对旋转与参考姿态相对旋转之差，取角度中位数
       （三种零偏模型取最优），零偏估计不过大；
    4. ``heading``：参考姿态的航向与参考位置坐标系一致（见 :func:`heading_consistency`）；
    另报告陀螺与参考姿态差分角速度的最佳常值旋转（应接近单位阵）与时间偏移（互相关），仅作诊断。
    """

    t_imu = np.asarray(raw.imu_time, dtype=np.float64)
    t_pose = np.asarray(raw.pose_time, dtype=np.float64)
    imu_ok = np.ones(len(t_imu), bool) if raw.imu_valid is None else np.asarray(raw.imu_valid, bool)
    pose_ok = np.ones(len(t_pose), bool) if raw.pose_valid is None else np.asarray(raw.pose_valid, bool)
    imu_ok &= np.isfinite(raw.gyroscope).all(1) & np.isfinite(raw.accelerometer).all(1)
    pose_ok &= np.isfinite(raw.position).all(1) & np.isfinite(raw.orientation).all(1)
    ti, gyro, acc = t_imu[imu_ok], raw.gyroscope[imu_ok], raw.accelerometer[imu_ok]
    tp, pos, ori = t_pose[pose_ok], raw.position[pose_ok], raw.orientation[pose_ok]

    out: dict = {"failures": []}
    lo, hi = max(ti[0], tp[0]), min(ti[-1], tp[-1])
    span_imu, span_pose = ti[-1] - ti[0], tp[-1] - tp[0]
    overlap = max(0.0, hi - lo)
    out["duration_s"] = float(overlap)
    out["overlap_ratio"] = float(overlap / max(min(span_imu, span_pose), 1e-9))
    out["imu_rate_hz"] = float(1.0 / np.median(np.diff(ti)))
    out["pose_rate_hz"] = float(1.0 / np.median(np.diff(tp)))
    if out["overlap_ratio"] < CHECK_MIN_OVERLAP or overlap < 2 * gyro_window:
        out["failures"].append(f"overlap {overlap:.1f}s ratio {out['overlap_ratio']:.2f}")
        return out

    out["pose_valid_fraction"] = float(pose_ok.mean())
    sel = (ti >= lo) & (ti <= hi)
    ti, gyro, acc = ti[sel], gyro[sel], acc[sel]
    q_imu = quat_interp(tp, ori, ti)
    # 只信任距最近有效位姿不超过 2 个位姿采样间隔的 IMU 样本（避免跨缺口插值的姿态参与判定）
    pose_dt = float(np.median(np.diff(tp)))
    j = np.clip(np.searchsorted(tp, ti), 1, len(tp) - 1)
    nearest = np.minimum(np.abs(ti - tp[j - 1]), np.abs(tp[j] - ti))
    near = nearest <= max(2.0 * pose_dt, 0.011)

    # 1) 重力
    acc_w = quat_rotate(q_imu[near], acc[near])
    mean_w = acc_w.mean(axis=0)
    out["acc_world_mean"] = [float(v) for v in mean_w]
    out["gravity_error"] = float(np.linalg.norm(mean_w - np.array([0.0, 0.0, GRAVITY])))
    out["gravity_horizontal"] = float(np.linalg.norm(mean_w[:2]))
    out["gravity_scale"] = float(np.linalg.norm(mean_w) / GRAVITY)
    out["gravity_tilt_deg"] = float(np.degrees(np.arctan2(np.linalg.norm(mean_w[:2]), mean_w[2])))
    if out["gravity_error"] > CHECK_GRAVITY_TOL:
        out["failures"].append(
            f"gravity mean {np.round(mean_w, 2).tolist()} (|g| ratio {out['gravity_scale']:.3f}, "
            f"tilt {out['gravity_tilt_deg']:.1f} deg)"
        )

    # 2) 速度（1 s 网格）。行人序列常含长时间静止（站立、坐下操作手机），
    #    因此范围检查用“运动秒”（> CHECK_STATIONARY_SPEED）的中位数，并单独报告静止比例。
    grid = np.arange(tp[0], tp[-1], 1.0)
    if len(grid) >= 3:
        p = np.stack([np.interp(grid, tp, pos[:, k]) for k in range(3)], axis=1)
        speed = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1)
        moving = speed > CHECK_STATIONARY_SPEED
        out["speed_median"] = float(np.median(speed))
        out["speed_moving_median"] = float(np.median(speed[moving])) if moving.any() else 0.0
        out["speed_p95"] = float(np.percentile(speed, 95))
        out["stationary_fraction"] = float(1.0 - moving.mean())
        out["path_length_m"] = float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))
        lo_s, hi_s = speed_range
        if not lo_s <= out["speed_moving_median"] <= hi_s:
            out["failures"].append(f"moving speed median {out['speed_moving_median']:.2f} m/s")
        if out["stationary_fraction"] > CHECK_MAX_STATIONARY:
            out["failures"].append(f"stationary fraction {out['stationary_fraction']:.2f}")
    else:
        out["failures"].append("pose too short for speed check")

    # 3) 陀螺相对姿态（只在 IMU 时间无大缺口、端点靠近有效位姿的窗口上比较）。
    #    手机陀螺常有 ~0.5°/s 的残余零偏且会缓慢漂移，10 s 窗口会累积 ~5°；该检查的目的是确认
    #    “同一刚体、轴约定正确”，因此同时评估几种零偏模型（见下），并报告原始误差与零偏估计。
    bias = estimate_gyro_bias(ti, gyro, tp, ori)
    out["gyro_bias_est"] = [float(v) for v in bias]
    out["gyro_bias_norm"] = float(np.linalg.norm(bias))
    if out["gyro_bias_norm"] > CHECK_MAX_GYRO_BIAS:
        out["failures"].append(f"gyro bias {out['gyro_bias_norm']:.3f} rad/s")
    starts = np.arange(ti[0], ti[-1] - gyro_window, gyro_window / 2)
    i0 = i1 = np.zeros(0, dtype=np.int64)
    if len(starts):
        i0 = np.searchsorted(ti, starts)
        i1 = np.minimum(np.searchsorted(ti, starts + gyro_window), len(ti) - 1)
        n_gaps = np.concatenate([[0], np.cumsum(np.diff(ti) > 0.1)])  # 截至每个样本的大缺口数
        clean = (n_gaps[i1] == n_gaps[i0]) & near[i0] & near[i1]
        i0, i1 = i0[clean], i1[clean]
    if len(i0):
        d_ref = quat_mul(quat_conj(q_imu[i0]), q_imu[i1])
        # 三种零偏模型：不去零偏 / 常值 / 60 s 分段常值；零偏估计会被微小失准“污染”（转向有偏好时
        # median(ω_gyro − ω_ref) ≠ 0），故取中位误差最小者判定。任何零偏模型都无法抵消与旋转量成正比的轴约定错误。
        models = {
            "none": gyro,
            "constant": gyro - bias,
            "segmented_60s": gyro - estimate_gyro_bias_segments(ti, gyro, tp, ori),
        }
        errors = {
            key: np.degrees(quat_angle(quat_mul(quat_conj(window_products(gyro_steps(ti, g), i0, i1)), d_ref)))
            for key, g in models.items()
        }
        best = min(errors, key=lambda key: np.median(errors[key]))
        err = errors[best]
        rot = np.degrees(quat_angle(d_ref))
        out["gyro_window_err_deg_median_raw"] = float(np.median(errors["none"]))
        out["gyro_window_err_deg_median_constant_bias"] = float(np.median(errors["constant"]))
        out["gyro_window_err_deg_median_segmented_bias"] = float(np.median(errors["segmented_60s"]))
        out["gyro_bias_model"] = best
        out["gyro_window_err_deg_median"] = float(np.median(err))
        out["gyro_window_err_deg_p90"] = float(np.percentile(err, 90))
        out["gyro_windows"] = int(len(err))
        out["ref_window_rot_deg_median"] = float(np.median(rot))
        if out["gyro_window_err_deg_median"] > CHECK_GYRO_TOL_DEG:
            out["failures"].append(
                f"gyro/ref window error {out['gyro_window_err_deg_median']:.1f} deg (best bias model: {best})"
            )
    else:
        out["failures"].append("no gap-free window for gyro check")
    # 4) 航向一致性：参考姿态的偏航与参考位置坐标系一致（拦截 90°/180° 级的坐标系错误）
    heading = heading_consistency(ti[near], quat_rotate(q_imu[near], acc[near]), tp, pos)
    out.update(heading)
    if (
        np.isfinite(heading["heading_corr"])
        and heading["heading_corr"] >= CHECK_HEADING_MIN_CORR
        and abs(heading["heading_offset_deg"]) > CHECK_HEADING_TOL_DEG
    ):
        out["failures"].append(
            f"heading offset {heading['heading_offset_deg']:.1f} deg between orientation and position frames "
            f"(corr {heading['heading_corr']:.2f})"
        )
    q_sb, rms, _ = estimate_body_extrinsic(ti, gyro, tp, ori)
    out["gyro_ref_extrinsic_deg"] = float(np.degrees(quat_angle(q_sb)))
    out["gyro_ref_extrinsic_resid"] = rms
    if estimate_offset:
        t_mid, omega_ref = body_rates_from_orientation(tp, ori)
        off, peak = estimate_time_offset(ti, np.linalg.norm(gyro, axis=1), t_mid, np.linalg.norm(omega_ref, axis=1))
        out["time_offset_s"] = off
        out["time_offset_corr"] = peak
    return out


# ---------------------------------------------------------------------------
# 合成运动（单元测试用）
# ---------------------------------------------------------------------------


def simulate_motion(
    duration: float = 60.0,
    rate: float = 200.0,
    seed: int = 0,
    speed: float = 1.2,
    times: Optional[np.ndarray] = None,
) -> dict:
    """生成一段平滑的行人式运动，返回真值（世界系重力对齐、z 向上，body→world）：

    ``t`` (N,)、``q_wb`` (N,4)、``euler`` (N,3)=(roll, pitch, yaw)、``gyro`` (N,3) 机体系角速度、
    ``specific_force`` (N,3) 机体系比力、``position`` (N,3)、``acc_world`` (N,3)。

    姿态为 ``R_z(yaw) ⊗ R_x(pitch) ⊗ R_y(roll)``（与 CoreMotion 约定一致），各角度含若干随机频率分量，
    避免互相关出现周期性歧义；导数用中心差分（步长 1e-4 s）计算，误差远小于测试容差。
    给定 ``times`` 时在这些时刻（相对秒）求值，忽略 ``duration`` / ``rate``；同一 ``seed`` 描述同一条连续运动。
    """

    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration, 1.0 / rate) if times is None else np.asarray(times, dtype=np.float64)
    radius = 6.0
    omega_path = speed / radius
    freqs = rng.uniform(0.05, 0.8, size=(3, 4))
    amps = np.array([[0.12], [0.1], [0.2]]) * rng.uniform(0.5, 1.0, size=(3, 4))
    phases = rng.uniform(0, 2 * np.pi, size=(3, 4))

    def euler(tt):
        tt = np.asarray(tt, float)[..., None]
        wobble = np.sum(amps * np.sin(2 * np.pi * freqs * tt[..., None, :] + phases), axis=-1)
        roll = 0.3 + wobble[..., 0]
        pitch = 0.8 + wobble[..., 1]
        yaw = omega_path * tt[..., 0] + np.pi / 2 + wobble[..., 2]
        return np.stack([roll, pitch, yaw], axis=-1)

    def quat(tt):
        e = euler(tt)
        axis = lambda k, a: quat_from_rotvec(np.eye(3)[k] * a[..., None])
        return quat_mul(quat_mul(axis(2, e[..., 2]), axis(0, e[..., 1])), axis(1, e[..., 0]))

    def pos(tt):
        tt = np.asarray(tt, float)
        return np.stack(
            [
                radius * np.cos(omega_path * tt),
                radius * np.sin(omega_path * tt),
                0.03 * np.sin(2 * np.pi * 1.8 * tt),
            ],
            axis=-1,
        )

    h = 1e-4
    q = quat(t)
    q_plus, q_minus = quat(t + h), quat(t - h)
    gyro = quat_to_rotvec(quat_mul(quat_conj(q_minus), q_plus)) / (2 * h)
    acc_world = (pos(t + h) - 2 * pos(t) + pos(t - h)) / h**2
    specific_force = quat_rotate(quat_conj(q), acc_world + np.array([0.0, 0.0, GRAVITY]))
    return {
        "t": t,
        "q_wb": q,
        "euler": euler(t),
        "gyro": gyro,
        "specific_force": specific_force,
        "position": pos(t),
        "acc_world": acc_world,
    }


# ---------------------------------------------------------------------------
# 转换器公共收尾
# ---------------------------------------------------------------------------

REQUIRED_ATTR_KEYS = (
    "subject_id",
    "device_id",
    "placement",
    "group_id",
    "position_source",
    "orientation_source",
    "device_orientation_source",
    "body_frame",
    "source_files",
)


def round_stats(stats: dict, digits: int = 4) -> dict:
    """把自检统计里的浮点数四舍五入（非有限值写 None），便于以 JSON 写入 notes。"""

    out = {}
    for key, value in stats.items():
        if isinstance(value, float):
            out[key] = round(value, digits) if np.isfinite(value) else None
        elif isinstance(value, list) and value and isinstance(value[0], float):
            out[key] = [round(v, digits) for v in value]
        else:
            out[key] = value
    return out


def finalize_sequence(raw, speed_range: tuple = CHECK_SPEED_RANGE, check: bool = True):
    """补齐有效掩码并运行物理自检；自检失败则把 ``raw.rejected`` 置为原因。

    自检统计以 ``physics_check {...}``（JSON）的形式追加到 ``raw.notes``，供转换报告与文档统计使用。
    """

    import json

    finite_imu = np.isfinite(raw.gyroscope).all(1) & np.isfinite(raw.accelerometer).all(1)
    finite_pose = np.isfinite(raw.position).all(1) & np.isfinite(raw.orientation).all(1)
    raw.imu_valid = finite_imu if raw.imu_valid is None else np.asarray(raw.imu_valid, bool) & finite_imu
    raw.pose_valid = finite_pose if raw.pose_valid is None else np.asarray(raw.pose_valid, bool) & finite_pose
    bad_imu, bad_pose = int((~finite_imu).sum()), int((~finite_pose).sum())
    if bad_imu or bad_pose:
        raw.notes.append(f"non-finite samples marked invalid: imu={bad_imu}, pose={bad_pose}")
    if raw.imu_valid.sum() < 2 or raw.pose_valid.sum() < 2:
        raw.rejected = raw.rejected or "fewer than two valid IMU or pose samples"
        return raw
    if not check:
        return raw
    stats = physics_check(raw, speed_range=speed_range)
    raw.notes.append("physics_check " + json.dumps(round_stats(stats), sort_keys=True))
    if stats["failures"] and raw.rejected is None:
        raw.rejected = "physics check failed: " + "; ".join(stats["failures"])
    return raw


def parse_physics_note(notes: Sequence[str]) -> Optional[dict]:
    """从 notes 中取回 ``finalize_sequence`` 写入的自检统计。"""

    import json

    for note in notes:
        if note.startswith("physics_check "):
            return json.loads(note[len("physics_check ") :])
    return None


def rejected_sequence(sequence_id: str, reason: str, attrs: Optional[dict] = None):
    """构造一条空的拒收序列（形状合法、属性齐全）。"""

    from .base import RawSequence

    full = {key: "unknown" for key in REQUIRED_ATTR_KEYS}
    full.update(attrs or {})
    return RawSequence(
        sequence_id=sequence_id,
        imu_time=np.zeros(0),
        gyroscope=np.zeros((0, 3)),
        accelerometer=np.zeros((0, 3)),
        pose_time=np.zeros(0),
        position=np.zeros((0, 3)),
        orientation=np.zeros((0, 4)),
        attrs=full,
        rejected=reason,
    )


def relative_source(path: Path, root: Optional[Path]) -> str:
    """源文件相对数据根目录的 POSIX 路径（无法相对化时退回文件名）。"""

    path = Path(path)
    if root is not None:
        try:
            return path.resolve().relative_to(Path(root).resolve()).as_posix()
        except ValueError:
            pass
    return path.name
