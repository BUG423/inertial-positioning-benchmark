"""转换器共享工具：四元数约定、姿态插值、安装旋转估计与物理自检。

本模块只服务于“解析层”（``converters/*.py``），不做重采样。所有四元数在接口上一律为
``wxyz``、``body_to_world``；内部借助 ``scipy.spatial.transform.Rotation``（其约定为 ``xyzw``）。

物理自检（``physical_checks``）回答五个问题：

1. 重力：机体系比力经参考姿态旋到世界系后，全段均值是否 ≈ ``[0, 0, +g]``；
2. 速度：参考位置差分得到的水平速度是否处于人体运动范围（用于发现单位错误）；
3. 陀螺：10 s 窗口内陀螺积分的相对姿态是否与参考姿态的相对变化一致（发现轴/方向错误）；
4. 加速度：参考位置的二阶差分是否与世界系比力减重力一致（诊断位置单位、时间尺度与同步）；
5. 时间：IMU 与参考位姿是否在同一时钟上充分重叠。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

STANDARD_GRAVITY = 9.80665  # m/s²，iOS 的 “G” 单位与本基准的重力标称值

# 物理自检硬阈值（违反即拒收，除非转换器给出了有据可查的解释）。
LIMITS = {
    "gravity_error_max": 0.5,  # m/s²，世界系比力均值与 [0,0,g] 的距离
    "gyro_window_error_median_max": 5.0,  # 度，10 s 窗口相对姿态误差中位数
    "speed_median_max": 4.5,  # m/s，水平速度中位数上限（人体跑步量级；单位错误会高出 2–3 个数量级）
    "speed_p95_max": 8.0,  # m/s，水平速度 95 分位上限
    "overlap_min": 0.9,  # IMU/位姿时间重叠占较短者时长的比例
    "min_windows": 1,  # 至少要有一个可评估的陀螺窗口
    "gyro_bias_max": 0.1,  # rad/s，可被接受为“未标定零偏”的常值零偏上限
}
# 软阈值：只写警告，不拒收。
SOFT_LIMITS = {
    "speed_median_min": 0.1,  # m/s，低于此值说明序列大部分时间静止
    "speed_median_walk_max": 2.5,  # m/s，高于此值说明以跑步等快速运动为主
    "gravity_error_warn": 0.3,  # m/s²
    "gyro_window_error_median_warn": 3.0,  # 度
    # 加速度一致性残差 / IMU 加速度幅值（高频参考噪声会抬高此值）
    "acc_consistency_ratio_warn": 1.0,
    "acc_scale_min": 0.7,  # 位置二阶差分 / IMU 加速度 的比例下限（位置被平滑或不同步时偏低）
    "acc_scale_max": 1.3,
    "time_offset_warn": 0.03,  # s，角速度互相关估计的时延
    "angular_rate_corr_min": 0.5,  # 角速度互相关峰值低于此值时时延估计不可信
}


# ---------------------------------------------------------------------------
# 四元数工具
# ---------------------------------------------------------------------------

def wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q[..., [1, 2, 3, 0]]


def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q[..., [3, 0, 1, 2]]


def as_rotation(q_wxyz: np.ndarray) -> Rotation:
    """``wxyz`` 四元数（body_to_world）→ ``Rotation``。"""

    return Rotation.from_quat(wxyz_to_xyzw(q_wxyz))


def from_rotation(rot: Rotation) -> np.ndarray:
    """``Rotation`` → ``wxyz`` 四元数（w ≥ 0 不强制，符号连续化由统一流水线负责）。"""

    return xyzw_to_wxyz(rot.as_quat())


def normalize_quaternions(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """逐元素 Hamilton 积 ``a ⊗ b``（``wxyz``，支持广播）。"""

    aw, ax, ay, az = np.moveaxis(np.asarray(a, dtype=np.float64), -1, 0)
    bw, bx, by, bz = np.moveaxis(np.asarray(b, dtype=np.float64), -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    q = np.array(q, dtype=np.float64, copy=True)
    q[..., 1:] *= -1.0
    return q


def rotvec_to_quat(v: np.ndarray) -> np.ndarray:
    """旋转向量 → ``wxyz`` 单位四元数（小角度数值稳定）。"""

    v = np.asarray(v, dtype=np.float64)
    angle = np.linalg.norm(v, axis=-1)
    half = 0.5 * angle
    small = angle < 1e-8
    safe = np.where(small, 1.0, angle)
    k = np.where(small, 0.5 - angle**2 / 48.0, np.sin(half) / safe)
    return np.concatenate([np.cos(half)[..., None], v * k[..., None]], axis=-1)


def quat_angle_deg(q: np.ndarray) -> np.ndarray:
    """单位四元数所表示旋转的转角（度，0–180）。"""

    q = np.asarray(q, dtype=np.float64)
    w = np.clip(np.abs(q[..., 0]), 0.0, 1.0)
    vec = np.linalg.norm(q[..., 1:], axis=-1)
    return np.degrees(2.0 * np.arctan2(vec, w))


def prefix_products(q: np.ndarray) -> np.ndarray:
    """前缀积 ``P[i] = q[0] ⊗ … ⊗ q[i]``，向量化递归倍增，O(N) 次运算、O(log N) 层。"""

    q = np.asarray(q, dtype=np.float64)
    n = len(q)
    if n == 0:
        return q.copy()
    if n == 1:
        return q.copy()
    m = n // 2
    paired = normalize_quaternions(quat_multiply(q[0 : 2 * m : 2], q[1 : 2 * m : 2]))
    sub = prefix_products(paired)  # sub[k] = q0 … q_{2k+1}
    out = np.empty_like(q)
    out[0] = q[0]
    out[1 : 2 * m : 2] = sub
    if n > 2:
        idx = np.arange(2, n, 2)
        out[idx] = normalize_quaternions(quat_multiply(sub[idx // 2 - 1], q[idx]))
    return out


# ---------------------------------------------------------------------------
# 时间与插值
# ---------------------------------------------------------------------------

def segment_ids(time: np.ndarray, gap: float) -> np.ndarray:
    """按缺口（相邻间隔 > ``gap``）切分连续段，返回每个样本的段编号。"""

    time = np.asarray(time, dtype=np.float64)
    if len(time) == 0:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate([[0], np.cumsum(np.diff(time) > gap)])


def interpolate_orientation(
    pose_time: np.ndarray,
    q_wxyz: np.ndarray,
    query_time: np.ndarray,
    pose_valid: Optional[np.ndarray] = None,
    gap: float = 0.05,
) -> tuple:
    """在 ``query_time`` 上 SLERP 参考姿态。

    返回 ``(rotation, mask)``：``mask`` 标记可插值的查询点（落在位姿时间范围内、两侧位姿都有效且
    两侧间隔不超过 ``gap``）；``rotation`` 只含 ``mask`` 为真的点，``mask`` 全假时为 ``None``。
    """

    pose_time = np.asarray(pose_time, dtype=np.float64)
    query_time = np.asarray(query_time, dtype=np.float64)
    valid = (
        np.ones(len(pose_time), dtype=bool) if pose_valid is None else np.asarray(pose_valid, bool)
    )
    valid = valid & np.isfinite(q_wxyz).all(axis=1) & (np.linalg.norm(q_wxyz, axis=1) > 0.5)
    right = np.searchsorted(pose_time, query_time, side="left")
    right = np.clip(right, 1, len(pose_time) - 1)
    left = right - 1
    exact = pose_time[right] == query_time
    left = np.where(exact, right, left)
    in_range = (query_time >= pose_time[0]) & (query_time <= pose_time[-1])
    span = pose_time[right] - pose_time[left]
    mask = in_range & valid[left] & valid[right] & (span <= gap)
    if not mask.any():
        return None, mask
    rot_all = as_rotation(np.where(valid[:, None], q_wxyz, [1.0, 0.0, 0.0, 0.0]))
    if np.array_equal(pose_time, query_time):
        return rot_all[mask], mask
    slerp = Slerp(pose_time, rot_all)
    return slerp(query_time[mask]), mask


# ---------------------------------------------------------------------------
# 角速度、安装旋转与时延
# ---------------------------------------------------------------------------

def reference_angular_velocity(
    pose_time: np.ndarray,
    q_wxyz: np.ndarray,
    grid: np.ndarray,
    delta: float,
    pose_valid=None,
    gap=0.05,
) -> tuple:
    """在 ``grid`` 上用 ``[t, t+delta]`` 的参考相对姿态差分出机体系角速度 (rad/s)。"""

    r0, m0 = interpolate_orientation(pose_time, q_wxyz, grid, pose_valid, gap)
    r1, m1 = interpolate_orientation(pose_time, q_wxyz, grid + delta, pose_valid, gap)
    both = m0 & m1
    omega = np.full((len(grid), 3), np.nan)
    if both.any():
        idx0 = np.cumsum(m0) - 1
        idx1 = np.cumsum(m1) - 1
        rel = r0[idx0[both]].inv() * r1[idx1[both]]
        omega[both] = rel.as_rotvec() / delta
    return omega, both


def mean_gyro_on_intervals(imu_time, gyro, start, delta, gap=0.05) -> tuple:
    """陀螺在 ``[start, start+delta]`` 上的平均值（梯形积分 / 时长）；跨缺口的区间无效。"""

    imu_time = np.asarray(imu_time, dtype=np.float64)
    gyro = np.asarray(gyro, dtype=np.float64)
    dt = np.diff(imu_time)
    area = np.concatenate(
        [np.zeros((1, 3)), np.cumsum(0.5 * (gyro[1:] + gyro[:-1]) * dt[:, None], axis=0)]
    )
    gaps = np.concatenate([[0], np.cumsum(dt > gap)])

    def integral(t):
        k = np.clip(np.searchsorted(imu_time, t, side="right") - 1, 0, len(imu_time) - 2)
        frac = ((t - imu_time[k]) / np.maximum(dt[k], 1e-12))[:, None]
        return area[k] + frac * (area[k + 1] - area[k]), k

    a0, k0 = integral(start)
    a1, k1 = integral(start + delta)
    ok = (
        (start >= imu_time[0])
        & (start + delta <= imu_time[-1])
        & (gaps[k0] == gaps[np.minimum(k1 + 1, len(gaps) - 1)])
    )
    return (a1 - a0) / delta, ok


@dataclass
class MountingEstimate:
    rotation: Rotation  # 把 IMU 系向量旋到参考机体系：omega_ref ≈ R · omega_imu
    rms_residual: float  # rad/s
    n: int
    angle_from_prior_deg: float = float("nan")


def estimate_mounting_rotation(
    imu_time,
    gyro,
    pose_time,
    q_wxyz,
    *,
    pose_valid=None,
    delta=0.05,
    min_rate=0.2,
    gap=0.05,
    prior: Optional[Rotation] = None,
) -> Optional[MountingEstimate]:
    """用陀螺与参考姿态角速度做常值旋转对齐（Wahba 问题），估计 IMU→参考机体系的安装旋转。"""

    t0 = max(imu_time[0], pose_time[0])
    t1 = min(imu_time[-1], pose_time[-1]) - delta
    if t1 <= t0:
        return None
    grid = np.arange(t0, t1, delta)
    w_ref, m_ref = reference_angular_velocity(pose_time, q_wxyz, grid, delta, pose_valid, gap)
    w_imu, m_imu = mean_gyro_on_intervals(imu_time, gyro, grid, delta, gap)
    ok = m_ref & m_imu & np.isfinite(w_imu).all(1)
    ok &= (np.linalg.norm(w_ref, axis=1) > min_rate) & (np.linalg.norm(w_ref, axis=1) < 20.0)
    if ok.sum() < 20:
        return None
    rot, _ = Rotation.align_vectors(w_ref[ok], w_imu[ok])
    resid = w_ref[ok] - rot.apply(w_imu[ok])
    est = MountingEstimate(rot, float(np.sqrt((resid**2).sum(1).mean())), int(ok.sum()))
    if prior is not None:
        est.angle_from_prior_deg = float(np.degrees((prior.inv() * rot).magnitude()))
    return est


def estimate_world_tilt(
    q_tilted: np.ndarray,
    q_level: np.ndarray,
    mask: Optional[np.ndarray] = None,
    outlier_deg: float = 5.0,
) -> tuple:
    """估计“参考 A 的世界系”相对重力的倾斜，重力方向取自同一机体上的水平参考 B（如 VIO）。

    对每个时刻，``u(t) = R_A(t) · R_B(t)ᵀ · e_z`` 是 B 认为的“上”方向在 A 世界系中的表示；
    B 的偏航漂移不影响 ``u``。返回 ``(leveling_rotation, tilt_deg, spread_deg)``：
    ``leveling_rotation`` 是把平均 ``u`` 转到 ``e_z`` 的最小旋转
    （左乘到 A 的姿态与位置上即可调平）。
    """

    ok = np.isfinite(q_tilted).all(1) & np.isfinite(q_level).all(1)
    if mask is not None:
        ok &= np.asarray(mask, bool)
    if ok.sum() < 10:
        return None, float("nan"), float("nan")
    ra = as_rotation(normalize_quaternions(q_tilted[ok]))
    rb = as_rotation(normalize_quaternions(q_level[ok]))
    up = ra.apply(rb.inv().apply(np.array([0.0, 0.0, 1.0])))
    mean = up.mean(0)
    mean /= np.linalg.norm(mean)
    ang = np.degrees(np.arccos(np.clip(up @ mean, -1, 1)))
    keep = ang < outlier_deg
    if keep.sum() >= 10:
        mean = up[keep].mean(0)
        mean /= np.linalg.norm(mean)
        ang = np.degrees(np.arccos(np.clip(up @ mean, -1, 1)))
    axis = np.cross(mean, [0.0, 0.0, 1.0])
    angle = float(np.arctan2(np.linalg.norm(axis), mean[2]))
    if np.linalg.norm(axis) < 1e-12:
        level = Rotation.identity()
    else:
        level = Rotation.from_rotvec(axis / np.linalg.norm(axis) * angle)
    return level, float(np.degrees(angle)), float(np.median(ang))


def estimate_gyro_bias(
    imu_time,
    gyro,
    pose_time,
    q_wxyz,
    *,
    imu_valid=None,
    pose_valid=None,
    window=1.0,
    gap=0.05,
    iterations=3,
) -> np.ndarray:
    """相对参考姿态的常值陀螺零偏（rad/s）。

    在 ``window`` 秒的不重叠窗口上比较陀螺积分与参考相对姿态：带零偏 ``b`` 的积分满足
    ``ΔR_gyro ≈ ΔR_ref · Exp(b·T)``，故残差旋转向量的中位数给出 ``−b·T``；
    迭代数次以消除大转动下的近似误差。
    仅用于诊断“陀螺与参考不一致”能否被常值零偏解释；不用于修改存储的 IMU。
    """

    gyro = np.asarray(gyro, dtype=np.float64)
    bias = np.zeros(3)
    for _ in range(iterations):
        err, dur = gyro_window_residuals(
            imu_time,
            gyro - bias,
            pose_time,
            q_wxyz,
            imu_valid=imu_valid,
            pose_valid=pose_valid,
            window=window,
            gap=gap,
        )
        if len(err) < 5:
            return np.full(3, np.nan)
        # err = ΔR_gyroᵀ ΔR_ref ≈ Exp(−δb·T)
        rotvec = as_rotation(err).as_rotvec()
        delta = -np.median(rotvec / dur[:, None], axis=0)
        bias = bias + delta
        if np.linalg.norm(delta) < 1e-5:
            break
    return bias


def estimate_time_offset(
    imu_time,
    gyro,
    pose_time,
    q_wxyz,
    *,
    pose_valid=None,
    max_lag=0.5,
    step=0.01,
    smooth=0.1,
    gap=0.05,
    max_points=60000,
) -> tuple:
    """用三轴角速度向量的互相关估计 IMU 相对参考位姿的时延。

    两路角速度先做 ``smooth`` 秒滑动平均（抑制参考姿态的高频抖动），
    再在 ``±max_lag`` 内以 ``step`` 分辨率搜索峰值。
    返回 ``(τ, corr)``：``imu_time + τ`` 与参考时钟最一致（负值表示 IMU 时间戳比参考晚），
    ``corr`` 是峰值处的归一化相关系数（偏低时 ``τ`` 不可信）。
    """

    from scipy.ndimage import uniform_filter1d

    imu_time = np.asarray(imu_time, dtype=np.float64)
    gyro = np.asarray(gyro, dtype=np.float64)
    t0 = max(imu_time[0], pose_time[0]) + max_lag
    t1 = min(imu_time[-1], pose_time[-1]) - max_lag - step
    if t1 - t0 < 10.0:
        return float("nan"), float("nan")
    grid = np.arange(t0, t1, step)
    if len(grid) > max_points:
        grid = grid[: max_points]
    w_ref, m_ref = reference_angular_velocity(pose_time, q_wxyz, grid, step, pose_valid, gap)
    if m_ref.sum() < 100:
        return float("nan"), float("nan")
    size = max(1, int(round(smooth / step)))
    w_ref = np.where(m_ref[:, None], w_ref, 0.0)
    w_ref, m_ref = _masked_moving_average(w_ref, m_ref, size)
    m_ref &= np.isfinite(w_ref).all(1)
    b = w_ref[m_ref] - w_ref[m_ref].mean(0)
    bb = float((b * b).sum())
    # 陀螺先在细网格上平滑一次，再按时移插值
    fine_t = np.arange(imu_time[0], imu_time[-1], step)
    fine_g = uniform_filter1d(
        np.stack([np.interp(fine_t, imu_time, gyro[:, k]) for k in range(3)], 1), size, axis=0
    )
    best, best_lag = -np.inf, float("nan")
    for lag in np.arange(-max_lag, max_lag + step / 2, step):
        # 参考角速度代表区间 [t, t+step] 的中点
        tq = grid[m_ref] + 0.5 * step - lag
        g = np.stack([np.interp(tq, fine_t, fine_g[:, k]) for k in range(3)], axis=1)
        a = g - g.mean(0)
        c = float((a * b).sum() / (np.sqrt((a * a).sum() * bb) + 1e-12))
        if c > best:
            best, best_lag = c, float(round(lag, 6))
    return best_lag, best


# ---------------------------------------------------------------------------
# 物理自检
# ---------------------------------------------------------------------------

def gyro_window_errors(
    imu_time, gyro, pose_time, q_wxyz, *, imu_valid=None, pose_valid=None, window=10.0, gap=0.05
) -> np.ndarray:
    """不重叠的 ``window`` 秒窗口内，陀螺积分相对姿态与参考相对姿态之间的角度误差（度）。"""

    err, _ = gyro_window_residuals(imu_time, gyro, pose_time, q_wxyz, imu_valid=imu_valid,
                                   pose_valid=pose_valid, window=window, gap=gap)
    return quat_angle_deg(err) if len(err) else np.zeros(0)


def gyro_window_residuals(
    imu_time, gyro, pose_time, q_wxyz, *, imu_valid=None, pose_valid=None, window=10.0, gap=0.05
) -> tuple:
    """返回每个窗口的残差旋转 ``ΔR_gyroᵀ·ΔR_ref``（wxyz，机体系）与窗口实际时长（秒）。"""

    empty = (np.zeros((0, 4)), np.zeros(0))
    imu_time = np.asarray(imu_time, dtype=np.float64)
    gyro = np.asarray(gyro, dtype=np.float64)
    m = len(imu_time)
    if m < 3:
        return empty
    dt = np.diff(imu_time)
    inc = rotvec_to_quat(0.5 * (gyro[1:] + gyro[:-1]) * dt[:, None])
    cum = np.concatenate([[[1.0, 0.0, 0.0, 0.0]], prefix_products(inc)])  # cum[j] = 0→j 的积分
    bad = dt > gap
    if imu_valid is not None:
        iv = np.asarray(imu_valid, bool)
        bad = bad | ~iv[1:] | ~iv[:-1]
    bad_cum = np.concatenate([[0], np.cumsum(bad)])
    t_start = max(imu_time[0], pose_time[0])
    t_end = min(imu_time[-1], pose_time[-1])
    starts = np.arange(t_start, t_end - window, window)
    if len(starts) == 0:
        return empty
    a = np.searchsorted(imu_time, starts, side="left")
    b = np.searchsorted(imu_time, starts + window, side="right") - 1
    ok = (b > a) & (bad_cum[b] - bad_cum[a] == 0)
    if not ok.any():
        return empty
    a, b = a[ok], b[ok]
    r_a, m_a = interpolate_orientation(pose_time, q_wxyz, imu_time[a], pose_valid, gap)
    r_b, m_b = interpolate_orientation(pose_time, q_wxyz, imu_time[b], pose_valid, gap)
    both = m_a & m_b
    if not both.any():
        return empty
    ia = (np.cumsum(m_a) - 1)[both]
    ib = (np.cumsum(m_b) - 1)[both]
    ref_rel = from_rotation(r_a[ia].inv() * r_b[ib])
    gyr_rel = quat_multiply(quat_conjugate(cum[a[both]]), cum[b[both]])
    err = quat_multiply(
        quat_conjugate(normalize_quaternions(gyr_rel)), normalize_quaternions(ref_rel)
    )
    return normalize_quaternions(err), imu_time[b[both]] - imu_time[a[both]]


def horizontal_speeds(
    pose_time, position, *, pose_valid=None, step=1.0, gap=0.05, max_points=20000
) -> np.ndarray:
    """参考位置在 ``step`` 秒间隔上的水平速度样本（m/s），不跨缺口。"""

    pose_time = np.asarray(pose_time, dtype=np.float64)
    position = np.asarray(position, dtype=np.float64)
    valid = np.isfinite(position).all(1)
    if pose_valid is not None:
        valid &= np.asarray(pose_valid, bool)
    seg = segment_ids(pose_time, gap)
    idx = np.flatnonzero(valid)
    if len(idx) < 2:
        return np.zeros(0)
    if len(idx) > max_points:
        idx = idx[:: int(np.ceil(len(idx) / max_points))]
    j = np.searchsorted(pose_time, pose_time[idx] + step, side="left")
    ok = j < len(pose_time)
    idx, j = idx[ok], j[ok]
    ok = valid[j] & (seg[j] == seg[idx])
    idx, j = idx[ok], j[ok]
    dtt = pose_time[j] - pose_time[idx]
    ok = (dtt > 0.5 * step) & (dtt < 1.5 * step)
    idx, j, dtt = idx[ok], j[ok], dtt[ok]
    return np.linalg.norm(position[j, :2] - position[idx, :2], axis=1) / dtt


def gravity_mean(
    imu_time,
    accelerometer,
    pose_time,
    q_wxyz,
    *,
    imu_valid=None,
    pose_valid=None,
    gap=0.05,
    max_points=200000,
):
    """机体系比力经参考姿态旋到世界系后的均值向量，以及参与平均的样本数。"""

    imu_time = np.asarray(imu_time, dtype=np.float64)
    acc = np.asarray(accelerometer, dtype=np.float64)
    sel = np.isfinite(acc).all(1)
    if imu_valid is not None:
        sel &= np.asarray(imu_valid, bool)
    idx = np.flatnonzero(sel)
    if len(idx) > max_points:
        idx = idx[:: int(np.ceil(len(idx) / max_points))]
    rot, mask = interpolate_orientation(pose_time, q_wxyz, imu_time[idx], pose_valid, gap)
    if not mask.any():
        return np.full(3, np.nan), 0
    world = rot.apply(acc[idx[mask]])
    return world.mean(axis=0), int(mask.sum())


def _interp_on_grid(time, values, valid, grid, gap):
    """分段线性插值到 ``grid``；跨缺口或两侧样本无效的网格点标记为无效。"""

    right = np.clip(np.searchsorted(time, grid, side="left"), 1, len(time) - 1)
    left = right - 1
    span = time[right] - time[left]
    ok = (grid >= time[0]) & (grid <= time[-1]) & (span <= gap) & valid[left] & valid[right]
    frac = np.clip((grid - time[left]) / np.maximum(span, 1e-12), 0.0, 1.0)[:, None]
    out = values[left] + frac * (values[right] - values[left])
    return out, ok


def _masked_moving_average(x: np.ndarray, mask: np.ndarray, size: int) -> tuple:
    from scipy.ndimage import uniform_filter1d

    w = uniform_filter1d(mask.astype(float), size, mode="constant")
    s = uniform_filter1d(np.where(mask[:, None], x, 0.0), size, axis=0, mode="constant")
    return s / np.maximum(w, 1e-12)[:, None], w > 0.5


def acceleration_consistency(
    raw,
    *,
    half_width: float = 0.25,
    detrend: float = 2.0,
    rate: float = 100.0,
    gap: float = 0.05,
    gravity: float = STANDARD_GRAVITY,
    accelerometer=None,
    time_shift: float = 0.0,
    max_points: int = 400000,
) -> dict:
    """比较参考位置的二阶中心差分与世界系比力减重力。

    ``(p(t+h) − 2p(t) + p(t−h)) / h²`` 等于加速度与半宽 ``h`` 的归一化三角核的卷积，
    所以把世界系 IMU 加速度用同一核平滑后两边可直接比较；再各自减去 ``detrend`` 秒滑动平均，
    去掉重力残差与零偏泄漏等低频成分（保留 0.5–4 Hz 的步态动态）。

    返回残差 RMS、信号 RMS、二者之比 ``acc_consistency_ratio``，以及最小二乘比例
    ``acc_scale``（``a_pos ≈ k·a_imu``；位置单位与时间同步正确时 k≈1，时移 0.1 s 即显著下降）。
    ``time_shift`` 把 IMU 时间戳平移后再比较（用于时延确认）。
    """

    nan = {"acc_resid_rms": float("nan"), "acc_signal_rms": float("nan"),
           "acc_consistency_ratio": float("nan"), "acc_scale": float("nan")}
    acc = np.asarray(raw.accelerometer if accelerometer is None else accelerometer, dtype=float)
    imu_time = np.asarray(raw.imu_time, dtype=float) + time_shift
    t0 = max(imu_time[0], raw.pose_time[0]) + half_width
    t1 = min(imu_time[-1], raw.pose_time[-1]) - half_width
    if t1 - t0 < max(4 * detrend, 10.0):
        return nan
    grid = np.arange(t0, t1, 1.0 / rate)
    if len(grid) > max_points:
        grid = grid[:max_points]
    imu_valid = np.isfinite(acc).all(1)
    if raw.imu_valid is not None:
        imu_valid &= np.asarray(raw.imu_valid, bool)
    acc_g, ok_acc = _interp_on_grid(imu_time, acc, imu_valid, grid, gap)
    rot, ok_rot = interpolate_orientation(raw.pose_time, raw.orientation, grid, raw.pose_valid, gap)
    ok_imu = ok_acc & ok_rot
    if ok_imu.sum() < 100:
        return nan
    world = np.zeros((len(grid), 3))
    world[ok_rot] = rot.apply(acc_g[ok_rot])
    world[:, 2] -= gravity
    world[~ok_imu] = 0.0
    k = max(1, int(round(half_width * rate)))
    kernel = np.concatenate([np.arange(1, k + 1), np.arange(k - 1, 0, -1)]).astype(float)
    kernel /= kernel.sum()
    a_imu = np.stack([np.convolve(world[:, i], kernel, mode="same") for i in range(3)], axis=1)
    ok_imu = ok_imu & (np.convolve((~ok_imu).astype(float), np.ones(len(kernel)), mode="same") == 0)

    position = np.asarray(raw.position, dtype=float)
    pos_valid = np.isfinite(position).all(1)
    if raw.pose_valid is not None:
        pos_valid &= np.asarray(raw.pose_valid, bool)
    pc, vc = _interp_on_grid(raw.pose_time, position, pos_valid, grid, gap)
    pm, vm = _interp_on_grid(raw.pose_time, position, pos_valid, grid - half_width, gap)
    pp, vp = _interp_on_grid(raw.pose_time, position, pos_valid, grid + half_width, gap)
    seg = segment_ids(raw.pose_time, gap)
    lo = np.clip(np.searchsorted(raw.pose_time, grid - half_width), 0, len(seg) - 1)
    hi = np.clip(np.searchsorted(raw.pose_time, grid + half_width) - 1, 0, len(seg) - 1)
    ok_pos = vc & vm & vp & (seg[lo] == seg[hi])
    a_pos = np.where(ok_pos[:, None], (pp - 2 * pc + pm) / half_width**2, 0.0)

    ok = ok_imu & ok_pos
    if detrend:
        size = max(3, int(round(detrend * rate)))
        ma_imu, ok1 = _masked_moving_average(a_imu, ok, size)
        ma_pos, ok2 = _masked_moving_average(a_pos, ok, size)
        a_imu, a_pos = a_imu - ma_imu, a_pos - ma_pos
        ok &= ok1 & ok2
    if ok.sum() < 200:
        return nan
    a_imu, a_pos = a_imu[ok], a_pos[ok]
    resid = float(np.sqrt(((a_pos - a_imu) ** 2).sum(1).mean()))
    signal = float(np.sqrt((a_imu**2).sum(1).mean()))
    scale = float((a_pos * a_imu).sum() / max((a_imu * a_imu).sum(), 1e-12))
    return {
        "acc_resid_rms": resid,
        "acc_signal_rms": signal,
        "acc_consistency_ratio": resid / signal if signal > 0 else float("nan"),
        "acc_scale": scale,
    }


def short_window_gyro_error(
    raw, time_shift: float = 0.0, window: float = 1.0, gap: float = 0.05
) -> float:
    """去除常值零偏后，``window`` 秒窗口陀螺积分与参考相对姿态误差的中位数（度）；
    IMU 时间可平移。
    """

    imu_time = np.asarray(raw.imu_time, dtype=np.float64) + time_shift
    bias = estimate_gyro_bias(
        imu_time,
        raw.gyroscope,
        raw.pose_time,
        raw.orientation,
        imu_valid=raw.imu_valid,
        pose_valid=raw.pose_valid,
        gap=gap,
    )
    if not np.isfinite(bias).all():
        return float("nan")
    errs = gyro_window_errors(
        imu_time,
        np.asarray(raw.gyroscope) - bias,
        raw.pose_time,
        raw.orientation,
        imu_valid=raw.imu_valid,
        pose_valid=raw.pose_valid,
        window=window,
        gap=gap,
    )
    return float(np.median(errs)) if len(errs) else float("nan")


def correct_time_offset_if_confirmed(
    raw,
    stats: dict,
    *,
    min_offset: float = 0.03,
    min_corr: float = 0.7,
    min_gain: float = 0.2,
    confirm: str = "acceleration",
) -> bool:
    """两个独立证据一致时修正参考位姿时间戳，返回是否修正。

    证据 1：角速度互相关给出 ``τ``（``|τ| ≥ min_offset`` 且峰值相关 ``≥ min_corr``）；
    证据 2（``confirm``）：

    * ``"acceleration"``：IMU 平移 ``τ`` 后，加速度一致性残差比相对下降至少 ``min_gain``，
      且比例因子更接近 1；
    * ``"gyro"``：IMU 平移 ``τ`` 后，1 s 窗口去偏陀螺误差中位数相对下降至少 ``min_gain``
      （用于参考位置被平滑、加速度一致性无信息的数据集）。

    修正方式：``pose_time ← pose_time − τ``（保持 IMU 时钟不变），写入 ``notes``。
    """

    tau = stats.get("time_offset_s", float("nan"))
    corr = stats.get("time_offset_corr", float("nan"))
    if not (np.isfinite(tau) and np.isfinite(corr)) or abs(tau) < min_offset or corr < min_corr:
        return False
    if confirm == "acceleration":
        base = acceleration_consistency(raw)
        shifted = acceleration_consistency(raw, time_shift=tau)
        r0, r1 = base["acc_consistency_ratio"], shifted["acc_consistency_ratio"]
        if not (np.isfinite(r0) and np.isfinite(r1)):
            return False
        closer = abs(shifted["acc_scale"] - 1.0) <= abs(base["acc_scale"] - 1.0)
        ok = r1 <= (1.0 - min_gain) * r0 and closer
        evidence = (f"acceleration residual ratio {r0:.2f} -> {r1:.2f}, "
                    f"scale {base['acc_scale']:.2f} -> {shifted['acc_scale']:.2f}")
    elif confirm == "gyro":
        e0, e1 = short_window_gyro_error(raw), short_window_gyro_error(raw, time_shift=tau)
        if not (np.isfinite(e0) and np.isfinite(e1)):
            return False
        ok = e1 <= (1.0 - min_gain) * e0
        evidence = f"debiased 1 s gyro/reference error median {e0:.2f} -> {e1:.2f} deg"
    else:
        raise ValueError(f"unknown confirmation mode {confirm!r}")
    if not ok:
        return False
    raw.pose_time = np.asarray(raw.pose_time, dtype=np.float64) - tau
    raw.notes.append(f"time offset corrected: pose_time shifted by {-tau:+.3f} s "
                     f"(angular-rate cross-correlation {corr:.2f}; {evidence})")
    return True


def time_overlap(imu_time, pose_time) -> dict:
    lo = max(imu_time[0], pose_time[0])
    hi = min(imu_time[-1], pose_time[-1])
    overlap = max(0.0, hi - lo)
    shorter = min(imu_time[-1] - imu_time[0], pose_time[-1] - pose_time[0])
    return {
        "overlap_s": float(overlap),
        "overlap_ratio": float(overlap / shorter) if shorter > 0 else 0.0,
    }


def physical_checks(
    raw,
    *,
    gravity: float = STANDARD_GRAVITY,
    window: float = 10.0,
    gap: float = 0.05,
    gyroscope=None,
    accelerometer=None,
    time_offset: bool = True,
) -> dict:
    """对一条 ``RawSequence`` 计算物理自检统计量（不修改输入）。

    ``gyroscope`` / ``accelerometer`` 可替换输入（例如仅用于诊断的零偏补偿版本）。
    """

    gyro = raw.gyroscope if gyroscope is None else gyroscope
    acc = raw.accelerometer if accelerometer is None else accelerometer
    stats = {}
    stats.update(time_overlap(raw.imu_time, raw.pose_time))
    stats["imu_duration_s"] = float(raw.imu_time[-1] - raw.imu_time[0])
    stats["pose_duration_s"] = float(raw.pose_time[-1] - raw.pose_time[0])
    stats["imu_rate_hz"] = float(1.0 / np.median(np.diff(raw.imu_time)))
    stats["imu_max_gap_s"] = float(np.max(np.diff(raw.imu_time)))
    stats["imu_nonmonotonic"] = int(np.sum(np.diff(raw.imu_time) <= 0))
    stats["pose_nonmonotonic"] = int(np.sum(np.diff(raw.pose_time) <= 0))
    qn = np.linalg.norm(raw.orientation, axis=1)
    stats["quat_norm_dev_max"] = float(np.nanmax(np.abs(qn - 1.0)))
    mean, n = gravity_mean(
        raw.imu_time,
        acc,
        raw.pose_time,
        raw.orientation,
        imu_valid=raw.imu_valid,
        pose_valid=raw.pose_valid,
        gap=gap,
    )
    stats["gravity_world_mean"] = [float(x) for x in mean]
    stats["gravity_error"] = (
        float(np.linalg.norm(mean - np.array([0.0, 0.0, gravity]))) if n else float("nan")
    )
    stats["gravity_tilt_deg"] = (
        float(np.degrees(np.arctan2(np.hypot(mean[0], mean[1]), mean[2]))) if n else float("nan")
    )
    speeds = horizontal_speeds(raw.pose_time, raw.position, pose_valid=raw.pose_valid, gap=gap)
    stats["speed_median"] = float(np.median(speeds)) if len(speeds) else float("nan")
    stats["speed_p95"] = float(np.percentile(speeds, 95)) if len(speeds) else float("nan")
    errs = gyro_window_errors(
        raw.imu_time,
        gyro,
        raw.pose_time,
        raw.orientation,
        imu_valid=raw.imu_valid,
        pose_valid=raw.pose_valid,
        window=window,
        gap=gap,
    )
    stats["gyro_windows"] = int(len(errs))
    stats["gyro_window_error_median_deg"] = float(np.median(errs)) if len(errs) else float("nan")
    stats["gyro_window_error_p90_deg"] = (
        float(np.percentile(errs, 90)) if len(errs) else float("nan")
    )
    # 诊断：参考姿态所隐含的常值陀螺零偏，以及去除它之后的窗口误差
    bias = estimate_gyro_bias(
        raw.imu_time,
        gyro,
        raw.pose_time,
        raw.orientation,
        imu_valid=raw.imu_valid,
        pose_valid=raw.pose_valid,
        gap=gap,
    )
    stats["gyro_bias_vs_ref"] = [float(x) for x in bias]
    if np.isfinite(bias).all() and len(errs):
        errs_db = gyro_window_errors(
            raw.imu_time,
            np.asarray(gyro) - bias,
            raw.pose_time,
            raw.orientation,
            imu_valid=raw.imu_valid,
            pose_valid=raw.pose_valid,
            window=window,
            gap=gap,
        )
        stats["gyro_window_error_median_deg_debiased"] = (
            float(np.median(errs_db)) if len(errs_db) else float("nan")
        )
    else:
        stats["gyro_window_error_median_deg_debiased"] = float("nan")
    stats.update(acceleration_consistency(raw, gap=gap, gravity=gravity, accelerometer=acc))
    if time_offset:
        lag, corr = estimate_time_offset(
            raw.imu_time, np.asarray(gyro), raw.pose_time, raw.orientation,
            pose_valid=raw.pose_valid, gap=gap,
        )
        stats["time_offset_s"] = lag
        stats["time_offset_corr"] = corr
    return stats


def evaluate_checks(stats: dict, limits: Optional[dict] = None, skip: tuple = ()) -> tuple:
    """返回 ``(failures, warnings)`` 两个字符串列表。NaN 统计量视为失败。

    ``skip`` 可包含 ``"acceleration_consistency"``：参考位置被数据集作者平滑时，该诊断没有信息量，
    由转换器在数据集文档中统一说明，而不是逐序列告警。
    """

    lim = dict(LIMITS)
    if limits:
        lim.update(limits)
    failures, warnings = [], []

    def bad(value, upper):
        return not np.isfinite(value) or value > upper

    if stats["imu_nonmonotonic"] or stats["pose_nonmonotonic"]:
        failures.append("non-monotonic timestamps")
    if stats["overlap_ratio"] < lim["overlap_min"]:
        failures.append(f"IMU/pose overlap {stats['overlap_ratio']:.2f} < {lim['overlap_min']}")
    if bad(stats["gravity_error"], lim["gravity_error_max"]):
        failures.append(
            f"gravity check {stats['gravity_error']:.3f} m/s^2 > {lim['gravity_error_max']}"
        )
    elif stats["gravity_error"] > SOFT_LIMITS["gravity_error_warn"]:
        warnings.append(
            f"gravity check {stats['gravity_error']:.3f} m/s^2 "
            f"(warn > {SOFT_LIMITS['gravity_error_warn']})"
        )
    gyro_limit = lim["gyro_window_error_median_max"]
    debiased = stats.get("gyro_window_error_median_deg_debiased", float("nan"))
    bias = np.asarray(stats.get("gyro_bias_vs_ref", [np.nan] * 3), dtype=float)
    bias_norm = float(np.linalg.norm(bias)) if np.isfinite(bias).all() else float("nan")
    if stats["gyro_windows"] < lim["min_windows"]:
        failures.append("no gap-free 10 s window for the gyro check")
    elif bad(stats["gyro_window_error_median_deg"], gyro_limit):
        message = (
            f"gyro/reference 10 s rotation error median "
            f"{stats['gyro_window_error_median_deg']:.2f} deg > {gyro_limit}"
        )
        # 常值零偏不能掩盖轴/方向/时间错误：去偏后通过且零偏量级合理时，归因于未标定陀螺零偏
        if (
            np.isfinite(debiased)
            and debiased <= gyro_limit
            and bias_norm <= lim.get("gyro_bias_max", 0.1)
        ):
            warnings.append(
                message
                + f"; explained by a constant gyro bias |b|={bias_norm:.4f} rad/s "
                f"(debiased median {debiased:.2f} deg; bias NOT removed from stored IMU)"
            )
        else:
            failures.append(message + (f" (debiased {debiased:.2f} deg, |b|={bias_norm:.4f} rad/s)"
                                       if np.isfinite(debiased) else ""))
    elif stats["gyro_window_error_median_deg"] > SOFT_LIMITS["gyro_window_error_median_warn"]:
        warnings.append(
            f"gyro/reference 10 s rotation error median "
            f"{stats['gyro_window_error_median_deg']:.2f} deg "
            f"(warn > {SOFT_LIMITS['gyro_window_error_median_warn']})"
        )
    if bad(stats["speed_median"], lim["speed_median_max"]):
        failures.append(
            f"horizontal speed median {stats['speed_median']:.2f} m/s > {lim['speed_median_max']}"
        )
    elif stats["speed_median"] < SOFT_LIMITS["speed_median_min"]:
        warnings.append(f"horizontal speed median {stats['speed_median']:.3f} m/s: mostly static")
    elif stats["speed_median"] > SOFT_LIMITS["speed_median_walk_max"]:
        warnings.append(
            f"horizontal speed median {stats['speed_median']:.2f} m/s: "
            f"fast motion (running pace)"
        )
    if bad(stats["speed_p95"], lim["speed_p95_max"]):
        failures.append(
            f"horizontal speed p95 {stats['speed_p95']:.2f} m/s > {lim['speed_p95_max']}"
        )
    ratio = stats.get("acc_consistency_ratio", float("nan"))
    scale = stats.get("acc_scale", float("nan"))
    if "acceleration_consistency" not in skip and np.isfinite(ratio) and (
            ratio > SOFT_LIMITS["acc_consistency_ratio_warn"]
            or not SOFT_LIMITS["acc_scale_min"] <= scale <= SOFT_LIMITS["acc_scale_max"]):
        warnings.append(
            f"reference acceleration (position 2nd difference) vs IMU: "
            f"residual ratio {ratio:.2f}, scale {scale:.2f}"
        )
    offset = stats.get("time_offset_s", float("nan"))
    corr = stats.get("time_offset_corr", float("nan"))
    if offset is not None and np.isfinite(offset):
        if np.isfinite(corr) and corr < SOFT_LIMITS["angular_rate_corr_min"]:
            warnings.append(
                f"gyro vs reference angular-rate correlation only {corr:.2f} "
                f"(noisy reference orientation)"
            )
        elif abs(offset) > SOFT_LIMITS["time_offset_warn"]:
            warnings.append(
                f"estimated IMU-to-reference time offset {offset:+.2f} s "
                f"(angular-rate cross-correlation {corr:.2f}); not corrected"
            )
    return failures, warnings


_NOTE_PREFIX = "physical_check: "


def _round(value):
    if isinstance(value, float):
        return None if not math.isfinite(value) else round(value, 5)
    if isinstance(value, (list, tuple)):
        return [_round(v) for v in value]
    return value


def format_stats(stats: dict) -> str:
    """把统计量写成一行 JSON 注释（NaN 写作 null）。"""

    return _NOTE_PREFIX + json.dumps({k: _round(v) for k, v in stats.items()}, sort_keys=True)


def parse_stats_note(notes) -> Optional[dict]:
    """从 ``notes`` 中取回 ``format_stats`` 写入的统计量（供文档统计脚本与测试使用）。"""

    for note in notes:
        if note.startswith(_NOTE_PREFIX):
            out = json.loads(note[len(_NOTE_PREFIX):])
            return {k: (float("nan") if v is None else v) for k, v in out.items()}
    return None


def rejected_sequence(sequence_id: str, reason: str, attrs: Optional[dict] = None, notes=None):
    """构造一条被拒收的空序列（必需属性缺失时填 ``unknown``）。"""

    from .base import RAW_REQUIRED_ATTRS, RawSequence

    full = {key: "unknown" for key in RAW_REQUIRED_ATTRS}
    full.update(attrs or {})
    empty3 = np.zeros((0, 3))
    return RawSequence(sequence_id, np.zeros(0), empty3, empty3.copy(), np.zeros(0), empty3.copy(),
                       np.zeros((0, 4)), attrs=full, notes=list(notes or []), rejected=reason)


def apply_checks(raw, stats: dict, failures: list, warnings: list) -> None:
    """把自检结果写入 ``raw.notes``；有硬失败时设置 ``raw.rejected``（已有拒收原因则保留）。"""

    raw.notes.append(format_stats(stats))
    for w in warnings:
        raw.notes.append("warning: " + w)
    if failures and raw.rejected is None:
        raw.rejected = "physical check failed: " + "; ".join(failures)


# ---------------------------------------------------------------------------
# 合成运动（单元测试用）
# ---------------------------------------------------------------------------

def simulate_rig_motion(
    duration: float = 60.0, rate: float = 200.0, seed: int = 0, gravity: float = STANDARD_GRAVITY
) -> dict:
    """生成物理一致的行人式运动。

    返回时间、``body_to_world`` 姿态（wxyz）、位置、速度、机体系角速度与机体系比力
    （含重力；世界系 z 轴向上）。
    """

    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration, 1.0 / rate)
    # 平滑的偏航（行走转弯）+ 小幅俯仰/横滚（手持晃动）
    yaw = 0.3 * t + 0.8 * np.sin(0.21 * t + rng.uniform(0, 6))
    pitch = 0.25 * np.sin(1.3 * t + rng.uniform(0, 6))
    roll = 0.2 * np.sin(0.9 * t + rng.uniform(0, 6)) + 0.3
    rot = Rotation.from_euler("ZYX", np.stack([yaw, pitch, roll], axis=1))
    speed = 1.2 + 0.2 * np.sin(1.8 * 2 * np.pi * t)
    heading = 0.3 * t
    vel = np.stack(
        [speed * np.cos(heading), speed * np.sin(heading), 0.05 * np.sin(1.8 * 2 * np.pi * t)],
        axis=1,
    )
    pos = np.concatenate([np.zeros((1, 3)), np.cumsum(0.5 * (vel[1:] + vel[:-1]) / rate, axis=0)])
    acc_world = np.gradient(vel, t, axis=0)
    # 机体系角速度：由相邻姿态的相对旋转得到（区间中点值再平均到样本点）
    w_mid = (rot[:-1].inv() * rot[1:]).as_rotvec() * rate
    gyro = np.empty((len(t), 3))
    gyro[1:-1] = 0.5 * (w_mid[1:] + w_mid[:-1])
    gyro[0], gyro[-1] = w_mid[0], w_mid[-1]
    specific_force = rot.inv().apply(acc_world + np.array([0.0, 0.0, gravity]))
    return {
        "time": t,
        "orientation": from_rotation(rot),
        "position": pos,
        "velocity": vel,
        "gyroscope": gyro,
        "accelerometer": specific_force,
    }
