"""统一采样率（DESIGN 2.3）：去重、缺口掩码、插值或多相重采样、SLERP。

规则摘要：

1. 剔除重复/倒序时间戳（保留严格大于此前最大值的样本），记录数量；
2. 相邻源样本间隔超过缺口阈值的区段不跨缺口插值，对应网格点 ``valid=False``；
   每个时钟的有效阈值为 ``max(gap_threshold, 2 × 中位采样间隔)``，以兼容慢速参考时钟；
3. ``|f_src / rate − 1| ≤ tolerance``（默认 3%）：IMU 在统一网格上直接线性插值；
4. 否则：先在“名义源频率”上线性均匀化，再用 ``scipy.signal.resample_poly``
   （Kaiser 窗 FIR，抗混叠/抗镜像）变换到目标频率；滤波支撑区内受缺口影响的网格点同样置为无效；
5. 位置/速度一律线性插值，四元数一律 SLERP，输出前做符号连续化与归一化；
6. （转换流水线）输出裁剪到首个/末个 IMU 与位姿同时有效的样本，首尾孤立的短有效片段
   （短于 ``DEFAULT_EDGE_MIN_RUN``、与主体之间隔着无效区）一并裁掉，见 ``valid_extent``。

时间比较使用 ``TIME_TOL``（1 µs）容差：Unix 秒量级的 float64 时间戳分辨率约 2.4e-7 s，
纳秒级容差会把恰好落在末样本上的网格点误判为越界。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Optional

import numpy as np
from scipy.signal import resample_poly

from ..utils.geometry import quat_interp, quat_make_continuous, quat_normalize

DEFAULT_RATE = 200.0
DEFAULT_GAP_THRESHOLD = 0.05
DEFAULT_TOLERANCE = 0.03
DEFAULT_EDGE_MIN_RUN = 1.0  # s，首尾孤立有效片段短于该值时裁掉
POLY_HALF_LEN = 10  # scipy 默认 FIR 半长系数：half_len = 10 · max(up, down)
TIME_TOL = 1e-6  # s，时间戳比较容差（远小于任何采样间隔，大于 Unix 秒 float64 的分辨率）


def clean_timestamps(t: np.ndarray) -> np.ndarray:
    """返回保留样本的布尔掩码：剔除非有限、重复与倒序时间戳。"""
    t = np.asarray(t, dtype=np.float64)
    keep = np.isfinite(t)
    if not keep.any():
        return keep
    # 非有限值不参与“此前最大值”的计算
    filled = np.where(keep, t, -np.inf)
    prev_max = np.maximum.accumulate(filled)
    keep[1:] &= t[1:] > prev_max[:-1]
    return keep


def measure_rate(t: np.ndarray) -> float:
    """实测采样率：中位采样间隔的倒数。"""
    dt = np.diff(np.asarray(t, dtype=np.float64))
    dt = dt[dt > 0]
    return float(1.0 / np.median(dt)) if len(dt) else float("nan")


def effective_gap_threshold(t: np.ndarray, gap_threshold: float) -> float:
    """每个时钟的缺口阈值：不小于 2 倍中位采样间隔。"""
    dt = np.diff(t)
    return max(gap_threshold, 2.0 * float(np.median(dt))) if len(dt) else gap_threshold


def uniform_grid(t_start: float, t_end: float, rate: float) -> np.ndarray:
    """``[t_start, t_end]`` 内以 ``rate`` 采样的网格（含起点）。"""
    if t_end < t_start:
        return np.zeros(0)
    n = int(np.floor((t_end - t_start + TIME_TOL) * rate)) + 1
    return t_start + np.arange(n, dtype=np.float64) / rate


def valid_on_grid(
    t_src: np.ndarray,
    grid: np.ndarray,
    valid_src: Optional[np.ndarray] = None,
    gap_threshold: float = DEFAULT_GAP_THRESHOLD,
) -> np.ndarray:
    """网格点有效 ⇔ 两侧源样本都有效且间隔不超过阈值（恰好落在源样本上时只看该样本）。"""
    t_src = np.asarray(t_src, dtype=np.float64)
    m = len(t_src)
    valid_src = np.ones(m, dtype=bool) if valid_src is None else np.asarray(valid_src, bool)
    out = np.zeros(len(grid), dtype=bool)
    if m == 0:
        return out
    tol = TIME_TOL  # 网格由浮点运算生成，允许微秒级偏差视为“恰好落在源样本上”
    left = np.searchsorted(t_src, grid + tol, side="right") - 1
    inside = (left >= 0) & (grid <= t_src[-1] + tol)
    li = np.clip(left, 0, m - 1)
    ri = np.clip(left + 1, 0, m - 1)
    exact = inside & (np.abs(t_src[li] - grid) <= tol)
    out[exact] = valid_src[li[exact]]
    between = inside & ~exact & (left + 1 < m)
    gap_ok = (t_src[ri] - t_src[li]) <= gap_threshold + 1e-12
    out[between] = (valid_src[li] & valid_src[ri] & gap_ok)[between]
    return out


def fill_invalid(t: np.ndarray, x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """用有效样本线性插值填补无效/非有限样本（仅为数值稳定，结果由掩码标记为无效）。"""
    x = np.array(x, dtype=np.float64, copy=True)
    good = np.asarray(valid, bool) & np.all(np.isfinite(x.reshape(len(x), -1)), axis=1)
    if good.all():
        return x
    if not good.any():
        return np.zeros_like(x)
    flat = x.reshape(len(x), -1)
    for c in range(flat.shape[1]):
        flat[~good, c] = np.interp(t[~good], t[good], flat[good, c])
    return flat.reshape(x.shape)


def interp_linear(t_src: np.ndarray, x_src: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """逐列线性插值（超出范围取端点值）。"""
    x_src = np.asarray(x_src, dtype=np.float64)
    if x_src.ndim == 1:
        return np.interp(grid, t_src, x_src)
    return np.stack([np.interp(grid, t_src, x_src[:, c]) for c in range(x_src.shape[1])], axis=1)


def poly_ratio(src_rate: float, rate: float, max_denominator: int = 20) -> tuple[int, int, float]:
    """返回 ``(up, down, nominal_src_rate)``，使 ``rate / nominal = up / down`` 为简单分数。"""
    frac = Fraction(src_rate / rate).limit_denominator(max_denominator)
    if frac <= 0:
        raise ValueError(f"invalid source rate {src_rate}")
    nominal = float(rate * frac)
    ratio = Fraction(rate / nominal).limit_denominator(1000)
    return ratio.numerator, ratio.denominator, nominal


def resample_poly_to_grid(
    t_src: np.ndarray, x_src: np.ndarray, grid: np.ndarray, rate: float, src_rate: float
) -> tuple[np.ndarray, dict]:
    """名义源频率均匀化 + ``resample_poly``，输出严格对齐 ``grid``（``grid[0]`` 为相位原点）。"""
    up, down, nominal = poly_ratio(src_rate, rate)
    support = POLY_HALF_LEN * max(up, down) / (nominal * up)  # FIR 半支撑宽度（秒）
    # 若源数据早于网格起点，保留整数个 down 块作为滤波上下文，且保持输出相位与 grid 对齐
    block = down / nominal
    m = int(min(np.floor((grid[0] - t_src[0]) / block + 1e-9), np.ceil(2 * support / block) + 1))
    m = max(m, 0)
    u = uniform_grid(grid[0] - m * block, t_src[-1], nominal)
    x_u = interp_linear(t_src, x_src, u)
    y = resample_poly(x_u, up, down, axis=0, padtype="line")[m * up:]
    n = len(grid)
    if len(y) < n:  # 仅在末端可能差一个样本：用端点值补齐
        pad = np.repeat(y[-1:], n - len(y), axis=0)
        y = np.concatenate([y, pad], axis=0)
    info = {
        "method": "resample_poly",
        "up": up,
        "down": down,
        "nominal_source_rate_hz": nominal,
        "support_s": support,  # 用于把缺口影响扩展到滤波支撑区
    }
    return y[:n], info


def dilate_invalid(t: np.ndarray, valid: np.ndarray, radius: float) -> np.ndarray:
    """把无效区在时间上向两侧各扩展 ``radius`` 秒。"""
    valid = np.asarray(valid, bool)
    if valid.all() or radius <= 0:
        return valid.copy()
    bad_t = t[~valid]
    idx = np.searchsorted(bad_t, t)
    lo = np.abs(t - bad_t[np.clip(idx - 1, 0, len(bad_t) - 1)])
    hi = np.abs(bad_t[np.clip(idx, 0, len(bad_t) - 1)] - t)
    return valid & (np.minimum(lo, hi) > radius)


def resample_signal(
    t_src: np.ndarray,
    x_src: np.ndarray,
    grid: np.ndarray,
    *,
    rate: float = DEFAULT_RATE,
    valid_src: Optional[np.ndarray] = None,
    gap_threshold: float = DEFAULT_GAP_THRESHOLD,
    tolerance: float = DEFAULT_TOLERANCE,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """把 IMU 类信号重采样到 ``grid``，返回 ``(x_grid, valid_grid, info)``。"""
    t_src = np.asarray(t_src, dtype=np.float64)
    valid_src = np.ones(len(t_src), bool) if valid_src is None else np.asarray(valid_src, bool)
    finite = np.all(np.isfinite(np.asarray(x_src, dtype=np.float64).reshape(len(t_src), -1)), 1)
    valid_src = valid_src & finite
    x_filled = fill_invalid(t_src, x_src, valid_src)
    thr = effective_gap_threshold(t_src, gap_threshold)
    valid = valid_on_grid(t_src, grid, valid_src, thr)
    src_rate = measure_rate(t_src)
    if abs(src_rate / rate - 1.0) <= tolerance:
        x = interp_linear(t_src, x_filled, grid)
        info = {"method": "linear"}
    else:
        x, info = resample_poly_to_grid(t_src, x_filled, grid, rate, src_rate)
        valid = dilate_invalid(grid, valid, info["support_s"])
    info.update(source_rate_hz=src_rate, gap_threshold_s=thr)
    return x, valid, info


@dataclass
class ResampleResult:
    """``resample_streams`` 的输出：统一网格上的全部数组与处理说明。"""

    timestamp: np.ndarray  # (N,) 从 0 开始
    start_time: float  # 网格起点在源时钟上的时间
    arrays: dict  # gyroscope/accelerometer/position/orientation/[velocity]/[device_orientation]
    valid_imu: np.ndarray
    valid_pose: np.ndarray
    info: dict = field(default_factory=dict)

    @property
    def description(self) -> str:
        return self.info.get("description", "")


def resample_streams(
    imu_time: np.ndarray,
    gyroscope: np.ndarray,
    accelerometer: np.ndarray,
    pose_time: np.ndarray,
    position: np.ndarray,
    orientation: np.ndarray,
    *,
    velocity: Optional[np.ndarray] = None,
    device_orientation: Optional[np.ndarray] = None,
    imu_valid: Optional[np.ndarray] = None,
    pose_valid: Optional[np.ndarray] = None,
    rate: float = DEFAULT_RATE,
    gap_threshold: float = DEFAULT_GAP_THRESHOLD,
    tolerance: float = DEFAULT_TOLERANCE,
) -> ResampleResult:
    """把 IMU 时钟与位姿时钟上的数据统一到 ``rate`` Hz 的公共网格（取两者时间重叠区）。"""
    imu_time = np.asarray(imu_time, dtype=np.float64)
    pose_time = np.asarray(pose_time, dtype=np.float64)
    keep_i = clean_timestamps(imu_time)
    keep_p = clean_timestamps(pose_time)
    dropped = {"imu": int((~keep_i).sum()), "pose": int((~keep_p).sum())}
    if keep_i.sum() < 2 or keep_p.sum() < 2:
        raise ValueError("fewer than two usable timestamps on the IMU or pose clock")

    ti = imu_time[keep_i]
    tp = pose_time[keep_p]
    iv = np.ones(len(imu_time), bool) if imu_valid is None else np.asarray(imu_valid, bool)
    pv = np.ones(len(pose_time), bool) if pose_valid is None else np.asarray(pose_valid, bool)
    iv, pv = iv[keep_i], pv[keep_p]

    t0, t1 = max(ti[0], tp[0]), min(ti[-1], tp[-1])
    grid = uniform_grid(t0, t1, rate)
    if len(grid) < 2:
        raise ValueError(f"IMU and pose clocks do not overlap ({ti[0]:.3f}-{ti[-1]:.3f} vs "
                         f"{tp[0]:.3f}-{tp[-1]:.3f})")

    gyro, gv, info_imu = resample_signal(
        ti, np.asarray(gyroscope)[keep_i], grid, rate=rate, valid_src=iv,
        gap_threshold=gap_threshold, tolerance=tolerance)
    acc, av, _ = resample_signal(
        ti, np.asarray(accelerometer)[keep_i], grid, rate=rate, valid_src=iv,
        gap_threshold=gap_threshold, tolerance=tolerance)
    valid_imu = gv & av

    arrays = {"gyroscope": gyro, "accelerometer": acc}
    if device_orientation is not None:
        qd = np.asarray(device_orientation, dtype=np.float64)[keep_i]
        qd_ok = _quat_ok(qd)
        arrays["device_orientation"] = _interp_quat(ti, qd, qd_ok, grid)
        # 设备姿态缺失处视为 IMU 无效：该姿态只用于“设备姿态”视图
        valid_imu &= valid_on_grid(ti, grid, qd_ok, effective_gap_threshold(ti, gap_threshold))

    pos = np.asarray(position, dtype=np.float64)[keep_p]
    quat = np.asarray(orientation, dtype=np.float64)[keep_p]
    pose_ok = pv & np.all(np.isfinite(pos), 1) & _quat_ok(quat)
    thr_p = effective_gap_threshold(tp, gap_threshold)
    valid_pose = valid_on_grid(tp, grid, pose_ok, thr_p)
    arrays["position"] = interp_linear(tp, fill_invalid(tp, pos, pose_ok), grid)
    arrays["orientation"] = _interp_quat(tp, quat, pose_ok, grid)
    if velocity is not None:
        vel = np.asarray(velocity, dtype=np.float64)[keep_p]
        vel_ok = pose_ok & np.all(np.isfinite(vel), 1)
        arrays["velocity"] = interp_linear(tp, fill_invalid(tp, vel, vel_ok), grid)
        valid_pose &= valid_on_grid(tp, grid, vel_ok, thr_p)

    pose_rate = measure_rate(tp)
    if info_imu["method"] == "linear":
        imu_desc = f"imu: linear interpolation {info_imu['source_rate_hz']:.2f}->{rate:g} Hz"
    else:
        imu_desc = (
            f"imu: resample_poly up={info_imu['up']} down={info_imu['down']} (kaiser, "
            f"padtype=line) from nominal {info_imu['nominal_source_rate_hz']:g} Hz "
            f"(measured {info_imu['source_rate_hz']:.2f} Hz)"
        )
    desc = (
        f"{imu_desc}; pose: linear position + SLERP orientation from {pose_rate:.2f} Hz; "
        f"gap_threshold={gap_threshold:g}s (imu {info_imu['gap_threshold_s']:.3f}s, "
        f"pose {thr_p:.3f}s); tolerance={tolerance:g}"
    )
    info = {
        "imu": info_imu,
        "pose_rate_hz": pose_rate,
        "pose_gap_threshold_s": thr_p,
        "dropped_timestamps": dropped,
        "description": desc,
        "rate_hz": rate,
    }
    return ResampleResult(
        timestamp=np.arange(len(grid), dtype=np.float64) / rate,  # 精确均匀，避免大时间戳相减误差
        start_time=float(grid[0]),
        arrays=arrays,
        valid_imu=valid_imu,
        valid_pose=valid_pose,
        info=info,
    )


def valid_extent(valid: np.ndarray, rate: float = DEFAULT_RATE,
                 min_run: float = DEFAULT_EDGE_MIN_RUN) -> tuple[int, int]:
    """返回要保留的样本区间 ``[start, stop)``。

    从第一个长度不短于 ``min_run`` 秒的连续有效片段开始，到最后一个这样的片段结束：
    首尾的无效样本，以及首尾被无效区隔开的短有效片段都被裁掉；中间的短片段保留。
    没有任何足够长的片段时只裁掉首尾无效样本；全部无效时不裁剪（交给校验拒收）。
    """
    valid = np.asarray(valid, bool)
    n = len(valid)
    if not valid.any():
        return 0, n
    edges = np.flatnonzero(np.diff(np.concatenate([[0], valid.astype(np.int8), [0]])))
    runs = list(zip(edges[::2].tolist(), edges[1::2].tolist()))
    long = [(a, b) for a, b in runs if b - a >= min_run * rate]
    if not long:
        return runs[0][0], runs[-1][1]
    return long[0][0], long[-1][1]


def trim_result(res: ResampleResult, start: int, stop: int) -> ResampleResult:
    """截取 ``[start, stop)``：时间戳重新从 0 开始，``start_time`` 相应后移。"""
    rate = float(res.info.get("rate_hz", DEFAULT_RATE))
    n = stop - start
    info = dict(res.info)
    info["trimmed_samples"] = {"start": int(start), "end": int(len(res.timestamp) - stop)}
    return ResampleResult(
        timestamp=np.arange(n, dtype=np.float64) / rate,
        start_time=float(res.start_time + start / rate),
        arrays={k: v[start:stop] for k, v in res.arrays.items()},
        valid_imu=res.valid_imu[start:stop],
        valid_pose=res.valid_pose[start:stop],
        info=info,
    )


def _quat_ok(q: np.ndarray) -> np.ndarray:
    """有限且范数接近 1 的四元数样本（范数偏离过大视为无效而非静默归一化）。"""
    finite = np.all(np.isfinite(q), axis=1)
    norm = np.linalg.norm(np.where(finite[:, None], q, 0.0), axis=1)
    return finite & (np.abs(norm - 1.0) < 0.1)


def _interp_quat(t: np.ndarray, q: np.ndarray, ok: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """只用有效样本做 SLERP（无效区由两侧有效样本球面插值填充），输出单位且符号连续。"""
    if not ok.any():
        out = np.zeros((len(grid), 4))
        out[:, 0] = 1.0
        return out
    q_ok = quat_normalize(q[ok])
    return quat_make_continuous(quat_normalize(quat_interp(t[ok], q_ok, grid)))
