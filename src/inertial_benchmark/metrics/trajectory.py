"""轨迹级指标（DESIGN 第 6 节，精确定义见 ``docs/METRICS.md``）。

约定：所有函数输入同一均匀时间轴上的 ``pred``/``gt`` 位置 ``(N, ≥dims)``，``valid`` 为参考位姿
有效掩码，``rate`` 为采样率（Hz）；默认只取前两维（水平面）。无法定义时返回 ``nan``。
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

NAN = float("nan")


def _prep(pred: np.ndarray, gt: np.ndarray, valid: Optional[np.ndarray], dims: int):
    pred = np.asarray(pred, dtype=np.float64)[:, :dims]
    gt = np.asarray(gt, dtype=np.float64)[:, :dims]
    if pred.shape != gt.shape:
        raise ValueError(f"pred {pred.shape} and gt {gt.shape} differ")
    valid = np.ones(len(gt), bool) if valid is None else np.asarray(valid, bool)
    return pred, gt, valid


def sample_indices(valid: np.ndarray, rate: float, resolution: Optional[float]) -> np.ndarray:
    """路径长度的采样点：从首个有效样本起每 ``resolution`` 秒取一点，并补上末个有效样本。"""
    idx_valid = np.flatnonzero(valid)
    if len(idx_valid) == 0:
        return idx_valid
    step = 1 if not resolution else max(int(round(resolution * rate)), 1)
    first, last = idx_valid[0], idx_valid[-1]
    idx = np.arange(first, last + 1, step)
    if idx[-1] != last:
        idx = np.append(idx, last)
    return idx


def path_length(
    pos: np.ndarray,
    valid: Optional[np.ndarray] = None,
    rate: float = 200.0,
    resolution: Optional[float] = 1.0,
    dims: int = 2,
    indices: Optional[np.ndarray] = None,
) -> float:
    """按固定时间分辨率采样后的折线长度；只累计两端点均有效的线段。"""
    pos = np.asarray(pos, dtype=np.float64)[:, :dims]
    valid = np.ones(len(pos), bool) if valid is None else np.asarray(valid, bool)
    idx = sample_indices(valid, rate, resolution) if indices is None else indices
    if len(idx) < 2:
        return 0.0
    seg = np.linalg.norm(np.diff(pos[idx], axis=0), axis=1)
    ok = valid[idx[:-1]] & valid[idx[1:]]
    return float(seg[ok].sum())


def ate(pred: np.ndarray, gt: np.ndarray, valid: Optional[np.ndarray] = None,
        dims: int = 2) -> float:
    """绝对轨迹误差 ``sqrt(mean_t ‖p̂_t − p_t‖²)``，不做对齐。"""
    pred, gt, valid = _prep(pred, gt, valid, dims)
    if not valid.any():
        return NAN
    return float(np.sqrt(np.mean(np.sum((pred[valid] - gt[valid]) ** 2, axis=1))))


def align_trajectory(pred: np.ndarray, gt: np.ndarray, valid: Optional[np.ndarray] = None,
                     dims: int = 2) -> np.ndarray:
    """最小二乘刚体对齐：绕 z 轴旋转 + 平移（2D 为 SE(2)，3D 为 4 自由度）。"""
    pred_d, gt_d, valid = _prep(pred, gt, valid, dims)
    p, g = pred_d[valid], gt_d[valid]
    pc, gc = p.mean(axis=0), g.mean(axis=0)
    a, b = p[:, :2] - pc[:2], g[:, :2] - gc[:2]
    theta = np.arctan2(np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]), np.sum(a * b))
    c, s = np.cos(theta), np.sin(theta)
    out = pred_d - pc
    x, y = out[:, 0].copy(), out[:, 1].copy()
    out[:, 0], out[:, 1] = c * x - s * y, s * x + c * y
    return out + gc


def ate_aligned(pred: np.ndarray, gt: np.ndarray, valid: Optional[np.ndarray] = None,
                dims: int = 2) -> float:
    """刚体对齐后的 ATE。"""
    _, _, valid = _prep(pred, gt, valid, dims)
    if valid.sum() < 2:
        return NAN
    return ate(align_trajectory(pred, gt, valid, dims), gt, valid, dims)


def rte(pred: np.ndarray, gt: np.ndarray, valid: Optional[np.ndarray] = None, rate: float = 200.0,
        delta: float = 60.0, dims: int = 2) -> float:
    """RoNIN 定义的相对轨迹误差。

    ``Δ = round(delta·rate)`` 个样本；对所有两端均有效的 ``(i, i+Δ)``：
    ``sqrt(mean ‖(p̂_{i+Δ}−p̂_i) − (p_{i+Δ}−p_i)‖²)``。序列时长短于 ``delta`` 时，
    取首末有效样本这一对，并按 ``delta / 实际跨度`` 线性放大。
    """
    pred, gt, valid = _prep(pred, gt, valid, dims)
    n = len(gt)
    d = int(round(delta * rate))
    idx = np.flatnonzero(valid)
    if len(idx) < 2 or d < 1:
        return NAN
    if n - 1 < d:
        i0, i1 = idx[0], idx[-1]
        err = (pred[i1] - pred[i0]) - (gt[i1] - gt[i0])
        return float(np.linalg.norm(err) * delta / ((i1 - i0) / rate))
    i = np.arange(0, n - d)
    j = i + d
    ok = valid[i] & valid[j]
    if not ok.any():
        return NAN
    i, j = i[ok], j[ok]
    err = (pred[j] - pred[i]) - (gt[j] - gt[i])
    return float(np.sqrt(np.mean(np.sum(err**2, axis=1))))


def cumulative_distance(gt: np.ndarray, valid: np.ndarray, rate: float,
                        resolution: Optional[float] = 1.0) -> np.ndarray:
    """参考轨迹累计行走距离（按 ``resolution`` 采样的折线，再线性插值回每个样本）。"""
    idx = sample_indices(valid, rate, resolution)
    s = np.zeros(len(gt))
    if len(idx) < 2:
        return s
    seg = np.linalg.norm(np.diff(gt[idx], axis=0), axis=1)
    seg[~(valid[idx[:-1]] & valid[idx[1:]])] = 0.0
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    return np.interp(np.arange(len(gt)), idx, cum)


def d_rte(pred: np.ndarray, gt: np.ndarray, valid: Optional[np.ndarray] = None,
          rate: float = 200.0, distance: float = 10.0, dims: int = 2,
          start_step: float = 1.0) -> float:
    """按距离的相对误差：参考轨迹每走过 ``distance`` 米的相对位移误差 RMSE。

    起点沿参考路径每 ``start_step`` 米取一个（避免静止段重复计数）；终点为累计距离首次
    达到 ``起点 + distance`` 的样本；两端都必须有效。
    """
    pred, gt, valid = _prep(pred, gt, valid, dims)
    s = cumulative_distance(gt, valid, rate)
    if s[-1] < distance:
        return NAN
    marks = np.arange(0.0, s[-1] - distance + 1e-9, start_step)
    i = np.searchsorted(s, marks, side="left")
    j = np.searchsorted(s, s[i] + distance, side="left")
    ok = (j < len(s))
    i, j = i[ok], j[ok]
    ok = valid[i] & valid[j]
    if not ok.any():
        return NAN
    i, j = i[ok], j[ok]
    err = (pred[j] - pred[i]) - (gt[j] - gt[i])
    return float(np.sqrt(np.mean(np.sum(err**2, axis=1))))


def drift(pred: np.ndarray, gt: np.ndarray, valid: Optional[np.ndarray] = None,
          rate: float = 200.0, dims: int = 2, min_length: float = 1.0) -> float:
    """终点漂移 PDE：``‖p̂_N − p_N‖ / L_ref × 100``（N 为末个有效样本，L_ref 为 1 s 分辨率长度）。"""
    pred, gt, valid = _prep(pred, gt, valid, dims)
    idx = np.flatnonzero(valid)
    if len(idx) == 0:
        return NAN
    length = path_length(gt, valid, rate, 1.0, dims)
    if length < min_length:
        return NAN
    last = idx[-1]
    return float(np.linalg.norm(pred[last] - gt[last]) / length * 100.0)


def length_ratios(pred: np.ndarray, gt: np.ndarray, valid: Optional[np.ndarray] = None,
                  rate: float = 200.0, oracle: Optional[np.ndarray] = None,
                  dims: int = 2) -> dict:
    """路径长度比：``plr``、``plr_dense``，给定 oracle 时另有 ``plr_oracle``、``oracle_ratio``。

    预测与 oracle 轨迹长度一律按 1 s 分辨率计算；``plr_dense`` 的参考长度取 200 Hz 稠密折线。
    恒等式：``plr_dense = oracle_ratio × plr_oracle``。
    """
    pred, gt, valid = _prep(pred, gt, valid, dims)
    idx = sample_indices(valid, rate, 1.0)
    l_pred = path_length(pred, valid, rate, dims=dims, indices=idx)
    l_ref = path_length(gt, valid, rate, dims=dims, indices=idx)
    l_dense = path_length(gt, valid, rate, None, dims)
    out = {
        "plr": l_pred / l_ref if l_ref > 0 else NAN,
        "plr_dense": l_pred / l_dense if l_dense > 0 else NAN,
    }
    if oracle is not None:
        l_oracle = path_length(np.asarray(oracle)[:, :dims], valid, rate, dims=dims, indices=idx)
        out["plr_oracle"] = l_pred / l_oracle if l_oracle > 0 else NAN
        out["oracle_ratio"] = l_oracle / l_dense if l_dense > 0 else NAN
    return out


def trajectory_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    valid: Optional[np.ndarray] = None,
    rate: float = 200.0,
    oracle: Optional[np.ndarray] = None,
    dims: int = 2,
    rte_delta: float = 60.0,
    t_rte: Iterable[float] = (1.0, 10.0),
    d_rte_distance: Iterable[float] = (10.0,),
) -> dict:
    """一次计算全部轨迹级指标，返回 ``{name: value}``。"""
    out = {
        "ate": ate(pred, gt, valid, dims),
        "ate_aligned": ate_aligned(pred, gt, valid, dims),
        "rte": rte(pred, gt, valid, rate, rte_delta, dims),
    }
    for tau in t_rte:
        out[f"t_rte_{tau:g}s"] = rte(pred, gt, valid, rate, tau, dims)
    for dist in d_rte_distance:
        out[f"d_rte_{dist:g}m"] = d_rte(pred, gt, valid, rate, dist, dims)
    out["pde"] = drift(pred, gt, valid, rate, dims)
    out.update(length_ratios(pred, gt, valid, rate, oracle, dims))
    if oracle is not None:
        out["ate_oracle"] = ate(oracle, gt, valid, dims)
    return out
