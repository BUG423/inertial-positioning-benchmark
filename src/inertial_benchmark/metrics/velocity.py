"""窗口级速度指标（DESIGN 第 6 节，定义见 ``docs/METRICS.md``）。

输入为世界系窗口速度 ``vel_pred``/``vel_target`` ``(K, ≥dims)`` 与窗口有效掩码。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

NAN = float("nan")


def angle_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """逐行夹角（度，范围 [0, 180]）；任一向量为零时记为 90°（不含方向信息）。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    dot = np.sum(a * b, axis=1)
    if a.shape[1] == 2:
        cross = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
    else:
        cross = np.linalg.norm(np.cross(a, b), axis=1)
    angle = np.degrees(np.arctan2(cross, dot))
    zero = (np.linalg.norm(a, axis=1) < 1e-9) | (np.linalg.norm(b, axis=1) < 1e-9)
    return np.where(zero, 90.0, angle)


def window_metrics(vel_pred: np.ndarray, vel_target: np.ndarray,
                   valid: Optional[np.ndarray] = None, dims: int = 2,
                   min_speed: float = 0.2) -> dict:
    """速度 RMSE、速度大小偏差（差值/比值）、方向误差、沿/横向误差。"""
    p = np.asarray(vel_pred, dtype=np.float64)[:, :dims]
    t = np.asarray(vel_target, dtype=np.float64)[:, :dims]
    if valid is not None:
        p, t = p[np.asarray(valid, bool)], t[np.asarray(valid, bool)]
    out = {"vel_rmse": NAN, "speed_bias": NAN, "speed_ratio": NAN, "dir_err_mean": NAN,
           "dir_err_median": NAN, "along_bias": NAN, "along_rmse": NAN, "cross_rmse": NAN,
           "num_windows": int(len(p)), "num_moving_windows": 0}
    if len(p) == 0:
        return out
    err = p - t
    sp = np.linalg.norm(p, axis=1)
    st = np.linalg.norm(t, axis=1)
    out["vel_rmse"] = float(np.sqrt(np.mean(np.sum(err**2, axis=1))))
    out["speed_bias"] = float(np.mean(sp - st))
    out["speed_ratio"] = float(sp.sum() / st.sum()) if st.sum() > 0 else NAN
    moving = st > min_speed
    out["num_moving_windows"] = int(moving.sum())
    if moving.any():
        angles = angle_between(p[moving], t[moving])
        out["dir_err_mean"] = float(np.mean(angles))
        out["dir_err_median"] = float(np.median(angles))
        u = t[moving] / st[moving, None]
        e = err[moving]
        along = np.sum(e * u, axis=1)
        cross = e - along[:, None] * u
        out["along_bias"] = float(np.mean(along))
        out["along_rmse"] = float(np.sqrt(np.mean(along**2)))
        out["cross_rmse"] = float(np.sqrt(np.mean(np.sum(cross**2, axis=1))))
    return out
