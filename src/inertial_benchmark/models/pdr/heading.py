"""设备航向与前向轴（``docs/algorithms/pdr.md`` 第 4 节第 7–8 步），供 PDR 与常速基线共用。

航向来自姿态：取机体前向轴 ``e_fwd`` 旋到世界系后的水平投影，水平投影太小（设备前向轴接近竖直）
时换用备用轴 ``e_alt``，再加标定得到的常值航向偏置 ``δ``。前向轴按序列根属性 ``body_frame`` 查表。
"""

from __future__ import annotations

from typing import Mapping, Optional

import numpy as np

from ...utils.geometry import quat_rotate

# body_frame → (主前向轴, 备用轴)。手机机体系为 “x 右、y 指向机顶、z 出屏”：
# 平放手持、机顶朝前时前向为 +y；竖持（机顶朝上）时机背朝前，备用轴为 −z。
BODY_AXES: dict = {
    "android_device": ((0.0, 1.0, 0.0), (0.0, 0.0, -1.0)),
    "ios_device": ((0.0, 1.0, 0.0), (0.0, 0.0, -1.0)),
}
CHUNK = 512  # 逐窗口航向的分块大小（避免一次展开整条序列的四元数）


def resolve_axes(body_frame: str, extra: Optional[Mapping[str, object]] = None) -> tuple:
    """按 ``body_frame`` 查前向轴；未知机体系必须显式配置（否则报错，不猜）。"""
    table = dict(BODY_AXES)
    for name, axes in (extra or {}).items():
        table[str(name)] = (tuple(axes[0]), tuple(axes[1]))  # type: ignore[index]
    if body_frame not in table:
        raise ValueError(
            f"unknown body_frame {body_frame!r}: the forward axis must be configured explicitly "
            f"(body_axes={{'{body_frame}': [[fx, fy, fz], [ax, ay, az]]}}); "
            f"known: {sorted(table)}")
    fwd, alt = table[body_frame]
    return np.asarray(fwd, dtype=np.float64), np.asarray(alt, dtype=np.float64)


def _horizontal_sum(q: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """一段（或一批段）姿态下前向轴水平分量之和，形状为 ``q.shape[:-2] + (2,)``。"""
    return quat_rotate(q, axis)[..., :2].sum(axis=-2)


def segment_heading(q: np.ndarray, fwd: np.ndarray, alt: np.ndarray,
                    min_norm: float = 0.2) -> tuple:
    """一段样本的航向（弧度）与是否使用了备用轴。"""
    total = _horizontal_sum(q, fwd)
    used_alt = bool(np.linalg.norm(total) < min_norm * len(q))
    if used_alt:
        total = _horizontal_sum(q, alt)
    return float(np.arctan2(total[1], total[0])), used_alt


def window_headings(q: np.ndarray, starts: np.ndarray, window: int, fwd: np.ndarray,
                    alt: np.ndarray, min_norm: float = 0.2) -> tuple:
    """逐窗口航向 ``(K,)``（弧度）与是否使用备用轴的掩码 ``(K,)``。"""
    starts = np.asarray(starts, dtype=np.int64)
    angles = np.zeros(len(starts))
    used_alt = np.zeros(len(starts), bool)
    frames = np.arange(int(window))
    for lo in range(0, len(starts), CHUNK):
        block = starts[lo:lo + CHUNK]
        rows = q[block[:, None] + frames]
        total = _horizontal_sum(rows, fwd)
        flip = np.linalg.norm(total, axis=-1) < min_norm * window
        if flip.any():
            total = np.where(flip[:, None], _horizontal_sum(rows, alt), total)
        angles[lo:lo + CHUNK] = np.arctan2(total[:, 1], total[:, 0])
        used_alt[lo:lo + CHUNK] = flip
    return angles, used_alt


def circular_mean(angles: np.ndarray) -> tuple:
    """圆均值与平均合向量长度 ``R̄ ∈ [0, 1]``（越小说明偏置越不恒定）。"""
    angles = np.asarray(angles, dtype=np.float64)
    if len(angles) == 0:
        return 0.0, 0.0
    vector = np.array([np.cos(angles).mean(), np.sin(angles).mean()])
    return float(np.arctan2(vector[1], vector[0])), float(np.linalg.norm(vector))


__all__ = ["BODY_AXES", "circular_mean", "resolve_axes", "segment_heading", "window_headings"]
