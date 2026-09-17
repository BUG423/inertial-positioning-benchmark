"""评测指标（DESIGN 第 6 节）。定义与边界情况见 ``docs/METRICS.md``。

轨迹级与窗口级指标只依赖 numpy；效率指标（``metrics.efficiency``）依赖 torch，需显式导入。
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from .trajectory import (
    align_trajectory,
    ate,
    ate_aligned,
    cumulative_distance,
    d_rte,
    drift,
    length_ratios,
    path_length,
    relative_error,
    rte,
    trajectory_metrics,
    valid_span,
)
from .velocity import angle_between, window_metrics

# 指标名 → (展示名, 单位, 越小越好)。越接近 1 越好的比值指标标记为 None。
METRIC_INFO = {
    "ate": ("ATE", "m", True),
    "ate_aligned": ("ATE (aligned)", "m", True),
    "rte": ("RTE", "m", True),
    "rte_scaled": ("RTE scaled", "", None),  # 该序列的 RTE 是否按有效跨度换算（聚合后=换算比例）
    "valid_span_s": ("Valid span", "s", None),
    "t_rte_1s": ("T-RTE@1s", "m", True),
    "t_rte_10s": ("T-RTE@10s", "m", True),
    "d_rte_10m": ("D-RTE@10m", "m", True),
    "pde": ("PDE", "%", True),
    "plr": ("PLR", "", None),
    "plr_dense": ("PLR (dense)", "", None),
    "plr_oracle": ("PLR (oracle)", "", None),
    "oracle_ratio": ("Oracle ratio", "", None),
    "ate_oracle": ("ATE (oracle)", "m", True),
    "vel_rmse": ("Vel RMSE", "m/s", True),
    "speed_bias": ("Speed bias", "m/s", None),
    "speed_ratio": ("Speed ratio", "", None),
    "dir_err_mean": ("Dir. err (mean)", "deg", True),
    "dir_err_median": ("Dir. err (median)", "deg", True),
    "along_bias": ("Along-track bias", "m/s", None),
    "along_rmse": ("Along-track RMSE", "m/s", True),
    "cross_rmse": ("Cross-track RMSE", "m/s", True),
    "loss": ("Loss", "", True),
    "params": ("Params", "", True),
    "flops": ("FLOPs", "", True),
}
# 默认报表列。``ate_oracle`` 是同一协议下的误差下限（用窗口目标本身积分），必须和 ``ate`` 一起看：
# 有多秒 valid 缺口的序列上它可以超过模型的 ATE（真实数据实例：RIDI huayi_bag2，
# oracle 5.37 m 对模型 3.31 m），此时 ATE 度量的是缺口而不是模型。
MAIN_METRICS = ("ate", "ate_oracle", "rte", "d_rte_10m", "pde", "plr", "vel_rmse",
                "dir_err_mean")
# 不能用于模型选择：效率指标不在 val 上计算，oracle 指标与模型无关（选它等于不选）
NON_FITNESS = ("params", "flops", "ate_oracle")


def display_name(key: str) -> str:
    name, unit, _ = METRIC_INFO.get(key, (key, "", True))
    return f"{name} ({unit})" if unit else name


# 跨序列聚合统计量的后缀：``<metric>`` 取逐序列均值（缺省），``<metric>_median`` 取中位数。
# 中位数型 fitness 用于 val 划分小、构成偏斜的数据集：RIDI 自动生成的 val 只有 1 名受试者，
# 其中 2 条携带方式在 train 中不存在，均值会被这两条主导，选模信号很噪。
MEDIAN_SUFFIX = "_median"


def split_metric_key(key: str, known: Optional[Iterable[str]] = None) -> tuple:
    """把 ``fitness`` 键拆成 ``(指标名, 统计量)``，统计量为 ``"mean"`` 或 ``"median"``。

    ``ate`` → ``("ate", "mean")``；``ate_median`` → ``("ate", "median")``。
    只有去掉后缀后仍是**已知指标名**时才按聚合键解析，因此不会和 ``dir_err_median``
    （序列内窗口角度的中位数，本身就是一个指标名）冲突；它的跨序列中位数是
    ``dir_err_median_median``。
    """
    names = set(known) if known is not None else set(METRIC_INFO)
    if key in names:
        return key, "mean"
    if key.endswith(MEDIAN_SUFFIX):
        base = key[: -len(MEDIAN_SUFFIX)]
        if base in names:
            return base, "median"
    return key, "mean"


def fitness_keys(t_rte: Iterable[float] = (1.0, 10.0),
                 d_rte: Iterable[float] = (10.0,)) -> list:
    """可用作 ``fitness`` 的验证指标名（越小越好）。

    含按配置生成的 T-RTE / D-RTE 键，以及每个键的中位数型变体 ``<key>_median``
    （逐序列该指标的中位数，见 :data:`MEDIAN_SUFFIX`）。
    """
    keys = {k for k, info in METRIC_INFO.items() if info[2] is True and k not in NON_FITNESS}
    keys |= {f"t_rte_{float(tau):g}s" for tau in t_rte}
    keys |= {f"d_rte_{float(dist):g}m" for dist in d_rte}
    # ``loss`` 是按窗口数加权的窗口级损失，没有“逐序列中位数”的语义
    keys |= {f"{k}{MEDIAN_SUFFIX}" for k in keys if k != "loss"}
    return sorted(keys)


def sequence_metrics(
    pos_pred: np.ndarray,
    pos_gt: np.ndarray,
    valid_pose: np.ndarray,
    vel_pred: np.ndarray,
    vel_target: np.ndarray,
    window_valid: np.ndarray,
    *,
    rate: float = 200.0,
    pos_oracle: Optional[np.ndarray] = None,
    dims: int = 2,
    rte_delta: float = 60.0,
    t_rte: Iterable[float] = (1.0, 10.0),
    d_rte_distance: Iterable[float] = (10.0,),
    min_speed: float = 0.2,
) -> dict:
    """一条序列的全部轨迹级与窗口级指标。"""
    out = trajectory_metrics(pos_pred, pos_gt, valid_pose, rate, pos_oracle, dims, rte_delta,
                             t_rte, d_rte_distance)
    out.update(window_metrics(vel_pred, vel_target, window_valid, dims, min_speed))
    return out


__all__ = [
    "MAIN_METRICS",
    "MEDIAN_SUFFIX",
    "METRIC_INFO",
    "NON_FITNESS",
    "align_trajectory",
    "angle_between",
    "ate",
    "ate_aligned",
    "cumulative_distance",
    "d_rte",
    "display_name",
    "drift",
    "fitness_keys",
    "length_ratios",
    "path_length",
    "relative_error",
    "rte",
    "sequence_metrics",
    "split_metric_key",
    "trajectory_metrics",
    "valid_span",
    "window_metrics",
]
