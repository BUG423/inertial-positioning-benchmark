"""等窗口预算：固定“看过的训练窗口总数”，由框架换算出每个数据集的 epoch 数。

固定 epoch 数在规模差一个数量级的数据集之间并不公平：实测 RIDI 的 train 划分约 9.6 万窗口/轮，
而 PedLocData 有约 89 h 数据（百万级窗口/轮），按同一个 epoch 数跑会让大数据集吃掉全部算力，
小数据集则训练不足。``train_windows_budget``（预算 ``B``）给每个 run 固定同一个
“训练窗口总数”，epoch 数由框架计算：

    epochs = clip(ceil(B / W), 1, budget_max_epochs)

其中 ``W`` 是该数据集**每轮的训练窗口数**，取 train 划分上实际可用（窗口内无无效样本）的
窗口数，与日志里 ``train: <W> windows`` 的数字、``len(train_set)`` 一致。上界
``budget_max_epochs``（缺省 200）避免极小数据集被换算成几千轮空转；下界 1 保证至少训练一轮。

这是**配置层**能力：模型 YAML 不得为此硬编码 ``epochs``。换算结果（预算、``W``、未截断的
轮数、最终轮数）写入 ``args.yaml`` 的 ``epochs`` 与 ``metrics.json`` 的 ``train_budget`` 块，
以便审计“每个 run 到底看过多少训练窗口”。

序列级模型（:class:`~inertial_benchmark.nn.base.SequenceModel`：PDR、常速基线等）没有梯度
训练，只在 train 划分上标定一次，因此不适用等窗口预算（``resolve_budget`` 由调用方跳过）。
"""

from __future__ import annotations

import math
from typing import Any, Optional

from ..utils import LOGGER

# (数据集, 视图/步长指纹) → 每轮训练窗口数；同一进程内多次规划不重复扫描 HDF5
_WINDOW_CACHE: dict = {}
# 决定 train 划分窗口数的键：窗口网格（输入跨度）与训练步长
_COUNT_KEYS = ("window", "frame", "orientation", "remove_gravity", "target", "dims", "rate",
               "history", "history_stride", "stride")


def budget_epochs(budget: int, windows_per_epoch: int, max_epochs: int = 200) -> int:
    """``clip(ceil(budget / windows_per_epoch), 1, max_epochs)``。

    Args:
        budget: 本次 run 允许看过的训练窗口总数（必须 > 0）。
        windows_per_epoch: 该数据集 train 划分每轮的可用窗口数（必须 > 0）。
        max_epochs: epoch 上限（必须 >= 1）。
    """
    budget, windows_per_epoch, max_epochs = int(budget), int(windows_per_epoch), int(max_epochs)
    if budget <= 0:
        raise ValueError(f"train_windows_budget must be > 0, got {budget}")
    if windows_per_epoch <= 0:
        raise ValueError(f"windows_per_epoch must be > 0, got {windows_per_epoch}")
    if max_epochs < 1:
        raise ValueError(f"budget_max_epochs must be >= 1, got {max_epochs}")
    return min(max(math.ceil(budget / windows_per_epoch), 1), max_epochs)


def resolve_budget(cfg: Any, windows_per_epoch: int) -> Optional[dict]:
    """按配置换算 epoch 数；``train_windows_budget <= 0`` 时返回 ``None``（沿用 ``epochs``）。

    返回的字典即 ``metrics.json`` 的 ``train_budget`` 块：预算、每轮窗口数、未截断与最终的
    epoch 数、上限，以及按最终轮数计算的 ``planned_windows``（该 run 实际会看过的窗口总数）。
    """
    get = cfg.get if hasattr(cfg, "get") else (lambda k, d=None: getattr(cfg, k, d))
    budget = int(get("train_windows_budget") or 0)
    if budget <= 0:
        return None
    max_epochs = int(get("budget_max_epochs") or 200)
    windows_per_epoch = int(windows_per_epoch)
    uncapped = math.ceil(budget / max(windows_per_epoch, 1)) if windows_per_epoch > 0 else 0
    epochs = budget_epochs(budget, windows_per_epoch, max_epochs)
    return {
        "train_windows_budget": budget,
        "windows_per_epoch": windows_per_epoch,
        "epochs_uncapped": int(uncapped),
        "epochs": int(epochs),
        "budget_max_epochs": max_epochs,
        "planned_windows": int(epochs * windows_per_epoch),
    }


def log_budget(name: str, budget: dict) -> None:
    """把换算过程写进日志（预算被上限截断时给出告警）。"""
    if budget["epochs"] < budget["epochs_uncapped"]:
        LOGGER.warning(
            f"train_windows_budget: {name} needs {budget['epochs_uncapped']} epochs for "
            f"{budget['train_windows_budget']:,} windows but budget_max_epochs="
            f"{budget['budget_max_epochs']} caps it; this run only sees "
            f"{budget['planned_windows']:,} windows")
    else:
        LOGGER.info(
            f"train_windows_budget: {budget['train_windows_budget']:,} windows / "
            f"{budget['windows_per_epoch']:,} windows per epoch on {name} "
            f"→ epochs={budget['epochs']} ({budget['planned_windows']:,} windows seen)")


def is_sequence_model(model: Any) -> bool:
    """模型是否为序列级/标定型（:class:`SequenceModel`）——这类模型不适用等窗口预算。

    只查注册表里的类，不构建模型（规划阶段不需要权重、也不占显存）。
    """
    from ..cfg import load_model_cfg
    from ..nn.base import SequenceModel
    from ..nn.registry import MODELS, import_models

    model_cfg = load_model_cfg(model)
    import_models()
    cls = MODELS.get(model_cfg.get("arch", model_cfg.get("name")))
    return bool(cls is not None and issubclass(cls, SequenceModel))


def count_train_windows(cfg: Any, spec: Any = None, split: str = "train") -> int:
    """``split`` 划分上的可用训练窗口数（规划用；惰性读取，不把序列载入内存）。

    只读 ``valid/*`` 掩码与时间戳长度，因此可以在 ``dry_run`` 里为每个 (模型, 数据集) 组合
    算出 epoch 数而不必真的开始训练；同一视图/步长的结果按进程缓存。
    """
    from ..data.build import build_dataset
    from ..data.manifest import resolve_dataset
    from ..utils import IterableSimpleNamespace

    spec = spec or resolve_dataset(cfg.data)
    get = cfg.get if hasattr(cfg, "get") else (lambda k, d=None: getattr(cfg, k, d))
    key = (str(spec.root), split, tuple(str(get(k)) for k in _COUNT_KEYS))
    if key not in _WINDOW_CACHE:
        lean = IterableSimpleNamespace(**{**dict(cfg.to_dict()), "cache": False, "augment": []})
        _WINDOW_CACHE[key] = len(build_dataset(lean, split, training=(split == "train"),
                                               spec=spec))
    return int(_WINDOW_CACHE[key])


__all__ = ["budget_epochs", "count_train_windows", "is_sequence_model", "log_budget",
           "resolve_budget"]
