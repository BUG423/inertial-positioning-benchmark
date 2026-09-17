"""通用损失：MSE、对角高斯 NLL、TLIO 式“先 MSE 后 NLL”调度。

损失函数签名统一为 ``fn(out: dict, target: Tensor, epoch: int) -> (loss, items)``。
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch

MIN_LOGSTD = math.log(1e-3)  # 与 TLIO 相同的下限
MAX_LOGSTD = math.log(1e3)


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """逐元素均方误差（与 ``nn.MSELoss()`` 相同）。"""
    return torch.mean((pred - target) ** 2)


def gaussian_nll(pred: torch.Tensor, logstd: torch.Tensor, target: torch.Tensor,
                 min_logstd: Optional[float] = MIN_LOGSTD,
                 max_logstd: Optional[float] = MAX_LOGSTD) -> torch.Tensor:
    """对角高斯负对数似然（省略常数 ``0.5·log 2π``）：

    ``mean( (t − μ)² / (2 σ²) + log σ )``，``log σ`` 先裁剪到 ``[min_logstd, max_logstd]``。
    """
    logstd = torch.clamp(logstd, min=min_logstd, max=max_logstd)
    return torch.mean((pred - target) ** 2 / (2.0 * torch.exp(2.0 * logstd)) + logstd)


class MSELoss:
    name = "mse"

    def __call__(self, out: dict, target: torch.Tensor, epoch: int = 0) -> tuple:
        loss = mse(out["vel"], target)
        return loss, {"mse": loss.detach()}


class GaussianNLLLoss:
    name = "gaussian_nll"

    def __init__(self, min_logstd: float = MIN_LOGSTD, max_logstd: float = MAX_LOGSTD) -> None:
        self.min_logstd = min_logstd
        self.max_logstd = max_logstd

    def __call__(self, out: dict, target: torch.Tensor, epoch: int = 0) -> tuple:
        if "logstd" not in out:
            raise KeyError("gaussian_nll needs the model to output 'logstd'")
        nll = gaussian_nll(out["vel"], out["logstd"], target, self.min_logstd, self.max_logstd)
        return nll, {"nll": nll.detach(), "mse": mse(out["vel"], target).detach()}


class MSEThenNLL:
    """TLIO 调度：``epoch < switch_epoch`` 用 MSE（不训练 logstd），之后用高斯 NLL。"""

    name = "mse_then_nll"

    def __init__(self, switch_epoch: int = 10, min_logstd: float = MIN_LOGSTD,
                 max_logstd: float = MAX_LOGSTD) -> None:
        self.switch_epoch = int(switch_epoch)
        self.nll = GaussianNLLLoss(min_logstd, max_logstd)

    def __call__(self, out: dict, target: torch.Tensor, epoch: int = 0) -> tuple:
        if epoch < self.switch_epoch:
            loss = mse(out["vel"], target)
            items = {"mse": loss.detach()}
            if "logstd" in out:
                # 让 logstd 分支参与计算图但梯度为零，避免 DDP/AMP 报未使用参数
                loss = loss + 0.0 * out["logstd"].sum()
            return loss, items
        return self.nll(out, target, epoch)

    def stage(self, epoch: int) -> str:
        return "mse" if epoch < self.switch_epoch else "nll"


LOSSES: dict = {"mse": MSELoss, "gaussian_nll": GaussianNLLLoss, "mse_then_nll": MSEThenNLL}


def build_loss(name: str, **kwargs) -> Callable:
    if name not in LOSSES:
        raise ValueError(f"unknown loss {name!r}; available: {sorted(LOSSES)}")
    return LOSSES[name](**kwargs)
