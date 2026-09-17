"""通用损失：MSE、对角高斯 NLL、TLIO 式“先 MSE 后 NLL”调度。

损失函数签名统一为 ``fn(out, target, epoch, mask) -> (loss, items)``。

目标可以是窗口级 ``(B, D)``、逐帧 ``(B, T, D)`` 或多步 ``(B, H, D)``（DESIGN 第 3 节）。
``mask`` 为逐输出的有效掩码（``(B,)``、``(B, T)`` 或 ``(B, H)``）：无效的帧/步不参与损失，
全部无效时返回 0（并保留计算图，避免优化器拿不到梯度）。
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch

MIN_LOGSTD = math.log(1e-3)  # 与 TLIO 相同的下限
MAX_LOGSTD = math.log(1e3)


def expand_mask(mask: Optional[torch.Tensor], like: torch.Tensor) -> Optional[torch.Tensor]:
    """把逐输出掩码广播到与预测同形：``(B,)``/``(B,R)`` → ``(B,[R,]D)``。"""
    if mask is None:
        return None
    mask = mask.to(dtype=like.dtype)
    while mask.ndim < like.ndim:
        mask = mask.unsqueeze(-1)
    return mask.expand_as(like)


def masked_mean(values: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """掩码平均；掩码全零时返回 ``0 · sum(values)``（保留计算图）。"""
    if mask is None:
        return values.mean()
    total = mask.sum()
    if bool(total == 0):
        return values.sum() * 0.0
    return (values * mask).sum() / total


def mse(pred: torch.Tensor, target: torch.Tensor,
        mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """逐元素均方误差（无掩码时与 ``nn.MSELoss()`` 相同）。"""
    return masked_mean((pred - target) ** 2, expand_mask(mask, pred))


def gaussian_nll(pred: torch.Tensor, logstd: torch.Tensor, target: torch.Tensor,
                 min_logstd: Optional[float] = MIN_LOGSTD,
                 max_logstd: Optional[float] = MAX_LOGSTD,
                 mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """对角高斯负对数似然（省略常数 ``0.5·log 2π``）：

    ``mean( (t − μ)² / (2 σ²) + log σ )``，``log σ`` 先裁剪到 ``[min_logstd, max_logstd]``。
    """
    logstd = torch.clamp(logstd, min=min_logstd, max=max_logstd)
    terms = (pred - target) ** 2 / (2.0 * torch.exp(2.0 * logstd)) + logstd
    return masked_mean(terms, expand_mask(mask, pred))


class MSELoss:
    name = "mse"

    def __call__(self, out: dict, target: torch.Tensor, epoch: int = 0,
                 mask: Optional[torch.Tensor] = None) -> tuple:
        loss = mse(out["vel"], target, mask)
        return loss, {"mse": loss.detach()}


class GaussianNLLLoss:
    name = "gaussian_nll"

    def __init__(self, min_logstd: float = MIN_LOGSTD, max_logstd: float = MAX_LOGSTD) -> None:
        self.min_logstd = min_logstd
        self.max_logstd = max_logstd

    def __call__(self, out: dict, target: torch.Tensor, epoch: int = 0,
                 mask: Optional[torch.Tensor] = None) -> tuple:
        if "logstd" not in out:
            raise KeyError("gaussian_nll needs the model to output 'logstd'")
        nll = gaussian_nll(out["vel"], out["logstd"], target, self.min_logstd, self.max_logstd,
                           mask)
        return nll, {"nll": nll.detach(), "mse": mse(out["vel"], target, mask).detach()}


class MSEThenNLL:
    """TLIO 调度：``epoch < switch_epoch`` 用 MSE（不训练 logstd），之后用高斯 NLL。"""

    name = "mse_then_nll"

    def __init__(self, switch_epoch: int = 10, min_logstd: float = MIN_LOGSTD,
                 max_logstd: float = MAX_LOGSTD) -> None:
        self.switch_epoch = int(switch_epoch)
        self.nll = GaussianNLLLoss(min_logstd, max_logstd)

    def __call__(self, out: dict, target: torch.Tensor, epoch: int = 0,
                 mask: Optional[torch.Tensor] = None) -> tuple:
        if epoch < self.switch_epoch:
            loss = mse(out["vel"], target, mask)
            items = {"mse": loss.detach()}
            if "logstd" in out:
                # 让 logstd 分支参与计算图但梯度为零，避免 DDP/AMP 报未使用参数
                loss = loss + 0.0 * out["logstd"].sum()
            return loss, items
        return self.nll(out, target, epoch, mask)

    def stage(self, epoch: int) -> str:
        return "mse" if epoch < self.switch_epoch else "nll"


LOSSES: dict = {"mse": MSELoss, "gaussian_nll": GaussianNLLLoss, "mse_then_nll": MSEThenNLL}


def build_loss(name: str, **kwargs) -> Callable:
    if name not in LOSSES:
        raise ValueError(f"unknown loss {name!r}; available: {sorted(LOSSES)}")
    return LOSSES[name](**kwargs)
