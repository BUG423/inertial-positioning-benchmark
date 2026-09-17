"""通用损失：MSE、对角高斯 NLL、TLIO 式两阶段调度，以及模型专用损失的注册入口。

损失函数签名统一为 ``fn(out: dict, target: Tensor, epoch: int) -> (loss, items)``。
模型包可以用 ``@register_loss(name)`` 注册专用损失（导入模型包时生效），模型 YAML 的
``loss``/``loss_kwargs`` 即可引用。
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

    ``mean( (t − μ)² / (2 σ²) + log σ )``，``log σ`` 先裁剪到 ``[min_logstd, max_logstd]``；
    界为 ``None`` 时该侧不裁剪（TLIO 官方只有下限）。
    """
    if min_logstd is not None or max_logstd is not None:
        logstd = torch.clamp(logstd, min=min_logstd, max=max_logstd)
    return torch.mean((pred - target) ** 2 / (2.0 * torch.exp(2.0 * logstd)) + logstd)


class MSELoss:
    name = "mse"

    def __call__(self, out: dict, target: torch.Tensor, epoch: int = 0) -> tuple:
        loss = mse(out["vel"], target)
        return loss, {"mse": loss.detach()}


class MSESumLoss:
    """逐维 MSE 之和 ``Σ_d mean_b (v̂_d − v_d)²``（= D × ``mse``）。

    对应 Keras 多输出头各自 ``'mse'`` 等权求和（TinyOdom），以及逐样本对分量求和的
    ``mean_i Σ_d (·)²``（RIO 速度项）。
    """

    name = "mse_sum"

    def __call__(self, out: dict, target: torch.Tensor, epoch: int = 0) -> tuple:
        loss = torch.mean(torch.sum((out["vel"] - target) ** 2, dim=-1))
        return loss, {"mse": mse(out["vel"], target).detach()}


class MSEPlusL1Loss:
    """``mse_weight · MSE + l1_weight · mean|v̂ − v|``（DeepILS 官方损失，两项权重均为 1）。"""

    name = "mse_l1"

    def __init__(self, mse_weight: float = 1.0, l1_weight: float = 1.0) -> None:
        self.mse_weight = float(mse_weight)
        self.l1_weight = float(l1_weight)

    def __call__(self, out: dict, target: torch.Tensor, epoch: int = 0) -> tuple:
        m = mse(out["vel"], target)
        l1 = torch.mean(torch.abs(out["vel"] - target))
        loss = self.mse_weight * m + self.l1_weight * l1
        return loss, {"mse": m.detach(), "l1": l1.detach()}


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


class DetachedNLLThenNLL:
    """TLIO 官方代码的实际调度：``epoch < switch_epoch`` 时仍用高斯 NLL，但 ``logstd`` 先
    ``detach``（协方差头不收梯度，均值误差按当前 ``exp(-2·logstd)`` 加权；``logstd≈0`` 时
    等价于 ``0.5·MSE``）；之后为完整 NLL。

    与 :class:`MSEThenNLL` 的区别仅在第一阶段。缺省只做下限裁剪 ``log(1e-3)``（TLIO 官方
    ``losses.py`` 没有上限）；``switch_epoch`` 按 IPB 的 0 起 epoch 计数。
    """

    name = "nll_detach_then_nll"

    def __init__(self, switch_epoch: int = 10, min_logstd: Optional[float] = MIN_LOGSTD,
                 max_logstd: Optional[float] = None) -> None:
        self.switch_epoch = int(switch_epoch)
        self.min_logstd = min_logstd
        self.max_logstd = max_logstd

    def __call__(self, out: dict, target: torch.Tensor, epoch: int = 0) -> tuple:
        if "logstd" not in out:
            raise KeyError(f"{self.name} needs the model to output 'logstd'")
        logstd = out["logstd"]
        detached = epoch < self.switch_epoch
        nll = gaussian_nll(out["vel"], logstd.detach() if detached else logstd, target,
                           self.min_logstd, self.max_logstd)
        items = {"nll": nll.detach(), "mse": mse(out["vel"], target).detach()}
        if detached:
            # logstd 分支留在计算图中但梯度为零，避免 DDP 报未使用参数
            nll = nll + 0.0 * logstd.sum()
        return nll, items

    def stage(self, epoch: int) -> str:
        return "nll_detached" if epoch < self.switch_epoch else "nll"


LOSSES: dict = {
    "mse": MSELoss,
    "mse_sum": MSESumLoss,
    "mse_l1": MSEPlusL1Loss,
    "gaussian_nll": GaussianNLLLoss,
    "mse_then_nll": MSEThenNLL,
    "nll_detach_then_nll": DetachedNLLThenNLL,
}
# 构造参数含 ``switch_epoch``、可由 ``loss_switch_epoch`` 配置的调度损失
SWITCH_LOSSES = ("mse_then_nll", "nll_detach_then_nll")


def register_loss(name: str):
    """类装饰器：注册模型专用损失（构造参数来自模型 YAML 的 ``loss_kwargs``）。"""

    def decorator(cls):
        if name in LOSSES and LOSSES[name] is not cls:
            raise KeyError(f"loss name {name!r} already registered by {LOSSES[name]}")
        LOSSES[name] = cls
        cls.name = name
        return cls

    return decorator


def build_loss(name: str, **kwargs) -> Callable:
    if name not in LOSSES:
        raise ValueError(f"unknown loss {name!r}; available: {sorted(LOSSES)}")
    return LOSSES[name](**kwargs)
