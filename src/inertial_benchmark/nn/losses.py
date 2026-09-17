"""通用损失：MSE、对角高斯 NLL、TLIO 式两阶段调度，以及模型专用损失的注册入口。

损失函数签名统一为 ``fn(out: dict, target: Tensor, epoch: int, mask: Tensor | None)
-> (loss, items)``。

目标可以是窗口级 ``(B, D)``、逐帧 ``(B, T, D)`` 或多步 ``(B, H, D)``（DESIGN 第 3 节）。
``mask`` 为逐输出的有效掩码（窗口级 ``(B, 1)`` 或 ``(B,)``、逐帧 ``(B, T)``、多步 ``(B, H)``）：
无效的帧/步不参与损失，全部无效时返回 0（并保留计算图，避免优化器拿不到梯度）。

模型包可以用 ``@register_loss(name)`` 注册专用损失（导入模型包时生效），模型 YAML 的
``loss``/``loss_kwargs`` 即可引用；专用损失同样按上面的四参数签名实现。
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


def masked_output_mean(values: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """对“每个输出一个标量”的张量做掩码平均。

    ``values`` 的形状是预测形状去掉通道维（窗口级 ``(B,)``、逐帧 ``(B, T)``、多步 ``(B, H)``）；
    逐输出掩码本身是 ``(B, R)``，因此先补回长度 1 的通道维再广播（窗口级的 ``(B, 1)`` 掩码
    正是这样对齐到 ``(B, D)`` 预测的）。
    """
    keep = values.unsqueeze(-1)
    return masked_mean(keep, expand_mask(mask, keep))


def mse(pred: torch.Tensor, target: torch.Tensor,
        mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """逐元素均方误差（无掩码时与 ``nn.MSELoss()`` 相同）。"""
    return masked_mean((pred - target) ** 2, expand_mask(mask, pred))


def gaussian_nll(pred: torch.Tensor, logstd: torch.Tensor, target: torch.Tensor,
                 min_logstd: Optional[float] = MIN_LOGSTD,
                 max_logstd: Optional[float] = MAX_LOGSTD,
                 mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """对角高斯负对数似然（省略常数 ``0.5·log 2π``）：

    ``mean( (t − μ)² / (2 σ²) + log σ )``，``log σ`` 先裁剪到 ``[min_logstd, max_logstd]``；
    界为 ``None`` 时该侧不裁剪（TLIO 官方只有下限）。
    """
    if min_logstd is not None or max_logstd is not None:
        logstd = torch.clamp(logstd, min=min_logstd, max=max_logstd)
    terms = (pred - target) ** 2 / (2.0 * torch.exp(2.0 * logstd)) + logstd
    return masked_mean(terms, expand_mask(mask, pred))


class MSELoss:
    name = "mse"

    def __call__(self, out: dict, target: torch.Tensor, epoch: int = 0,
                 mask: Optional[torch.Tensor] = None) -> tuple:
        loss = mse(out["vel"], target, mask)
        return loss, {"mse": loss.detach()}


class MSESumLoss:
    """逐维 MSE 之和 ``Σ_d mean_b (v̂_d − v_d)²``（= D × ``mse``）。

    对应 Keras 多输出头各自 ``'mse'`` 等权求和（TinyOdom），以及逐样本对分量求和的
    ``mean_i Σ_d (·)²``（RIO 速度项）。逐帧/多步目标下按 ``mask`` 只平均有效的帧/步。
    """

    name = "mse_sum"

    def __call__(self, out: dict, target: torch.Tensor, epoch: int = 0,
                 mask: Optional[torch.Tensor] = None) -> tuple:
        loss = masked_output_mean(torch.sum((out["vel"] - target) ** 2, dim=-1), mask)
        return loss, {"mse": mse(out["vel"], target, mask).detach()}


class MSEPlusL1Loss:
    """``mse_weight · MSE + l1_weight · mean|v̂ − v|``（DeepILS 官方损失，两项权重均为 1）。"""

    name = "mse_l1"

    def __init__(self, mse_weight: float = 1.0, l1_weight: float = 1.0) -> None:
        self.mse_weight = float(mse_weight)
        self.l1_weight = float(l1_weight)

    def __call__(self, out: dict, target: torch.Tensor, epoch: int = 0,
                 mask: Optional[torch.Tensor] = None) -> tuple:
        m = mse(out["vel"], target, mask)
        l1 = masked_mean(torch.abs(out["vel"] - target), expand_mask(mask, out["vel"]))
        loss = self.mse_weight * m + self.l1_weight * l1
        return loss, {"mse": m.detach(), "l1": l1.detach()}


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

    def __call__(self, out: dict, target: torch.Tensor, epoch: int = 0,
                 mask: Optional[torch.Tensor] = None) -> tuple:
        if "logstd" not in out:
            raise KeyError(f"{self.name} needs the model to output 'logstd'")
        logstd = out["logstd"]
        detached = epoch < self.switch_epoch
        nll = gaussian_nll(out["vel"], logstd.detach() if detached else logstd, target,
                           self.min_logstd, self.max_logstd, mask)
        items = {"nll": nll.detach(), "mse": mse(out["vel"], target, mask).detach()}
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
    """类装饰器：注册模型专用损失（构造参数来自模型 YAML 的 ``loss_kwargs``）。

    被注册的类必须按模块开头的四参数签名实现 ``__call__``（含 ``mask``）。
    """

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
