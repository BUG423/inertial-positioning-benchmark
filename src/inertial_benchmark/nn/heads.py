"""输出头：速度头、对角高斯头、极坐标头（速度大小 + 单位方向）。

所有头接收特征 ``(B, F)``，返回 dict，``vel`` 始终为 ``(B, dims)`` 的速度（或模型目标量）。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class VelocityHead(nn.Module):
    """线性回归头：``{"vel"}``。"""

    def __init__(self, in_features: int, dims: int = 2) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, dims)

    def forward(self, x: torch.Tensor) -> dict:
        return {"vel": self.linear(x)}


class GaussianHead(nn.Module):
    """对角高斯头：均值与 ``log σ``（``{"vel", "logstd"}``）；裁剪在损失中进行。"""

    def __init__(self, in_features: int, dims: int = 2) -> None:
        super().__init__()
        self.mean = nn.Linear(in_features, dims)
        self.logstd = nn.Linear(in_features, dims)

    def forward(self, x: torch.Tensor) -> dict:
        return {"vel": self.mean(x), "logstd": self.logstd(x)}


class PolarHead(nn.Module):
    """极坐标头：速度大小 ``s = softplus(·) ≥ 0`` 与单位方向 ``u``，``vel = s·u``。

    返回 ``{"vel": (B,D), "speed": (B,), "direction": (B,D)}``。方向由线性层输出归一化得到
    （范数下限 ``eps`` 防止除零）；``beta`` 为 softplus 的锐度。
    """

    def __init__(self, in_features: int, dims: int = 2, eps: float = 1e-6,
                 beta: float = 1.0) -> None:
        super().__init__()
        self.speed = nn.Linear(in_features, 1)
        self.direction = nn.Linear(in_features, dims)
        self.eps = eps
        self.beta = beta

    def forward(self, x: torch.Tensor) -> dict:
        speed = F.softplus(self.speed(x), beta=self.beta).squeeze(-1)
        raw = self.direction(x)
        direction = raw / raw.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        return {"vel": speed.unsqueeze(-1) * direction, "speed": speed, "direction": direction}


HEADS = {"velocity": VelocityHead, "gaussian": GaussianHead, "polar": PolarHead}


def build_head(name: str, in_features: int, dims: int, **kwargs) -> nn.Module:
    if name not in HEADS:
        raise ValueError(f"unknown head {name!r}; available: {sorted(HEADS)}")
    return HEADS[name](in_features, dims, **kwargs)
