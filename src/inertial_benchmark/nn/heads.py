"""输出头：速度头、对角高斯头、极坐标头（速度大小 + 单位方向）、门控头、分箱头。

所有头接收特征 ``(B, F)``，返回 dict，``vel`` 始终为 ``(B, dims)`` 的速度（或模型目标量）。

``gated``（GNIO 的门控预测头）与 ``bins``（VeloBins 的分箱解码器）做成**可复用**输出头，
因此同一个门控/分箱机制可以插到任意骨干上做消融（"换头不换骨干"）。
"""

from __future__ import annotations

import math
from typing import Optional

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


# 门控头的幅值/门控激活（GNIO 论文 Table IV 扫过的组合）
SCALE_FNS = {
    "softplus": F.softplus,                       # 论文默认，也是 Table IV 的最优
    "pos_elu": lambda x: F.elu(x) + 1.0,
    "abs": torch.abs,
    "exp": torch.exp,
    "linear": lambda x: x,                        # 退化为普通线性回归（幅值可为负）
}
GATE_FNS = {"tanh": torch.tanh, "sigmoid": torch.sigmoid}
# 幅值恒非负的激活：只有这些取值才保证 ``|vel| <= scale``
NONNEGATIVE_SCALE_FNS = ("softplus", "pos_elu", "abs", "exp")


class GatedHead(nn.Module):
    """门控预测头（GNIO，`docs/algorithms/gnio.md` §4.2）：``vel = s̃ ⊙ g``。

    ``s̃ = scale_fn(W_s h + b_s)``（逐轴幅值提案）与 ``g = gate_fn(W_g h + b_g)``（逐轴门控）
    都是 ``dims`` 维，**不是**“模长 × 单位方向”的极坐标分解；两个支路参数独立，因此参数量是
    普通线性头的两倍（``2·(dims·F + dims)``，F=512、dims=3 时 1,539 → 3,078）。

    结构性质：``scale_fn`` 非负时 ``|vel| ≤ s̃``（逐元素）、``vel_i = 0 ⟺ g_i = 0``，
    静止段可以靠 ``g → 0`` 做“软 ZUPT”。``gate_fn="tanh"`` 时符号由门控决定（可翻转方向），
    ``sigmoid`` 只能衰减不能翻转（论文 Table IV 因此更差）。

    ``zero_gate_bias=True``（默认）把 ``b_g`` 初始化为 0，使初始门控在 0 附近、初始输出接近 0
    （论文未规定，卡片 §4.3 的建议）。返回 ``{"vel", "scale", "gate"}``。
    """

    def __init__(self, in_features: int, dims: int = 2, scale_fn: str = "softplus",
                 gate_fn: str = "tanh", zero_gate_bias: bool = True) -> None:
        super().__init__()
        if scale_fn not in SCALE_FNS:
            raise ValueError(f"unknown scale_fn {scale_fn!r}; available: {sorted(SCALE_FNS)}")
        if gate_fn not in GATE_FNS:
            raise ValueError(f"unknown gate_fn {gate_fn!r}; available: {sorted(GATE_FNS)}")
        self.scale_linear = nn.Linear(in_features, dims)
        self.gate_linear = nn.Linear(in_features, dims)
        self.scale_fn_name = scale_fn
        self.gate_fn_name = gate_fn
        if zero_gate_bias:
            nn.init.zeros_(self.gate_linear.bias)

    @property
    def bounded(self) -> bool:
        """``|vel| ≤ scale`` 是否成立（幅值激活非负时成立）。"""
        return self.scale_fn_name in NONNEGATIVE_SCALE_FNS

    def forward(self, x: torch.Tensor) -> dict:
        scale = SCALE_FNS[self.scale_fn_name](self.scale_linear(x))
        gate = GATE_FNS[self.gate_fn_name](self.gate_linear(x))
        return {"vel": scale * gate, "scale": scale, "gate": gate}


DECODES = ("expectation", "argmax", "topk_expectation")
BIN_ENCODINGS = ("spe", "pe", "direct")


def sinusoidal_encoding(values: torch.Tensor, channels: int,
                        temperature: float = 10000.0) -> torch.Tensor:
    """Transformer 式正弦位置编码：``(N,) → (N, channels)``，偶数通道 sin、奇数通道 cos。"""
    if channels % 2:
        raise ValueError(f"channels must be even, got {channels}")
    index = torch.arange(channels // 2, dtype=values.dtype, device=values.device)
    omega = temperature ** (-2.0 * index / channels)
    angle = values.unsqueeze(-1) * omega
    out = torch.zeros(values.shape + (channels,), dtype=values.dtype, device=values.device)
    out[..., 0::2] = torch.sin(angle)
    out[..., 1::2] = torch.cos(angle)
    return out


class BinsHead(nn.Module):
    """分箱输出头（VeloBins，`docs/algorithms/velobins.md` §4.2）：逐轴 ``N`` 箱分类 + 解码。

    每个轴 ``i`` 用“箱坐标分类器”给出箱上的分布，参数量与箱数 ``N`` **无关**：

    ``SPE(b) = sin(γ ⊙ PE(b)) ∈ R^{N×C}``（``γ ∈ R^C`` 可学习的逐通道频率）；
    ``e_n = φ(SPE(b_n)) ∈ R^D``（``φ = Linear(C, D)``，所有轴共享）；
    ``q_i = θ_i(h) ∈ R^D``（每轴一个 ``Linear(F, D)``）；``π_i = softmax_n(q_i · e_n)``。

    参数量 ``C + D(C+1) + dims·D(F+1)``（``F=48, C=64, D=32, dims=3`` → 6,848）。
    对照：直接 ``Linear(F, dims·N)`` 在 N=512 时需要 75,264 个参数。

    解码（``decode``）：

    * ``expectation``（论文默认，式 1）：``v̂_i = Σ_n π_{i,n} b_n``；
    * ``argmax``：``v̂_i = b_{argmax π_i}``（量化误差 ≤ Δ/2）；
    * ``topk_expectation``：只在概率最大的 ``topk`` 个箱上归一化后求期望（``topk ≥ N`` 时与
      ``expectation`` 逐位相等）。

    方差解码一律用**完整分布**围绕解码均值的二阶矩（式 2）：``σ̂_i² = Σ_n π_{i,n}(b_n − v̂_i)²``；
    ``clamp_std=True`` 时再取下限 ``Δ/10``（与训练标签的 σ 下限一致），避免 ``logstd = −∞``。

    ``encoding``：``spe``（默认，可学习频率）/ ``pe``（固定正弦编码，无 ``γ``）/
    ``direct``（消融用：直接 ``Linear(F, dims·N)`` 出 logits，参数量随 N 增长）。

    返回 ``{"vel", "logstd", "probs" (B,dims,N), "logits" (B,dims,N)}``。
    """

    def __init__(self, in_features: int, dims: int = 2, bins: int = 512,
                 value_range: float = 5.0, code_dim: int = 64, query_dim: int = 32,
                 encoding: str = "spe", decode: str = "expectation", topk: int = 5,
                 clamp_std: bool = True, temperature: float = 10000.0) -> None:
        super().__init__()
        if int(bins) < 2:
            raise ValueError(f"bins must be >= 2, got {bins}")
        if float(value_range) <= 0:
            raise ValueError(f"value_range must be > 0, got {value_range}")
        if encoding not in BIN_ENCODINGS:
            raise ValueError(f"unknown encoding {encoding!r}; available: {list(BIN_ENCODINGS)}")
        self.bins = int(bins)
        self.dims = int(dims)
        self.encoding = encoding
        self.temperature = float(temperature)
        self.clamp_std = bool(clamp_std)
        self.set_decode(decode, topk)
        # 均匀箱中心 b_n = −R + (n + ½)Δ，Δ = 2R/N；箱边界取相邻中心的中点
        step = 2.0 * float(value_range) / self.bins
        centers = -float(value_range) + (torch.arange(self.bins, dtype=torch.float32) + 0.5) * step
        self.register_buffer("centers", centers)
        self.value_range = float(value_range)
        self.bin_width = step
        if encoding == "direct":
            self.logits_linear = nn.Linear(in_features, self.dims * self.bins)
        else:
            if encoding == "spe":
                self.freq = nn.Parameter(torch.ones(int(code_dim)))
            self.code = nn.Linear(int(code_dim), int(query_dim))
            self.queries = nn.ModuleList(
                nn.Linear(in_features, int(query_dim)) for _ in range(self.dims))
            self.register_buffer("bin_encoding",
                                 sinusoidal_encoding(centers, int(code_dim), self.temperature))

    def set_decode(self, decode: str, topk: Optional[int] = None) -> None:
        """切换解码方式（消融用；不改变任何参数）。"""
        if decode not in DECODES:
            raise ValueError(f"unknown decode {decode!r}; available: {list(DECODES)}")
        self.decode = decode
        if topk is not None:
            if int(topk) < 1:
                raise ValueError(f"topk must be >= 1, got {topk}")
            self.topk = int(topk)

    def bin_logits(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, F)`` → ``(B, dims, N)`` 的未归一化 logits。"""
        if self.encoding == "direct":
            return self.logits_linear(x).reshape(-1, self.dims, self.bins)
        code = self.bin_encoding
        if self.encoding == "spe":
            code = torch.sin(self.freq * code)
        embed = self.code(code.to(x.dtype))                      # (N, D)
        query = torch.stack([q(x) for q in self.queries], dim=1)  # (B, dims, D)
        return query @ embed.transpose(0, 1)

    def decode_probs(self, probs: torch.Tensor) -> tuple:
        """``(B, dims, N)`` 的分布 → ``(vel, std)``（式 1–2 与可切换解码）。"""
        centers = self.centers.to(probs.dtype)
        mean = (probs * centers).sum(-1)
        if self.decode == "argmax":
            vel = centers[probs.argmax(-1)]
        elif self.decode == "topk_expectation" and self.topk < self.bins:
            top, index = probs.topk(self.topk, dim=-1)
            vel = (top * centers[index]).sum(-1) / top.sum(-1).clamp_min(1e-12)
        else:
            vel = mean
        var = (probs * (centers - vel.unsqueeze(-1)) ** 2).sum(-1)
        std = var.clamp_min(0.0).sqrt()
        if self.clamp_std:
            std = std.clamp_min(self.bin_width / 10.0)
        return vel, std

    def forward(self, x: torch.Tensor) -> dict:
        logits = self.bin_logits(x)
        probs = torch.softmax(logits, dim=-1)
        vel, std = self.decode_probs(probs)
        return {"vel": vel, "logstd": torch.log(std.clamp_min(1e-12)),
                "probs": probs, "logits": logits}


def bin_edges(centers: torch.Tensor, value_range: float) -> tuple:
    """箱的下/上边界：相邻中心的中点，首尾箱外边界延伸到 ``∓value_range``。"""
    mid = 0.5 * (centers[:-1] + centers[1:])
    lower = torch.cat([centers.new_full((1,), -float(value_range)), mid])
    upper = torch.cat([mid, centers.new_full((1,), float(value_range))])
    return lower, upper


def normal_cdf(x: torch.Tensor) -> torch.Tensor:
    """标准正态分布函数 ``Φ(x) = ½(1 + erf(x/√2))``。"""
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


HEADS = {"velocity": VelocityHead, "gaussian": GaussianHead, "polar": PolarHead,
         "gated": GatedHead, "bins": BinsHead}


def build_head(name: str, in_features: int, dims: int, **kwargs) -> nn.Module:
    if name not in HEADS:
        raise ValueError(f"unknown head {name!r}; available: {sorted(HEADS)}")
    return HEADS[name](in_features, dims, **kwargs)
