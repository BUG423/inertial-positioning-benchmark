"""iMoT：变量 token 的惯性运动 Transformer（编码器 PSD/APE/ASC + 运动粒子解码器 + DSM）。

论文：Son Minh Nguyen, Linh Duy Tran, Duc Viet Le, Paul J.M Havinga,
"iMoT: Inertial Motion Transformer for Inertial Navigation", AAAI 2025 (39(6):6209–6217);
https://arxiv.org/abs/2412.12190（本文件依据 arXiv v1，含附录）。
官方仓库：https://github.com/Minh-Son-Nguyen/iMoT —— **只有 LICENSE 与 README**，README 写着
"will be made publicly available soon"；许可：仓库 LICENSE 为 MIT（无代码，对本复现无实际约束）。
fidelity：`paper-only`——本文件依据 `docs/algorithms/imot.md` 规格卡实现，卡中标 [A] 的项
（头数、FFN 宽度、归一化位置、各 MLP 的深度与共享关系、DSM 的归一化方式等）取卡片推荐默认值。

.. warning::

   **参数量无法由论文唯一确定。** 论文 Table 3 报告 200 Hz 下 **14.49 M**；本卡参考规格
   （T=200、D=3、P=128、N=M=2、8 头、FFN 宽度 4T）推导出 **7,137,482**，只有它的约 49%。
   把 FFN 宽度改成 2048 也只有 9,139,274。论文未给出的宽度或模块必然更大，差距无法归因
   （卡 §4、§10-1）。单元测试锁定的是**本卡规格**的 7,137,482，不是论文值。

token 布局固定为 18 个变量 token（iTransformer 式：每个通道的整条序列是一个 token）：
``[acc_raw(3), acc_trend(3), acc_seasonal(3), gyro_raw(3), gyro_trend(3), gyro_seasonal(3)]``。

编码器（N=2 层，每层 2,005,400）：

1. **PSD**（Progressive Series Decoupler，无参数）：对当前的 6 个 raw 槽做两次居中滑动平均
   （k₁=9 → k₂=3，replicate padding，保持长度）得到趋势 ``A_t``，``A_s = A − A_t``；
   trend/seasonal 槽被新值替换（每层只对 raw 槽重新分解，否则 token 数会涨到 54，卡 §10-3）；
2. **APE**（Adaptive Positional Encoding）：基础正弦编码 ``E`` 的位置取**轴序号** 0–2、特征维为
   ``T``，沿 token 维平铺 3 份得到 ``(9, T)``；``Ẽ = [MLP_a(Ã_a) ⊙ E, MLP_g(Ã_g) ⊙ E]``；
3. 多头自注意力：``q = k = Ã + Ẽ``、``v = Ã``（式 3）；
4. 两个残差连接上各挂一个 **ASC**（Adaptive Spatial-Channel）分支：把 acc 与 gyro 的对应槽沿
   时间维拼成 ``(B, 9, 2T)`` → 沿 token 方向的 ``Conv1d(2T, 2T, 3)`` → ``GAP_T + Conv1d(2T, 2T, 1)
   + sigmoid`` 的通道门控 → 拆回 ``(B, 18, T)`` 后逐 token ``Linear(T, T) + GELU``；
5. FFN（``T → 4T → T``），post-LN。

解码器（M=2 层，每层 1,486,200）：可学习的**运动粒子** ``v̂⁰ ∈ R^{P×2}``（P=128，初值 U[−1,1]）
与全零内容特征 ``C⁰``；每层做 粒子位置编码（DAB-DETR 式正弦 + 共享 ``MLP_pos``）→ 自注意力 →
位置缩放 ``qpos = MLP_s(C) ⊙ E_v̂`` → **两路交叉注意力**（acc / gyro，query 与 key 为 2T 维、
value 为 T 维，Conditional-DETR 式）→ ``MLP_c`` 融合 → FFN → 共享 ``MLP_Δ`` 细化粒子。
最后 **DSM**（式 10）：``S = softmax_P(MLP_dsm(v̂ᵀ))``，``v_m = Σ_p S·v̂``（逐轴凸组合）。

损失：只用 ``J_vel = mean_b ‖v_GT − v_m‖²``（论文：“can be efficiently optimized using only the
velocity loss”）——即 IPB 的 ``mse_sum``（逐样本对分量求和再对样本取均值）。

与论文的差异及理由（引用规格卡）：

1. 通道置换：论文 token 顺序为 ``{A_a, A_g}``（**加计在前**），IPB 输入为 ``[gyro, acc]``，
   在模型内部置换（卡 §2、§6）；
2. T=200：与官方 200 Hz 配置相同；OxIOD/IDOL 官方用 T=100，IPB 重采样后为 T=200，token 维随之
   翻倍、参数量也随之变化（这是论文自身按采样率设定的规则，卡 §6）。``T=100`` 时 8 头无法整除
   100，必须改用 4 头（卡 §4）；
3. 输入系、目标定义与步长论文均未说明：取 ``frame=gravity_world``、``target=avg_velocity``、
   ``dims=2``、``stride=10``（卡 §2、§3、§6 的假设）；
4. 式 6 虽写作 “SelfAttn”，实际是交叉注意力；``E_A ∈ R^{D×T}`` 与 ``MLP(Ã_a) ∈ R^{3D×T}`` 逐元素
   相乘需要平铺/广播；这些记号不自洽处按卡 §10-2 的读法实现；
5. DSM 的归一化论文未说明，取沿粒子维 softmax（使 ``v_m`` 是凸组合；否则尺度无约束，卡 §10-5）；
6. 论文称模型 “permutation-invariant to token orders”，但 ASC 沿 token 维做 k=3 卷积，实际并不
   置换不变，因此**不设**这一测试（卡 §10-6）；
7. 不带 DSM 的变体（式 8–9 的熵损失 + 测试期粒子平均池化）只作为消融提及，未实现（卡 §5）。
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from ...nn.base import BaseModel, InputSpec
from ...nn.heads import sinusoidal_encoding
from ...nn.registry import register_model

TOKENS_PER_MODALITY = 9      # 每个模态 3 个槽（raw / trend / seasonal）× 3 轴
MODALITY_TOKENS = 18


def moving_average(x: torch.Tensor, kernel: int) -> torch.Tensor:
    """居中滑动平均（stride 1、replicate padding、保持长度），无参数。"""
    pad = kernel // 2
    padded = F.pad(x, (pad, kernel - 1 - pad), mode="replicate")
    return F.avg_pool1d(padded, kernel, stride=1)


def two_layer_mlp(in_features: int, hidden: int, out_features: int,
                  activation: str = "relu") -> nn.Sequential:
    act = nn.ReLU() if activation == "relu" else nn.GELU()
    return nn.Sequential(nn.Linear(in_features, hidden), act, nn.Linear(hidden, out_features))


class ProgressiveSeriesDecoupler(nn.Module):
    """PSD（无参数）：``A_t = MA_{k2}(MA_{k1}(A))``、``A_s = A − A_t``，重组为 18 个 token。"""

    def __init__(self, first_kernel: int = 9, second_kernel: int = 3) -> None:
        super().__init__()
        self.first_kernel = int(first_kernel)
        self.second_kernel = int(second_kernel)

    def decompose(self, raw: torch.Tensor) -> tuple:
        trend = moving_average(moving_average(raw, self.first_kernel), self.second_kernel)
        return trend, raw - trend

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        """``(B, 6, T)`` 的 raw 槽 → ``(B, 18, T)``，模态内顺序为 raw / trend / seasonal。"""
        trend, seasonal = self.decompose(raw)
        parts = []
        for start in (0, 3):
            block = slice(start, start + 3)
            parts += [raw[:, block], trend[:, block], seasonal[:, block]]
        return torch.cat(parts, dim=1)

    def raw_slots(self, tokens: torch.Tensor) -> torch.Tensor:
        """从 18 个 token 中取回 6 个 raw 槽（``(B, 6, T)``）。"""
        return torch.cat([tokens[:, 0:3], tokens[:, TOKENS_PER_MODALITY:TOKENS_PER_MODALITY + 3]],
                         dim=1)


class AdaptivePositionalEncoding(nn.Module):
    """APE：``Ẽ = [MLP_a(Ã_a) ⊙ E, MLP_g(Ã_g) ⊙ E]``，``E`` 为按轴序号的正弦编码平铺 3 份。"""

    def __init__(self, length: int, temperature: float = 10000.0) -> None:
        super().__init__()
        base = sinusoidal_encoding(torch.arange(3, dtype=torch.float32), length, temperature)
        self.register_buffer("base", base.repeat(3, 1))       # (9, T)
        self.acc_mlp = two_layer_mlp(length, length, length)
        self.gyro_mlp = two_layer_mlp(length, length, length)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        base = self.base.to(tokens.dtype)
        acc = self.acc_mlp(tokens[:, :TOKENS_PER_MODALITY]) * base
        gyro = self.gyro_mlp(tokens[:, TOKENS_PER_MODALITY:]) * base
        return torch.cat([acc, gyro], dim=1)


class AdaptiveSpatialChannel(nn.Module):
    """ASC：跨模态时间拼接 → token 维卷积 → 通道门控 → 投影回 token 空间 + GELU。"""

    def __init__(self, length: int, kernel: int = 3) -> None:
        super().__init__()
        self.length = int(length)
        self.conv = nn.Conv1d(2 * length, 2 * length, kernel, padding=kernel // 2)
        self.gate = nn.Conv1d(2 * length, 2 * length, 1)
        self.project = nn.Linear(length, length)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        acc = tokens[:, :TOKENS_PER_MODALITY]
        gyro = tokens[:, TOKENS_PER_MODALITY:]
        merged = torch.cat([acc, gyro], dim=-1).transpose(1, 2)        # (B, 2T, 9)
        spatial = self.conv(merged)
        gate = torch.sigmoid(self.gate(spatial.mean(dim=-1, keepdim=True)))
        merged = (spatial * gate).transpose(1, 2)                      # (B, 9, 2T)
        restored = torch.cat([merged[..., : self.length], merged[..., self.length:]], dim=1)
        return F.gelu(self.project(restored))


class EncoderLayer(nn.Module):
    """编码层：PSD → APE → 自注意力（+ASC 残差）→ FFN（+ASC 残差），post-LN。"""

    def __init__(self, length: int, heads: int = 8, feedforward: int = 0, dropout: float = 0.1,
                 psd_kernels: Sequence[int] = (9, 3), asc_kernel: int = 3) -> None:
        super().__init__()
        self.decoupler = ProgressiveSeriesDecoupler(*psd_kernels)
        self.positional = AdaptivePositionalEncoding(length)
        self.attention = nn.MultiheadAttention(length, heads, dropout=dropout, batch_first=True)
        self.asc1 = AdaptiveSpatialChannel(length, asc_kernel)
        self.norm1 = nn.LayerNorm(length)
        hidden = int(feedforward) or 4 * length
        self.feedforward = nn.Sequential(
            nn.Linear(length, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, length))
        self.asc2 = AdaptiveSpatialChannel(length, asc_kernel)
        self.norm2 = nn.LayerNorm(length)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> tuple:
        tokens = self.decoupler(self.decoupler.raw_slots(tokens))
        encoding = self.positional(tokens)
        query = tokens + encoding
        attended, _ = self.attention(query, query, tokens, need_weights=False)
        tokens = self.norm1(tokens + self.dropout(attended) + self.asc1(tokens))
        tokens = self.norm2(tokens + self.dropout(self.feedforward(tokens)) + self.asc2(tokens))
        return tokens, encoding


class ConditionalCrossAttention(nn.Module):
    """Conditional-DETR 式交叉注意力：query/key 为 ``2T`` 维、value 为 ``T`` 维。"""

    def __init__(self, length: int, heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        if (2 * length) % heads or length % heads:
            raise ValueError(f"heads={heads} must divide both {length} and {2 * length}")
        self.heads = int(heads)
        self.query = nn.Linear(2 * length, 2 * length)
        self.key = nn.Linear(2 * length, 2 * length)
        self.value = nn.Linear(length, length)
        self.out = nn.Linear(length, length)
        self.dropout = nn.Dropout(dropout)
        self.head_dim = (2 * length) // heads

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, features = x.shape
        return x.reshape(batch, tokens, self.heads, features // self.heads).transpose(1, 2)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> tuple:
        q = self._split(self.query(query))
        k = self._split(self.key(key))
        v = self._split(self.value(value))
        weights = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.head_dim), dim=-1)
        context = self.dropout(weights) @ v
        batch, _, tokens, _ = context.shape
        merged = context.transpose(1, 2).reshape(batch, tokens, -1)
        return self.out(merged), weights


class DecoderLayer(nn.Module):
    """解码层：自注意力 → 位置缩放 → 两路交叉注意力 → 融合 → FFN（每步残差 + LN）。"""

    def __init__(self, length: int, heads: int = 8, feedforward: int = 0,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(length, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(length)
        self.scale_mlp = two_layer_mlp(length, length, length)
        self.cross_acc = ConditionalCrossAttention(length, heads, dropout)
        self.cross_gyro = ConditionalCrossAttention(length, heads, dropout)
        self.fuse = two_layer_mlp(2 * length, length, length)
        self.norm2 = nn.LayerNorm(length)
        hidden = int(feedforward) or 4 * length
        self.feedforward = nn.Sequential(
            nn.Linear(length, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, length))
        self.norm3 = nn.LayerNorm(length)
        self.dropout = nn.Dropout(dropout)

    def forward(self, content: torch.Tensor, particle_encoding: torch.Tensor,
                memory: torch.Tensor, memory_encoding: torch.Tensor) -> tuple:
        query = content + particle_encoding
        attended, _ = self.attention(query, query, content, need_weights=False)
        content_sa = self.norm1(content + self.dropout(attended))
        qpos = self.scale_mlp(content) * particle_encoding
        joint_query = torch.cat([content_sa, qpos], dim=-1)
        acc, gyro = memory[:, :TOKENS_PER_MODALITY], memory[:, TOKENS_PER_MODALITY:]
        acc_key = torch.cat([acc, memory_encoding[:, :TOKENS_PER_MODALITY]], dim=-1)
        gyro_key = torch.cat([gyro, memory_encoding[:, TOKENS_PER_MODALITY:]], dim=-1)
        acc_context, acc_map = self.cross_acc(joint_query, acc_key, acc)
        gyro_context, _ = self.cross_gyro(joint_query, gyro_key, gyro)
        fused = self.fuse(torch.cat([acc_context, gyro_context], dim=-1))
        content = self.norm2(content_sa + self.dropout(fused))
        content = self.norm3(content + self.dropout(self.feedforward(content)))
        return content, acc_map


@register_model("imot")
class IMoT(BaseModel):
    """iMoT：变量 token 编码器 + 运动粒子解码器 + 动态打分（DSM）。"""

    default_loss = "mse_sum"
    saved_outputs = ()

    def __init__(
        self,
        input_spec: InputSpec,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        particles: int = 128,
        heads: int = 8,
        feedforward: int = 0,
        dropout: float = 0.1,
        psd_kernels: Sequence[int] = (9, 3),
        asc_kernel: int = 3,
        particle_scale: float = 2.0 * math.pi,
        particle_temperature: float = 10000.0,
    ) -> None:
        super().__init__(input_spec)
        if input_spec.output_layout != "window":
            raise ValueError("imot predicts one velocity segment per window")
        if input_spec.sub_windows:
            raise ValueError("imot takes a single window (history=0)")
        length = int(input_spec.window)
        if length % heads:
            raise ValueError(f"heads={heads} must divide the token length {length} "
                             f"(T=100 needs heads=4, see the spec card)")
        self.length = length
        self.particle_scale = float(particle_scale)
        self.particle_temperature = float(particle_temperature)
        self.decoupler = ProgressiveSeriesDecoupler(*psd_kernels)
        self.encoder = nn.ModuleList(
            EncoderLayer(length, heads, feedforward, dropout, psd_kernels, asc_kernel)
            for _ in range(encoder_layers))
        self.particles = nn.Parameter(torch.empty(int(particles), input_spec.dims).uniform_(-1, 1))
        # MLP_pos 与 MLP_Δ 在各解码层之间**共享**（论文 [P]），MLP_s 每层独立
        self.position_mlp = two_layer_mlp(length, length, length)
        self.refine_mlp = two_layer_mlp(length, length, input_spec.dims)
        self.decoder = nn.ModuleList(
            DecoderLayer(length, heads, feedforward, dropout) for _ in range(decoder_layers))
        self.scoring = two_layer_mlp(int(particles), int(particles), int(particles))

    def tokenize(self, imu: torch.Tensor) -> torch.Tensor:
        """IPB 的 ``[gyro, acc]`` → 论文的 ``[acc, gyro]`` raw 槽 → 18 个变量 token。"""
        self.check_input(imu)
        return self.decoupler(torch.cat([imu[:, 3:6], imu[:, 0:3]], dim=1))

    def particle_encoding(self, particles: torch.Tensor) -> torch.Tensor:
        """DAB-DETR 式粒子位置编码：每个轴 ``T/2`` 维正弦编码，再过共享的 ``MLP_pos``。"""
        dims = particles.shape[-1]
        per_axis = self.length // dims
        codes = [sinusoidal_encoding(particles[..., d] * self.particle_scale, per_axis,
                                     self.particle_temperature) for d in range(dims)]
        code = torch.cat(codes, dim=-1)
        if code.shape[-1] < self.length:     # dims 不整除 T 时补零到 T
            code = F.pad(code, (0, self.length - code.shape[-1]))
        return self.position_mlp(code)

    def dynamic_scores(self, particles: torch.Tensor) -> torch.Tensor:
        """DSM（式 10）：``S = softmax_P(MLP_dsm(v̂ᵀ))``，形状 ``(B, dims, P)``。"""
        return torch.softmax(self.scoring(particles.transpose(1, 2)), dim=-1)

    def forward(self, imu: torch.Tensor) -> dict:
        tokens = self.tokenize(imu)
        encoding = None
        for layer in self.encoder:
            tokens, encoding = layer(tokens)
        batch = imu.shape[0]
        particles = self.particles.to(imu.dtype).unsqueeze(0).expand(batch, -1, -1)
        content = torch.zeros(batch, particles.shape[1], self.length, dtype=imu.dtype,
                              device=imu.device)
        attention_map = None
        for layer in self.decoder:
            position = self.particle_encoding(particles)
            content, attention_map = layer(content, position, tokens, encoding)
            particles = particles + self.refine_mlp(content)
        scores = self.dynamic_scores(particles)
        velocity = (scores * particles.transpose(1, 2)).sum(dim=-1)
        return {"vel": velocity,
                "aux": {"particles": particles, "scores": scores, "attention": attention_map}}
