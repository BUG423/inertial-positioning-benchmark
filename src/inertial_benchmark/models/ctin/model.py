"""CTIN：Contextual Transformer 惯性网络，逐帧 2D 速度 + 对角 ``log σ`` + 多任务不确定度加权。

论文：B. Rao, E. Kazemi, Y. Ding, D. M. Shila, F. M. Tucker, L. Wang, "CTIN: Robust Contextual
Transformer Network for Inertial Navigation", AAAI 2022, 36(5):5413-5421.
https://arxiv.org/abs/2112.02143（本文件依据 arXiv v2，含附录 A.2/A.3）。
官方仓库：https://github.com/bingrao/ctin @ ``1441c726811ac87283903471562023fd429e3a08``
——**只有 README 与 LICENSE，没有任何源码**（README 说代码与数据归 Unknot.id 所有，待批准后发布）。
许可：仓库 LICENSE 为 MIT（README 徽章写 Apache-2.0，两者不一致）；由于没有代码，许可对本复现
没有实际约束。
fidelity：``paper-only``。本文件依据规格卡 ``docs/algorithms/ctin.md`` 的**参考规格**实现，
卡中标注为 **[A]（假设）** 的宽度都做成构造参数并取卡片推荐默认值。

**参数量与论文的差异**：参考规格（``d=64, w=64, heads=8, ff=512, groups=4, reduction=2``）合计
**477,572**（加 2 个多任务权重标量为 477,574），而论文 Table 3 报告 ``0.5571×10⁶``，
本实现为论文值的 **85.7%**（少 14.3%）。原因：论文只给出模块级描述，没有给出任何通道数、
隐层维度与 FFN 宽度，因此无法唯一复原该数字（规格卡 §4、§7）。例如只把 FFN 宽度改成 640 就得到
543,620（−2.4%），但没有任何证据支持该取值，所以仍以 ``ff=512`` 为默认值。**单元测试锁定的是
本卡推导值，不是论文值，不能作为忠实性证据**；结果表中须注明“参数量由规格推导”。

结构（规格卡 §4）：

* **空间嵌入**：``Conv1d(6→64, k3, p1) → BatchNorm1d → 逐时刻 Linear(64→64)``；
* **空间编码器（Nx=1）**：CoTNet 风格的 bottleneck——``Conv1d(k1)+BN+ReLU`` → 局部注意力
  （3×3 分组卷积的 key ``C1``、1×1 的 value、拼接 ``[X, C1]`` 后两层 1×1 生成沿时间维 softmax 的
  权重 ``γ``、``C2 = γ⊙V·m``、再用 split-attention 把 ``C1`` 与 ``C2`` 融合）→ ``BN+ReLU`` →
  8 头全局自注意力（三个独立 1×1 卷积产生 Q/K/V，各头拼接后**不做输出投影**）→
  ``Conv1d(k1)+BN`` → 残差相加 + ReLU + Dropout(0.5)；**不做时间降采样**（逐帧输出）；
* **时间嵌入**：对原始 IMU 的单层双向 LSTM（每方向 32）+ 可训练位置编码 ``Embedding(200, 64)``；
* **时间解码器（Nx=4）**：标准 post-norm Transformer 解码器层（因果掩码的自注意力 + 以编码器
  输出为 memory 的交叉注意力 + FFN 64→512→64），dropout 0.05；
* **输出头**：两个 ``Linear(64→64) → LayerNorm(64) → Linear(64→2)`` 分支，分别给速度与 ``log σ``；
* **多任务权重**：两个可学习标量 ``u_v = ln δv``、``u_c = ln δc``，初值 0。

与论文的差异及理由（规格卡 §6、§10）：

1. **逐样本旋转**：附录 A.2 一处说“按窗口起点姿态旋转”，另一处又说数据加载遵循 RoNIN 协议
   （逐样本旋转）。本实现取逐样本旋转（IPB ``frame=gravity_world``），因为 IPB 没有
   “按窗口起点整体旋转”的视图（卡 §10.2）；
2. **“linear + LayerNorm” 头不能按字面实现**：先 ``Linear(64→2)`` 再对 2 维输出做 LayerNorm 会把
   速度尺度抹掉（输出只能是 ``±γ+β``），因此改为 ``Linear → LN(64) → Linear(→2)``（卡 §10.3）；
3. **ResNet-18 “bottleneck”**：ResNet-18 本身用 BasicBlock，论文所述实为 CoTNet 风格的
   ``1×1 → 注意力 → 1×1``；取步长 1、不降采样，与逐帧 seq2seq 输出相容（卡 §10.4）；
4. **非因果**：解码器自注意力虽有上三角掩码，但 tgt 来自双向 LSTM、memory 来自非因果编码器，
   因此整体输出**不是**因果的（卡 §8、§10.5）；单元测试对此有显式断言，避免被误当作因果模型；
5. **不加阶段切换**：论文没有 MSE→NLL 热身（第三方 iMoT 复现时加过），``official`` 配方从第 0 轮
   起就联合训练速度损失与协方差损失（卡 §5、§10.7）；
6. **IVL 的积分位置项**用逐帧速度目标的累积和代替真值位置（``p_t − p_1 ≈ dt·Σ_{τ≤t} v_τ``），
   因为 IPB 的 ``target=frame_velocity`` 只提供逐帧速度（卡 §5 的推荐公式）；
7. **逐数据集的姿态选择器与步长（20/50/10）** 改为 benchmark 全局的 ``orientation`` 与
   ``stride``，否则不同数据集的数字不可比（卡 §6）；
8. **协方差参数化**取 ``log σ``（论文只说“由速度的两个系数参数化”，卡 §10.8）。
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn

from ...nn.base import BaseModel, InputSpec
from ...nn.losses import masked_output_mean, mse, register_loss
from ...nn.registry import register_model


# --------------------------------------------------------------------------- 专用损失
def _sum_squares(error: torch.Tensor) -> torch.Tensor:
    """逐时刻对分量求平方和，返回 ``(B, T)``。"""
    return (error ** 2).sum(dim=-1)


@register_loss("ctin_multitask")
class CTINMultiTaskLoss:
    """CTIN 的多任务损失（规格卡 §5 的推荐公式，式 6-8）。

    ``dt = 1/rate``，窗口内 ``t = 0..T−1``：

    * **IVL**：``Lpv = mean_t ‖dt·Σ_{τ≤t}(v̂_τ − v_τ)‖²``（积分位置误差）、
      ``Lev = mean_t ‖v̂_t − v_t‖²``（逐帧速度误差），``Lv = Lpv + Lev``；
    * **CNL**：``Lc = mean_t Σ_k [½(v_{t,k} − v̂_{t,k})²·e^{−2u_{t,k}} + u_{t,k}]``；
    * **同方差不确定度加权**（Kendall）：
      ``L = ½·e^{−2u_v}·Lv + ½·e^{−2u_c}·Lc + u_v + u_c``，``u_v``/``u_c`` 是模型里的两个
      可学习标量（初值 0，此时 ``L = ½(Lv + Lc)``）。

    掩码（DESIGN §3.1）：无效帧的误差先置零；``Lev``/``Lc`` 的第 ``t`` 项按 ``mask[:, t]`` 计入，
    ``Lpv`` 的第 ``t`` 项要求帧 ``0..t`` 全部有效；全部无效时返回 0 且保留计算图。
    视图只给窗口级目标时自动降级：``Lpv`` 消失、``Lv`` 与 ``Lc`` 用窗口级 ``vel``/``logstd``
    计算（卡 §6 的降级模式，结果中须标注 ``degraded``）。
    """

    name = "ctin_multitask"

    def __init__(self, rate: float = 200.0, position_weight: float = 1.0,
                 velocity_weight: float = 1.0) -> None:
        self.dt = 1.0 / float(rate)
        self.position_weight = float(position_weight)
        self.velocity_weight = float(velocity_weight)

    @staticmethod
    def _weights(out: dict, pred: torch.Tensor) -> tuple:
        """多任务权重 ``(u_v, u_c)``；模型未提供时退化为固定的 ``½(Lv + Lc)``。"""
        log_sigma = out.get("log_sigma")
        if log_sigma is None:
            zero = torch.zeros((), device=pred.device, dtype=pred.dtype)
            return zero, zero
        flat = log_sigma.reshape(-1, log_sigma.shape[-1]).mean(dim=0)
        return flat[0], flat[1]

    def _covariance(self, pred: torch.Tensor, logstd: Optional[torch.Tensor],
                    target: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if logstd is None:
            raise KeyError("ctin_multitask needs the model to output 'logstd'")
        terms = 0.5 * (target - pred) ** 2 * torch.exp(-2.0 * logstd) + logstd
        return masked_output_mean(terms.sum(dim=-1), mask)

    def __call__(self, out: dict, target: torch.Tensor, epoch: int = 0,
                 mask: Optional[torch.Tensor] = None) -> tuple:
        pred, logstd = out["vel"], out.get("logstd")
        u_v, u_c = self._weights(out, pred)
        items = {"mse": mse(pred, target, mask).detach()}
        if pred.ndim < 3:  # 窗口级降级布局：没有逐帧序列，积分位置项不存在
            lv = self.velocity_weight * masked_output_mean(_sum_squares(pred - target), mask)
            lc = self._covariance(pred, logstd, target, mask)
            items["degraded"] = torch.ones((), device=pred.device)
        else:
            error = pred - target
            if mask is not None:
                error = torch.where(mask.bool().unsqueeze(-1), error, torch.zeros_like(error))
            integrated = self.dt * torch.cumsum(error, dim=1)
            prefix_mask = None if mask is None else (
                torch.cumprod(mask.to(torch.bool).to(pred.dtype), dim=1) > 0.5)
            lpv = masked_output_mean(_sum_squares(integrated), prefix_mask)
            lev = masked_output_mean(_sum_squares(error), mask)
            lv = self.position_weight * lpv + self.velocity_weight * lev
            lc = self._covariance(pred, logstd, target, mask)
            items["lpv"] = lpv.detach()
            items["lev"] = lev.detach()
        loss = 0.5 * torch.exp(-2.0 * u_v) * lv + 0.5 * torch.exp(-2.0 * u_c) * lc + u_v + u_c
        items.update({"lv": lv.detach(), "lc": lc.detach(),
                      "u_v": u_v.detach(), "u_c": u_c.detach()})
        return loss, items


# --------------------------------------------------------------------------- 网络积木
class SpatialEmbedding(nn.Module):
    """``Conv1d(C→d, k3, p1) → BatchNorm1d → 逐时刻 Linear(d→d)``（规格卡 §4 的 S1-S3）。"""

    def __init__(self, in_channels: int, dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, dim, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(dim)
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(self.conv(x))
        return self.linear(x.transpose(1, 2)).transpose(1, 2)


class LocalAttention(nn.Module):
    """CoT 式局部注意力（规格卡 §4 的 E2a-E2e）。

    分组卷积 key + 1×1 value + 沿时间维 softmax 的权重 + split-attention 融合。
    """

    def __init__(self, dim: int, kernel_size: int = 3, groups: int = 4,
                 reduction: int = 2) -> None:
        super().__init__()
        hidden = max(dim // reduction, 1)
        self.key = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=groups, bias=False),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
        )
        self.value = nn.Sequential(nn.Conv1d(dim, dim, 1, bias=False), nn.BatchNorm1d(dim))
        self.attend = nn.Sequential(
            nn.Conv1d(2 * dim, hidden, 1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, dim, 1),
        )
        self.fuse = nn.Sequential(
            nn.Linear(dim, hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2 * dim),
        )
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]
        c1 = self.key(x)
        value = self.value(x)
        gamma = torch.softmax(self.attend(torch.cat([x, c1], dim=1)), dim=-1)
        c2 = gamma * value * length
        weights = self.fuse((c1 + c2).mean(dim=-1)).reshape(-1, 2, self.dim)
        weights = torch.softmax(weights, dim=1).unsqueeze(-1)
        return weights[:, 0] * c1 + weights[:, 1] * c2


class GlobalAttention(nn.Module):
    """多头缩放点积自注意力：Q/K/V 各一个 1×1 卷积，各头拼接后**不做输出投影**（规格卡 §4 E4）。"""

    def __init__(self, dim: int, heads: int = 8) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        self.query = nn.Conv1d(dim, dim, 1)
        self.key = nn.Conv1d(dim, dim, 1)
        self.value = nn.Conv1d(dim, dim, 1)
        self.heads = int(heads)
        self.scale = 1.0 / math.sqrt(dim // heads)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        batch, dim, length = x.shape
        return x.reshape(batch, self.heads, dim // self.heads, length).transpose(-1, -2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self._split(self.query(x)), self._split(self.key(x)), self._split(self.value(x))
        weights = torch.softmax(q @ k.transpose(-1, -2) * self.scale, dim=-1)
        out = (weights @ v).transpose(-1, -2)
        return out.reshape(x.shape)


class SpatialEncoderBlock(nn.Module):
    """CoTNet 风格 bottleneck：``1×1 → 局部注意力 → 全局自注意力 → 1×1``，残差 + ReLU + Dropout。"""

    def __init__(self, dim: int, heads: int = 8, kernel_size: int = 3, groups: int = 4,
                 reduction: int = 2, dropout: float = 0.5) -> None:
        super().__init__()
        self.reduce = nn.Sequential(nn.Conv1d(dim, dim, 1, bias=False), nn.BatchNorm1d(dim),
                                    nn.ReLU(inplace=True))
        self.local = LocalAttention(dim, kernel_size, groups, reduction)
        self.norm = nn.Sequential(nn.BatchNorm1d(dim), nn.ReLU(inplace=True))
        self.glob = GlobalAttention(dim, heads)
        self.expand = nn.Sequential(nn.Conv1d(dim, dim, 1, bias=False), nn.BatchNorm1d(dim))
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.expand(self.glob(self.norm(self.local(self.reduce(x)))))
        return self.dropout(self.relu(out + x))


class TemporalEmbedding(nn.Module):
    """原始 IMU 的单层双向 LSTM + 可训练位置编码（规格卡 §4 的 T1、T2）。"""

    def __init__(self, in_channels: int, dim: int, window: int) -> None:
        super().__init__()
        if dim % 2:
            raise ValueError(f"dim={dim} must be even for a bidirectional LSTM")
        self.lstm = nn.LSTM(in_channels, dim // 2, 1, batch_first=True, bidirectional=True)
        self.position = nn.Embedding(window, dim)
        self.window = int(window)

    def forward(self, imu: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(imu.transpose(1, 2))
        index = torch.arange(out.shape[1], device=imu.device)
        return out + self.position(index).unsqueeze(0)


class OutputHead(nn.Module):
    """``Linear(d→d) → LayerNorm(d) → Linear(d→dims)``（规格卡 §10.3 的修正版“linear + LN”头）。"""

    def __init__(self, dim: int, dims: int = 2) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.norm(self.proj(x)))


@register_model("ctin")
class CTIN(BaseModel):
    """CTIN：空间编码器 + 时间解码器 → 逐帧 2D 速度与 ``log σ``（参考规格 477,574 参数）。"""

    default_loss = "ctin_multitask"
    saved_outputs: tuple = ()  # 多任务权重与逐帧序列不逐窗口存盘

    def __init__(
        self,
        input_spec: InputSpec,
        dim: int = 64,
        heads: int = 8,
        encoder_layers: int = 1,
        decoder_layers: int = 4,
        ff_dim: int = 512,
        kernel_size: int = 3,
        groups: int = 4,
        reduction: int = 2,
        encoder_dropout: float = 0.5,
        decoder_dropout: float = 0.05,
    ) -> None:
        super().__init__(input_spec)
        window, dims = int(input_spec.window), int(input_spec.dims)
        self.spatial_embedding = SpatialEmbedding(input_spec.num_channels, dim, kernel_size)
        self.encoder = nn.ModuleList([
            SpatialEncoderBlock(dim, heads, kernel_size, groups, reduction, encoder_dropout)
            for _ in range(encoder_layers)])
        self.temporal_embedding = TemporalEmbedding(input_spec.num_channels, dim, window)
        self.decoder = nn.ModuleList([
            nn.TransformerDecoderLayer(dim, heads, ff_dim, decoder_dropout, activation="relu",
                                       batch_first=True)
            for _ in range(decoder_layers)])
        self.velocity_head = OutputHead(dim, dims)
        self.covariance_head = OutputHead(dim, dims)
        # u_v = ln δv、u_c = ln δc（Kendall 同方差不确定度加权），初值 0
        self.log_sigma = nn.Parameter(torch.zeros(1, 2))

    @staticmethod
    def causal_mask(length: int, device: torch.device) -> torch.Tensor:
        """严格上三角的 ``-inf`` 掩码（第 ``t`` 步只能看到 ``≤ t``）。"""
        return torch.triu(torch.full((length, length), float("-inf"), device=device), diagonal=1)

    def encode(self, imu: torch.Tensor) -> torch.Tensor:
        """空间编码器输出 ``(B, T, d)``（解码器的 memory）。"""
        x = self.spatial_embedding(imu)
        for block in self.encoder:
            x = block(x)
        return x.transpose(1, 2)

    def forward(self, imu: torch.Tensor) -> dict:
        self.check_input(imu)
        memory = self.encode(imu)
        tgt = self.temporal_embedding(imu)
        mask = self.causal_mask(tgt.shape[1], imu.device)
        for layer in self.decoder:
            tgt = layer(tgt, memory, tgt_mask=mask)
        vel_seq, logstd_seq = self.velocity_head(tgt), self.covariance_head(tgt)
        out = {"log_sigma": self.log_sigma.expand(imu.shape[0], 2)}
        if self.input_spec.output_layout == "frame":
            out.update({"vel": vel_seq, "logstd": logstd_seq})
            return out
        # 窗口级降级布局（规格卡 §3）：速度取均值，σ 取“误差完全相关”的保守近似
        out.update({"vel": vel_seq.mean(dim=1),
                    "logstd": 0.5 * torch.log(torch.exp(2.0 * logstd_seq).mean(dim=1)),
                    "vel_seq": vel_seq, "logstd_seq": logstd_seq})
        return out


__all__ = ["CTIN", "CTINMultiTaskLoss", "GlobalAttention", "LocalAttention", "OutputHead",
           "SpatialEmbedding", "SpatialEncoderBlock", "TemporalEmbedding"]
