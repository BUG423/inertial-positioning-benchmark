"""GNIO：Motion Bank（全局运动原型注意力）+ 门控预测头的 TLIO 式惯性里程计。

论文：Dapeng Feng*, Yizhen Yin*, Zhiqiang Chen, Yuhua Qi, Hongbo Chen,
"GNIO: Gated Neural Inertial Odometry", arXiv:2603.15281v1 [cs.RO], 2026-03-16（投 RA-L）。
官方仓库：**无**（论文未给出代码链接，也未承诺发布）；许可：不适用（无代码，arXiv 非独占分发许可）。
fidelity：`paper-only`——本文件依据 `docs/algorithms/gnio.md` 规格卡实现，卡中标 [A] 的细节
（注意力实现与维度、池化方式、`u` 的下限、增强与调度）都作为可配置项并取卡片推荐默认值。

结构（卡 §4，参数量 **4,934,153** 由单元测试锁定；论文 Table I 报告 4.90 M）：

1. 骨干与 `tlio` 完全相同的 `ResNet1D(BasicBlock1D, [2,2,2,2], base_plane=64)`（3,846,144），
   输出 ``(B, 512, 7)``；
2. 时间维**全局平均池化** → ``f_k ∈ R^512``（卡 §10-4 的假设：论文只说得到 `f_k ∈ R^D`，
   用 TLIO 式展平 FC 颈部会多出 787,712 个参数、与 Table I 的 4.90 M 不符）；
3. **Motion Bank**：可学习原型 ``M ∈ R^{m×512}``（m=64，32,768 个参数）+ 一次多头注意力
   ``c_k = MHA(Q=f_k, K=V=M)``（``nn.MultiheadAttention(512, 8)``，1,050,624 个参数）。
   查询长度为 1，**没有时间维注意力**（卡 §10-8）；
4. 残差融合 ``h_k = f_k + c_k``（式 7）；
5. **门控预测头**（可复用的 `nn.heads.GatedHead`，式 8–10）：``s̃ = Softplus(W_s h)``、
   ``g = Tanh(W_g h)``、``d̂ = s̃ ⊙ g``（3,078），另一条线性支路给 ``u``（式 11，1,539）。

分解：3,846,144 + 32,768 + 1,050,624 + 3,078 + 1,539 = 4,934,153；``m ∈ {16,32,128}`` 时分别为
4,909,577 / 4,917,769 / 4,966,921，``dims=2`` 时 4,932,614（卡 §4.3）。

损失（式 13–14，`gnio_static_weighted`）：
``L = λ_MSE·‖d − d̂‖² + λ_NLL·(½‖d − d̂‖²_Σ̂ + ½log det Σ̂)``，``λ_MSE = 1e2``、``λ_NLL = 1e-4``。
**没有按 epoch 的阶段切换**：论文正文称 "two-stage training"，但式 13 是固定权重的静态加和，
所谓“分阶段”只来自 10⁶ 倍的权重比（卡 §10-1）。

与论文的差异及理由（引用规格卡）：

1. 通道顺序：式 (1) 写 ``m_t = [a_t, ω_t]``（加计在前），正文又说 "Following TLIO"（陀螺在前）。
   默认按 TLIO 顺序直接吃 IPB 的 ``[gyro, acc]``；``channel_order="acc_gyro"`` 可在模型内部置换
   （骨干首层是全通道卷积，顺序不影响参数量与表达能力，卡 §2、§10-5）；
2. 注意力的 ``d_k``、头数、是否有偏置/输出投影论文全未给出；本实现固定为
   ``nn.MultiheadAttention(embed_dim=512, num_heads=8)``（含 in/out 投影与偏置），这是卡 §4.3
   四个候选中与 4.90 M 最接近的一个（+0.7%），并在配置中显式声明；
3. ``u`` 没有 TLIO 的 ``max(u, log 1e-3)`` 下限（论文未提）；``min_logstd`` 作为可选保护，
   **默认关闭**（卡 §6）；
4. 不实现随机克隆 EKF（论文全部数值都是 EKF 融合后的结果）：IPB v1 只评“网络 + DESIGN §5
   统一积分”，与 `tlio`、`llio` 同一口径，因此**不能**与论文 Table II 直接对照（卡 §7）；
5. 偏置补偿：官方在旋转前扣除 EKF 估计的偏置，IPB 数据格式没有偏置字段，改用 TLIO 式的
   ``bias_shift`` 增强覆盖（卡 §2、§6）；
6. ``vel`` 键按 IPB 约定给出**目标量本身**（窗口位移 ``d̂``，单位 m），预测器再除以窗口跨度
   ``(T−1)·dt = 0.995 s`` 换算为平均速度，``logstd_vel = u − log(0.995)``（卡 §3）；
7. ``frame=gravity_yaw_local`` 以窗口**末端**偏航为锚，论文（继承 TLIO）以窗口起点为锚，
   两者只差一个全局偏航（卡 §10-6）。
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import nn

from ...nn.base import BaseModel, InputSpec
from ...nn.heads import GatedHead
from ...nn.losses import expand_mask, masked_mean, masked_output_mean, mse, register_loss
from ...nn.modules.resnet1d import ResNet1DBackbone, init_resnet_weights
from ...nn.registry import register_model

CHANNEL_ORDERS = ("gyro_acc", "acc_gyro")


class MotionBank(nn.Module):
    """Motion Bank（GNIO 式 3–7）：``m`` 个可学习全局运动原型 + 一次多头注意力 + 残差融合。

    ``forward(f)``：``f (B, D)`` → ``h = f + MHA(Q=f, K=V=M)``，另返回注意力权重 ``(B, m)``。
    查询序列长度恒为 1（对 ``m`` 个原型做一次加权平均），**不是**对时间步的注意力。
    """

    def __init__(self, dim: int = 512, prototypes: int = 64, heads: int = 8,
                 dropout: float = 0.0) -> None:
        super().__init__()
        if prototypes < 1:
            raise ValueError(f"prototypes must be >= 1, got {prototypes}")
        self.bank = nn.Parameter(torch.randn(int(prototypes), int(dim)))
        self.attention = nn.MultiheadAttention(int(dim), int(heads), dropout=float(dropout),
                                               batch_first=True)
        self.dim = int(dim)
        self.prototypes = int(prototypes)

    def forward(self, features: torch.Tensor) -> tuple:
        memory = self.bank.to(features.dtype).unsqueeze(0).expand(features.shape[0], -1, -1)
        context, weights = self.attention(features.unsqueeze(1), memory, memory,
                                          need_weights=True)
        return features + context.squeeze(1), weights.squeeze(1)


@register_loss("gnio_static_weighted")
class GNIOStaticWeightedLoss:
    """GNIO 式 13–14：``λ_MSE·L_MSE + λ_NLL·L_NLL`` 的**静态**加权和（无 epoch 切换）。

    ``L_MSE = mean_b ‖d − d̂‖²``（逐样本对 3 个轴求和再对 batch 取均值，式 13 的 ``‖·‖²``）；
    ``L_NLL = mean_b (½ Σ_i (d_i − d̂_i)²/exp(2u_i) + Σ_i u_i)``（式 14 的
    ``½‖·‖²_Σ̂ + ½log det Σ̂``，``Σ̂ = diag(exp 2u)``，省略 ``log 2π`` 常数）。

    ``min_logstd``/``max_logstd`` 缺省为 ``None``（论文没有 TLIO 的 ``log σ ≥ log 1e-3`` 下限）；
    两项都按逐输出掩码 ``mask`` 平均。``λ_MSE/λ_NLL`` 的 10⁶ 倍比值由单元测试锁定。
    """

    def __init__(self, mse_weight: float = 1.0e2, nll_weight: float = 1.0e-4,
                 min_logstd: Optional[float] = None,
                 max_logstd: Optional[float] = None) -> None:
        self.mse_weight = float(mse_weight)
        self.nll_weight = float(nll_weight)
        self.min_logstd = min_logstd
        self.max_logstd = max_logstd

    def __call__(self, out: dict, target: torch.Tensor, epoch: int = 0,
                 mask: Optional[torch.Tensor] = None) -> tuple:
        if "logstd" not in out:
            raise KeyError("gnio_static_weighted needs the model to output 'logstd'")
        pred, logstd = out["vel"], out["logstd"]
        if self.min_logstd is not None or self.max_logstd is not None:
            logstd = torch.clamp(logstd, min=self.min_logstd, max=self.max_logstd)
        error = (pred - target) ** 2
        squared = masked_output_mean(error.sum(dim=-1), mask)
        terms = 0.5 * error / torch.exp(2.0 * logstd) + logstd
        nll = masked_mean(terms.sum(dim=-1, keepdim=True),
                          expand_mask(mask, terms.sum(dim=-1, keepdim=True)))
        loss = self.mse_weight * squared + self.nll_weight * nll
        return loss, {"sq": squared.detach(), "nll": nll.detach(),
                      "mse": mse(pred, target, mask).detach()}


@register_model("gnio")
class GNIO(BaseModel):
    """GNIO：TLIO ResNet 骨干 + GAP + Motion Bank + 门控预测头 + 线性不确定度支路。"""

    default_loss = "gnio_static_weighted"
    # 预测文件只额外保存门控/幅值（注意力权重是 (B, m) 的诊断量，需要时用 saved_outputs 打开）
    saved_outputs = ("scale", "gate")

    def __init__(
        self,
        input_spec: InputSpec,
        group_sizes: Sequence[int] = (2, 2, 2, 2),
        base_plane: int = 64,
        kernel_size: int = 3,
        prototypes: int = 64,
        attention_heads: int = 8,
        attention_dropout: float = 0.0,
        motion_bank: bool = True,
        gated_head: bool = True,
        scale_fn: str = "softplus",
        gate_fn: str = "tanh",
        channel_order: str = "gyro_acc",
    ) -> None:
        super().__init__(input_spec)
        if channel_order not in CHANNEL_ORDERS:
            raise ValueError(f"channel_order must be one of {CHANNEL_ORDERS}, "
                             f"got {channel_order!r}")
        if input_spec.output_layout != "window":
            raise ValueError("gnio predicts one displacement per window "
                             "(target=displacement / avg_velocity / velocity_at_end)")
        self.channel_order = channel_order
        self.backbone = ResNet1DBackbone(input_spec.num_channels, group_sizes, base_plane,
                                         kernel_size)
        self.feature_length = self.backbone.output_length(input_spec.window)
        if self.feature_length < 1:
            raise ValueError(f"window {input_spec.window} is too short for the GNIO backbone")
        dim = self.backbone.out_channels
        self.motion_bank = MotionBank(dim, prototypes, attention_heads,
                                      attention_dropout) if motion_bank else None
        self.head = GatedHead(dim, input_spec.dims, scale_fn, gate_fn) if gated_head \
            else nn.Linear(dim, input_spec.dims)
        self.logstd_head = nn.Linear(dim, input_spec.dims)
        init_resnet_weights(self, zero_init_residual=False)
        if isinstance(self.head, GatedHead):
            # 门控偏置重新置零：init_resnet_weights 把所有 Linear 偏置置 0，这里只做显式说明
            nn.init.zeros_(self.head.gate_linear.bias)

    def order_channels(self, imu: torch.Tensor) -> torch.Tensor:
        """按论文式 (1) 把 IPB 的 ``[gyro, acc]`` 置换为 ``[acc, gyro]``（``acc_gyro`` 时）。"""
        if self.channel_order == "gyro_acc":
            return imu
        return torch.cat([imu[:, 3:6], imu[:, 0:3]], dim=1)

    def features(self, imu: torch.Tensor) -> tuple:
        """返回 ``(h_k, 注意力权重)``：骨干 → 时间维 GAP → Motion Bank 残差融合。"""
        self.check_input(imu)
        pooled = self.backbone(self.order_channels(imu)).mean(dim=-1)
        if self.motion_bank is None:
            return pooled, None
        return self.motion_bank(pooled)

    def forward(self, imu: torch.Tensor) -> dict:
        hidden, weights = self.features(imu)
        out = self.head(hidden) if isinstance(self.head, GatedHead) \
            else {"vel": self.head(hidden)}
        # vel 键按 IPB 约定给出目标量本身（窗口位移，m）；预测器再换算为平均速度
        out["logstd"] = self.logstd_head(hidden)
        if weights is not None:
            out["aux"] = {"attention": weights}
        return out
