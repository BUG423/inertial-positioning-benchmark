"""IONext：自适应动态卷积（ADM）+ 自适应门控（AGU）的惯性里程计骨干。

论文：Shanshan Zhang, Qi Zhang, Siyue Wang, Tianshui Wen, Liqin Wu, Ziheng Zhou, Xuemin Hong,
Ao Peng, Lingxiang Zheng, Yu Yang, "IONext: Unlocking the Next Era of Inertial Odometry",
arXiv:2507.17089（v1 2025-07-23，**v2 2025-10-11**；截至规格卡编写时仍是在审预印本）。
官方仓库：**无**（v2 称审稿结束后发布）；许可：无代码，arXiv 预印本。
fidelity：`paper-only`——本文件依据 `docs/algorithms/ionext.md` 规格卡实现，**以 v2 为准**，
v2 删去的结构表（每阶段通道/核组合）取自 v1 表 I。

结构（卡 §4.1，参数量 **10,641,506** 为本卡的解析计数，由单元测试锁定；论文 v1 表 I 报告
``1.1×10⁷``，两位有效数字一致）：

``stem(Conv k4 s4 6→96 + BN)`` →（4 个阶段，深度 ``[2,2,6,2]``，通道 ``[96,192,384,768]``，
阶段间 ``BN + Conv k2 s2`` 非重叠下采样）→ ``BN + 时间维 GAP + Linear(768, dims)``。

ADE 块（``C`` 通道，``h = C/2``，卡 §4.2，单块 ``4.5C² + 35C``）：

``X' = X + ADM(BN₁(X))``；``X_out = X' + AGU(BN₂(X'))``（v2 式 (1)(2) 的预归一化残差）

* **ADM**：沿通道均分为两翼；每翼 3 条并行**深度卷积**（翼 0 核 ``(1,3,11)``、翼 1 核
  ``(1,5,17)``，``padding = k//2`` 保持长度），融合权重由 ``GAP_T → Conv1d(h, 3h, 1)`` 给出、
  在**分支轴**上 softmax（``ω`` 形状 ``(B,3,h,1)``，沿分支求和为 1）；两翼结果拼接后过
  ``Conv1d(C, C, 1)``；
* **AGU**：门控支路 ``concat(GAP_T(Z), GMP_T(Z)) → Conv1d(2C, C, 1) → sigmoid`` 得到
  ``ξ ∈ (0,1)^{C×1}``（**通道轴**门控，v2 的定义），乘到 ``Conv1d(C, C, 3, groups=C)`` 的输出上。

网络中**没有** ReLU/GELU：非线性只有 softmax 融合权重、sigmoid 门控与乘法交互（卡 §4.1）。

标 [假设] 的部分（论文未写，卡 §4.1 按 ConvNeXt 惯例补全，共 4,074 个参数，不影响与
``1.1×10⁷`` 的核对）：stem 后的归一化、每个下采样卷积前的归一化、头部（BN + GAP + Linear）、
AGU 值支路的核大小 3、softmax 的轴。``softmax_axis`` 与 ``value_kernel`` 保留为配置项。

与论文的差异及理由（引用规格卡）：

1. ``dims=2``：论文未写明输出维度，与 RoNIN 系基线一致取 2（``dims=3`` 时 10,642,275，卡 §8）；
2. 输入直接用 IPB 的 ``[gyro, acc]``：stem 是 6 输入通道的全通道卷积，**输入通道的任意置换都能
   被 stem 权重吸收**，从头训练时完全等价（卡 §2）；
3. ``frame=gravity_world``（假设）：论文未明示输入系，其对比基线都用 RoNIN 式重力对齐世界系；
4. **划分**：论文把 6 个数据集**全部随机重划分 8:1:1**，IPB 用官方划分（官方优先 + 分组防泄漏），
   因此论文数值与 IPB **不可直接比较**；论文的指标还做了长度归一化并乘以 100（卡 §7）；
5. v1/v2 结构描述不一致（v1 的门控在时间轴、式 (5) 每翼前多一个 BN）：本实现按 **v2**，理由是
   只有 v2 的通道轴门控能和 v1 表 I 的 ``1.1×10⁷`` 对上（v1 读法约 6.3×10⁶，卡 §10-1）；
6. softmax 的轴：v2 写 “along the channel dimension”，但权重是 ``3h`` 维、之后切成 3 段。
   “沿全部 ``3h`` 个通道 softmax” 会让每个系数小到约 ``1/(3h)``，与“多尺度融合系数”的语义不符；
   本实现取**每个通道在 3 个分支之间 softmax**，并保留 ``softmax_axis`` 配置项（卡 §10-2）；
7. 调度器、增强、训练步长、卷积 bias、初始化论文均未给出，按卡 §5 的推断填充并标注（卡 §10-4）。
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from ...nn.base import BaseModel, InputSpec
from ...nn.registry import register_model

SOFTMAX_AXES = ("branch", "channel")


class AdaptiveDynamicModule(nn.Module):
    """ADM：通道二分 → 每翼 3 条并行深度卷积 → 自适应融合 → 逐点卷积（卡 §4.2 a–h）。"""

    def __init__(self, channels: int, wing0_kernels: Sequence[int] = (1, 3, 11),
                 wing1_kernels: Sequence[int] = (1, 5, 17),
                 softmax_axis: str = "branch") -> None:
        super().__init__()
        if channels % 2:
            raise ValueError(f"ADM splits the channels in two halves, got {channels}")
        if softmax_axis not in SOFTMAX_AXES:
            raise ValueError(f"softmax_axis must be one of {SOFTMAX_AXES}, got {softmax_axis!r}")
        half = channels // 2
        self.half = half
        self.softmax_axis = softmax_axis
        self.kernels = (tuple(wing0_kernels), tuple(wing1_kernels))
        self.wings = nn.ModuleList(
            nn.ModuleList(nn.Conv1d(half, half, k, padding=k // 2, groups=half) for k in kernels)
            for kernels in self.kernels)
        self.weights = nn.ModuleList(
            nn.Conv1d(half, len(kernels) * half, 1) for kernels in self.kernels)
        self.project = nn.Conv1d(channels, channels, 1)

    def fusion_weights(self, wing: int, x: torch.Tensor) -> torch.Tensor:
        """``(B, h, T) → (B, branches, h, 1)``，沿所选轴 softmax（分支轴求和为 1）。"""
        branches = len(self.kernels[wing])
        raw = self.weights[wing](x.mean(dim=-1, keepdim=True))
        raw = raw.reshape(x.shape[0], branches, self.half, 1)
        if self.softmax_axis == "branch":
            return torch.softmax(raw, dim=1)
        return torch.softmax(raw.reshape(x.shape[0], branches * self.half, 1),
                             dim=1).reshape(x.shape[0], branches, self.half, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        halves = (x[:, : self.half], x[:, self.half:])
        fused = []
        for wing, part in enumerate(halves):
            weights = self.fusion_weights(wing, part)
            branches = torch.stack([conv(part) for conv in self.wings[wing]], dim=1)
            fused.append((branches * weights).sum(dim=1))
        return self.project(torch.cat(fused, dim=1))


class AdaptiveGatingUnit(nn.Module):
    """AGU：``ξ = sigmoid(W3([GAP_T(Z), GMP_T(Z)]))``（通道轴门控）乘上深度卷积（卡 §4.2 k–m）。"""

    def __init__(self, channels: int, value_kernel: int = 3) -> None:
        super().__init__()
        self.gate = nn.Conv1d(2 * channels, channels, 1)
        self.value = nn.Conv1d(channels, channels, value_kernel, padding=value_kernel // 2,
                               groups=channels)

    def gating(self, x: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat([x.mean(dim=-1, keepdim=True),
                            x.amax(dim=-1, keepdim=True)], dim=1)
        return torch.sigmoid(self.gate(pooled))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gating(x) * self.value(x)


class ADEBlock(nn.Module):
    """ADE 块：``X' = X + ADM(BN₁(X))``、``X_out = X' + AGU(BN₂(X'))``（v2 式 (1)(2)）。"""

    def __init__(self, channels: int, wing0_kernels: Sequence[int] = (1, 3, 11),
                 wing1_kernels: Sequence[int] = (1, 5, 17), value_kernel: int = 3,
                 softmax_axis: str = "branch") -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm1d(channels)
        self.adm = AdaptiveDynamicModule(channels, wing0_kernels, wing1_kernels, softmax_axis)
        self.norm2 = nn.BatchNorm1d(channels)
        self.agu = AdaptiveGatingUnit(channels, value_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.adm(self.norm1(x))
        return x + self.agu(self.norm2(x))


@register_model("ionext")
class IONext(BaseModel):
    """IONext 骨干：``stem → 4 个 ADE 阶段（含下采样）→ BN + GAP + 线性头``。"""

    default_loss = "mse"

    def __init__(
        self,
        input_spec: InputSpec,
        depths: Sequence[int] = (2, 2, 6, 2),
        widths: Sequence[int] = (96, 192, 384, 768),
        stem_kernel: int = 4,
        downsample_kernel: int = 2,
        wing0_kernels: Sequence[int] = (1, 3, 11),
        wing1_kernels: Sequence[int] = (1, 5, 17),
        value_kernel: int = 3,
        softmax_axis: str = "branch",
    ) -> None:
        super().__init__(input_spec)
        if len(depths) != len(widths):
            raise ValueError(f"depths {list(depths)} and widths {list(widths)} must match")
        if input_spec.output_layout != "window":
            raise ValueError("ionext predicts one average velocity per window")
        block = dict(wing0_kernels=wing0_kernels, wing1_kernels=wing1_kernels,
                     value_kernel=value_kernel, softmax_axis=softmax_axis)
        # stem：非重叠卷积（借鉴 Swin/ConvNeXt），后接 BN [假设]
        self.stem = nn.Sequential(
            nn.Conv1d(input_spec.num_channels, widths[0], stem_kernel, stem_kernel),
            nn.BatchNorm1d(widths[0]),
        )
        stages, downsamples = [], []
        for index, (depth, width) in enumerate(zip(depths, widths)):
            if index:
                # 下采样前的归一化 [假设]（ConvNeXt 惯例）
                downsamples.append(nn.Sequential(
                    nn.BatchNorm1d(widths[index - 1]),
                    nn.Conv1d(widths[index - 1], width, downsample_kernel, downsample_kernel),
                ))
            stages.append(nn.Sequential(*[ADEBlock(width, **block) for _ in range(depth)]))
        self.stages = nn.ModuleList(stages)
        self.downsamples = nn.ModuleList(downsamples)
        self.norm = nn.BatchNorm1d(widths[-1])                      # 末端归一化 [假设]
        self.head = nn.Linear(widths[-1], input_spec.dims)          # 头部 [假设]
        self.stage_lengths = self._stage_lengths(int(input_spec.window), stem_kernel,
                                                 downsample_kernel, len(depths))
        if self.stage_lengths[-1] < 1:
            raise ValueError(f"window {input_spec.window} is too short for IONext")
        self.feature_length = self.stage_lengths[-1]

    @staticmethod
    def _stage_lengths(window: int, stem_kernel: int, downsample_kernel: int,
                       stages: int) -> list:
        length = (window - stem_kernel) // stem_kernel + 1
        lengths = [length]
        for _ in range(stages - 1):
            length = (length - downsample_kernel) // downsample_kernel + 1
            lengths.append(length)
        return lengths

    def features(self, imu: torch.Tensor) -> torch.Tensor:
        self.check_input(imu)
        x = self.stem(imu)
        for index, stage in enumerate(self.stages):
            if index:
                x = self.downsamples[index - 1](x)
            x = stage(x)
        return self.norm(x)

    def forward(self, imu: torch.Tensor) -> dict:
        return {"vel": self.head(self.features(imu).mean(dim=-1))}
