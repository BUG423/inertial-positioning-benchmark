"""TinyOdom：因果 TCN + 线性头的窗口位移回归网络（MCU 友好）。

论文：S. S. Saha, S. S. Sandha, L. A. Garcia, M. Srivastava, "TinyOdom: Hardware-Aware Efficient
Neural Inertial Navigation", ACM IMWUT 6(2), Article 71, 2022（DOI 10.1145/3534594）。
官方仓库：https://github.com/nesl/tinyodom（BSD-3-Clause），核对提交 7958781baedd；
骨干来自 `keras-tcn==3.3.0`（MIT）。本文件依据 `docs/algorithms/tinyodom.md` 规格卡从零编写
（洁净室，PyTorch 版），未阅读、未复制官方代码。
fidelity：official-code（结构取自规格卡固定的 RoNIN 候选：``nb_filters=30, kernel_size=12,
dilations=[1,2,4,8,64]``，无 skip 求和、无归一化；官方未公布 NAS 最终结构）。

结构（规格卡 §4，IPB 配置参数量 100,481 由单元测试锁定）：

* 通道置换 ``(B, 6, T)[gyro, acc] → [acc, gyro]``（官方通道顺序，下标 ``[3,4,5,0,1,2]``）；
* 5 个 keras-tcn 残差块（膨胀 1/2/4/8/64）：每块
  ``因果 Conv(k12) → ReLU → SpatialDropout1D → 因果 Conv(k12) → ReLU → SpatialDropout1D
  → ReLU（冗余）``，捷径在 ``C_in ≠ F`` 时为 1×1 卷积（只有 block0），最后 ``ReLU(捷径 + 主支)``；
* 取**最后一个时间步** ``x[:, :, −1]`` → 视作长度 30 的序列做 ``MaxPool1d(2, 2)``（**沿滤波器轴**
  两两取最大）→ 展平 15 维 → ``Linear(15, 32)``（**线性激活**）→ ``dims`` 个 ``Linear(32, 1)`` 头。

因果卷积在左侧补 ``(k−1)·d`` 个零，输出长度保持 ``T``；实际感受野为
``1 + 2·(k−1)·Σd = 1739`` 个样本（keras-tcn 自带的 ``receptive_field`` 公式有误，不要用）。

与官方的差异及理由（引用卡片 §6）：

1. **输入通道 10 → 6**：IPB 格式没有磁力计，``step_mask`` 依赖整条序列的非因果统计量与第三方包
   （``pydometer``），无法在窗口内复现；因此 ``block0`` 的首卷积与捷径卷积的 ``in_channels``
   由 10 改为 6（少 1,560 个参数）。**这是改变结构**；
2. **视图 ``frame=body, dims=3``**：官方靠磁力计从机体系输入推断世界系航向；没有磁力计时世界系
   航向对网络不可观测，所以改为预测窗口末端**机体系**平均速度（DESIGN §3 要求 ``body`` 配
   ``dims=3``），并新增第三个头 ``velz``（+33 参数）。网络本身仍不接收任何姿态，姿态只用于目标
   构造与轨迹重建；
3. 目标由官方的 1.995 s 窗口位移改为 IPB 平均速度（线性缩放，不改变结构）；
4. 轨迹用 DESIGN §5 的时间轴梯形积分，替代官方 ``L_t = L_{t−1} + v_t/19``（系数偏大约 5%，且
   官方是把重建后的参考轨迹当真值比较）；
5. dropout 取 0.0（NAS 搜索范围 {0,…,0.4} 的下界；推理图中无法得知官方取值）；
6. 选模用 val ATE，替代官方的 ``monitor='loss'``（训练损失）；
7. 损失为各头 MSE 等权求和（``mse_sum``，与官方 ``{'velx':'mse','vely':'mse'}`` 一致）；
8. 权重初始化照 Keras：卷积 ``he_normal``（截断正态，σ = √(2/fan_in)/0.8796，±2σ 截断）、
   偏置 0；``Dense`` 为 ``glorot_uniform``、偏置 0；
9. 官方 NAS（Mango + 硬件在环）不属于 benchmark 范畴，不实现；可选的 ``physics_channel``
   （窗口内统计量版 step mask）也未实现，需要时另立变体。
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from ...nn.base import BaseModel, InputSpec
from ...nn.registry import register_model

# IPB 的 [gyro_xyz, acc_xyz] → 官方的 [acc_xyz, gyro_xyz]
CHANNEL_PERMUTATION = (3, 4, 5, 0, 1, 2)


def keras_he_normal_(weight: torch.Tensor, fan_in: int) -> None:
    """Keras ``he_normal``：截断正态，σ = √(2/fan_in)/0.8796，截断界 ±2σ。"""
    std = math.sqrt(2.0 / fan_in) / 0.8796256610342398
    nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)


class TCNResidualBlock(nn.Module):
    """keras-tcn 3.3.0 的残差块（因果卷积 ×2 + 通道级 dropout + 可选 1×1 捷径）。"""

    def __init__(self, in_channels: int, filters: int, kernel_size: int, dilation: int,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1D_0 = nn.Conv1d(in_channels, filters, kernel_size, dilation=dilation)
        self.conv1D_1 = nn.Conv1d(filters, filters, kernel_size, dilation=dilation)
        self.matching_conv1D = (nn.Conv1d(in_channels, filters, 1)
                                if in_channels != filters else None)
        self.dropout = nn.Dropout1d(dropout)     # SpatialDropout1D：整条通道一起丢弃
        self.pad = (kernel_size - 1) * dilation
        for conv in (self.conv1D_0, self.conv1D_1):
            keras_he_normal_(conv.weight, conv.in_channels * kernel_size)
            nn.init.zeros_(conv.bias)
        if self.matching_conv1D is not None:
            keras_he_normal_(self.matching_conv1D.weight, in_channels)
            nn.init.zeros_(self.matching_conv1D.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dropout(F.relu(self.conv1D_0(F.pad(x, (self.pad, 0)))))
        y = self.dropout(F.relu(self.conv1D_1(F.pad(y, (self.pad, 0)))))
        y = F.relu(y)                            # keras-tcn 主支末尾的冗余 ReLU
        shortcut = x if self.matching_conv1D is None else self.matching_conv1D(x)
        return F.relu(shortcut + y)


@register_model("tinyodom")
class TinyOdom(BaseModel):
    """TinyOdom TCN（IPB 适配：6 通道输入、机体系 3 维输出）。"""

    default_loss = "mse_sum"

    def __init__(
        self,
        input_spec: InputSpec,
        filters: int = 30,
        kernel_size: int = 12,
        dilations: Sequence[int] = (1, 2, 4, 8, 64),
        dropout: float = 0.0,
        pre_units: int = 32,
        pool_size: int = 2,
        permute_channels: bool = True,
    ) -> None:
        super().__init__(input_spec)
        blocks = []
        channels = input_spec.num_channels
        for dilation in dilations:
            blocks.append(TCNResidualBlock(channels, filters, kernel_size, dilation, dropout))
            channels = filters
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.MaxPool1d(pool_size, pool_size)
        pooled = filters // pool_size
        self.pre = nn.Linear(pooled, pre_units)
        self.heads = nn.ModuleList(nn.Linear(pre_units, 1) for _ in range(input_spec.dims))
        for linear in [self.pre, *self.heads]:
            nn.init.xavier_uniform_(linear.weight)   # Keras Dense: glorot_uniform
            nn.init.zeros_(linear.bias)
        self.permute_channels = bool(permute_channels)
        self.receptive_field = 1 + 2 * (kernel_size - 1) * sum(dilations)

    def forward(self, imu: torch.Tensor) -> dict:
        x = imu[:, list(CHANNEL_PERMUTATION)] if self.permute_channels else imu
        for block in self.blocks:
            x = block(x)
        x = x[:, :, -1].unsqueeze(1)             # 末时间步 → (B, 1, F)，沿滤波器轴池化
        x = self.pool(x).flatten(1)
        x = self.pre(x)                          # 线性激活（无非线性）
        return {"vel": torch.cat([head(x) for head in self.heads], dim=-1)}
