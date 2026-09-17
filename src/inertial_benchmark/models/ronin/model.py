"""RoNIN ResNet：一维 ResNet 平均速度回归网络。

论文：S. Herath, H. Yan, Y. Furukawa, "RoNIN: Robust Neural Inertial Navigation in the Wild:
Benchmark, Evaluations, & New Methods", ICRA 2020. https://arxiv.org/abs/1905.12853
官方仓库：https://github.com/Sachini/ronin（GPL-3.0），核对提交 805b7f0f28bb。
本文件依据论文与官方仓库公开的网络结构从零实现，未复制官方代码。

结构（与官方 ``ResNet1D + BasicBlock1D + FCOutputModule`` 一致，参数量由单元测试锁定）：

* 输入层 ``Conv1d(6, 64, k7, s2, p3) → BN → ReLU → MaxPool(k3, s2, p1)``；
* 4 个残差阶段，每阶段 ``group_sizes[i]`` 个 BasicBlock，通道 64/128/256/512，步长 1/2/2/2；
* 输出头 ``Conv1d(512, 128, k1) → BN → Flatten → FC(128·L, 512) → ReLU → Dropout(0.5)
  → FC(512, 512) → ReLU → Dropout(0.5) → FC(512, dims)``。

与官方实现的差异及理由：

1. 展平长度 ``L`` 按卷积公式由窗口长度精确计算；官方写作 ``window // 32 + 1``，两者在 200 帧时
   相同（L=7），但官方公式在部分窗口长度（如 64）下与真实长度不符并会报错；
2. 输出维度由 ``input_spec.dims`` 决定（官方 RoNIN 为 2D）；
3. 窗口目标按 IPB 协议定义为窗口首末样本的平均速度（官方为 ``p[i+200] − p[i]`` 除以对应时长），
   协议差异由 benchmark 统一处理，不属于模型结构。
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from ...nn.base import BaseModel, InputSpec
from ...nn.heads import VelocityHead
from ...nn.modules.resnet1d import ResNet1DBackbone, init_resnet_weights
from ...nn.registry import register_model


@register_model("ronin_resnet")
class RoNINResNet(BaseModel):
    """RoNIN 1D ResNet（resnet18/50/101 由 ``group_sizes``、``fc_dim`` 区分）。"""

    def __init__(
        self,
        input_spec: InputSpec,
        group_sizes: Sequence[int] = (2, 2, 2, 2),
        base_plane: int = 64,
        kernel_size: int = 3,
        fc_dim: int = 512,
        trans_planes: int = 128,
        dropout: float = 0.5,
        zero_init_residual: bool = False,
    ) -> None:
        super().__init__(input_spec)
        self.backbone = ResNet1DBackbone(input_spec.num_channels, group_sizes, base_plane,
                                         kernel_size)
        length = self.backbone.output_length(input_spec.window)
        if length < 1:
            raise ValueError(f"window {input_spec.window} is too short for RoNIN ResNet")
        self.transition = nn.Sequential(
            nn.Conv1d(self.backbone.out_channels, trans_planes, kernel_size=1, bias=False),
            nn.BatchNorm1d(trans_planes),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(trans_planes * length, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.head = VelocityHead(fc_dim, input_spec.dims)
        self.feature_length = length
        init_resnet_weights(self, zero_init_residual)

    def forward(self, imu: torch.Tensor) -> dict:
        x = self.backbone(imu)
        x = self.fc(self.transition(x))
        return self.head(x)
