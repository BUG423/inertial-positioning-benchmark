"""一维 ResNet 积木（torchvision ResNet 思路的 1D 版本）。"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import nn


def conv1d_output_length(length: int, kernel: int, stride: int = 1, padding: int = 0,
                         dilation: int = 1) -> int:
    """``Conv1d``/``MaxPool1d`` 的输出长度公式。"""
    return (length + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


class BasicBlock1D(nn.Module):
    """两层 ``k×1`` 卷积的残差块；``stride`` 作用在第一层，``downsample`` 对齐捷径。"""

    expansion = 1

    def __init__(self, in_planes: int, planes: int, kernel_size: int = 3, stride: int = 1,
                 downsample: Optional[nn.Module] = None) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size, stride, pad, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size, 1, pad, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)

    def output_length(self, length: int) -> int:
        return conv1d_output_length(length, self.kernel_size, self.stride, self.kernel_size // 2)


class ResNet1DBackbone(nn.Module):
    """输入层（``Conv k7 s2`` + BN + ReLU + ``MaxPool k3 s2``）+ 若干残差阶段。

    第 ``i`` 个阶段通道数为 ``base_plane · 2^i``，首阶段步长 1，其余 2。输出 ``(B, C, L)``。
    """

    def __init__(self, in_channels: int, group_sizes: Sequence[int] = (2, 2, 2, 2),
                 base_plane: int = 64, kernel_size: int = 3) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_plane, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base_plane),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        stages = []
        in_planes = base_plane
        for i, blocks in enumerate(group_sizes):
            planes = base_plane * 2**i
            stride = 1 if i == 0 else 2
            downsample = None
            if stride != 1 or in_planes != planes:
                downsample = nn.Sequential(
                    nn.Conv1d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm1d(planes),
                )
            layers = [BasicBlock1D(in_planes, planes, kernel_size, stride, downsample)]
            layers += [BasicBlock1D(planes, planes, kernel_size) for _ in range(1, blocks)]
            stages.append(nn.Sequential(*layers))
            in_planes = planes
        self.stages = nn.Sequential(*stages)
        self.out_channels = in_planes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stages(self.stem(x))

    def output_length(self, length: int) -> int:
        length = conv1d_output_length(length, 7, 2, 3)
        length = conv1d_output_length(length, 3, 2, 1)
        for stage in self.stages:
            for block in stage:
                length = block.output_length(length)
        return length


def init_resnet_weights(module: nn.Module, zero_init_residual: bool = False) -> None:
    """卷积 Kaiming（fan_out）、BN 置 1/0、全连接 N(0, 0.01)，与 RoNIN 官方初始化一致。"""
    for m in module.modules():
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.01)
            nn.init.constant_(m.bias, 0.0)
    if zero_init_residual:
        for m in module.modules():
            if isinstance(m, BasicBlock1D):
                nn.init.constant_(m.bn2.weight, 0.0)
