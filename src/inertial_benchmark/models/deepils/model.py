"""DeepILS：深度可分离残差 + CBAM 式注意力的窗口速度回归网络（RoNIN-ResNet18 的轻量变体）。

论文：O. Tariq, M. B. A. Dastagir, M. Bilal, D. Han, "DeepILS: Toward Accurate Domain-Invariant
AIoT-Enabled Inertial Localization System", IEEE IoT Journal 12(11):17153–17168, 2025
（DOI 10.1109/JIOT.2025.3538938）。论文全文在付费墙后，规格卡以官方代码行为为准。
官方仓库：https://github.com/OmerTariq-KAIST/DeepILS-IoT-Journal-2025，核对提交 4fd65aa95640。
许可：**无 LICENSE 文件**，且 `src/models/ResNet1D_dws.py` 等文件头写有 “All Rights Reserved /
Unauthorized copying or redistribution … is strictly prohibited”。因此本文件依据
`docs/algorithms/deepils.md` 规格卡**完全独立**编写（洁净室），未阅读、未复制官方代码，
也不分发官方权重。
fidelity：official-code（结构、损失与配方取自规格卡）。

结构（规格卡 §4，参数量 2,290,226 由单元测试锁定）：

* ``Conv1d(6, 64, k5, s2, p3) → BN → ReLU → MaxPool(k3, s2, p1)``（k=5 配 p=3 的非对称
  “same”，长度 200 → 101 → 51，必须照此实现）；
* 4 个残差组（每组 2 个 ``DWBlock``，通道 64/128/256/512，步长 1/2/2/2）。``DWBlock``：
  深度可分离卷积 → BN → ReLU → 深度可分离卷积 → BN →（无 ReLU）→ 通道注意力 CA →
  时间注意力 SA → 与捷径相加 → ReLU；
* ``Conv1d(512, 128, k1) → BN`` → 展平 ``128·7`` → ``FC 512 → ReLU → Dropout(0.5)
  → FC 512 → ReLU → Dropout(0.5) → FC dims``。

CA 为共享 MLP 的 ``sigmoid(MLP(avgpool) + MLP(maxpool))``，隐层宽度恒为 ``C // 16``
（官方 ``ratio`` 形参无效）；SA 为 ``sigmoid(Conv1d(2, 1, k7, p3)([mean_c, max_c]))``。

与官方的差异及理由（引用卡片）：

1. 损失为 ``MSE + mean|·|``（``mse_l1``，两项权重 1，卡 §5）；
2. 目标为 IPB 的 ``(p[s+T−1] − p[s]) / ((T−1)·dt)``，官方跨 200 个采样间隔（差 0.5%，卡 §6）；
3. 轨迹重建用 IPB 预测器（窗口中心时间戳 + 梯形积分），官方把速度记在窗口起点（卡 §3、§6）；
4. 姿态：官方测试用设备姿态（game RV），IPB 默认 ``reference``，另可设 ``orientation=device``
   单独报告（卡 §6）；
5. 官方随机平移带 ``max(window, ·)`` 钳位（前 200 帧的窗口被折叠），IPB 用对称 ``time_shift``
   并丢弃越界窗口（卡 §6）；
6. 官方在 RIDI/IMUNet/KIOD/INAIOD 上用测试集当验证集选模（KIOD/INAIOD 的 train 与 test 列表
   完全相同），IPB 严格分离并按 val 选模（卡 §5、§10）；
7. 官方按 val 平均 L1 选模、按 val MSE 调 ``ReduceLROnPlateau``；IPB 的 official 配方统一用
   验证损失（``mse + l1``）既选模也驱动调度器，这是两者唯一的选模差异（卡 §5）；
8. 全连接头的输入长度 ``128 × L``：``feature_length`` 缺省锁定官方的 7（只有 ``T ∈ [191, 222]``
   成立），设为 ``null`` 时按窗口自动推导（改变结构，需重新锁定参数量，卡 §6）。
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import nn

from ...nn.base import BaseModel, InputSpec
from ...nn.modules.resnet1d import conv1d_output_length, init_resnet_weights
from ...nn.registry import register_model


def depthwise_separable(in_channels: int, out_channels: int, kernel_size: int = 3,
                        stride: int = 1) -> nn.Sequential:
    """``depthwise(k, stride) → pointwise(1×1)``，均无偏置。"""
    return nn.Sequential(
        nn.Conv1d(in_channels, in_channels, kernel_size, stride, kernel_size // 2,
                  groups=in_channels, bias=False),
        nn.Conv1d(in_channels, out_channels, 1, bias=False),
    )


class ChannelAttention(nn.Module):
    """CBAM 通道注意力：``sigmoid(MLP(avgpool) + MLP(maxpool))``，隐层 ``C // 16``。"""

    def __init__(self, channels: int, ratio: int = 16) -> None:
        super().__init__()
        hidden = max(channels // 16, 1)  # 官方写死 16，ratio 形参无效
        self.fc = nn.Sequential(
            nn.Conv1d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.fc(x.mean(dim=2, keepdim=True))
        mx = self.fc(x.max(dim=2, keepdim=True).values)
        return torch.sigmoid(avg + mx)


class TemporalAttention(nn.Module):
    """CBAM “空间”（此处为时间）注意力：``sigmoid(Conv1d(2, 1, k)([mean_c, max_c]))``。"""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stats = torch.cat([x.mean(dim=1, keepdim=True), x.max(dim=1, keepdim=True).values], dim=1)
        return torch.sigmoid(self.conv1(stats))


class DWBlock(nn.Module):
    """DeepILS 残差块：两层深度可分离卷积 + CA + SA + 捷径。"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, downsample: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.conv1 = depthwise_separable(in_channels, out_channels, kernel_size, stride)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = depthwise_separable(out_channels, out_channels, kernel_size, 1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.ca = ChannelAttention(out_channels)
        self.sa = TemporalAttention()
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = int(stride)
        self.kernel_size = int(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        y = y * self.ca(y)
        y = y * self.sa(y)
        return self.relu(y + identity)

    def output_length(self, length: int) -> int:
        return conv1d_output_length(length, self.kernel_size, self.stride, self.kernel_size // 2)


@register_model("deepils")
class DeepILS(BaseModel):
    """DeepILS（官方 ``src/models/DeepILS.py`` 的洁净室重实现）。"""

    default_loss = "mse_l1"

    def __init__(
        self,
        input_spec: InputSpec,
        group_sizes: Sequence[int] = (2, 2, 2, 2),
        base_plane: int = 64,
        kernel_size: int = 3,
        input_kernel: int = 5,
        input_padding: int = 3,
        fc_dim: int = 512,
        trans_planes: int = 128,
        dropout: float = 0.5,
        feature_length: Optional[int] = 7,
    ) -> None:
        super().__init__(input_spec)
        self.input_block = nn.Sequential(
            nn.Conv1d(input_spec.num_channels, base_plane, input_kernel, 2, input_padding,
                      bias=False),
            nn.BatchNorm1d(base_plane),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, 2, 1),
        )
        length = conv1d_output_length(input_spec.window, input_kernel, 2, input_padding)
        length = conv1d_output_length(length, 3, 2, 1)
        groups = []
        in_planes = base_plane
        for i, blocks in enumerate(group_sizes):
            planes = base_plane * 2**i
            stride = 1 if i == 0 else 2
            downsample = None
            if stride != 1 or in_planes != planes:
                downsample = nn.Sequential(
                    nn.Conv1d(in_planes, planes, 1, stride, bias=False),
                    nn.BatchNorm1d(planes),
                )
            layers = [DWBlock(in_planes, planes, kernel_size, stride, downsample)]
            layers += [DWBlock(planes, planes, kernel_size) for _ in range(1, blocks)]
            for layer in layers:
                length = layer.output_length(length)
            groups.append(nn.Sequential(*layers))
            in_planes = planes
        self.residual_groups = nn.Sequential(*groups)
        if feature_length is not None and length != feature_length:
            raise ValueError(
                f"DeepILS' FC head is built for a temporal length of {feature_length} "
                f"(window 200 → 101 → 51 → 26 → 13 → 7); window={input_spec.window} gives "
                f"{length}. Pass model_args={{feature_length: null}} to derive it from the "
                "window (this changes the parameter count)")
        self.feature_length = length
        self.transition = nn.Sequential(
            nn.Conv1d(in_planes, trans_planes, 1, bias=False),
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
            nn.Linear(fc_dim, input_spec.dims),
        )
        init_resnet_weights(self)

    def forward(self, imu: torch.Tensor) -> dict:
        x = self.residual_groups(self.input_block(imu))
        return {"vel": self.fc(self.transition(x))}
