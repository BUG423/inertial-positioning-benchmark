"""IMUNet 仓库中的 4 个一维移动端骨干基线：MobileNetV1 / MobileNetV2 / MnasNet / EfficientNetB0。

论文：这 4 个网络是 *IMUNet: Efficient Regression Architecture for Inertial IMU Navigation and
Positioning*（B. Zeinali, H. Zanddizari, J. M. Chang, IEEE TIM 73, 2024；arXiv:2208.00068）
中的对比基线；原始骨干分别为 Howard et al. 2017（arXiv:1704.04861）、Sandler et al. CVPR 2018、
Tan et al. CVPR 2019、Tan & Le ICML 2019。
官方仓库：https://github.com/BehnamZeinali/IMUNet，核对提交 c57f14d0f4f5。
许可：**官方仓库没有 LICENSE 文件**（默认保留全部权利）；`RONIN_torch/` 中的这些文件又分别声明
改编自第三方教程/仓库，许可链不清。本文件依据规格卡
`docs/algorithms/imunet_mobilenet.md`、`imunet_mobilenetv2.md`、`imunet_mnasnet.md`、
`imunet_efficientnetb0.md` 从零编写，未阅读、未复制官方代码，也不分发官方权重。
fidelity：official-code（结构取自规格卡的逐层表，参数量由单元测试锁定）。

共同点（规格卡各 §2/§3/§5）：输入 ``(B, 6, 200)``（世界系 ``[gyro, acc]``）、输出 2 维窗口平均
速度、训练配方与 `imunet` 完全相同（MSE、Adam 1e-4、``ReduceLROnPlateau(0.1, 10)``、batch 128、
300 epoch、``RandomHoriRotate(2π)`` + ±5 样本随机平移）。4 个骨干都是全局池化头，因此参数量与
窗口长度无关（``window=100/400`` 也能前向）。

与官方的差异及理由：

1. **这些网络不是标准视觉版本**：EfficientNetB0 变体没有 SE、没有 drop-connect、``t=1`` 的块仍
   保留 1×1 扩张卷积、分类头 dropout 为 0.5（卡 `imunet_efficientnetb0.md` §4）；MnasNet 变体
   没有 SE，分类头含 ``Dropout(0.5)``；MobileNetV2 变体没有 dropout。一律保持官方差异；
2. **初始化**：官方 ``_initialize_weights()`` 只匹配 ``Conv2d/BatchNorm2d/Linear``，对一维模型
   只有 ``Linear`` 生效（``weight ~ N(0, 0.01²)``、``bias = 0``），卷积与 BN 保持 PyTorch 默认；
   MobileNetV1 没有自定义初始化（全部默认）。照此实现，不套用 He 初始化；
3. 输出维度取 ``input_spec.dims``（官方写死 2）；
4. 目标定义、姿态处理、推理时间戳与选模的协议差异同 `imunet`（见 `models/imunet/model.py`）；
5. Swish 用 ``nn.SiLU`` 实现（数值相同；官方因 PyTorch 1.4 无 SiLU 才自定义，卡 §6）。
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from ...nn.base import BaseModel, InputSpec
from ...nn.registry import register_model


def _init_linear_only(module: nn.Module) -> None:
    """官方 ``_initialize_weights`` 在一维模型上的实际效果：只初始化 ``Linear``。"""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)


class DSConv1d(nn.Module):
    """MobileNetV1 的深度可分离块：``DW → BN → ReLU → PW → BN → ReLU``（无残差）。"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 kernel_size: int = 3) -> None:
        super().__init__()
        self.feature = nn.Sequential()
        self.feature.add_module("dconv", nn.Conv1d(in_channels, in_channels, kernel_size, stride,
                                                   kernel_size // 2, groups=in_channels,
                                                   bias=False))
        self.feature.add_module("bn1", nn.BatchNorm1d(in_channels))
        self.feature.add_module("act1", nn.ReLU(inplace=True))
        self.feature.add_module("pconv", nn.Conv1d(in_channels, out_channels, 1, bias=False))
        self.feature.add_module("bn2", nn.BatchNorm1d(out_channels))
        self.feature.add_module("act2", nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature(x)


@register_model("imunet_mobilenet")
class IMUNetMobileNet(BaseModel):
    """一维 MobileNetV1（通道 ``[32, 64, 128, 256, 512, 1024]``，13 个 DS 块，全局平均池化）。"""

    default_loss = "mse"

    def __init__(self, input_spec: InputSpec, channels: Sequence[int] = (32, 64, 128, 256, 512,
                                                                         1024),
                 width_multiplier: float = 1.0, kernel_size: int = 3) -> None:
        super().__init__(input_spec)
        c = [int(v * width_multiplier) for v in channels]
        self.conv = nn.Module()
        self.conv.conv = nn.Conv1d(input_spec.num_channels, c[0], kernel_size, 2,
                                   kernel_size // 2, bias=False)
        self.conv.bn = nn.BatchNorm1d(c[0])
        self.conv.act = nn.ReLU(inplace=True)
        specs = [("dsconv1", c[0], c[1], 1), ("dsconv2", c[1], c[2], 2),
                 ("dsconv3", c[2], c[2], 1), ("dsconv4", c[2], c[3], 2),
                 ("dsconv5", c[3], c[3], 1), ("dsconv6", c[3], c[4], 2)]
        specs += [(f"dsconv7_{s}", c[4], c[4], 1) for s in "abcde"]
        specs += [("dsconv8", c[4], c[5], 2), ("dsconv9", c[5], c[5], 1)]
        self.features = nn.Sequential()
        for name, cin, cout, stride in specs:
            self.features.add_module(name, DSConv1d(cin, cout, stride, kernel_size))
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(c[5], input_spec.dims)

    def forward(self, imu: torch.Tensor) -> dict:
        x = self.conv.act(self.conv.bn(self.conv.conv(imu)))
        x = self.features(x)
        return {"vel": self.linear(self.avgpool(x).flatten(1))}


def conv_bn(in_channels: int, out_channels: int, stride: int, activation: nn.Module,
            kernel_size: int = 3) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False),
        nn.BatchNorm1d(out_channels),
        activation,
    )


def conv_1x1_bn(in_channels: int, out_channels: int, activation: nn.Module) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, 1, bias=False),
        nn.BatchNorm1d(out_channels),
        activation,
    )


class InvertedResidual1d(nn.Module):
    """倒残差块：``[PW→BN→act] → DW→BN→act → PW→BN``（线性瓶颈），``s=1 且通道相同`` 时加残差。

    ``expand_always=False``（MobileNetV2/MnasNet）时 ``t=1`` 的块跳过首个 1×1 扩张卷积；
    ``expand_always=True``（IMUNet 仓库的 EfficientNetB0 变体）时始终保留它。
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int, expand: int,
                 kernel_size: int = 3, activation: str = "relu6",
                 expand_always: bool = False) -> None:
        super().__init__()
        hidden = in_channels * expand
        self.use_res = stride == 1 and in_channels == out_channels
        act = nn.ReLU6(inplace=True) if activation == "relu6" else nn.SiLU(inplace=True)
        layers: list = []
        if expand != 1 or expand_always:
            layers += [nn.Conv1d(in_channels, hidden, 1, bias=False), nn.BatchNorm1d(hidden), act]
        layers += [
            nn.Conv1d(hidden, hidden, kernel_size, stride, kernel_size // 2, groups=hidden,
                      bias=False),
            nn.BatchNorm1d(hidden),
            act,
            nn.Conv1d(hidden, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x) if self.use_res else self.conv(x)


class _MobileBackbone(BaseModel):
    """``stem → 倒残差阶段 → 1×1 卷积(1280) → 池化 → 分类头`` 的共同骨架。"""

    default_loss = "mse"
    settings: Sequence[Sequence[int]] = ()
    activation = "relu6"
    expand_always = False
    head_dropout = 0.0
    sep_conv_first = False

    def __init__(self, input_spec: InputSpec, stem_channels: int = 32,
                 last_channels: int = 1280) -> None:
        super().__init__(input_spec)
        act = (nn.ReLU6(inplace=True) if self.activation == "relu6" else nn.SiLU(inplace=True))
        blocks = [conv_bn(input_spec.num_channels, stem_channels, 2, act)]
        channels = stem_channels
        for t, c, n, s, k in self.settings:
            for i in range(n):
                stride = s if i == 0 else 1
                if self.sep_conv_first and not blocks[1:]:
                    blocks.append(nn.Sequential(
                        nn.Conv1d(channels, channels, k, stride, k // 2, groups=channels,
                                  bias=False),
                        nn.BatchNorm1d(channels),
                        act,
                        nn.Conv1d(channels, c, 1, bias=False),
                        nn.BatchNorm1d(c),
                    ))
                else:
                    blocks.append(InvertedResidual1d(channels, c, stride, t, k, self.activation,
                                                     self.expand_always))
                channels = c
        blocks.append(conv_1x1_bn(channels, last_channels, act))
        self.features = nn.Sequential(*blocks)
        self.last_channels = last_channels
        self.classifier = self._make_classifier(last_channels, input_spec.dims)
        _init_linear_only(self)

    def _make_classifier(self, in_features: int, dims: int):
        if self.head_dropout > 0:
            return nn.Sequential(nn.Dropout(self.head_dropout), nn.Linear(in_features, dims))
        return nn.Linear(in_features, dims)

    def pool(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=2)

    def forward(self, imu: torch.Tensor) -> dict:
        x = self.features(imu)
        return {"vel": self.classifier(self.pool(x))}


@register_model("imunet_mobilenetv2")
class IMUNetMobileNetV2(_MobileBackbone):
    """一维 MobileNetV2（``(t, c, n, s)`` 与论文原表一致，无 dropout，时间维取均值）。"""

    settings = ((1, 16, 1, 1, 3), (6, 24, 2, 2, 3), (6, 32, 3, 2, 3), (6, 64, 4, 2, 3),
                (6, 96, 3, 1, 3), (6, 160, 3, 2, 3), (6, 320, 1, 1, 3))


@register_model("imunet_mnasnet")
class IMUNetMnasNet(_MobileBackbone):
    """一维 MnasNet-B1（无 SE；首块为 SepConv；分类头 ``Dropout(0.5) + Linear``）。"""

    settings = ((1, 16, 1, 1, 3), (3, 24, 3, 2, 3), (3, 40, 3, 2, 5), (6, 80, 3, 2, 5),
                (6, 96, 2, 1, 3), (6, 192, 4, 2, 5), (6, 320, 1, 1, 3))
    head_dropout = 0.5
    sep_conv_first = True

    def pool(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.adaptive_avg_pool1d(x, 1).flatten(1)


@register_model("imunet_efficientnetb0")
class IMUNetEfficientNetB0(_MobileBackbone):
    """IMUNet 仓库的一维 “EfficientNetB0”：Swish 激活、**无 SE / 无 drop-connect**、
    ``t=1`` 的块保留扩张卷积、分类头 ``Dropout(0.5) + Linear``。"""

    settings = ((1, 16, 1, 1, 3), (6, 24, 2, 2, 3), (6, 40, 2, 2, 5), (6, 80, 3, 2, 3),
                (6, 112, 3, 1, 5), (6, 192, 4, 2, 5), (6, 320, 1, 1, 3))
    activation = "swish"
    expand_always = True
    head_dropout = 0.5

    def pool(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.adaptive_avg_pool1d(x, 1).flatten(1)
