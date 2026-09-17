"""IMUNet：深度可分离残差网络 + “噪声层”的窗口速度回归网络。

论文：B. Zeinali, H. Zanddizari, J. M. Chang, "IMUNet: Efficient Regression Architecture for
Inertial IMU Navigation and Positioning", IEEE TIM 73, 2024（DOI 10.1109/TIM.2024.3381717；
预印本 arXiv:2208.00068）。
官方仓库：https://github.com/BehnamZeinali/IMUNet，核对提交 c57f14d0f4f5。
许可：**官方仓库没有 LICENSE 文件**（默认保留全部权利）。本文件依据
`docs/algorithms/imunet.md` 规格卡从零编写，未阅读、未复制官方代码，也不分发官方权重。
fidelity：official-code（结构与训练配方均取自规格卡）。

结构（规格卡 §4，参数量 3,661,618 由单元测试锁定）：

* ``Conv1d(6, 64, k7, s2, p3) → BN → ReLU``（**ReLU**，其余块为 ELU）``→ MaxPool(k3, s2, p1)``；
* 12 个深度可分离块：``DSR(64) ×2 → DSP(64→64, s1) → DSR(64) → DSP(64→128, s2) → DSR(128)
  → DSP(128→256, s2) → DSR(256) → DSP(256→512, s2) → DSR(512) → DSP(512→1024, s2)
  → DSR(1024)``；``DSR`` 为恒等捷径、``DSP`` 为 1×1 投影捷径，两者主路径第二个卷积后已有
  ELU，相加后**再**过一次 ELU；
* ``Conv1d(1024, 400, k2, bias=True) → BN(400)``（无激活）→ 通道优先展平 ``(B, 1200)``；
* “噪声层” ``z = f − W ⊙ flatten(x) + b``（``W, b ∈ R^{1×1200}``，``W ~ N(0,1)``、``b = 0``）
  → ELU → ``Linear(1200, dims)``。

与官方的差异及理由（引用卡片）：

1. 结构、初始化（PyTorch 默认 + ``noise.W ~ N(0,1)``）与“按展平索引配对”的噪声层完全照卡实现；
   因此 ``window × channels`` 必须等于 1200（卡 §4、§10），构造时显式校验；
2. 输出维度取 ``input_spec.dims``（官方写死 2，卡 §4）；
3. 目标为 IPB 的 ``(p[s+T−1] − p[s]) / ((T−1)·dt)``，官方跨 200 个采样间隔（差 0.5%，卡 §6）；
4. 姿态：官方用设备姿态并在首帧做整旋转对齐，IPB 只补偿常值偏航；``input.orientation`` 取
   ``reference``（IPB 默认协议），复现官方测试时设 ``orientation=device``（卡 §2、§6）；
5. 推理时间戳取窗口中心并用梯形积分，修正官方的半窗错位（卡 §3、§6）；
6. 官方在训练前会以 ``train()`` 模式跑一遍前向来记录初始损失（会更新 BN 滑动统计），IPB 不复现
   这一步（卡 §5）；
7. 官方在 RIDI / 自采 / OxIOD 上以测试集作验证集选模，IPB 严格分离 val/test（卡 §5、§10）。
"""

from __future__ import annotations

import torch
from torch import nn

from ...nn.base import BaseModel, InputSpec
from ...nn.heads import VelocityHead
from ...nn.modules.resnet1d import conv1d_output_length
from ...nn.registry import register_model


class DSConvRegular(nn.Module):
    """``DSConv_Regular``：恒等捷径的深度可分离块（``C_in = C_out``、步长 1）。"""

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.depth_wise = nn.Conv1d(channels, channels, kernel_size, 1, pad, groups=channels,
                                    bias=False)
        self.bn_1 = nn.BatchNorm1d(channels)
        self.point_wise = nn.Conv1d(channels, channels, 1, bias=False)
        self.bn_2 = nn.BatchNorm1d(channels)
        self.act = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.bn_1(self.depth_wise(x)))
        y = self.act(self.bn_2(self.point_wise(y)))
        return self.act(y + x)

    @staticmethod
    def output_length(length: int) -> int:
        return length


class DSConvProject(nn.Module):
    """``DSConv``：带 1×1 投影捷径的深度可分离块（可改变通道与长度）。"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 kernel_size: int = 3) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.depth_wise = nn.Conv1d(in_channels, in_channels, kernel_size, stride, pad,
                                    groups=in_channels, bias=False)
        self.bn_1 = nn.BatchNorm1d(in_channels)
        self.point_wise = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.bn_2 = nn.BatchNorm1d(out_channels)
        self.downsample_ = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, stride, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.act = nn.ELU()
        self.stride = int(stride)
        self.kernel_size = int(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.bn_1(self.depth_wise(x)))
        y = self.act(self.bn_2(self.point_wise(y)))
        return self.act(y + self.downsample_(x))

    def output_length(self, length: int) -> int:
        return conv1d_output_length(length, self.kernel_size, self.stride, self.kernel_size // 2)


class NoiseLayer(nn.Module):
    """官方 ``CustomLayer``：``z = f − W ⊙ u + b``，``u`` 为展平后的**原始输入**。

    特征与输入按展平索引（通道优先）配对，二者没有语义对应关系，但必须照此复现（卡 §4）。
    """

    def __init__(self, features: int) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.randn(1, features))
        self.b = nn.Parameter(torch.zeros(1, features))

    def forward(self, features: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        return features - self.W * raw + self.b


@register_model("imunet")
class IMUNet(BaseModel):
    """IMUNet（官方 PyTorch 版，``RONIN_torch/IMUNet.py``）。"""

    default_loss = "mse"

    def __init__(
        self,
        input_spec: InputSpec,
        base_plane: int = 64,
        kernel_size: int = 3,
        trans_planes: int = 400,
        trans_kernel: int = 2,
        official_features: int = 1200,
    ) -> None:
        super().__init__(input_spec)
        c = base_plane
        flat = input_spec.num_channels * input_spec.window
        # 噪声层的参数在官方 __init__ 中最先创建，注册顺序必须保持在最前（夹具 param_shapes）
        self.noise = NoiseLayer(flat)
        self.input_block = nn.Sequential(
            nn.Conv1d(input_spec.num_channels, base_plane, 7, 2, 3, bias=False),
            nn.BatchNorm1d(base_plane),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, 2, 1),
        )
        self.conv_1_1 = DSConvRegular(c, kernel_size)
        self.conv_1_2 = DSConvRegular(c, kernel_size)
        self.conv_3_1 = DSConvProject(c, c, 1, kernel_size)
        self.conv_3_2 = DSConvRegular(c, kernel_size)
        self.conv_4_1 = DSConvProject(c, 2 * c, 2, kernel_size)
        self.conv_4_2 = DSConvRegular(2 * c, kernel_size)
        self.conv_5_1 = DSConvProject(2 * c, 4 * c, 2, kernel_size)
        self.conv_5_2 = DSConvRegular(4 * c, kernel_size)
        self.conv_6_1 = DSConvProject(4 * c, 8 * c, 2, kernel_size)
        self.conv_6_2 = DSConvRegular(8 * c, kernel_size)
        self.conv_7_1 = DSConvProject(8 * c, 16 * c, 2, kernel_size)
        self.conv_7_2 = DSConvRegular(16 * c, kernel_size)
        self.output_block = nn.Sequential(
            nn.Conv1d(16 * c, trans_planes, trans_kernel, bias=True),
            nn.BatchNorm1d(trans_planes),
        )
        self.act = nn.ELU()
        self.head = VelocityHead(flat, input_spec.dims)
        self.feature_length = self._feature_length(input_spec.window)
        if flat != official_features or trans_planes * self.feature_length != flat:
            raise ValueError(
                f"IMUNet's noise layer pairs the flattened features with the raw input by flat "
                f"index, so the official implementation is hard-wired to "
                f"{official_features} = 6×200 features (400 channels × length 3); got "
                f"{input_spec.num_channels}×{input_spec.window} inputs and "
                f"{trans_planes}×{self.feature_length} features")

    def _feature_length(self, window: int) -> int:
        length = conv1d_output_length(window, 7, 2, 3)
        length = conv1d_output_length(length, 3, 2, 1)
        for block in (self.conv_1_1, self.conv_1_2, self.conv_3_1, self.conv_3_2, self.conv_4_1,
                      self.conv_4_2, self.conv_5_1, self.conv_5_2, self.conv_6_1, self.conv_6_2,
                      self.conv_7_1, self.conv_7_2):
            length = block.output_length(length)
        conv = self.output_block[0]
        return conv1d_output_length(length, conv.kernel_size[0], conv.stride[0], 0)

    def features(self, imu: torch.Tensor) -> torch.Tensor:
        x = self.input_block(imu)
        for block in (self.conv_1_1, self.conv_1_2, self.conv_3_1, self.conv_3_2, self.conv_4_1,
                      self.conv_4_2, self.conv_5_1, self.conv_5_2, self.conv_6_1, self.conv_6_2,
                      self.conv_7_1, self.conv_7_2):
            x = block(x)
        return self.output_block(x)

    def forward(self, imu: torch.Tensor) -> dict:
        self.check_input(imu)
        x = self.features(imu).reshape(imu.shape[0], -1)
        x = self.act(self.noise(x, imu.reshape(imu.shape[0], -1)))
        return self.head(x)
