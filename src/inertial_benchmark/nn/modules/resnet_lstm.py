"""ResNet-LSTM 序列骨干：逐子窗一维 ResNet + 跨子窗单向 LSTM + 两个全连接输出块。

RNIN-VIO（`docs/algorithms/rnin.md` §4）与 TartanIMU（`docs/algorithms/tartanimu.md` §4）的
主干结构完全相同，只是子窗长度、LSTM 宽度/层数与全连接宽度不同，因此在这里做成通用积木：

* ``input_block``：``Conv1d(C, 64, k7, s2, p3, bias=False) → BatchNorm1d → ReLU``；
  **没有 MaxPool**（这是它与 :class:`~inertial_benchmark.nn.modules.resnet1d.ResNet1DBackbone`
  ——TLIO/RoNIN 风格骨干——的关键区别）；
* ``residual_groups``：4 组，每组 ``layer_sizes[i]`` 个 :class:`ResBlock1D`，通道
  ``64/128/256/512``，组内首块步长 ``1/2/2/2``；捷径在通道或步长变化时为 ``Conv1d(k1) + BN``；
* ``resnet_post_pro``：``Conv1d(512, 128, k1, bias=False) → BatchNorm1d``（**无激活**）；
* 展平成 ``(B, S, 128·L)`` 后送入单向 ``LSTM``（``batch_first``，初始状态每次前向置零）；
* ``output_block1`` / ``output_block2``：两个结构相同、参数不共享的 :class:`FcBlock`，
  分别给出均值与 ``log σ``。

参数注册顺序即上面的顺序，与两张规格卡的 ``param_shapes`` 夹具一致。
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import nn

from .resnet1d import conv1d_output_length


class ResBlock1D(nn.Module):
    """残差块：``convs = [Conv, BN, ReLU, Conv, BN]``，捷径可选 ``downsample = [Conv1x1, BN]``。

    子模块名与官方一致（``convs.0/1/3/4``、``downsample.0/1``），前向为
    ``relu(convs(x) + shortcut(x))``。
    """

    def __init__(self, in_planes: int, planes: int, kernel_size: int = 3,
                 stride: int = 1) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.convs = nn.Sequential(
            nn.Conv1d(in_planes, planes, kernel_size, stride, pad, bias=False),
            nn.BatchNorm1d(planes),
            nn.ReLU(inplace=True),
            nn.Conv1d(planes, planes, kernel_size, 1, pad, bias=False),
            nn.BatchNorm1d(planes),
        )
        self.downsample: Optional[nn.Module] = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm1d(planes),
            )
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        return self.relu(self.convs(x) + identity)

    def output_length(self, length: int) -> int:
        return conv1d_output_length(length, self.kernel_size, self.stride, self.kernel_size // 2)


class FcBlock(nn.Module):
    """三层全连接输出块 ``fcs = [Linear, ReLU, Dropout, Linear, ReLU, Dropout, Linear]``。"""

    def __init__(self, in_features: int, fc_dim: int, out_dim: int,
                 dropout: float = 0.2) -> None:
        super().__init__()
        self.fcs = nn.Sequential(
            nn.Linear(in_features, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fcs(x)


class ResNetLSTMSeqNet(nn.Module):
    """逐子窗 ResNet + 序列 LSTM + 两个 :class:`FcBlock`。

    ``forward`` 与 :meth:`encode` 都接收 ``(B, S, C, L)``：``S`` 个子窗口先合并到批维独立过
    ResNet，再按序列送入 LSTM，因此 ``S`` 可变、参数量与 ``S`` 无关。
    """

    def __init__(
        self,
        in_channels: int,
        out_dim: int,
        sub_window: int,
        layer_sizes: Sequence[int] = (2, 2, 2, 2),
        base_plane: int = 64,
        post_channels: int = 128,
        kernel_size: int = 3,
        lstm_size: int = 256,
        lstm_layers: int = 1,
        lstm_dropout: float = 0.0,
        fc_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_block = nn.Sequential(
            nn.Conv1d(in_channels, base_plane, 7, 2, 3, bias=False),
            nn.BatchNorm1d(base_plane),
            nn.ReLU(inplace=True),
        )
        groups = []
        in_planes = base_plane
        for i, blocks in enumerate(layer_sizes):
            planes = base_plane * 2**i
            stride = 1 if i == 0 else 2
            layers = [ResBlock1D(in_planes, planes, kernel_size, stride)]
            layers += [ResBlock1D(planes, planes, kernel_size) for _ in range(1, blocks)]
            groups.append(nn.Sequential(*layers))
            in_planes = planes
        self.residual_groups = nn.Sequential(*groups)
        self.resnet_post_pro = nn.Sequential(
            nn.Conv1d(in_planes, post_channels, 1, bias=False),
            nn.BatchNorm1d(post_channels),
        )
        length = self.feature_length(sub_window)
        if length < 1:
            raise ValueError(f"sub-window {sub_window} is too short for the ResNet trunk")
        self.sub_window = int(sub_window)
        self.resnet_code = post_channels * length
        self.lstm = nn.LSTM(self.resnet_code, lstm_size, lstm_layers, batch_first=True,
                            dropout=lstm_dropout)
        self.lstm_size = int(lstm_size)
        self.output_block1 = FcBlock(lstm_size, fc_dim, out_dim, dropout)
        self.output_block2 = FcBlock(lstm_size, fc_dim, out_dim, dropout)

    def feature_length(self, length: int) -> int:
        """ResNet 输出的时间长度（按卷积公式精确计算）。"""
        length = conv1d_output_length(length, 7, 2, 3)
        for group in self.residual_groups:
            for block in group:
                length = block.output_length(length)
        return length

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, S, C, L)`` → ``(B, S, lstm_size)``；LSTM 初始状态每次前向置零。"""
        if x.ndim != 4:
            raise ValueError(f"expected (B, S, C, L) sub-window input, got {tuple(x.shape)}")
        batch, steps = x.shape[0], x.shape[1]
        feat = self.resnet_post_pro(self.residual_groups(self.input_block(x.flatten(0, 1))))
        feat = feat.reshape(batch, steps, -1)
        if feat.shape[-1] != self.resnet_code:
            raise ValueError(f"sub-window length {x.shape[-1]} gives a flattened ResNet code of "
                             f"{feat.shape[-1]}, expected {self.resnet_code}")
        out, _ = self.lstm(feat)
        return out

    def forward(self, x: torch.Tensor, predict_cov: bool = True) -> dict:
        """``(B, S, C, L)`` → ``{"mean": (B, S, out_dim)[, "logstd": (B, S, out_dim)]}``。"""
        feat = self.encode(x)
        flat = feat.reshape(-1, feat.shape[-1])
        shape = feat.shape[:2] + (-1,)
        out = {"mean": self.output_block1(flat).reshape(shape)}
        if predict_cov:
            out["logstd"] = self.output_block2(flat).reshape(shape)
        return out


def init_resnet_lstm_weights(module: nn.Module) -> None:
    """卷积 Kaiming(fan_out)、BN 1/0、全连接 ``N(0, 0.01²)``、LSTM 正交 + 遗忘门偏置置 1。

    与 RNIN（`rnin.md` §4）和 TartanIMU（`tartanimu.md` §4）的 ``_initialize`` 一致。
    PyTorch 的门顺序为 ``i, f, g, o``，因此遗忘门偏置是 ``bias_hh[H:2H]``。
    """
    for m in module.modules():
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.01)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LSTM):
            hidden = m.hidden_size
            for name, param in m.named_parameters():
                if name.startswith("weight"):
                    nn.init.orthogonal_(param)
                elif name.startswith("bias"):
                    nn.init.constant_(param, 0.0)
            for layer in range(m.num_layers):
                bias_hh = getattr(m, f"bias_hh_l{layer}")
                with torch.no_grad():
                    bias_hh[hidden:2 * hidden].fill_(1.0)
