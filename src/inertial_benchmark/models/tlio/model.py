"""TLIO ResNet：1 s 窗口 → 3D 位移 + 对角 log σ 的一维 ResNet。

论文：W. Liu, D. Caruso, E. Ilg, J. Dong, A. I. Mourikis, K. Daniilidis, V. Kumar, J. Engel,
"TLIO: Tight Learned Inertial Odometry", IEEE RA-L 5(4):5653–5660, 2020.
https://arxiv.org/abs/2007.01867
官方仓库：https://github.com/CathIAS/TLIO（BSD-3-Clause），核对提交 d7051c4ff148。
fidelity：official-code（网络、损失与阶段切换）。本文件依据 `docs/algorithms/tlio.md`
规格卡从零实现，未阅读、未复制官方代码。

结构（规格卡 §4，参数量 5,424,646 由单元测试锁定）：

* 输入块 ``Conv1d(6, 64, k7, s2, p3) → BN → ReLU → MaxPool(k3, s2, p1)``；
* 4 个残差组（每组 2 个 BasicBlock，通道 64/128/256/512，步长 1/2/2/2）；
* **两个互不共享的输出块**接在同一特征上：``Conv1d(512, 128, k1) → BN`` →（无激活）展平
  ``128·L`` → ``FC 512 → ReLU → Dropout(0.5) → FC 512 → ReLU → Dropout(0.5) → FC dims``；
  第一个输出块给均值（位移），第二个给 ``logstd``。

与官方的差异及理由（引用规格卡）：

1. 展平长度 ``L`` 由卷积公式精确计算；官方写作 ``(past+window+future)//32 + 1``，在 100/200/400
   帧下与真实长度一致（卡 §4、§8），窗口取其他值时官方公式会形状不匹配；
2. 目标为 IPB 的 ``target=displacement``（``p[s+T−1] − p[s]``，与官方 ``targ_dt_World[:, -1]``
   是同一对样本，卡 §6）。按 IPB 模型接口，``forward`` 的 ``vel`` 键给出的是**目标量本身**
   （即位移，单位 m），损失因此和官方一样在位移尺度上计算；框架的预测器再按 DESIGN §5 把它
   除以窗口跨度 ``(T−1)·dt = 0.995 s`` 换算为窗口平均速度（官方 ``test.py`` 除以 1.0 s，
   带来约 0.5% 的系统性低估，IPB 不沿用）；
3. ``frame=gravity_yaw_local`` 以窗口**末端**偏航为锚，官方 EKF 以窗口起点为锚、训练数据为
   世界系 + U[−π, π] 随机偏航；三者只差一个全局偏航（卡 §2、§6）；
4. 保留 3 维输出（``dims=3``）：改成 2 维会改变 ``fc3`` 形状（参数量 5,423,620）与 NLL 的轴数；
5. 不实现随机克隆 EKF（卡 §6）：v1 只评网络 + IPB 积分，等价于论文的 “3D-RONIN”；
6. 数据侧不做 IMU 偏置补偿（IPB 格式无偏置字段），靠 ``bias_shift`` 增强覆盖（卡 §2、§6）；
7. ``Bottleneck`` 误用 ``BatchNorm2d``、``_initialize`` 引用未定义类等官方死代码不移植（卡 §4）。
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from ...nn.base import BaseModel, InputSpec
from ...nn.modules.resnet1d import ResNet1DBackbone, init_resnet_weights
from ...nn.registry import register_model


class FcBlock(nn.Module):
    """TLIO 输出块：``1×1 卷积 + BN``（无激活）→ 展平 → 3 层全连接（前两层 ReLU + Dropout）。"""

    def __init__(self, in_channels: int, length: int, out_dim: int, trans_planes: int = 128,
                 fc_dim: int = 512, dropout: float = 0.5) -> None:
        super().__init__()
        self.prep1 = nn.Conv1d(in_channels, trans_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(trans_planes)
        self.fc1 = nn.Linear(trans_planes * length, fc_dim)
        self.fc2 = nn.Linear(fc_dim, fc_dim)
        self.fc3 = nn.Linear(fc_dim, out_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn1(self.prep1(x)).flatten(1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        return self.fc3(x)


@register_model("tlio")
class TLIOResNet(BaseModel):
    """TLIO ``ResNet1D``：位移均值头 + ``logstd`` 头（两头结构相同、参数不共享）。"""

    default_loss = "nll_detach_then_nll"

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
            raise ValueError(f"window {input_spec.window} is too short for TLIO ResNet")
        self.feature_length = length
        head_args = (self.backbone.out_channels, length, input_spec.dims, trans_planes, fc_dim,
                     dropout)
        self.mean_head = FcBlock(*head_args)
        self.logstd_head = FcBlock(*head_args)
        init_resnet_weights(self, zero_init_residual)

    def forward(self, imu: torch.Tensor) -> dict:
        x = self.backbone(imu)
        # vel 键按 IPB 约定给出“目标量”，即窗口位移（m）；预测器再换算为平均速度
        return {"vel": self.mean_head(x), "logstd": self.logstd_head(x)}
