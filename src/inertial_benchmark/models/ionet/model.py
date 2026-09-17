"""IONet：双层双向 LSTM + 极坐标输出头的惯性里程计网络。

论文：C. Chen, X. Lu, A. Markham, N. Trigoni, "IONet: Learning to Cure the Curse of Drift in
Inertial Odometry", AAAI 2018. https://arxiv.org/abs/1802.02209
官方代码：**没有公开仓库**（RoNIN 论文 §5.1 明确写 “we use our local implementation, as the code
is not publicly available”）。fidelity 为 ``paper-only``：本文件依据 `docs/algorithms/ionet.md`
规格卡实现，卡中标为 [A] 的假设（dropout 位置、``h_n`` 读出、κ、batch、初始化）取卡片推荐默认值。

结构（规格卡 §4，按规格推导的参数量 302,978 由单元测试锁定）：

* 通道置换 ``[gyro, acc] → [acc, gyro]``（论文式 (18) 的 ``(a, w)``）；
* IPB 适配的无参数抽取 ``avg_pool1d(k=2, s=2)``：200 Hz×400（2 s）→ 100 Hz×200，即论文的
  “window length of 200 frames (2 s)”；
* ``BiLSTM(6 → 96/方向) → Dropout(0.25)`` → ``BiLSTM(192 → 96/方向)``，读出**``h_n`` 的前向与
  反向拼接**（TF ``Bidirectional(LSTM(return_sequences=False))`` 的语义，不是 ``out[:, -1]``）
  → ``Dropout(0.25)`` → ``Linear(192, 2)``；
* 极坐标头（无参数）：``s = out[:, 0]``、``ψ = out[:, 1]``，``vel = s·[cos ψ, sin ψ]``。

IPB 适配（规格卡 §6，**改变语义**，结果表必须注明 “IONet (gravity_world adaptation)”）：

1. 输入系由 ``body`` 改为 ``gravity_world``：官方的 ``Δψ`` 是相对航向变化，需要状态累加与初始航向；
   IPB 协议要求窗口独立、每个窗口给出世界系速度，而绝对航向只有在重力对齐系输入下才可观测；
2. 输出语义由 ``(Δl, Δψ)`` 改为 ``(s, ψ)``：``s`` 为窗口平均水平速度的模长（m/s）、``ψ`` 为其在
   世界系中的方向角；全连接层仍是 ``192 → 2``，参数量不变；
3. 损失改为 ``‖s̃ − s‖² + κ·m·wrap(ψ̃ − ψ)²``（``ionet_polar``），``m = 1[‖v̄‖ > 0.1 m/s]``、
   ``wrap(x) = atan2(sin x, cos x)``：近静止时方向没有定义，且 ``wrap`` 避免 ±π 跳变；
4. 卡 §6.3 的“忠实协议模式”（``frame=body``、``(Δl, Δψ)``、有状态航向累加）需要预测器的序列级
   钩子（`docs/ALGORITHMS.md` §3 的待落地扩展），v1 未实现。

与论文的其他差异：``hidden=96``、``lr=1.5e-3``（AAAI 版）；OxIOD 的 DeepIO 变体为 ``hidden=128``、
``lr=1e-4``（``model_args={hidden: 128}`` 即可，参数量 535,042）。论文“每种放置方式单独训练一个
模型”的做法属于数据划分，不在模型内实现。
"""

from __future__ import annotations

import torch
from torch import nn

from ...nn.base import BaseModel, InputSpec
from ...nn.losses import register_loss
from ...nn.registry import register_model

# IPB 的 [gyro_xyz, acc_xyz] → 论文的 (a, w) = [acc_xyz, gyro_xyz]
CHANNEL_PERMUTATION = (3, 4, 5, 0, 1, 2)


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    """把角度包裹到 ``(−π, π]``。"""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


@register_loss("ionet_polar")
class IONetPolarLoss:
    """极坐标损失 ``mean(s − s̃)² + κ·mean(m·wrap(ψ − ψ̃)²)``（规格卡 §6.1）。

    ``m = 1[‖v̄‖ > min_speed]``：近静止窗口的方向没有定义，不参与角度项。
    """

    def __init__(self, kappa: float = 1.0, min_speed: float = 0.1) -> None:
        self.kappa = float(kappa)
        self.min_speed = float(min_speed)

    def __call__(self, out: dict, target: torch.Tensor, epoch: int = 0) -> tuple:
        speed_target = target.norm(dim=-1)
        angle_target = torch.atan2(target[:, 1], target[:, 0])
        speed_term = torch.mean((out["speed"] - speed_target) ** 2)
        mask = (speed_target > self.min_speed).to(target.dtype)
        angle_term = torch.mean(mask * wrap_angle(out["heading"] - angle_target) ** 2)
        loss = speed_term + self.kappa * angle_term
        return loss, {"speed": speed_term.detach(), "heading": angle_term.detach(),
                      "mse": torch.mean((out["vel"] - target) ** 2).detach()}


@register_model("ionet")
class IONet(BaseModel):
    """IONet（双层 BiLSTM + 极坐标头；IPB 默认适配输出世界系水平速度）。"""

    default_loss = "ionet_polar"

    def __init__(
        self,
        input_spec: InputSpec,
        hidden: int = 96,
        dropout: float = 0.25,
        decimation: int = 2,
        permute_channels: bool = True,
    ) -> None:
        super().__init__(input_spec)
        if input_spec.dims != 2:
            raise ValueError("ionet's polar head predicts a horizontal vector and needs dims=2")
        self.decimation = int(decimation)
        if self.decimation > 1 and input_spec.window % self.decimation:
            raise ValueError(f"window {input_spec.window} is not a multiple of "
                             f"decimation {self.decimation}")
        self.lstm1 = nn.LSTM(input_spec.num_channels, hidden, batch_first=True,
                             bidirectional=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(2 * hidden, hidden, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden, 2)
        self.hidden = int(hidden)
        self.permute_channels = bool(permute_channels)

    def features(self, imu: torch.Tensor) -> torch.Tensor:
        """返回 ``h_n`` 拼接后的 ``(B, 2·hidden)`` 特征（未经 dropout）。"""
        x = imu[:, list(CHANNEL_PERMUTATION)] if self.permute_channels else imu
        if self.decimation > 1:
            x = nn.functional.avg_pool1d(x, self.decimation, self.decimation)
        sequence, _ = self.lstm1(x.transpose(1, 2))
        _, (state, _) = self.lstm2(self.dropout1(sequence))
        return torch.cat([state[0], state[1]], dim=-1)

    def forward(self, imu: torch.Tensor) -> dict:
        self.check_input(imu)
        polar = self.fc(self.dropout2(self.features(imu)))
        speed, heading = polar[:, 0], polar[:, 1]
        vel = speed.unsqueeze(-1) * torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
        return {"vel": vel, "speed": speed, "heading": heading}
