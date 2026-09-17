"""RIO：旋转等变自监督的惯性里程计（RoNIN 骨干 + GroupNorm + 共轭旋转辅助损失）。

论文：X. Cao, C. Zhou, D. Zeng, Y. Wang, "RIO: Rotation-Equivariance Supervised Learning of
Robust Inertial Odometry", CVPR 2022, pp. 6614–6623；arXiv:2111.11676。
官方代码：**未找到公开仓库**（CVPR 版链接指向华为云 AI Gallery 数据集页，需 JS/登录，未能核实）。
因此 fidelity 为 ``paper-only``：本文件依据 `docs/algorithms/rio.md` 规格卡实现，卡中标注为假设的
细节（GN 组数、TTT 学习率、stop-gradient、epoch）都作为可配置项并取卡片推荐默认值。
骨干沿用 RoNIN（官方代码 GPL-3.0），我们使用的是 IPB 自己的 `models/ronin` 实现。

结构（规格卡 §4，参数量 4,634,882 与 `ronin_resnet18` 夹具相同）：RoNIN-ResNet18 把全部 21 个
``BatchNorm1d`` 换成 ``GroupNorm(32, C)``（为了小批量测试时训练）；GN 与 BN 的可学习参数量都是
``2C``，所以参数量不变，但模型**不再有任何 buffer**（没有 running statistics）。

训练（规格卡 §5.2，Algorithm 1）：每个样本抽 ``φ ~ U(0, 2π]``，把窗口绕 z 轴旋转 φ 得到共轭
输入；``forward`` 在训练模式下把原始与共轭窗口拼成 ``2B`` 的一批同时前向，返回
``vel``、``vel_conj`` 与 ``phi``。损失（``rio_joint``）为

    ``L = mean_i Σ_d (v̂_i − v_i^gt)² + w · mean_i gate_i · D(R(φ_i) v̂_i, v̂_i^c)``，

``D(a, b) = −⟨a, b⟩ / (‖a‖‖b‖ + eps)``（负余弦，完全一致时为 −1），
``gate_i = 1[‖v̂_i‖ > 0.5 m/s]``（用预测速度的范数，``detach``；测试时也只能如此）。

与论文/卡片的差异及理由：

1. 论文 Algorithm 1 写 ``Σ_{x,y,z}``，但 RoNIN 骨干是 2 维输出；按 ``dims`` 求和（卡 §6、§10.2）；
2. 论文目标公式漏除时长，按正文文字取 IPB ``avg_velocity``（卡 §10.3）；
3. 速度项按“逐样本对分量求和、再对样本取均值”实现（``mse_sum``），**不是** RoNIN 的
   ``nn.MSELoss()``（那会把速度项相对辅助项缩小 D 倍，卡 §5.2、§10.7）；
4. 论文未提 stop-gradient，默认两支都回传（损失的 ``stop_grad=false``，卡 §5.1）；
5. 门控默认用预测速度（``gate_on="pred"``），可切到 ``target``（仅训练期可用，卡 §10.4）；
6. GN 组数默认 32（Wu & He 2018 的默认值，卡 §4）；通道不能被整除时退化为 ``gcd``；
7. 验证/推理时没有共轭前向，损失只剩速度项（辅助项需要第二次前向，框架的验证器只提供目标）；
8. A-TTT（卡 §5.3）实现在 `models/rio/ttt.py`，是独立的工具类：IPB v1 的预测器还没有“序列级
   有状态钩子”（ALGORITHMS.md §3 的待落地扩展），因此 TTT 暂不接入 ``ipb val``。
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
from torch import nn

from ...nn.base import BaseModel, InputSpec
from ...nn.losses import register_loss
from ...nn.registry import register_model
from ..ronin.model import RoNINResNet


def rotate_imu_z(imu: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """绕世界 z 轴旋转 ``(B, 6, T)`` 的陀螺与加计（z 分量不变）。"""
    cos, sin = torch.cos(angle), torch.sin(angle)
    cos = cos.reshape(-1, 1)
    sin = sin.reshape(-1, 1)
    out = imu.clone()
    for base in (0, 3):
        x, y = imu[:, base], imu[:, base + 1]
        out[:, base] = cos * x - sin * y
        out[:, base + 1] = sin * x + cos * y
    return out


def rotate_vector_z(vec: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """绕 z 轴旋转 ``(B, D)`` 速度的水平分量（``D = 3`` 时 z 分量不变）。"""
    cos, sin = torch.cos(angle), torch.sin(angle)
    out = vec.clone()
    out[:, 0] = cos * vec[:, 0] - sin * vec[:, 1]
    out[:, 1] = sin * vec[:, 0] + cos * vec[:, 1]
    return out


def negative_cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """差异度量 ``D(a, b) = −⟨a, b⟩ / (‖a‖‖b‖ + eps)``（论文式 (2)）。"""
    norm = a.norm(dim=-1) * b.norm(dim=-1) + eps
    return -(a * b).sum(dim=-1) / norm


def replace_batchnorm_with_groupnorm(module: nn.Module, groups: int = 32) -> int:
    """就地把所有 ``BatchNorm1d`` 换成 ``GroupNorm``（保持注册顺序），返回替换个数。"""
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm1d):
            channels = child.num_features
            g = groups if channels % groups == 0 else math.gcd(groups, channels)
            setattr(module, name, nn.GroupNorm(max(g, 1), channels, eps=child.eps))
            count += 1
        else:
            count += replace_batchnorm_with_groupnorm(child, groups)
    return count


@register_loss("rio_joint")
class RIOJointLoss:
    """RIO 联合损失：速度项（逐分量求和）+ 旋转等变自监督项（负余弦、按速度门控）。"""

    def __init__(self, ssl_weight: float = 1.0, speed_gate: float = 0.5,
                 gate_on: str = "pred", stop_grad: bool = False, eps: float = 1e-8) -> None:
        if gate_on not in ("pred", "target"):
            raise ValueError(f"gate_on must be 'pred' or 'target', got {gate_on!r}")
        self.ssl_weight = float(ssl_weight)
        self.speed_gate = float(speed_gate)
        self.gate_on = gate_on
        self.stop_grad = bool(stop_grad)
        self.eps = float(eps)

    def gate(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        reference = target if self.gate_on == "target" else pred.detach()
        return (reference.norm(dim=-1) > self.speed_gate).to(pred.dtype)

    def ssl(self, out: dict, target: torch.Tensor) -> torch.Tensor:
        pred = out["vel"].detach() if self.stop_grad else out["vel"]
        rotated = rotate_vector_z(pred, out["phi"])
        terms = negative_cosine(rotated, out["vel_conj"], self.eps)
        return (self.gate(out["vel"], target) * terms).mean()

    def __call__(self, out: dict, target: torch.Tensor, epoch: int = 0) -> tuple:
        velocity = torch.mean(torch.sum((out["vel"] - target) ** 2, dim=-1))
        items = {"vel_loss": velocity.detach(),
                 "mse": torch.mean((out["vel"] - target) ** 2).detach()}
        loss = velocity
        if "vel_conj" in out and "phi" in out:
            ssl = self.ssl(out, target)
            items["ssl"] = ssl.detach()
            loss = loss + self.ssl_weight * ssl
        return loss, items


@register_model("rio")
class RIOResNet(BaseModel):
    """RIO 的 J-ResNet：GroupNorm 版 RoNIN-ResNet18，训练时额外前向一个共轭旋转窗口。"""

    default_loss = "rio_joint"

    def __init__(
        self,
        input_spec: InputSpec,
        group_sizes: Sequence[int] = (2, 2, 2, 2),
        base_plane: int = 64,
        kernel_size: int = 3,
        fc_dim: int = 512,
        trans_planes: int = 128,
        dropout: float = 0.5,
        gn_groups: int = 32,
        conjugate: bool = True,
    ) -> None:
        super().__init__(input_spec)
        if input_spec.frame == "body":
            raise ValueError("rio's rotation-equivariance loss needs a gravity-aligned frame")
        self.backbone = RoNINResNet(input_spec, group_sizes, base_plane, kernel_size, fc_dim,
                                    trans_planes, dropout)
        self.num_groupnorm = replace_batchnorm_with_groupnorm(self.backbone, gn_groups)
        self.feature_length = self.backbone.feature_length
        self.conjugate = bool(conjugate)

    def predict(self, imu: torch.Tensor) -> torch.Tensor:
        """只做一次前向，返回窗口速度（不含共轭分支）。"""
        return self.backbone(imu)["vel"]

    def conjugate_pass(self, imu: torch.Tensor, angle: torch.Tensor) -> tuple:
        """原始与共轭窗口拼成 ``2B`` 的一批前向（GroupNorm 使之与分别前向等价）。"""
        both = torch.cat([imu, rotate_imu_z(imu, angle)], dim=0)
        vel = self.backbone(both)["vel"]
        return vel[: imu.shape[0]], vel[imu.shape[0]:]

    def sample_angles(self, imu: torch.Tensor) -> torch.Tensor:
        """``φ ~ U(0, 2π]``，每个样本独立（K = 1）。"""
        return (1.0 - torch.rand(imu.shape[0], device=imu.device,
                                 dtype=imu.dtype)) * (2.0 * math.pi)

    def forward(self, imu: torch.Tensor, angle: Optional[torch.Tensor] = None) -> dict:
        if not (self.training and self.conjugate) and angle is None:
            return {"vel": self.predict(imu)}
        angle = self.sample_angles(imu) if angle is None else angle
        vel, vel_conj = self.conjugate_pass(imu, angle)
        return {"vel": vel, "vel_conj": vel_conj, "phi": angle}
