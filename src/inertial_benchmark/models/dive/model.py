"""DIVE：局部重力对齐系的姿态 + 去重力加速度输入，回归窗口末端 3D 速度。

论文：A. Bajwa, C. C. Cossette, M. A. Shalaby, J. R. Forbes,
"DIVE: Deep Inertial-Only Velocity Aided Estimation for Quadrotors",
IEEE RA-L 9(4):3728–3734, 2024. DOI 10.1109/LRA.2024.3370006（无 arXiv 版本）。
官方仓库：https://github.com/decargroup/DIVE，核对提交 e982ddd0643c；许可：**MIT**（宽松）。
本文件依据 `docs/algorithms/dive.md` 规格卡独立实现（洁净室），未阅读、未复制官方代码。
fidelity：official-code（网络、输入构造、损失与阶段切换）。

.. warning::

   **IPB 适配改变了结构。** 官方是 400 Hz、3.5 s 窗（1400 样本、3D 输出，12,367,110 参数）；
   IPB 统一 200 Hz，按卡 §6 取 ``window=200``（1.0 s，与官方 DIDO EKF 脚本的默认窗长一致）
   与 ``dims=2``，两个头的 ``fc3`` 由 512→3 变成 512→2。参数量取 **IPB 变体的夹具值 7,516,420**；
   官方配置（1400 样本、3D）为 12,367,110，200 样本 3D 为 7,517,446
   （夹具 ``variant_total_params``）。
   官方论文的数值来自四旋翼数据集且是 EKF 融合后的结果，**不能**与 IPB 直接对照。

输入通道**不是** ``[gyro, acc]``，而是 ``[φ_k(3), a_k(3)]``（卡 §2）：

* ``γ`` 为窗口末端姿态的偏航，``C_γ = R_z(γ)``；``D_{T−1} = C_γᵀ C_end``（只含横滚/俯仰）；
* 向后积分 ``D_k = D_{k+1}·Exp(ω^b_k Δt)ᵀ``（只用**窗口末端一个姿态** + 陀螺，与官方一致：
  在线时这正是 EKF 能提供的量）；
* ``φ_k = Log(D_k)``（so(3) 对数，主值，角度 ∈ [0, π]）；
* ``a_k = D_k f^b_k − [0, 0, g]``，``g = 9.80665``（官方用 ``scipy.constants.g``，**不是** IPB 的
  9.81；差 1.5e-3 m/s²，作为 ``gravity`` 参数显式声明）；
* 陀螺本身**不作为通道输入**，只通过 ``φ`` 间接进入。

由于 IPB 的视图只给 ``[gyro, acc]``，模型用 ``extra_inputs=[orientation]``（逐样本姿态，非特权）
先把视图坐标系的 IMU 旋回机体系，再按上式重建官方的 6 个通道；这一步是恒等变换的组合，
数学上与官方“机体系 IMU + 末端姿态”的输入完全一致。

结构（卡 §4，参数量由夹具锁定）：TLIO 风格 ``ResNet1D(BasicBlock1D, [3,3,3,3])``（注意深度是
**3**，不是 TLIO 的 2）+ 两个结构相同、参数独立的 ``FcBlock``（``prep1`` k1 512→128 + BN → 展平
896 → fc 512 → fc 512 → fc dims），分别给 ``vel`` 与 ``logstd``。

损失（`dive_mse_then_nll`，卡 §5）：``epoch < 10`` 为 MSE，``epoch ≥ 10`` 为
``mean((v̂−v)²/(2·exp(2(s+1e-7))) + (s+1e-7))``（官方 ``loss.py`` 的 1e-7 偏移，无截断）；
MSE 阶段 ``logstd`` 头不在计算图中、不收梯度。

与官方的差异及理由（引用规格卡）：

1. 采样率与窗长（见上）：改变结构，参数量取 IPB 变体的夹具值；
2. ``dims=2``：IPB 指标在水平面，局部系 z 轴与世界 z 一致，水平分量可独立监督；对角 NLL 按轴
   独立，2D 化不改变损失形式（卡 §6）；
3. 偏航定义：官方取 ``γ = atan2(C[1,0], C[0,0])``（ZYX 偏航，俯仰接近 ±90° 时跳变）；IPB 按
   DESIGN §3 统一使用绕世界 z 轴的**扭转分量** ``ψ = 2·atan2(q_z, q_w)``（连续、偏航等变）。
   两者在 roll = 0 时逐位相同，一般姿态下相差约 ``pitch·roll/2``（卡 §10-9）；
4. 训练输入用数据集的真实 IMU：官方 ``self_augment=True`` 用**真值合成**的无噪 IMU 再加噪
   （IPB 数据格式没有真值 IMU 字段），噪声/偏置注入保留为增强（卡 §6、§10-2）；
5. 官方验证集同样逐次随机加噪、且第 0–9 轮与之后比较两种量纲的 ``val_loss``；IPB 的诚实协议要求
   验证确定，增强只在 train 上生效（卡 §10-4）；
6. AdamW 的默认 ``weight_decay=0.01`` 在官方实际生效（配置里没写出来），official 配方显式写出
   （卡 §5、§10-5）；
7. 不实现 EKF（IPB v1 只评网络 + DESIGN §5 积分）：官方**不做**网络直接积分，全部数值都来自
   SE_2(3) EKF，因此口径差异更大（卡 §3）；
8. 行人适用性：论文针对四旋翼（速度平滑、与姿态强相关），行人手持/口袋时姿态与运动方向解耦，
   本算法在 IPB 中作为“姿态显式编码”的跨平台对照（卡 §6）。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional, Sequence

import torch

from ...nn.base import BaseModel, InputSpec
from ...nn.losses import expand_mask, gaussian_nll, masked_mean, mse, register_loss
from ...nn.modules.lie_events import quat_to_matrix, so3_exp
from ...nn.registry import register_model

# 官方使用 scipy.constants.g（IPB 的 utils.geometry.GRAVITY 为 9.81，差 1.5e-3 m/s²）
SCIPY_GRAVITY = 9.80665
# DIVE 的输入通道：局部重力对齐系中的姿态对数与去重力加速度
DIVE_CHANNELS = ("phi_x", "phi_y", "phi_z", "acc_ga_x", "acc_ga_y", "acc_ga_z")


def so3_log_principal(rot: torch.Tensor) -> torch.Tensor:
    """``Log(R)`` 的主值（角度 ∈ [0, π]），与 PyPose / scipy 的 ``as_rotvec`` 一致。

    与 `nn.modules.lie_events.so3_log` 的区别：那里复刻了 NIO 官方“``|θ−π| < 1e-3`` 直接返回 0”
    的特判（该实现的既有缺陷），这里是数值稳定的通用实现，θ → π 时用对称部分恢复轴。
    """
    trace = rot[..., 0, 0] + rot[..., 1, 1] + rot[..., 2, 2]
    theta = torch.arccos(torch.clamp(0.5 * (trace - 1.0), -1.0, 1.0))
    antisymmetric = rot - rot.transpose(-1, -2)
    vee = 0.5 * torch.stack([antisymmetric[..., 2, 1], antisymmetric[..., 0, 2],
                             antisymmetric[..., 1, 0]], dim=-1)
    small = theta < 1e-6
    scale = torch.where(small, torch.ones_like(theta),
                        theta / torch.sin(theta).clamp_min(1e-30))
    phi = vee * scale.unsqueeze(-1)
    # θ → π 时 sin θ → 0，改用 (R + Rᵀ)/2 = I + (cos θ − 1)(I − nnᵀ) 恢复轴的方向
    near_pi = theta > torch.pi - 1e-3
    if bool(near_pi.any()):
        symmetric = 0.5 * (rot + rot.transpose(-1, -2))
        eye = torch.eye(3, dtype=rot.dtype, device=rot.device).expand(symmetric.shape)
        cosine = torch.cos(theta).unsqueeze(-1).unsqueeze(-1)
        outer = (symmetric - cosine * eye) / (1.0 - cosine).clamp_min(1e-30)
        axis = torch.sqrt(torch.diagonal(outer, dim1=-2, dim2=-1).clamp_min(0.0))
        axis = axis * torch.sign(torch.where(vee.abs() > 1e-12, vee, torch.ones_like(vee)))
        phi = torch.where(near_pi.unsqueeze(-1), axis * theta.unsqueeze(-1), phi)
    return phi


def heading_from_quaternion(quat: torch.Tensor) -> torch.Tensor:
    """绕世界 z 轴的扭转分量 ``ψ = 2·atan2(q_z, q_w)``（DESIGN §3 的航向定义，torch 版）。"""
    return 2.0 * torch.atan2(quat[..., 3], quat[..., 0])


def rotation_z(yaw: torch.Tensor) -> torch.Tensor:
    """``R_z(ψ)``，``(...) → (..., 3, 3)``。"""
    cos, sin = torch.cos(yaw), torch.sin(yaw)
    zero, one = torch.zeros_like(cos), torch.ones_like(cos)
    rows = [torch.stack([cos, -sin, zero], -1),
            torch.stack([sin, cos, zero], -1),
            torch.stack([zero, zero, one], -1)]
    return torch.stack(rows, dim=-2)


def body_measurements(imu: torch.Tensor, rot: torch.Tensor) -> tuple:
    """视图坐标系的 ``[gyro, acc]`` + 逐样本姿态 → 机体系量测 ``(ω^b, f^b)``，各 ``(B, T, 3)``。"""
    return (torch.einsum("btji,btj->bti", rot, imu[:, 0:3].transpose(1, 2)),
            torch.einsum("btji,btj->bti", rot, imu[:, 3:6].transpose(1, 2)))


def local_attitude_frames(body_gyro: torch.Tensor, end_rotation: torch.Tensor,
                          end_yaw: torch.Tensor, dt: float) -> torch.Tensor:
    """向后积分得到 ``D_k = C_γᵀ C_k``，``(B, T, 3, 3)``（官方 ``dido_preprocessor`` 的做法）。

    ``D_{T−1} = C_γᵀ C_end``，``D_k = D_{k+1}·Exp(ω^b_k Δt)ᵀ``。只用**窗口末端一个姿态**，
    其余时刻全部由陀螺推出——在线时这正是 EKF 能提供的量。
    """
    current = rotation_z(end_yaw).transpose(-1, -2) @ end_rotation
    frames = [current]
    for k in range(body_gyro.shape[1] - 2, -1, -1):
        current = current @ so3_exp(body_gyro[:, k] * dt).transpose(-1, -2)
        frames.append(current)
    return torch.stack(frames[::-1], dim=1)


def gravity_aligned_input(imu: torch.Tensor, quat: torch.Tensor, dt: float,
                          gravity: float = SCIPY_GRAVITY) -> torch.Tensor:
    """官方 ``generate_gravity_aligned_input`` 的等价实现，返回 ``(B, 6, T)`` 的 ``[φ, a]``。

    ``imu`` 为视图坐标系的 ``[gyro, acc]``，``quat`` 为同一坐标系的逐样本姿态 ``(B, T, 4)``。
    """
    rot = quat_to_matrix(quat)                                   # (B, T, 3, 3)
    body_gyro, body_acc = body_measurements(imu, rot)
    frames = local_attitude_frames(body_gyro, rot[:, -1], heading_from_quaternion(quat[:, -1]), dt)
    phi = so3_log_principal(frames)
    acc = torch.einsum("btij,btj->bti", frames, body_acc)
    acc = acc - torch.tensor([0.0, 0.0, gravity], dtype=acc.dtype, device=acc.device)
    return torch.cat([phi, acc], dim=-1).transpose(1, 2)


@register_loss("dive_mse_then_nll")
class DIVEMSEThenNLL:
    """DIVE 调度（官方 ``loss.py`` + ``modules.py``）：前 ``switch_epoch`` 轮 MSE，之后对角 NLL。

    NLL 为 ``mean((v̂−v)²/(2·exp(2(s+1e-7))) + (s+1e-7))``——官方在 ``s`` 上加了 1e-7 的偏移且
    **不做截断**；``min_logstd``/``max_logstd`` 缺省为 ``None`` 以保持这一行为。
    MSE 阶段 ``logstd`` 头不接收梯度（以 ``0·logstd.sum()`` 保留计算图）。
    """

    def __init__(self, switch_epoch: int = 10, offset: float = 1e-7,
                 min_logstd: Optional[float] = None,
                 max_logstd: Optional[float] = None) -> None:
        self.switch_epoch = int(switch_epoch)
        self.offset = float(offset)
        self.min_logstd = min_logstd
        self.max_logstd = max_logstd

    def stage(self, epoch: int) -> str:
        return "mse" if epoch < self.switch_epoch else "nll"

    def __call__(self, out: dict, target: torch.Tensor, epoch: int = 0,
                 mask: Optional[torch.Tensor] = None) -> tuple:
        if "logstd" not in out:
            raise KeyError("dive_mse_then_nll needs the model to output 'logstd'")
        pred, logstd = out["vel"], out["logstd"]
        if self.stage(epoch) == "mse":
            loss = masked_mean((pred - target) ** 2, expand_mask(mask, pred))
            return loss + 0.0 * logstd.sum(), {"mse": loss.detach()}
        nll = gaussian_nll(pred, logstd + self.offset, target, self.min_logstd, self.max_logstd,
                           mask)
        return nll, {"nll": nll.detach(), "mse": mse(pred, target, mask).detach()}


@register_model("dive")
class DIVE(BaseModel):
    """DIVE：``[φ, a]`` 输入的 TLIO 风格 ResNet1D（``group_sizes=[3,3,3,3]``）+ 两个独立的头。"""

    default_loss = "dive_mse_then_nll"

    def __init__(
        self,
        input_spec: InputSpec,
        group_sizes: Sequence[int] = (3, 3, 3, 3),
        base_plane: int = 64,
        kernel_size: int = 3,
        fc_dim: int = 512,
        trans_planes: int = 128,
        dropout: float = 0.5,
        gravity: float = SCIPY_GRAVITY,
    ) -> None:
        from ..tlio.model import TLIOResNet

        super().__init__(input_spec)
        if "orientation" not in input_spec.extra_inputs:
            raise ValueError("dive builds its input channels from the per-sample attitude; "
                             "declare extra_inputs: [orientation]")
        if input_spec.frame == "body":
            raise ValueError("dive works in a gravity-aligned frame (frame=gravity_yaw_local)")
        if input_spec.output_layout != "window":
            raise ValueError("dive predicts one velocity per window (target=velocity_at_end)")
        if input_spec.sub_windows:
            raise ValueError("dive takes a single window (history=0)")
        self.gravity = float(gravity)
        self.backbone = TLIOResNet(replace(input_spec, channels=DIVE_CHANNELS), group_sizes,
                                   base_plane, kernel_size, fc_dim, trans_planes, dropout)
        self.feature_length = self.backbone.feature_length

    def preprocess(self, imu: torch.Tensor, extra: dict) -> torch.Tensor:
        """``(B, 6, T)`` 的 ``[gyro, acc]`` + 逐样本姿态 → 官方的 ``[φ, a]`` 通道。"""
        self.check_input(imu)
        return gravity_aligned_input(imu, extra["orientation"].to(imu.dtype),
                                     self.input_spec.dt, self.gravity)

    def forward(self, imu: torch.Tensor, extra: dict) -> dict:
        return self.backbone(self.preprocess(imu, extra))
