"""O(2) 等变积木与规范帧网络（EqNIO 系列使用）。

向量张量形状为 ``(..., 2, C)``（倒数第二维是水平面的 x/y，最后一维是通道），标量张量为
``(..., C)``。在 ``Q ∈ O(2)`` 作用下向量按 ``V ↦ Q V``、标量不变；本模块的所有算子都保持这一
性质，因此规范帧 ``F`` 满足 ``F ↦ Q F``，用 ``Fᵀ`` 规范化后的特征对 O(2) 不变。

用法（见 ``models/eqnio``）：``o2_frame_features`` 把 ``(B, 6, T)`` 的 ``[gyro, acc]`` 变成
向量/标量特征，``O2FrameNet`` 输出两个向量，``gram_schmidt_frame`` 把它们正交化为规范帧。
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

EPS_NORM = 1e-6
EPS_FRAME = 1e-8
EPS_PREPROCESS = 1e-7


class VNLinear(nn.Module):
    """向量通道线性层：``V ↦ V Wᵀ``（无偏置，x/y 两个分量共享权重）。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.vector_linear = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, vec: torch.Tensor) -> torch.Tensor:
        return self.vector_linear(vec)


class EqLayerNorm(nn.Module):
    """LayerNorm，但偏移 ``beta`` 是恒为 0 的 buffer（不参与训练），只有 ``gamma`` 可学。"""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))
        self.dim = int(dim)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (self.dim,), self.gamma, self.beta, self.eps)


class VNLayerNorm(nn.Module):
    """向量 LayerNorm：对逐通道范数做 LayerNorm，方向保持不变。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.ln = EqLayerNorm(channels)

    def forward(self, vec: torch.Tensor) -> torch.Tensor:
        norm = torch.linalg.vector_norm(vec, dim=-2)
        return vec / norm.clamp_min(EPS_NORM).unsqueeze(-2) * self.ln(norm).unsqueeze(-2)


class VNNonLinearity(nn.Module):
    """等变非线性：``z = LN(ReLU(W [‖V‖ ‖ s]))``；``V' = z_v ⊙ V/‖V‖``，``s' = z_s``。

    ``sca_out = 0`` 时只返回向量分支（标量输出被丢弃）。
    """

    def __init__(self, vec_channels: int, sca_channels: int, sca_out: int) -> None:
        super().__init__()
        self.vec_channels = int(vec_channels)
        self.sca_out = int(sca_out)
        self.linear = nn.Linear(vec_channels + sca_channels, vec_channels + sca_out, bias=False)
        self.layer_norm = EqLayerNorm(vec_channels + sca_out)

    def forward(self, vec: torch.Tensor, sca: torch.Tensor) -> tuple:
        norm = torch.linalg.vector_norm(vec, dim=-2)
        z = self.layer_norm(F.relu(self.linear(torch.cat([norm, sca], dim=-1))))
        gain = z[..., : self.vec_channels]
        out_vec = gain.unsqueeze(-2) * vec / norm.clamp_min(EPS_NORM).unsqueeze(-2)
        out_sca = z[..., self.vec_channels:] if self.sca_out else None
        return out_vec, out_sca


class EqConv(nn.Module):
    """沿时间维的等变卷积：向量与标量各一条无偏置卷积，x/y 共享权重。

    ``padding='same'`` + ``padding_mode='replicate'``；偶数核的左右填充为
    ``((k−1)//2, k//2)``（k=32 时为 15/16），与官方实现一致。
    """

    def __init__(self, vec_in: int, vec_out: int, sca_in: int, sca_out: int,
                 kernel_size: int = 32) -> None:
        super().__init__()
        self.conv_layer_vec = nn.Conv2d(vec_in, vec_out, (kernel_size, 1), bias=False)
        self.conv_layer_sca = nn.Conv2d(sca_in, sca_out, (kernel_size, 1), bias=False)
        self.pad = ((kernel_size - 1) // 2, kernel_size // 2)

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        return F.pad(x, (0, 0, self.pad[0], self.pad[1]), mode="replicate")

    def forward(self, vec: torch.Tensor, sca: torch.Tensor) -> tuple:
        v = vec.permute(0, 3, 1, 2)                     # (B, C, T, 2)
        v = self.conv_layer_vec(self._pad(v)).permute(0, 2, 3, 1)
        s = sca.permute(0, 2, 1).unsqueeze(-1)          # (B, C, T, 1)
        s = self.conv_layer_sca(self._pad(s)).squeeze(-1).permute(0, 2, 1)
        return v, s


class O2FrameNet(nn.Module):
    """EqNIO 的 O(2) 规范帧网络：向量/标量特征 → ``out_vectors`` 个等变向量。

    ``(B, T, 2, vec_in)`` 与 ``(B, T, sca_in)`` → ``(B, 2, out_vectors)``（按列排）。
    结构顺序（与官方注册顺序一致）：输入线性 → 非线性 → ``depth`` 个
    ``[时间卷积 → 非线性 → 向量 LN → 标量 LN]`` → 时间均值 → 线性 → 非线性 → 线性 →
    向量 LN → 输出线性。
    """

    def __init__(self, vec_in: int = 3, sca_in: int = 9, hidden: int = 64, depth: int = 2,
                 kernel_size: int = 32, out_vectors: int = 2) -> None:
        super().__init__()
        self.vnlinear_layer0 = VNLinear(vec_in, hidden)
        self.slinear_layer0 = nn.Linear(sca_in, hidden, bias=False)
        self.nonlinearity0 = VNNonLinearity(hidden, hidden, hidden)
        self.layers = nn.ModuleList(
            nn.ModuleList([
                EqConv(hidden, hidden, hidden, hidden, kernel_size),
                VNNonLinearity(hidden, hidden, hidden),
                VNLayerNorm(hidden),
                EqLayerNorm(hidden),
            ])
            for _ in range(depth)
        )
        self.vnlinear_layer1 = VNLinear(hidden, hidden)
        self.slinear_layer1 = nn.Linear(hidden, hidden, bias=False)
        self.nonlinearity1 = VNNonLinearity(hidden, hidden, 0)
        self.vnlinear_layer2 = VNLinear(hidden, hidden)
        self.vector_ln1 = VNLayerNorm(hidden)
        self.vnoutput_layer = VNLinear(hidden, out_vectors)

    def forward(self, vec: torch.Tensor, sca: torch.Tensor) -> torch.Tensor:
        v = self.vnlinear_layer0(vec)
        s = self.slinear_layer0(sca)
        v, s = self.nonlinearity0(v, s)
        for conv, nonlinearity, vector_norm, scalar_norm in self.layers:
            v, s = conv(v, s)
            v, s = nonlinearity(v, s)
            v, s = vector_norm(v), scalar_norm(s)
        v, s = v.mean(dim=1), s.mean(dim=1)             # 时间维均值池化
        v = self.vnlinear_layer1(v)
        s = self.slinear_layer1(s)
        v, _ = self.nonlinearity1(v, s)
        v = self.vnlinear_layer2(v)
        v = self.vector_ln1(v)
        return self.vnoutput_layer(v)


def gram_schmidt_frame(vectors: torch.Tensor) -> torch.Tensor:
    """两个向量 ``(B, 2, 2)``（按列）→ 正交规范帧 ``F = [e1 e2]``（按列，``det F = ±1``）。"""
    u1, u2 = vectors[..., 0], vectors[..., 1]
    e1 = u1 / torch.linalg.vector_norm(u1, dim=-1, keepdim=True).clamp_min(EPS_FRAME)
    rest = u2 - (u2 * e1).sum(dim=-1, keepdim=True) * e1
    e2 = rest / torch.linalg.vector_norm(rest, dim=-1, keepdim=True).clamp_min(EPS_FRAME)
    return torch.stack([e1, e2], dim=-1)


def o2_frame_features(imu: torch.Tensor) -> tuple:
    """``(B, 6, T)`` 的 ``[gyro, acc]`` → O(2) 向量/标量/原始标量特征。

    ``w̃ = (−ω_y, ω_x, 0)``；``ω_xy ≠ 0`` 时 ``u1 = ω × w̃``，否则 ``u1 = e_y × ω``；
    ``u2 = ω × u1``；``v_i = u_i ‖ω‖ / max(‖u_i‖, 1e-7)``。两个 ``v_i`` 在反射下按**普通矢量**
    变换（因为陀螺是赝矢量，两次叉积把符号消掉），因此：

    * 向量特征 ``(B, T, 2, 3)``：列为 ``[a_xy, v1_xy, v2_xy]``，按 ``Q`` 变换；
    * 标量特征 ``(B, T, 9)``：``[a_z, v1_z, v2_z, ‖a_xy‖, ‖v1_xy‖, ‖v2_xy‖, a·v1, v1·v2, a·v2]``；
    * 原始标量 ``(B, T, 3)``：``[a_z, v1_z, v2_z]``。
    """
    x = imu.transpose(1, 2)
    gyro, acc = x[..., 0:3], x[..., 3:6]
    tilde = torch.stack([-gyro[..., 1], gyro[..., 0], torch.zeros_like(gyro[..., 0])], dim=-1)
    ey = torch.zeros_like(gyro)
    ey[..., 1] = 1.0
    degenerate = ((gyro[..., 0] == 0) & (gyro[..., 1] == 0)).unsqueeze(-1)
    u1 = torch.where(degenerate, torch.linalg.cross(ey, gyro, dim=-1),
                     torch.linalg.cross(gyro, tilde, dim=-1))
    u2 = torch.linalg.cross(gyro, u1, dim=-1)
    scale = torch.linalg.vector_norm(gyro, dim=-1, keepdim=True)
    v1 = u1 * scale / torch.linalg.vector_norm(u1, dim=-1,
                                               keepdim=True).clamp_min(EPS_PREPROCESS)
    v2 = u2 * scale / torch.linalg.vector_norm(u2, dim=-1,
                                               keepdim=True).clamp_min(EPS_PREPROCESS)
    plane = [acc[..., :2], v1[..., :2], v2[..., :2]]
    vec = torch.stack(plane, dim=-1)
    norms = [torch.linalg.vector_norm(p, dim=-1) for p in plane]
    dots = [(plane[0] * plane[1]).sum(-1), (plane[1] * plane[2]).sum(-1),
            (plane[0] * plane[2]).sum(-1)]
    original = torch.stack([acc[..., 2], v1[..., 2], v2[..., 2]], dim=-1)
    sca = torch.cat([original, torch.stack(norms, dim=-1), torch.stack(dots, dim=-1)], dim=-1)
    return vec, sca, original


def canonicalise(vec: torch.Tensor, original: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
    """用规范帧把特征转到规范系：返回骨干输入 ``(B, 6, T)``，通道 ``[a'_xy, a_z, s·ω'_xy, s·ω_z]``。

    ``V_c = Fᵀ V``；``a_c = [V_c[...,0], a_z]``、``w1 = [V_c[...,1], v1_z]``、
    ``w2 = [V_c[...,2], v2_z]``；``ω_c = (w1 × w2) / max(‖w1‖, 1e-8) = det(F)·[Fᵀω_xy, ω_z]``。
    """
    vec_c = torch.einsum("bji,btjk->btik", frame, vec)
    parts = [torch.cat([vec_c[..., k], original[..., k:k + 1]], dim=-1) for k in range(3)]
    acc_c, w1, w2 = parts
    gyro_c = torch.linalg.cross(w1, w2, dim=-1) / torch.linalg.vector_norm(
        w1, dim=-1, keepdim=True).clamp_min(EPS_FRAME)
    return torch.cat([acc_c, gyro_c], dim=-1).transpose(1, 2)


def rotation_matrix_2d(angles: Sequence[float], dtype=torch.float64) -> torch.Tensor:
    """``R(θ)``，形状 ``(len(angles), 2, 2)``（测试与等变性检查用）。"""
    theta = torch.as_tensor(angles, dtype=dtype)
    cos, sin = torch.cos(theta), torch.sin(theta)
    return torch.stack([torch.stack([cos, -sin], -1), torch.stack([sin, cos], -1)], dim=-2)


def reflection_matrix_2d(angles: Sequence[float], dtype=torch.float64) -> torch.Tensor:
    """绕方向角 ``φ`` 的轴做反射：``Q = R(φ) diag(1, −1) R(φ)ᵀ``，``det Q = −1``。"""
    rot = rotation_matrix_2d(angles, dtype)
    flip = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=dtype)
    return rot @ flip @ rot.transpose(-1, -2)


def act_o2(imu: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """让 ``Q ∈ O(2)`` 作用在 ``(B, 6, T)`` 上：``a' = M a``，``ω' = det(Q)·M ω``（赝矢量）。"""
    det = torch.linalg.det(matrix).reshape(-1, 1, 1)
    gyro, acc = imu[:, 0:3], imu[:, 3:6]

    def rotate(v: torch.Tensor) -> torch.Tensor:
        plane = torch.einsum("bij,bjt->bit", matrix.to(v.dtype), v[:, :2])
        return torch.cat([plane, v[:, 2:3]], dim=1)

    return torch.cat([det.to(gyro.dtype) * rotate(gyro), rotate(acc)], dim=1)
