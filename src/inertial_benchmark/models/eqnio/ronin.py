"""EqNIO（RoNIN 骨干）：O(2) 规范帧网络 + RoNIN-ResNet18 的次等变惯性里程计。

论文：R. K. Jayanth*, Y. Xu*, Z. Wang, E. Chatzipantazis, K. Daniilidis, D. Gehrig,
"EqNIO: Subequivariant Neural Inertial Odometry", ICLR 2025. https://arxiv.org/abs/2408.06321
官方仓库：https://github.com/RoyinaJayanth/EqNIO，核对提交 97b7a60a5e5c（子目录 `RONIN/`）。
许可：**官方仓库没有 LICENSE 文件**（默认保留全部权利），且 `RONIN/source/` 派生自 GPL-3.0 的
RoNIN 代码却未声明许可。本文件依据 `docs/algorithms/eqnio_ronin.md` 规格卡**完全独立**编写
（洁净室），未阅读、未复制官方代码，也不分发官方权重。
fidelity：official-code（结构与等变语义取自规格卡，参数量 5,230,466 由单元测试锁定）。

数据流（规格卡 §2.1、§4）：

1. ``o2_frame_features``：由 ``[ω, a]`` 构造向量特征 ``[a_xy, v1_xy, v2_xy]``、9 维标量特征与
   3 维原始标量，其中 ``v1 = ‖ω‖·(ω × w̃)/‖·‖``、``v2 = ‖ω‖·(ω × v1)/‖·‖``、``w̃ = (−ω_y, ω_x, 0)``；
2. ``O2FrameNet``（hidden 64、depth 2、时间核 32）输出两个等变向量，Gram–Schmidt 得到规范帧
   ``F = [e1 e2]``（``det F = ±1``）；
3. 规范化：``V_c = Fᵀ V``，``ω_c = (w1 × w2)/‖w1‖ = det(F)·[Fᵀω_xy, ω_z]``，骨干输入通道为
   ``[a'_x, a'_y, a_z, s·ω'_x, s·ω'_y, s·ω_z]``（``s = det F``，**加计在前**，与 RoNIN 原版的
   陀螺在前不同）；
4. RoNIN-ResNet18（6→2）输出规范系速度 ``v_c``，最终 ``vel = F v_c``。

等变性：输入按 ``a' = M a``、``ω' = det(Q)·M ω``（陀螺是**赝矢量**）变换时，``vel' = Q vel``、
``frame' = frame Qᵀ``，骨干输入不变；单元测试在 float64 下锁定误差 ≤ 1e-8（骨干输入 ≤ 1e-7），
并包含“把陀螺当普通矢量”的负向测试（误差必须显著大于 1e-2）。float32 下等变性只是近似成立：
规范帧网络的 LayerNorm 链在少数窗口上可放大到 ~1e6 倍，误差中位数约 1e-6、最坏可达 1e-2
（架构固有，卡 §10.6）。

与官方的差异及理由（引用卡片）：

1. 论文写 ``F(ω) = (‖ω‖w1/‖w1‖, ‖ω‖w2/‖w2‖)``，代码实际用的是两个向量对调且其中一个取反；
   本实现照**代码**（卡 §2.1、§10.2）；
2. 预处理使用输入自身的 dtype；官方因 ``torch.zeros`` 固定为 float32（卡 §6、§10.7）；
3. 官方发布的训练循环写成 ``range(start_epoch, 1)``，只跑 1 个 epoch；official 配方按官方权重
   ``config.json`` 的 120 epoch（卡 §5、§6、§10.1）；
4. 等变架构不使用随机水平旋转增强，只保留 ``time_shift``（卡 §2、§5）；
5. 姿态：官方测试用设备姿态（game RV），训练按 20° 规则回退；IPB 默认 ``reference``，可设
   ``orientation=device`` 单独报告（卡 §2、§6）；
6. 轨迹重建与指标按 DESIGN §5/§6（官方把速度记在窗口起点、ATE 按分量平均，卡 §3、§10.9）；
7. 只实现 O(2) 变体（README 训练命令与论文表 2 的最优行）；SO(2) 变体（6,456,194 参数）与单向量
   变体不注册（卡 §4.4）。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import torch

from ...nn.base import BaseModel, InputSpec
from ...nn.modules.equivariant import (
    O2FrameNet,
    canonicalise,
    gram_schmidt_frame,
    o2_frame_features,
)
from ...nn.registry import register_model
from ..ronin.model import RoNINResNet


@register_model("eqnio_ronin")
class EqNIORoNIN(BaseModel):
    """EqNIO O(2) 规范帧网络 + RoNIN-ResNet18 骨干（``forward`` 另给出 ``aux.frame``）。"""

    default_loss = "mse"

    def __init__(
        self,
        input_spec: InputSpec,
        hidden: int = 64,
        depth: int = 2,
        frame_kernel: int = 32,
        group_sizes: Sequence[int] = (2, 2, 2, 2),
        base_plane: int = 64,
        kernel_size: int = 3,
        fc_dim: int = 512,
        trans_planes: int = 128,
        dropout: float = 0.5,
    ) -> None:
        super().__init__(input_spec)
        if input_spec.frame == "body":
            raise ValueError("eqnio_ronin needs a gravity-aligned frame (frame=gravity_world)")
        if input_spec.dims != 2:
            raise ValueError("eqnio_ronin canonicalises the horizontal plane and needs dims=2")
        self.frame_net = O2FrameNet(vec_in=3, sca_in=9, hidden=hidden, depth=depth,
                                    kernel_size=frame_kernel, out_vectors=2)
        self.ronin = RoNINResNet(replace(input_spec, dims=2), group_sizes, base_plane,
                                 kernel_size, fc_dim, trans_planes, dropout)
        self.feature_length = self.ronin.feature_length

    def frame(self, imu: torch.Tensor) -> tuple:
        """返回 ``(F, 骨干输入)``：``F`` 为按列排的规范帧 ``(B, 2, 2)``。"""
        vec, sca, original = o2_frame_features(imu)
        frame = gram_schmidt_frame(self.frame_net(vec, sca))
        return frame, canonicalise(vec, original, frame)

    def forward(self, imu: torch.Tensor) -> dict:
        frame, canonical = self.frame(imu)
        vel_c = self.ronin(canonical)["vel"]
        vel = torch.einsum("bij,bj->bi", frame, vel_c)
        # aux.frame 为官方返回的 Fᵀ（每行是一个规范基向量），便于等变性测试与诊断
        return {"vel": vel, "aux": {"frame": frame.transpose(1, 2)}}
