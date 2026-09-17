"""EqNIO（TLIO 骨干）：O(2) 规范帧网络 + TLIO ResNet 的次等变惯性里程计。

论文：R. K. Jayanth*, Y. Xu*, Z. Wang, E. Chatzipantazis, K. Daniilidis, D. Gehrig,
"EqNIO: Subequivariant Neural Inertial Odometry", ICLR 2025. https://arxiv.org/abs/2408.06321
官方仓库：https://github.com/RoyinaJayanth/EqNIO，核对提交 97b7a60a5e5c（子目录 `TLIO-master/`）。
许可：**官方仓库没有 LICENSE 文件**（默认保留全部权利）；`TLIO-master` 派生自 BSD-3 的 TLIO，
EqNIO 新增文件未声明许可。本文件依据 `docs/algorithms/eqnio_tlio.md` 规格卡**完全独立**编写
（洁净室），未阅读、未复制官方代码，也不分发官方权重。
fidelity：official-code（结构与等变语义取自规格卡，参数量 6,020,230 由夹具锁定）。

数据流（卡 §4，与 `eqnio_ronin` 共用 `nn/modules/equivariant.py` 的全部积木）：

1. ``o2_frame_features``：由 ``[ω, a]`` 构造向量特征 ``[a_xy, v1_xy, v2_xy]``、9 维标量特征与
   3 维原始标量（语义定义见 `eqnio_ronin` 卡 §2.1，两份官方代码逐通道一致）；
2. ``O2FrameNet``（hidden 64、depth 2、时间核 32，595,584 参数）输出两个等变向量，
   Gram–Schmidt 得到规范帧 ``F = [e1 e2]``（``det F = ±1``）；
3. ``canonicalise``：骨干输入通道为 ``[a'_x, a'_y, a_z, s·ω'_x, s·ω'_y, s·ω_z]``
   （``s = det F``，**加计在前**，与 TLIO 原版的陀螺在前相反）；
4. TLIO ResNet（6→3 位移 + 3 logstd，展平长度 7，5,424,646 参数）输出**规范帧中**的位移
   ``d_c`` 与对角 ``logstd_c``；
5. 回到视图坐标系：``d_w = [F d_c,xy, d_c,z]``，``Σ_w = blkdiag(F diag(σ_x², σ_y²) Fᵀ, σ_z²)``。
   名称 "fullCov" 指的正是这件事：规范帧里是对角协方差，旋回世界系后 xy 块成为满阵（卡 §10-9）。

等变性：输入按 ``a' = M a``、``ω' = det(Q)·M ω``（陀螺是**赝矢量**）变换时，``d_w' = M̃ d_w``、
``Σ_w' = M̃ Σ_w M̃ᵀ``（``M̃ = diag(M, 1)``）、``frame' = frame Qᵀ``，而 ``d_c``、``logstd_c``
与骨干输入**不变**。单元测试在 float64 下锁定误差 ≤ 1e-8（骨干输入 ≤ 1e-7），并包含“把陀螺当
普通矢量”的负向测试（误差必须显著大于 1e-2）；float32 下等变性只是近似成立，只锁中位数
（规范帧网络的 LayerNorm 链可放大少数窗口的误差，`eqnio_ronin` 卡 §10.6）。

与官方的差异及理由（引用卡片）：

1. 坐标系：官方按窗口**起点**偏航去偏航（`gravity_yaw_local` 的起点版），IPB 该取值按窗口末端
   去偏航。因为模型对 O(2) 严格等变，直接用 ``frame=gravity_world`` 与官方在数值上等价，
   同时回避了参考点不一致的问题（卡 §6 的推荐做法）；
2. 目标：官方 ``p[199] − p[0]`` 与 IPB 的 ``target=displacement`` 定义完全相同。``vel`` 键按
   IPB 约定给出**目标量本身**（位移，m），预测器再除以 ``(T−1)·dt = 0.995 s``；官方 `test.py`
   除以 1.0 s，属于协议层差异（卡 §3、§6）；
3. 保留 3 维输出（``dims=3``）：改成 2 维会改变 ``fc3`` 的形状与对角 NLL 的轴数（卡 §6）；
4. 损失 `eqnio_canonical_nll` 按官方做法**在规范帧中**逐轴计算（MSE → 对角 NLL，第 10 个
   epoch 切换，MSE 阶段 ``logstd`` 被 detach）。它与“世界系满协方差 NLL 除以 3”数学恒等，
   单元测试锁定这一恒等式（卡 §5、§8）；
5. 官方 `do_train` 对所有 frame 架构**从不调用 ``optimizer.zero_grad()``**（梯度跨步累积后再裁剪
   到 0.1），学习率调度器也从未 ``step()``，且 ``range(start+1, epochs)`` 少跑一个 epoch。
   这些是配方/训练器层面的官方缺陷，IPB 的 official 配方只保留可由配置表达的部分（恒定学习率、
   ``grad_clip=0.1``），梯度累积缺陷**不复现**（卡 §5、§10-1…3）；
6. 官方对 O(2) 在**验证集**上也做随机重力/偏置扰动，导致验证损失是随机的；IPB 的诚实协议要求
   验证确定，增强只在 train 上生效（卡 §6、§10-4）；
7. 只实现 O(2) 变体（README 训练命令与论文表 1 的最优行）；SO(2) 变体（8,884,870 参数）不注册
   （卡 §4.3）；
8. 不实现 EKF（IPB v1 只评网络 + DESIGN §5 积分，卡 §6）。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional, Sequence

import torch

from ...nn.base import BaseModel, InputSpec
from ...nn.losses import MIN_LOGSTD, expand_mask, gaussian_nll, masked_mean, mse, register_loss
from ...nn.modules.equivariant import (
    O2FrameNet,
    canonicalise,
    gram_schmidt_frame,
    o2_frame_features,
)
from ...nn.registry import register_model
from ..tlio.model import TLIOResNet


def frame_to_world_vector(frame: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """规范帧 → 视图坐标系：``[F v_xy, v_z]``（``frame`` 按列排，``(B,2,2)``）。"""
    plane = torch.einsum("bij,bj->bi", frame, vec[:, :2])
    return torch.cat([plane, vec[:, 2:]], dim=-1)


def canonical_diagonal_to_world_cov(frame: torch.Tensor, logstd: torch.Tensor) -> torch.Tensor:
    """``Σ_w = blkdiag(F diag(σ_x², σ_y²) Fᵀ, σ_z²)``，返回 ``(B, 3, 3)``。"""
    var = torch.exp(2.0 * logstd)
    plane = torch.einsum("bij,bj,bkj->bik", frame, var[:, :2], frame)
    cov = torch.zeros(var.shape[0], 3, 3, dtype=var.dtype, device=var.device)
    cov[:, :2, :2] = plane
    cov[:, 2, 2] = var[:, 2]
    return cov


@register_loss("eqnio_canonical_nll")
class EqNIOCanonicalNLL:
    """EqNIO 官方损失：在**规范帧**中逐轴计算，第 ``switch_epoch`` 个 epoch 起换成对角 NLL。

    * ``epoch < switch_epoch``：``mean((d_c − targ_c)²)``，``logstd`` **不接收梯度**
      （官方实测：协方差头的梯度为 None；本实现用 ``0·logstd.sum()`` 保留计算图并置零梯度）；
    * 之后：``mean(e²/(2σ²) + log σ)``，``log σ`` 只做下限截断 ``log(1e-3)``（官方无上限）。

    目标先按 ``targ_c = [Fᵀ t_xy, t_z]`` 旋进规范帧，与官方 `train.py` 一致；这与“在视图坐标系
    里算满协方差 NLL 再除以 3”数学恒等（单元测试锁定）。``switch_epoch`` 按 IPB 的 0 起计数。
    """

    def __init__(self, switch_epoch: int = 9, min_logstd: Optional[float] = MIN_LOGSTD,
                 max_logstd: Optional[float] = None) -> None:
        self.switch_epoch = int(switch_epoch)
        self.min_logstd = min_logstd
        self.max_logstd = max_logstd

    def canonical_target(self, out: dict, target: torch.Tensor) -> torch.Tensor:
        """``targ_c = [Fᵀ t_xy, t_z]``；``frame_columns`` 是**张量**，训练与验证两条路径都能拿到
        （Predictor 会过滤掉 ``aux`` 这类非张量项，DESIGN §4 的损失契约）。"""
        plane = torch.einsum("bji,bj->bi", out["frame_columns"], target[:, :2])
        return torch.cat([plane, target[:, 2:]], dim=-1)

    def stage(self, epoch: int) -> str:
        return "mse" if epoch < self.switch_epoch else "nll"

    def __call__(self, out: dict, target: torch.Tensor, epoch: int = 0,
                 mask: Optional[torch.Tensor] = None) -> tuple:
        for key in ("disp_canonical", "logstd_canonical"):
            if key not in out:
                raise KeyError(f"eqnio_canonical_nll needs the model to output {key!r}")
        pred, logstd = out["disp_canonical"], out["logstd_canonical"]
        targ = self.canonical_target(out, target)
        if self.stage(epoch) == "mse":
            loss = masked_mean((pred - targ) ** 2, expand_mask(mask, pred))
            items = {"mse": loss.detach()}
            # 协方差头留在计算图中但梯度为零（官方在这一阶段对 logstd 做 detach）
            return loss + 0.0 * logstd.sum(), items
        nll = gaussian_nll(pred, logstd, targ, self.min_logstd, self.max_logstd, mask)
        return nll, {"nll": nll.detach(),
                     "mse": mse(out["vel"], target, mask).detach()}


@register_model("eqnio_tlio")
class EqNIOTLIO(BaseModel):
    """EqNIO O(2) 规范帧网络 + TLIO ResNet 骨干（位移均值 + 对角 logstd，规范帧内）。"""

    default_loss = "eqnio_canonical_nll"
    # 预测文件保存世界系满协方差；``vel``/``logstd`` 由框架自动保存
    saved_outputs = ("cov",)

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
            raise ValueError("eqnio_tlio needs a gravity-aligned frame (frame=gravity_world)")
        if input_spec.dims != 3:
            raise ValueError("eqnio_tlio predicts 3-D displacement and needs dims=3")
        if input_spec.output_layout != "window":
            raise ValueError("eqnio_tlio predicts one displacement per window")
        self.frame_net = O2FrameNet(vec_in=3, sca_in=9, hidden=hidden, depth=depth,
                                    kernel_size=frame_kernel, out_vectors=2)
        self.tlio = TLIOResNet(replace(input_spec, dims=3), group_sizes, base_plane, kernel_size,
                               fc_dim, trans_planes, dropout)
        self.feature_length = self.tlio.feature_length

    def frame(self, imu: torch.Tensor) -> tuple:
        """返回 ``(F, 骨干输入)``：``F`` 为按列排的规范帧 ``(B, 2, 2)``。"""
        self.check_input(imu)
        vec, sca, original = o2_frame_features(imu)
        frame = gram_schmidt_frame(self.frame_net(vec, sca))
        return frame, canonicalise(vec, original, frame)

    def forward(self, imu: torch.Tensor) -> dict:
        frame, canonical = self.frame(imu)
        out = self.tlio(canonical)
        disp_c, logstd_c = out["vel"], out["logstd"]
        # vel 键按 IPB 约定给出目标量本身（窗口位移，m）；预测器再换算为平均速度
        vel = frame_to_world_vector(frame, disp_c)
        cov = canonical_diagonal_to_world_cov(frame, logstd_c)
        return {
            "vel": vel,
            # 视图坐标系中的**边缘**标准差（Σ_w 的对角），仅用于报告；损失走规范帧的对角 NLL
            "logstd": 0.5 * torch.log(torch.diagonal(cov, dim1=1, dim2=2).clamp_min(1e-30)),
            "cov": cov,
            "disp_canonical": disp_c,
            "logstd_canonical": logstd_c,
            "frame_columns": frame,
            # aux.frame 为官方返回的 Fᵀ（每行是一个规范基向量），便于等变性测试与诊断
            "aux": {"frame": frame.transpose(1, 2), "frame_columns": frame},
        }
