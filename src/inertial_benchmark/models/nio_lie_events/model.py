"""NIO from Lie Events：SE(3) 李事件表示 + RoNIN / TLIO 骨干。

论文：Royina Karegoudra Jayanth, Yinshuang Xu, Evangelos Chatzipantazis, Kostas Daniilidis,
Daniel Gehrig, "Neural Inertial Odometry from Lie Events", RSS 2025.
https://www.roboticsproceedings.org/rss21/p143.pdf, https://arxiv.org/abs/2505.09780
官方仓库：https://github.com/RoyinaJayanth/NIO_Lie_Events，核对提交 1e87470f6479。
许可：**官方仓库没有 LICENSE 文件**（默认保留全部权利）；`RONIN/` 派生自 GPL-3 的 RoNIN、
`TLIO/` 派生自 BSD-3 的 TLIO，均未重新声明许可。本文件依据 `docs/algorithms/nio_lie_events.md`
规格卡**完全独立**编写（洁净室），未阅读、未复制官方代码，也不分发官方权重。
fidelity：official-code（结构与事件语义取自规格卡；参数量与事件生成 golden 由夹具锁定）。

两个注册名共用本文件，因为方法的全部贡献都在**输入表示**（SE(3) 李事件与事件堆叠），骨干只是
把第一层卷积的输入通道由 6 改为 12：

* ``nio_lie_events_ronin``：RoNIN-ResNet18 骨干，2D 世界系平均速度，θ=0.1，**4,637,570** 参数；
* ``nio_lie_events_tlio``：TLIO ResNet 骨干，3D 位移 + 对角 logstd，θ=0.01，**5,427,334** 参数。

事件表示（卡 §2.2，实现在 `nn/modules/lie_events.py`）：把窗口的世界系 IMU 用逐样本姿态转回机体系
→ 从 ``(R_0, p0, v0)`` 出发做右端矩形法预积分 → 每当预积分位姿相对参考位姿的 ``‖Log(T_ref⁻¹T)‖``
跨过 θ 的整数倍就发出一个事件（时间由测地线上的插值参数 β 给出，量测按 β 线性插值，极性为
6 维单位切向量旋到世界系）→ 首尾补两条伪事件 → 按**序号**（不是时间戳）均分到 200 个桶
→ 每桶取量测均值与联合归一化的极性均值，得到 ``(12, 200)``。

**特权输入（必须单列报告）**：``extra_inputs: [orientation, init_velocity]``。预积分需要逐样本姿态
与窗口起点速度 ``v0``；``v0`` 来自参考真值，属于 DESIGN §3.3 的**特权输入**，框架会把它写进
``metrics.json`` 的 ``privileged_inputs``，``ipb report`` 因此把本方法单列一张表，不与纯 IMU
方法混排。

与官方的差异及理由（引用规格卡）：

1. **`v0` 协议**：官方 RoNIN 测试是**有状态递归**（第 1 个窗口用真值、之后用上一个窗口的预测），
   官方 TLIO 纯网络测试则每个窗口都直接用真值。IPB v1 的 Predictor 是无状态批量推理，无法表达
   递归协议，因此**训练与推理都用视图给出的 ``init_velocity``**（窗口首样本的参考真值速度），
   并显式标记为特权输入。这比官方 RoNIN 的测试协议**更乐观**，报告时必须注明（卡 §3、§6、§10-2）；
2. 官方 RoNIN 训练用的是 ``glob_v[ind−200]``（**前一个窗口**的真值平均速度），与测试语义不一致
   且在 ``ind < 200`` 时会负索引回绕；IPB 统一用窗口首样本的速度，物理上才是预积分的初值
   （卡 §10-3）；
3. ``v0`` 噪声 ``U(±0.5)`` 与极性噪声 ``U(±0.5)`` 是方法自带的正则（不是通用增强），实现在模型
   内部且**只在训练模式**生效；官方把它们烘焙进缓存、而且**验证集也带噪**，IPB 的诚实协议要求
   验证确定（卡 §5、§6、§10-8）。RoNIN 变体还把 ``v0_z`` 恒置 0（官方行为）；
4. 坐标系：RoNIN 变体用 ``frame=gravity_world``（与官方一致）；TLIO 变体官方按窗口**起点**偏航
   去偏航，IPB 的 ``gravity_yaw_local`` 按**末端**偏航，两者差一个常值偏航。事件表示对绕 z 的
   旋转严格等变（卡 §2.3），因此直接在视图坐标系里生成事件 ≡ 在世界系生成后整体旋转，**不需要**
   官方那一步额外的 ``R_z(ψ0)ᵀ``（卡 §2.2 第 8 步）；骨干本身不等变，所以参考点的选择仍会影响
   结果，官方靠偏航增强缓解（卡 §6）；
5. TLIO 伪事件的量测官方保持为 0（会把首尾桶的 IMU 均值拉向 0），RoNIN 用首/末样本的实测值。
   unified 口径统一为实测值，``zero_pseudo_measurements=true`` 可复现官方 TLIO 行为
   （卡 §6、§10-6）；
6. 静止或零位移时官方 ``u = w/0`` 产生 NaN 并污染此后全部事件；本实现取 ``u = 0``、不产生事件
   （卡 §8、§10-5），单元测试锁定输出有限；
7. 偏置增强统一用均匀分布 ``U(±r)``；官方 TLIO 数据加载器误用了 ``randn − 0.5``（高斯且均值为负，
   卡 §6、§10-7）；
8. 官方两处 ``generate_event_stack`` 的调用签名有误、发布代码按原样会抛 ``TypeError``；本实现按
   函数体语义（只按序号分桶、不使用时间戳）实现（卡 §6、§10-1）；
9. 通道顺序：论文写 ``[â‖ω̂]``（加计在前），代码为 ``[ω, a]``，**以代码为准**（卡 §10-9）；
10. 不实现 EKF（IPB v1 只评网络 + DESIGN §5 积分）。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import torch

from ...nn.base import BaseModel, InputSpec
from ...nn.modules.lie_events import lie_events, quat_to_matrix
from ...nn.registry import register_model
from ..ronin.model import RoNINResNet
from ..tlio.model import TLIOResNet

# 事件堆叠的 12 个通道（世界系）：陀螺、比力、se(3) 单位极性的平移段与旋转段
EVENT_CHANNELS = ("ev_gyro_x", "ev_gyro_y", "ev_gyro_z", "ev_acc_x", "ev_acc_y", "ev_acc_z",
                  "ev_rho_x", "ev_rho_y", "ev_rho_z", "ev_phi_x", "ev_phi_y", "ev_phi_z")
POLARITY_EPS = 1e-4


class LieEventModel(BaseModel):
    """事件表示 + 骨干的公共部分：把 ``(B, 6, T)`` 与额外输入变成 ``(B, 12, T)`` 再送入骨干。"""

    #: 子类给出的注册名与骨干构造
    threshold: float = 0.1

    def __init__(self, input_spec: InputSpec, threshold: float = 0.1,
                 v0_noise: float = 0.5, polarity_noise: float = 0.5,
                 zero_v0_z: bool = False, zero_pseudo_measurements: bool = False,
                 max_events_per_step: int = 64) -> None:
        super().__init__(input_spec)
        required = ("orientation", "init_velocity")
        missing = [name for name in required if name not in input_spec.extra_inputs]
        if missing:
            raise ValueError(f"{type(self).__name__} needs extra_inputs {list(required)} "
                             f"(missing {missing}); init_velocity is a privileged input")
        if input_spec.sub_windows:
            raise ValueError("lie-event models take a single window (history=0)")
        if input_spec.output_layout != "window":
            raise ValueError("lie-event models predict one output per window")
        self.threshold = float(threshold)
        self.v0_noise = float(v0_noise)
        self.polarity_noise = float(polarity_noise)
        self.zero_v0_z = bool(zero_v0_z)
        self.zero_pseudo_measurements = bool(zero_pseudo_measurements)
        self.max_events_per_step = int(max_events_per_step)

    @property
    def event_spec(self) -> InputSpec:
        """骨干看到的输入规格：12 个事件通道（视图仍然只提供 ``[gyro, acc]``）。"""
        return replace(self.input_spec, channels=EVENT_CHANNELS)

    def initial_velocity(self, extra: dict, dtype: torch.dtype) -> torch.Tensor:
        """把视图给的 ``init_velocity``（``(B, dims)``，**特权输入**）补成 3D 并按需加噪。"""
        v0 = extra["init_velocity"].to(dtype)
        if v0.shape[-1] == 2:
            v0 = torch.cat([v0, torch.zeros_like(v0[:, :1])], dim=-1)
        if self.zero_v0_z:
            v0 = torch.cat([v0[:, :2], torch.zeros_like(v0[:, :1])], dim=-1)
        if self.training and self.v0_noise > 0:
            noise = (torch.rand_like(v0) - 0.5) * (2.0 * self.v0_noise)
            v0 = v0 + noise
            if self.zero_v0_z:
                v0 = torch.cat([v0[:, :2], torch.zeros_like(v0[:, :1])], dim=-1)
        return v0

    def perturb_polarity(self, stack: torch.Tensor) -> torch.Tensor:
        """训练时对**非零**极性项加 ``U(±polarity_noise)``，再按桶重新联合归一化（卡 §5）。"""
        if not self.training or self.polarity_noise <= 0:
            return stack
        polarity = stack[:, 6:]
        active = (polarity != 0).to(polarity.dtype)
        noise = (torch.rand_like(polarity) - 0.5) * (2.0 * self.polarity_noise)
        noisy = polarity + noise * active
        norm = torch.linalg.vector_norm(noisy, dim=1, keepdim=True)
        renormalised = noisy / (norm + POLARITY_EPS)
        keep = (active.sum(dim=1, keepdim=True) > 0).to(polarity.dtype)
        return torch.cat([stack[:, :6], renormalised * keep], dim=1)

    def event_stack(self, imu: torch.Tensor, extra: dict) -> torch.Tensor:
        """``(B, 6, T)`` + ``{orientation, init_velocity}`` → 事件堆叠 ``(B, 12, T)``。

        事件生成是**输入表示**（没有可学习参数），因此全程 ``no_grad`` 并在 float64 下计算，
        与夹具的 golden 值对齐后再转回骨干的 dtype。
        """
        self.check_input(imu)
        spec = self.input_spec
        with torch.no_grad():
            gyro = imu[:, 0:3].transpose(1, 2).double()
            acc = imu[:, 3:6].transpose(1, 2).double()
            rotation = quat_to_matrix(extra["orientation"].double())
            result = lie_events(gyro, acc, rotation, self.initial_velocity(extra, torch.float64),
                                spec.dt, self.threshold, bins=int(spec.window),
                                max_events_per_step=self.max_events_per_step)
            stack = result["stack"]
            if self.zero_pseudo_measurements:
                # 官方 TLIO 路径把首尾伪事件的量测保持为 0（RoNIN 用实测值）
                stack = stack.clone()
                empty = (result["counts"] == 0).reshape(-1, 1)
                for edge in (0, int(spec.window) - 1):
                    single = (stack[:, 6:, edge].abs().sum(1) > 0) & ~empty.reshape(-1)
                    stack[:, :6, edge] = torch.where(single.reshape(-1, 1),
                                                     torch.zeros_like(stack[:, :6, edge]),
                                                     stack[:, :6, edge])
        return self.perturb_polarity(stack.to(imu.dtype))


@register_model("nio_lie_events_ronin")
class NIOLieEventsRoNIN(LieEventModel):
    """RoNIN-ResNet18 骨干（12 输入通道），2D 世界系平均速度；官方阈值 θ = 0.1。"""

    default_loss = "mse"

    def __init__(
        self,
        input_spec: InputSpec,
        threshold: float = 0.1,
        v0_noise: float = 0.5,
        polarity_noise: float = 0.5,
        group_sizes: Sequence[int] = (2, 2, 2, 2),
        base_plane: int = 64,
        kernel_size: int = 3,
        fc_dim: int = 512,
        trans_planes: int = 128,
        dropout: float = 0.5,
        max_events_per_step: int = 64,
    ) -> None:
        super().__init__(input_spec, threshold, v0_noise, polarity_noise, zero_v0_z=True,
                         zero_pseudo_measurements=False,
                         max_events_per_step=max_events_per_step)
        if input_spec.frame == "body":
            raise ValueError("nio_lie_events_ronin needs a gravity-aligned frame")
        self.backbone = RoNINResNet(self.event_spec, group_sizes, base_plane, kernel_size,
                                    fc_dim, trans_planes, dropout)
        self.feature_length = self.backbone.feature_length

    def forward(self, imu: torch.Tensor, extra: dict) -> dict:
        return self.backbone(self.event_stack(imu, extra))


@register_model("nio_lie_events_tlio")
class NIOLieEventsTLIO(LieEventModel):
    """TLIO ResNet 骨干（12 输入通道），3D 位移 + 对角 logstd；官方阈值 θ = 0.01。"""

    default_loss = "nll_detach_then_nll"

    def __init__(
        self,
        input_spec: InputSpec,
        threshold: float = 0.01,
        v0_noise: float = 0.5,
        polarity_noise: float = 0.5,
        group_sizes: Sequence[int] = (2, 2, 2, 2),
        base_plane: int = 64,
        kernel_size: int = 3,
        fc_dim: int = 512,
        trans_planes: int = 128,
        dropout: float = 0.5,
        zero_pseudo_measurements: bool = False,
        max_events_per_step: int = 64,
    ) -> None:
        super().__init__(input_spec, threshold, v0_noise, polarity_noise, zero_v0_z=False,
                         zero_pseudo_measurements=zero_pseudo_measurements,
                         max_events_per_step=max_events_per_step)
        if input_spec.dims != 3:
            raise ValueError("nio_lie_events_tlio predicts 3-D displacement and needs dims=3")
        self.backbone = TLIOResNet(self.event_spec, group_sizes, base_plane, kernel_size,
                                   fc_dim, trans_planes, dropout)
        self.feature_length = self.backbone.feature_length

    def forward(self, imu: torch.Tensor, extra: dict) -> dict:
        # vel 键按 IPB 约定给出目标量本身（窗口位移，m）；预测器再换算为平均速度
        return self.backbone(self.event_stack(imu, extra))
