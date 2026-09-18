"""AirIO：机体系 IMU + 逐样本姿态编码的序列到序列速度回归。

论文：Y. Qiu, C. Xu, Y. Chen, S. Zhao, J. Geng, S. Scherer,
"AirIO: Learning Inertial Odometry with Enhanced IMU Feature Observability",
IEEE RA-L 10(9):9368–9375, 2025. https://arxiv.org/abs/2501.15659
官方仓库：https://github.com/Air-IO/Air-IO，核对提交 7214e8a6ddee；许可：**BSD-3-Clause**（宽松）。
本文件依据 `docs/algorithms/airio.md` 规格卡独立实现（洁净室），未阅读、未复制官方代码。
fidelity：official-code（网络与损失；参数量 387,014 由夹具锁定）。

核心主张：**不要**把 IMU 旋到全局系，而是保留**机体系**原始 IMU（含重力），并把姿态作为
**单独的一路输入**显式编码。因此视图用 ``frame=body``（DESIGN §3 规定 body 系目标必须
``dims=3``），并用 ``extra_inputs=[orientation]``（逐样本姿态，**非特权**）提供姿态支路。

结构（卡 §4，``CodeNetMotionwithRot``，``propcov=True``）：

* ``feature_encoder``：``[acc, gyro]`` ``(B,6,T)`` → Conv k7 s3 p3 6→32 → BN → GELU →
  Conv k7 s3 p3 32→64 → BN → GELU → Dropout(0.5) → ``(B,64,L)``；
* ``ori_encoder``：姿态的 so(3) 对数 ``(B,3,T)`` 走同样的结构（3→32→64）；
* 两路在特征维拼接（**IMU 在前**）→ ``fcn2`` Linear 128→64 → ``batchnorm2``（通道维）→ GELU；
* ``gru1`` 双向 GRU(64→64) → ``gru2`` 双向 GRU(128→128) → ``veldecoder``/``velcov_decoder``
  （Linear 256→128 → GELU → Linear 128→3），``cov = exp(raw − 5)`` 是逐轴**方差**（不是 logstd）。

``L = ⌊(⌊(T−1)/3⌋+1−1)/3⌋+1``：T=1000 → 112，T=200 → 23（夹具 ``output_length_by_T``）。

**未参与前向的注册参数（必须保留）**：``CodeNetMotionwithRot`` 调用父类构造函数留下了未使用的
``cnn``（15,968），又定义了未使用的 ``fcn1``（16,512）与 ``batchnorm1``（256），共 **32,736** 个
参数计入总量；前向实际使用的有效参数为 **354,278**。移植时必须注册它们才能与官方参数量和
``state_dict`` 一致，但不得在前向中使用——反传后它们的 ``grad is None``（夹具
``params_without_grad_in_forward`` 逐项锁定）。注册顺序也必须一致：``cnn`` → ``gru1`` → ``gru2``
→ ``veldecoder`` → ``velcov_decoder`` → ``feature_encoder`` → ``ori_encoder`` → ``fcn1`` →
``batchnorm1`` → ``fcn2`` → ``batchnorm2``（子类重新赋值的模块保留父类注册的位置）。

损失（`airio_huber_cov`，卡 §5）：
``L = weight·(Huber_{δ=0.005}(v̂ − v) + cov_weight·mean((v̂−v)²/σ² + ln σ²))``，
``weight = 1e2``、``cov_weight = 1e-4``；协方差项**没有** ½ 系数，且 ``covaug=True`` 时残差
**不** detach（梯度也流向速度头）。无阶段切换。

与官方的差异及理由（引用规格卡）：

1. **监督粒度**：官方是序列到序列监督（``get_label`` 在窗口内按下标 ``14 + 9j`` 取逐样本机体系
   速度标签）。IPB 的视图只提供窗口级目标，因此本实现**只监督最后一个 token**
   （``vel = net_vel[:, -1]``，对应 ``target=velocity_at_end``、``frame=body``）。其余 token 照常
   前向但不进入损失，这是与官方最主要的差异，会削弱监督密度；``label_indices`` 保留官方的取样
   规则以备将来视图支持逐样本目标（卡 §6）；
2. 官方标签数组有 ``W+1`` 帧、补齐用下标 ``W``；IPB 窗口只有 ``T`` 个位姿样本，末端下标相差
   1 个样本（200 Hz 下 5 ms，卡 §6）；
3. 通道顺序：官方在网络内部 ``cat([acc, gyro])``（**加计在前**），IPB 输入为 ``[gyro, acc]``，
   在 ``forward`` 中置换（卡 §2、§10-2）；
4. ``window=1000``（5 s，与论文 EuRoC 设置相同）；网络全卷积 + GRU，参数量与窗口无关（卡 §6）；
5. 推理方式：官方 ``--whole`` 恒为真，整条序列一次前向（双向 GRU 会看到未来数据，是离线非因果
   估计）；IPB 统一为滑窗推理，窗口内仍是双向（卡 §6、§10-5）；
6. 选模：官方用 ``test`` 集的 RMSE 同时做 ReduceLROnPlateau 与选模，而 EuRoC 配置里 ``test`` 与
   ``train`` 是同一批序列；IPB 的诚实协议改用 val（卡 §6、§10-4）；
7. 轨迹积分官方用前向欧拉，IPB 按 DESIGN §5 用梯形积分（卡 §10-6）；
8. 姿态编码含**绝对偏航**且官方无偏航增强，跨序列世界系偏航任意时有过拟合风险。卡 §6 建议
   unified 加随机偏航增强，本实现**不采纳**：DESIGN §4 要求 ``augment`` 在两个配方中一致，
   而且 IPB 的 ``random_yaw`` 会同时旋转 IMU 与目标——在 ``frame=body`` 下两者本来就不随世界
   偏航改变，直接套用会制造自相矛盾的样本（真正需要的是“只旋转姿态支路”的算子，v1 没有）。
   **不要**为本模型写“全局偏航不变性”测试：它本来就不成立（卡 §8、§10-9）；
9. 不实现 EKF（IPB v1 只评网络 + DESIGN §5 积分）；
10. 行人适用性：论文针对多旋翼（机体系与运动方向强耦合），行人手持/口袋时两者解耦，本算法在
    IPB 中作为“坐标系表示”的对照（卡 §6）。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from ...nn.base import BaseModel, InputSpec
from ...nn.losses import expand_mask, masked_mean, masked_output_mean, mse, register_loss
from ...nn.modules.lie_events import quat_to_matrix
from ...nn.registry import register_model
from ..dive.model import so3_log_principal

COV_LOG_OFFSET = 5.0     # cov = exp(raw − 5)（官方 model/code.py）


class CNNEncoder(nn.Module):
    """AirIO 的 ``CNNEncoder``：两层 ``Conv1d(k7, s3, p3) → BN → GELU``，末层后接 Dropout。"""

    def __init__(self, in_channels: int, hidden: int = 32, out_channels: int = 64,
                 kernel_size: int = 7, stride: int = 3, padding: int = 3,
                 dropout: float = 0.5) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size, stride, padding),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Conv1d(hidden, out_channels, kernel_size, stride, padding),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @staticmethod
    def output_length(length: int, kernel_size: int = 7, stride: int = 3,
                      padding: int = 3) -> int:
        for _ in range(2):
            length = (length + 2 * padding - kernel_size) // stride + 1
        return length


@register_loss("airio_huber_cov")
class AirIOHuberCovLoss:
    """AirIO 损失：``weight·(Huber_δ(v̂−v) + cov_weight·mean((v̂−v)²/σ² + ln σ²))``。

    协方差项**没有** ½ 系数（官方 ``loss_func.py`` 的写法）；``covaug=True``（默认）时残差不
    detach，梯度也流向速度头。无阶段切换：协方差项从第 0 轮起就生效。
    """

    def __init__(self, weight: float = 1.0e2, cov_weight: float = 1.0e-4, delta: float = 0.005,
                 covaug: bool = True) -> None:
        self.weight = float(weight)
        self.cov_weight = float(cov_weight)
        self.delta = float(delta)
        self.covaug = bool(covaug)

    def __call__(self, out: dict, target: torch.Tensor, epoch: int = 0,
                 mask: Optional[torch.Tensor] = None) -> tuple:
        if "logstd" not in out:
            raise KeyError("airio_huber_cov needs the model to output 'logstd'")
        pred, logstd = out["vel"], out["logstd"]
        residual = pred - target
        huber = masked_mean(
            F.huber_loss(pred, target, reduction="none", delta=self.delta),
            expand_mask(mask, pred))
        squared = residual**2 if self.covaug else residual.detach() ** 2
        terms = squared / torch.exp(2.0 * logstd) + 2.0 * logstd
        cov = masked_output_mean(terms.mean(dim=-1), mask)
        loss = self.weight * (huber + self.cov_weight * cov)
        return loss, {"huber": huber.detach(), "cov": cov.detach(),
                      "mse": mse(pred, target, mask).detach()}


@register_model("airio")
class AirIO(BaseModel):
    """AirIO ``CodeNetMotionwithRot``：IMU 支路 + 姿态支路 + 双向 GRU + 速度/协方差解码器。"""

    default_loss = "airio_huber_cov"
    saved_outputs = ("variance",)

    def __init__(
        self,
        input_spec: InputSpec,
        hidden: int = 32,
        encoder_dim: int = 64,
        gru_hidden: int = 64,
        decoder_dim: int = 128,
        dropout: float = 0.5,
        kernel_size: int = 7,
        stride: int = 3,
        padding: int = 3,
    ) -> None:
        super().__init__(input_spec)
        if "orientation" not in input_spec.extra_inputs:
            raise ValueError("airio encodes the per-sample attitude; "
                             "declare extra_inputs: [orientation]")
        if input_spec.frame != "body":
            raise ValueError("airio keeps the IMU in the body frame (frame=body, dims=3)")
        if input_spec.output_layout != "window":
            raise ValueError("airio maps its last token to one window velocity "
                             "(target=velocity_at_end)")
        if input_spec.sub_windows:
            raise ValueError("airio takes a single window (history=0)")
        encoder = dict(hidden=hidden, out_channels=encoder_dim, kernel_size=kernel_size,
                       stride=stride, padding=padding, dropout=dropout)
        # 注册顺序必须与官方一致（父类先注册 cnn/gru/解码器，子类再注册编码器与 fcn/BN）
        self.cnn = CNNEncoder(input_spec.num_channels, **encoder)          # 未参与前向
        self.gru1 = nn.GRU(gru_hidden, gru_hidden, num_layers=1, bidirectional=True,
                           batch_first=True)
        self.gru2 = nn.GRU(2 * gru_hidden, 2 * gru_hidden, num_layers=1, bidirectional=True,
                           batch_first=True)
        self.veldecoder = self._decoder(4 * gru_hidden, decoder_dim, input_spec.dims)
        self.velcov_decoder = self._decoder(4 * gru_hidden, decoder_dim, input_spec.dims)
        self.feature_encoder = CNNEncoder(input_spec.num_channels, **encoder)
        self.ori_encoder = CNNEncoder(3, **encoder)
        self.fcn1 = nn.Sequential(nn.Linear(2 * encoder_dim, 2 * encoder_dim))   # 未参与前向
        self.batchnorm1 = nn.BatchNorm1d(2 * encoder_dim)                        # 未参与前向
        self.fcn2 = nn.Sequential(nn.Linear(2 * encoder_dim, gru_hidden))
        self.batchnorm2 = nn.BatchNorm1d(gru_hidden)
        self.gelu = nn.GELU()
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding
        self.feature_length = CNNEncoder.output_length(int(input_spec.window), kernel_size,
                                                       stride, padding)
        if self.feature_length < 1:
            raise ValueError(f"window {input_spec.window} is too short for the AirIO encoders")

    @staticmethod
    def _decoder(in_features: int, hidden: int, out_features: int) -> nn.Sequential:
        return nn.Sequential(nn.Linear(in_features, hidden), nn.GELU(),
                             nn.Linear(hidden, out_features))

    def label_indices(self, window: int) -> list:
        """官方 ``get_label`` 的取样下标：``14 + 9j``，不足 ``T_out`` 时用末端下标补齐。

        官方在长度 ``W+1`` 的标签数组上取样（补齐用下标 ``W``）；IPB 的窗口只有 ``T`` 个位姿
        样本，因此补齐用 ``T−1``（末端相差一个样本，卡 §6）。仅供诊断与将来的逐样本监督使用：
        IPB v1 只监督最后一个 token。
        """
        k0, s0, p0, p1 = self.kernel_size, self.stride, self.padding, self.padding
        start = (k0 - p0) + s0 * (k0 - 1 - p1) + 1
        step = s0 * s0
        length = CNNEncoder.output_length(int(window), k0, s0, p0)
        indices = [i for i in range(start, int(window), step)][:length]
        return indices + [int(window) - 1] * (length - len(indices))

    def attitude_channels(self, extra: dict, dtype: torch.dtype) -> torch.Tensor:
        """逐样本姿态 ``(B, T, 4)`` → so(3) 对数 ``(B, 3, T)``（与 PyPose ``SO3.Log`` 一致）。"""
        return so3_log_principal(quat_to_matrix(extra["orientation"].to(dtype))).transpose(1, 2)

    def sequence(self, imu: torch.Tensor, extra: dict) -> tuple:
        """返回逐 token 的 ``(net_vel (B,L,3), cov (B,L,3))``（官方的序列到序列输出）。"""
        self.check_input(imu)
        # IPB 的 [gyro, acc] → 官方的 [acc, gyro]
        features = self.feature_encoder(torch.cat([imu[:, 3:6], imu[:, 0:3]], dim=1))
        attitude = self.ori_encoder(self.attitude_channels(extra, imu.dtype))
        merged = torch.cat([features, attitude], dim=1).transpose(1, 2)   # (B, L, 128)
        merged = self.gelu(self.batchnorm2(self.fcn2(merged).transpose(1, 2)).transpose(1, 2))
        merged, _ = self.gru1(merged)
        merged, _ = self.gru2(merged)
        return self.veldecoder(merged), torch.exp(self.velcov_decoder(merged) - COV_LOG_OFFSET)

    def forward(self, imu: torch.Tensor, extra: dict) -> dict:
        velocity, variance = self.sequence(imu, extra)
        # 窗口级输出取最后一个 token（对应 target=velocity_at_end，机体系）
        return {"vel": velocity[:, -1],
                "logstd": 0.5 * torch.log(variance[:, -1].clamp_min(1e-30)),
                "variance": variance[:, -1]}
