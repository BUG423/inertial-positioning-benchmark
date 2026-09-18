"""TartanIMU ``Foundation_Model``：10 个 1 s 子窗（各抽到 40 点）→ 逐子窗机体系 3D 速度 + log σ。

论文：S. Zhao, S. Zhou, R. Blanchard, Y. Qiu, W. Wang, S. Scherer, "Tartan IMU: A Light
Foundation Model for Inertial Positioning in Robotics", CVPR 2025, pp. 22520-22529.
官方仓库：https://github.com/superxslam/TartanIMU @ ``8a8025000f85d2c6b0692b88e15c2c91e761f7d7``
（``tartan_imu/model/lstm/trunks.py``、``heads.py``、``common/losses.py``、
``config/resnet_lstm_multihead.yaml``）。
许可：代码 Apache-2.0（HF 权重的许可自相矛盾，本仓库不保存也不再分发权重）。
**本文件依据规格卡 ``docs/algorithms/tartanimu.md`` 独立实现，未阅读、未复制官方代码**。
fidelity：``official-code``（公开的 ResNet-LSTM 版本；参数量 5,698,346 由单元测试锁定）。
论文的 LoRA 适配（Stage 2）、在线自适应（Stage 3）与 Transformer 版核心均未公开，不实现。

结构（规格卡 §4）：主干与 RNIN 同构（逐子窗一维 ResNet **无 MaxPool** →
``Conv1d(512,128,k1)+BN`` → 展平 ``128×3 = 384`` → ``LSTM(384, 128, 2 层, 层间 dropout 0.1)``），
之后是 4 个平台头（注册顺序 ``dog, human, car, drone``），每个头含
``velocity_scale (1,3)``、``output_block1``（xy）、``output_block2``（log σ）与
``output_block1_z``（z）。
主干里还有两个**从不被调用**的 ``output_block1/2``（199,174 参数），官方已注册，为保证参数量与
权重可 ``strict`` 加载，这里同样保留。

IPB 协议映射与差异（规格卡 §2、§6，DESIGN §3.1/§5）：

1. **多步布局**：官方一个样本是 10 个首尾相接的 1 s 子窗，网络对每个子窗输出机体系平均速度；
   IPB 用 ``window=2000`` + ``target=multi_displacement`` + ``output_steps=10``，把 2000 样本窗口
   等分成 10 段，每段的目标是该段位移、时间戳为段中心。模型把每个子窗的速度乘以该段时长
   变成位移（官方 ``pred *= window_time``、``log σ += log(window_time)``，``window_time=1`` s），
   ``output_scales`` 再换算回速度。这样官方的 ``20·L1`` 可以同时监督 10 个子窗（与官方 seq2seq
   训练一致），推理也使用全部 10 个输出（与官方挑战赛脚本一致；``test.py`` 只取最后一个子窗，
   相当于 ``output_steps=1`` 的降级，本实现用窗口级布局支持，见第 2 条）；
2. **窗口级降级布局**（规格卡 §6 方案 A/B）：配成窗口级目标（``avg_velocity``）时输出
   ``vel = human[:, -1]``，即官方 ``test.py`` 的“取最后一个子窗”，时间戳为主窗口中心；
   ``window=200``（S=1）时等价于方案 B ``tartanimu@seq1``（LSTM 只跑一步）；
3. **子窗抽取在模型内部完成**：``(B,6,2000) → (B,10,6,200) → [..., ::5] → (B,10,6,40)``。
   官方直接丢样本、**不做抗混叠滤波**（规格卡 §10.6），必须照做；抽取后长度不在 ``[33, 48]``
   内时 ResNet 展平宽度会变，:class:`ResNetLSTMSeqNet` 会直接报错；
4. **坐标系**：``frame=body``、``dims=3``、``remove_gravity=false``（官方 ``use_local_coord=True``，
   机体系且保留重力）；论文式 (1) 的“去重力 + 去零偏”未在公开代码中实现，以代码为准；
5. **目标定义**：官方子窗目标是“逐帧机体速度的均值”，IPB 是“该段位移旋到段中心姿态的机体系”。
   两者在子窗内有转动时不同，属协议效应（规格卡 §6、§10.9），由 oracle 轨迹量化；
   IPB 按**每段自己的中心姿态**旋转，与官方积分时取 ``ind_intg = j + 100 + 200m`` 的真值姿态
   相差至多半个采样；
6. **协方差头**（规格卡 §5、§10.1）：发布配置下损失恒为 ``20·L1``，协方差头**拿不到梯度**，
   ``logstd`` 不是有效的不确定度。``official`` 配方用 ``predict_cov=false`` 精确复现
   （``output_block2`` 不进入计算图、损失与 epoch 无关）；``unified`` 配方用 ``predict_cov=true``
   打开协方差头，损失从 ``start_cov_epoch`` 起按 ``cov_ramp_epochs`` 线性引入 NLL，
   即论文式 (2)+(3) 描述、公开代码在 body 分支下从未启用的版本；
7. **行人数据固定走 ``human`` 头**（``motion_type=4``）；其余三头在行人训练中没有梯度，
   但参数保留以匹配夹具并支持官方权重的本地核对；
8. **不复刻测试时的 3 点滑动平均**（依赖 batch 切分，规格卡 §10.7）与 **20 m 分段重锚定评测**
   （规格卡 §10.8）：IPB 对所有模型都不做输出平滑，也不做任何对齐（DESIGN §5/§6）；
9. **增强**：官方在机体系里做偏航与倾斜旋转，框架自带的 ``random_yaw``/``gravity_perturb``
   在 ``frame=body`` 下会被跳过，因此本包注册了 ``body_yaw`` / ``body_tilt``（见 ``augment.py``）。
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
from torch import nn

from ...nn.base import BaseModel, InputSpec
from ...nn.losses import expand_mask, gaussian_nll, masked_mean, mse, register_loss
from ...nn.modules.resnet_lstm import FcBlock, ResNetLSTMSeqNet, init_resnet_lstm_weights
from ...nn.registry import register_model

# 官方 motion_type 映射（car=1, dog=2, drone=3, human=4）与 ModuleDict 的注册顺序不同
HEAD_ORDER = ("dog", "human", "car", "drone")
MOTION_TYPES = {"car": 1, "dog": 2, "drone": 3, "human": 4}


@register_loss("tartanimu_velocity")
class TartanIMUVelocityLoss:
    """TartanIMU 的机体系速度损失（规格卡 §5）。

    官方 ``use_local_coord=True`` 分支：``L = l1_weight · mean(|pred − targ|)``，
    ``l1_weight = 20``，**与 epoch 无关**，协方差头不参与（复现该行为时模型不输出 ``logstd``）。

    ``logstd`` 存在时按论文式 (3) 追加对角高斯 NLL，权重从 ``start_cov_epoch`` 起在
    ``cov_ramp_epochs`` 个 epoch 内线性升到 ``cov_weight``（官方代码里的 5 轮线性引入）。
    掩码按 DESIGN §3.1：无效的帧/步不计入，全部无效时返回 0 并保留计算图。
    """

    name = "tartanimu_velocity"

    def __init__(self, l1_weight: float = 20.0, cov_weight: float = 1.0,
                 start_cov_epoch: int = 0, cov_ramp_epochs: int = 5,
                 min_logstd: Optional[float] = None,
                 max_logstd: Optional[float] = None) -> None:
        self.l1_weight = float(l1_weight)
        self.cov_weight = float(cov_weight)
        self.start_cov_epoch = int(start_cov_epoch)
        self.cov_ramp_epochs = max(int(cov_ramp_epochs), 1)
        self.min_logstd = min_logstd
        self.max_logstd = max_logstd

    def cov_ramp(self, epoch: int) -> float:
        """NLL 权重的线性引入系数（``epoch < start_cov_epoch`` 时为 0）。"""
        if epoch < self.start_cov_epoch:
            return 0.0
        ramp = (epoch - self.start_cov_epoch + 1) / self.cov_ramp_epochs
        return float(min(max(ramp, 0.0), 1.0)) * self.cov_weight

    def __call__(self, out: dict, target: torch.Tensor, epoch: int = 0,
                 mask: Optional[torch.Tensor] = None) -> tuple:
        pred = out["vel"]
        l1 = masked_mean(torch.abs(pred - target), expand_mask(mask, pred))
        loss = self.l1_weight * l1
        items = {"l1": l1.detach(), "mse": mse(pred, target, mask).detach()}
        logstd = out.get("logstd")
        if logstd is not None:
            weight = self.cov_ramp(epoch)
            nll = gaussian_nll(pred, logstd, target, self.min_logstd, self.max_logstd, mask)
            items["nll"] = nll.detach()
            items["cov_weight"] = torch.full((), weight, device=pred.device)
            # 权重为 0 时也让协方差分支留在计算图中（梯度为零），避免 DDP 报未使用参数
            loss = loss + (weight * nll if weight > 0.0 else 0.0 * logstd.sum())
        return loss, items


class OutputHead(nn.Module):
    """平台头：``velocity_scale`` + xy 均值块 + ``log σ`` 块 + z 均值块（``split_z=True``）。

    参数注册顺序与官方一致：``nn.Module`` 先注册自己的 ``Parameter``，因此
    ``velocity_scale`` 排在三个 :class:`FcBlock` 之前。
    """

    def __init__(self, in_features: int, dims: int = 3, fc_dim: int = 256,
                 dropout: float = 0.3) -> None:
        super().__init__()
        if dims < 2:
            raise ValueError(f"tartanimu heads split z from xy and need dims >= 2, got {dims}")
        self.velocity_scale = nn.Parameter(torch.ones(1, dims))
        self.output_block1 = FcBlock(in_features, fc_dim, 2, dropout)
        self.output_block2 = FcBlock(in_features, fc_dim, dims, dropout)
        self.output_block1_z = FcBlock(in_features, fc_dim, dims - 2, dropout)

    def forward(self, feat: torch.Tensor, predict_cov: bool = False) -> tuple:
        """``feat (B, S, F)`` → ``(mean (B, S, D), logstd (B, S, D) 或 None)``。"""
        flat = feat.reshape(-1, feat.shape[-1])
        shape = feat.shape[:2] + (-1,)
        mean = torch.cat([self.output_block1(flat), self.output_block1_z(flat)], dim=-1)
        mean = mean.reshape(shape) * self.velocity_scale
        logstd = self.output_block2(flat).reshape(shape) if predict_cov else None
        return mean, logstd


@register_model("tartanimu")
class TartanIMUFoundation(BaseModel):
    """TartanIMU ``Foundation_Model``：共享 ResNet-LSTM 主干 + 4 个平台头。"""

    default_loss = "tartanimu_velocity"
    saved_outputs: tuple = ()  # 窗口级布局下的 ``seq_vel`` 不逐窗口存盘

    def __init__(
        self,
        input_spec: InputSpec,
        sub_window: int = 200,
        sample_step: int = 5,
        layer_sizes: Sequence[int] = (2, 2, 2, 2),
        base_plane: int = 64,
        post_channels: int = 128,
        lstm_size: int = 128,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.1,
        fc_dim: int = 256,
        dropout: float = 0.3,
        head: str = "human",
        predict_cov: bool = False,
    ) -> None:
        super().__init__(input_spec)
        if head not in HEAD_ORDER:
            raise ValueError(f"unknown head {head!r}; available: {list(HEAD_ORDER)}")
        window = int(input_spec.window)
        self.sub_window = int(sub_window)
        self.sample_step = int(sample_step)
        if self.sub_window < 1 or self.sample_step < 1:
            raise ValueError("sub_window and sample_step must be >= 1")
        if window % self.sub_window:
            raise ValueError(f"window {window} must be a multiple of sub_window {self.sub_window}")
        self.sub_windows = window // self.sub_window
        layout = input_spec.output_layout
        if layout == "frame":
            raise ValueError("tartanimu predicts per-sub-window velocity, not per-frame velocity")
        if layout == "steps" and int(input_spec.output_steps) != self.sub_windows:
            raise ValueError(f"output_steps={input_spec.output_steps} must equal the number of "
                             f"sub-windows {self.sub_windows} (window / sub_window)")
        sub_len = -(-self.sub_window // self.sample_step)  # ceil，与 x[..., ::step] 一致
        self.model = ResNetLSTMSeqNet(input_spec.num_channels, input_spec.dims, sub_len,
                                      layer_sizes, base_plane, post_channels, 3, lstm_size,
                                      lstm_layers, lstm_dropout, fc_dim, dropout)
        self.heads = nn.ModuleDict({name: OutputHead(lstm_size, input_spec.dims, fc_dim, dropout)
                                    for name in HEAD_ORDER})
        self.head = head
        self.motion_type = MOTION_TYPES[head]
        self.predict_cov = bool(predict_cov)
        # 官方只对主干做自定义初始化，平台头保持 PyTorch 默认（规格卡 §4）
        init_resnet_lstm_weights(self.model)
        durations = np.ones(1, dtype=np.float64)
        if layout == "steps":
            lo, hi = input_spec.step_bounds[:, 0], input_spec.step_bounds[:, 1]
            durations = (hi - lo) * input_spec.dt
        self.register_buffer("step_durations",
                             torch.tensor(durations, dtype=torch.float32).reshape(1, -1, 1),
                             persistent=False)

    def split_sub_windows(self, imu: torch.Tensor) -> torch.Tensor:
        """``(B, 6, S·W)`` → ``(B, S, 6, ceil(W/step))``。

        第 ``k`` 个子窗恰为 ``x[:, :, kW : (k+1)W : sample_step]``（规格卡 §8 的抽取检查）。
        """
        batch, channels, length = imu.shape
        steps = length // self.sub_window
        x = imu.reshape(batch, channels, steps, self.sub_window)[..., ::self.sample_step]
        return x.permute(0, 2, 1, 3).contiguous()

    def forward(self, imu: torch.Tensor) -> dict:
        self.check_input(imu)
        feat = self.model.encode(self.split_sub_windows(imu))
        mean, logstd = self.heads[self.head](feat, self.predict_cov)
        if self.input_spec.output_layout == "steps":
            # 官方 pred *= window_time、log σ += log(window_time)：这里按每段自己的时长换算
            out = {"vel": mean * self.step_durations}
            if logstd is not None:
                out["logstd"] = logstd + torch.log(self.step_durations)
            return out
        out = {"vel": mean[:, -1], "seq_vel": mean}
        if logstd is not None:
            out["logstd"] = logstd[:, -1]
        return out
