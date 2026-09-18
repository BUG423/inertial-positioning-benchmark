"""RNIN（RNIN-VIO 的惯性网络）：10 个 1 s 子窗 → 10 步 3D 位移 + 对角 ``log σ``。

论文：D. Chen, N. Wang, R. Xu, W. Xie, H. Bao, G. Zhang, "RNIN-VIO: Robust Neural Inertial
Navigation Aided Visual-Inertial Odometry in Challenging Scenes", ISMAR 2021, pp. 275-283,
DOI 10.1109/ISMAR52148.2021.00043.
官方仓库：https://github.com/zju3dv/rnin-vio @ ``b030ecc94f151159973a9086c8ba76e22bdbc56e``
（``model/model_lstm.py``、``model/losses.py``、``config/default.yaml`` 合并
``config/resnet_lstm.yaml``）。
许可：Apache-2.0（知识产权属 SenseTime）。**本文件依据规格卡 ``docs/algorithms/rnin.md``
独立实现，未阅读、未复制官方代码**。
fidelity：``official-code``（网络、专用损失与训练配方；参数量 5,358,342 由单元测试锁定）。
RNIN-VIO 的平方根逆滤波融合未开源，也不属于纯惯性 benchmark 范围，不实现。

结构（规格卡 §4）：逐子窗一维 ResNet（**无 MaxPool**）→ ``Conv1d(512,128,k1)+BN`` → 展平
``128×7 = 896`` → 单向 ``LSTM(896, 256, 1 层)`` → 两个独立的三层全连接块（均值 / ``log σ``）。
公共积木见 :mod:`inertial_benchmark.nn.modules.resnet_lstm`（与 TartanIMU 共用）。

IPB 协议映射与差异（规格卡 §2、§6，DESIGN §3.1/§3.2/§5）：

1. **多步位移布局**：官方一个训练样本是 10 个首尾相接的 1 s 子窗（100 Hz、每窗 100 样本），
   网络一次输出 10 个子窗位移。IPB 用 ``window=2000`` + ``target=multi_displacement`` +
   ``output_steps=10``（DESIGN §3.1 明确把“RNIN 式 10 步位移”定义为 ``output_steps=10``）：
   视图把 2000 样本窗口**等分**成 10 段，第 ``h`` 段的目标就是该段的位移，时间戳为段中心，
   ``output_scales`` 再把它换算成速度。这样官方的 RL + 8·AL 损失可以**原样**作用在 10 个
   目标上（累积位移严格等于位置差），而不需要额外的“模型专用目标”协议扩展；
   代价是等分边界为 ``round(linspace(0, 1999, 11))``，中间几段是 199/200 样本而非整 200，
   与子窗输入相差至多 1 个样本（5 ms）。
   规格卡 §6 与夹具 ``benchmark_config`` 写的是另一种映射（``history=1800`` 的窗口级输出
   ``vel = mean[:, -1]``）；那是历史上下文协议尚未落地时的降级方案，本实现同样支持
   （配成 ``target=displacement`` 或 ``avg_velocity`` 的窗口级布局时自动启用，只监督最后一个
   子窗，AL 项失效），但默认配方用多步布局，因为它保住了 RNIN 的核心贡献——绝对损失 AL；
2. **100 Hz 抽取在模型内部完成**：``x[..., ::2]``。100 Hz 网格是 IPB 200 Hz 网格的子集，与官方
   “线性插值到 100 Hz 且不做抗混叠滤波”等价（规格卡 §2）；
3. **推理时官方只取最后一个子窗**并把时间戳记在该子窗中心；IPB 把 10 个输出分别放到各自的
   段中心，重叠部分按 ``overlap=mean`` 合并（DESIGN §5），因此时间轴覆盖更密、不需要
   “序列前 9.5 s 不评测”的特例；
4. **协方差头**（规格卡 §5、§10.1）：官方 ``start_cov_epochs=2000 > epochs=201``，
   训练全程走 ``forward(x, 'dp')``，协方差头**从未训练**、测试时 σ 恒为 1。
   ``official`` 配方用 ``predict_cov=false`` 精确复现该行为：``output_block2`` 不进入计算图，
   其参数 ``.grad is None``，模型也不输出 ``logstd``；``unified`` 配方用 ``predict_cov=true``
   打开协方差头，损失从 ``start_cov_epoch`` 起改用 NLL（AL 项按独立假设用累积方差），
   即论文 §4.4 描述、公开配置从未启用的 MSE→NLL 版本；
5. **不做 VIO 零偏补偿**（IPB 格式没有零偏字段），由 ``bias_shift`` 增强覆盖（规格卡 §2、§6）；
6. **不做 4 m/s 子窗速度过滤**：该过滤只在参考位姿来自 VIO 的数据集上触发，且窗口级速度上限
   过滤尚未在协议层落地（``docs/ALGORITHMS.md`` §3）；
7. 官方每个 epoch 都在测试集上推理并记日志（规格卡 §10.5），IPB 不沿用。
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from ...nn.base import BaseModel, InputSpec
from ...nn.losses import masked_mean, mse, register_loss
from ...nn.modules.resnet_lstm import ResNetLSTMSeqNet, init_resnet_lstm_weights
from ...nn.registry import register_model


def cumulative_valid(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """``out[:, k]``：第 ``0..k`` 步是否全部有效（累积位移项的掩码）。"""
    if mask is None:
        return None
    return torch.cumprod(mask.to(torch.bool).to(torch.float32), dim=1) > 0.5


@register_loss("rnin_displacement")
class RNINDisplacementLoss:
    """RNIN 的“相对位移 + 累积位移”损失（规格卡 §5）。

    位移阶段（官方默认、``official`` 配方）::

        ℓ₁ = (d̂ − d)²                                       # (B, S, 3)   相对损失 RL
        ℓ₂ = w · (cumsum(d̂)[:,1:] − cumsum(d)[:,1:])²        # (B, S−1, 3) 绝对损失 AL
        loss = mean(cat([ℓ₁, ℓ₂], dim=1))                    # 对 B×(2S−1)×3 个元素求平均

    NLL 阶段（``logstd`` 存在且 ``epoch >= start_cov_epoch``，即论文所述、公开配置从未启用的
    第二阶段）::

        ℓ₁ = (d̂ − d)²/(2·e^{2u}) + u
        Σ  = cumsum(e^{2u})                                   # 独立假设下的累积方差
        ℓ₂ = w · [(D̂ − D)²/(2·Σ[:,1:]) + ½·log Σ[:,1:]]

    官方既不 detach 也不截断 ``log σ``，因此 ``min_logstd``/``max_logstd`` 默认为 ``None``。
    掩码：无效步的误差先置零；RL 第 ``k`` 项按 ``mask[:, k]`` 加权，AL 第 ``k`` 项要求第
    ``0..k`` 步全部有效；全部无效时返回 0 并保留计算图。
    窗口级布局（目标 ``(B, D)``）下只剩 RL，等价于 ``MSE``/``NLL``，并标记 ``degraded=1``。
    """

    name = "rnin_displacement"

    def __init__(self, absolute_weight: float = 8.0, start_cov_epoch: int = 0,
                 min_logstd: Optional[float] = None,
                 max_logstd: Optional[float] = None) -> None:
        self.absolute_weight = float(absolute_weight)
        self.start_cov_epoch = int(start_cov_epoch)
        self.min_logstd = min_logstd
        self.max_logstd = max_logstd

    def _logstd(self, out: dict, epoch: int) -> Optional[torch.Tensor]:
        logstd = out.get("logstd")
        if logstd is None or epoch < self.start_cov_epoch:
            return None
        if self.min_logstd is not None or self.max_logstd is not None:
            logstd = torch.clamp(logstd, min=self.min_logstd, max=self.max_logstd)
        return logstd

    def __call__(self, out: dict, target: torch.Tensor, epoch: int = 0,
                 mask: Optional[torch.Tensor] = None) -> tuple:
        pred = out["vel"]
        logstd = self._logstd(out, epoch)
        items = {"mse": mse(pred, target, mask).detach()}
        if pred.ndim < 3:  # 窗口级布局：没有 10 步序列，AL 项不存在（降级模式）
            rel = (pred - target) ** 2
            if logstd is not None:
                rel = rel / (2.0 * torch.exp(2.0 * logstd)) + logstd
            loss = masked_mean(rel, None if mask is None else mask.to(rel.dtype).expand_as(rel))
            items["degraded"] = torch.ones((), device=pred.device)
            items["relative"] = loss.detach()
            return loss, items
        error = pred - target
        keep = None if mask is None else mask.to(torch.bool).unsqueeze(-1)
        if keep is not None:
            error = torch.where(keep, error, torch.zeros_like(error))
        cum_error = torch.cumsum(error, dim=1)[:, 1:]
        if logstd is None:
            rel = error ** 2
            absolute = self.absolute_weight * cum_error ** 2
            if out.get("logstd") is not None:  # 让协方差分支留在计算图中但梯度为零
                rel = rel + 0.0 * out["logstd"].sum()
        else:
            variance = torch.exp(2.0 * logstd)
            rel = error ** 2 / (2.0 * variance) + logstd
            cum_var = torch.cumsum(variance, dim=1)[:, 1:]
            absolute = self.absolute_weight * (cum_error ** 2 / (2.0 * cum_var)
                                               + 0.5 * torch.log(cum_var))
        terms = torch.cat([rel, absolute], dim=1)
        if mask is None:
            weights = None
        else:
            valid = mask.to(torch.bool)
            weights = torch.cat([valid, cumulative_valid(valid)[:, 1:]], dim=1)
            weights = weights.to(terms.dtype).unsqueeze(-1).expand_as(terms)
        loss = masked_mean(terms, weights)
        items["relative"] = rel.detach().mean()
        items["absolute"] = absolute.detach().mean()
        items["stage"] = torch.full((), float(logstd is not None), device=pred.device)
        return loss, items


@register_model("rnin")
class RNINResNetLSTM(BaseModel):
    """RNIN ``ResNetLSTMSeqNet``：``(B, 6, 2000)`` → 10 个 1 s 子窗 → 10 步 3D 位移（+ σ）。"""

    default_loss = "rnin_displacement"
    saved_outputs: tuple = ()  # 窗口级布局下的 ``seq_disp`` 不逐窗口存盘

    def __init__(
        self,
        input_spec: InputSpec,
        sub_window: int = 200,
        decimate: int = 2,
        layer_sizes: Sequence[int] = (2, 2, 2, 2),
        base_plane: int = 64,
        post_channels: int = 128,
        lstm_size: int = 256,
        lstm_layers: int = 1,
        lstm_dropout: float = 0.0,
        fc_dim: int = 256,
        dropout: float = 0.2,
        predict_cov: bool = False,
    ) -> None:
        super().__init__(input_spec)
        window = int(input_spec.window)
        self.sub_window = int(sub_window)
        self.decimate = int(decimate)
        if self.sub_window < 1 or self.decimate < 1:
            raise ValueError("sub_window and decimate must be >= 1")
        if window % self.sub_window:
            raise ValueError(f"window {window} must be a multiple of sub_window {self.sub_window}")
        self.sub_windows = window // self.sub_window
        layout = input_spec.output_layout
        if layout == "frame":
            raise ValueError("rnin predicts per-sub-window displacements, not per-frame velocity")
        if layout == "steps" and int(input_spec.output_steps) != self.sub_windows:
            raise ValueError(f"output_steps={input_spec.output_steps} must equal the number of "
                             f"sub-windows {self.sub_windows} (window / sub_window)")
        self.trunk = ResNetLSTMSeqNet(input_spec.num_channels, input_spec.dims,
                                      self.sub_window // self.decimate, layer_sizes, base_plane,
                                      post_channels, 3, lstm_size, lstm_layers, lstm_dropout,
                                      fc_dim, dropout)
        self.predict_cov = bool(predict_cov)
        init_resnet_lstm_weights(self)

    def split_sub_windows(self, imu: torch.Tensor) -> torch.Tensor:
        """``(B, 6, S·W)`` → ``(B, S, 6, W/decimate)``。

        第 ``k`` 个子窗恰为 ``x[:, :, kW : (k+1)W : decimate]``（规格卡 §8 的抽取/切分检查）。
        """
        batch, channels, length = imu.shape
        steps = length // self.sub_window
        x = imu.reshape(batch, channels, steps, self.sub_window)[..., ::self.decimate]
        return x.permute(0, 2, 1, 3).contiguous()

    def forward(self, imu: torch.Tensor) -> dict:
        self.check_input(imu)
        out = self.trunk(self.split_sub_windows(imu), predict_cov=self.predict_cov)
        mean, logstd = out["mean"], out.get("logstd")
        if self.input_spec.output_layout == "steps":
            return {"vel": mean} if logstd is None else {"vel": mean, "logstd": logstd}
        # 窗口级降级布局（规格卡 §6）：只用最后一个子窗，其余步保留在 seq_disp 中
        result = {"vel": mean[:, -1], "seq_disp": mean}
        if logstd is not None:
            result["logstd"] = logstd[:, -1]
        return result
