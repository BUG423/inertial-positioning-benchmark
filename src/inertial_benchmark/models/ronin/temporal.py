"""RoNIN 的两个 seq2seq 模型：``ronin_lstm``（Bilinear + 单向 LSTM）与 ``ronin_tcn``（因果 TCN）。

论文：S. Herath, H. Yan, Y. Furukawa, "RoNIN: Robust Neural Inertial Navigation in the Wild:
Benchmark, Evaluations, & New Methods", ICRA 2020. https://arxiv.org/abs/1905.12853
官方仓库：https://github.com/Sachini/ronin @ ``805b7f0f28bb164ce89ada9ac05a9470dbe3d715``
（``source/model_temporal.py``、``source/tcn.py``、``source/ronin_lstm_tcn.py``）。
许可：官方代码 GPL-3.0（``source/tcn.py`` 改编自 locuslab/TCN，MIT）。**本文件依据规格卡
``docs/algorithms/ronin_lstm.md`` 与 ``docs/algorithms/ronin_tcn.md`` 独立实现，未阅读、未复制
官方代码**，因此不受 copyleft 约束。
fidelity：``official-code``（结构、参数量、专用损失与训练配方；参数量由 ``tests/models``
的夹具锁定：LSTM 216,620 / TCN 540,488）。

两者共用同一套任务视图：200 Hz、400 帧窗口、重力对齐世界系、**逐帧 2D 速度**目标
（``target=frame_velocity``，输出布局 ``(B, T, 2)``，DESIGN §3.1）。

与官方的差异及理由：

1. **``lstm_bi`` 是 bilinear 不是 bidirectional**（``ronin_lstm.md`` §1、§10.1）：官方
   ``--type lstm_bi`` 对应 ``BilinearLSTMSeqNetwork``，LSTM 本身单向。IPB 的 ``ronin_lstm``
   即该模型；不带 bilinear 的 ``--type lstm``（205,832 参数）只作为变体记录，不注册；
2. **隐藏状态不再绑定构造时的 batch**（``ronin_lstm.md`` §6、§10.2）：官方在构造时固定
   ``batch_size`` 并每次前向新建零状态，数值上等价于 ``nn.LSTM`` 的 ``hx=None``；IPB 去掉这一
   耦合，因此不需要 ``drop_last`` 才能推理；
3. **逐帧目标的定义**：官方逐帧目标为前向差分 ``(p[k+1] − p[k])/dt``，IPB ``frame_velocity``
   为中心差分（有 ``pose/velocity`` 时直接取，DESIGN §3.1）。两者都是同一帧的平均速度估计，
   差别只在半个采样间隔的相位，属于协议效应；
4. **推理方式**：官方测试对整条序列一次前向（LSTM 隐藏状态贯穿全序列、TCN 感受野 253 帧），
   IPB 默认对每个 400 帧窗口独立推理，再按 ``overlap=mean`` 合并重叠的逐帧输出（DESIGN §5）。
   :meth:`forward_sequence` 保留了官方的整序列推理（``ronin_*@stream`` 的钩子），
   窗口模式下 TCN 的前 252 帧与官方训练时的条件相同（左侧等价于补零）；
5. **不复刻官方的两个 bug**（``ronin_lstm.md`` §10.4/§10.5）：平滑参数的键名写错
   （``'feature_sigma,'``）使平滑永不生效——IPB 同样不做平滑；论文提到的线性层 dropout
   （keep 0.8）官方代码没有实现——IPB 也不加；
6. **TCN 的 weight norm 初始化**（``ronin_tcn.md`` §4、§10.2）：官方对 ``conv.weight.data`` 做
   ``N(0, 0.01)``，但 weight norm 早已把权重拆成 ``g, v``，第一次前向会用 ``g·v/‖v‖`` 覆盖，
   所以该初始化**不生效**。IPB 忠实复现“实际生效的行为”：卷积有效权重服从 Conv1d 默认初始化
   （``kaiming_uniform_(a=√5)``）、``g = ‖v‖``；只有 1×1 捷径卷积与输出层的 ``N(0, 0.01)``
   / ``N(0, 0.001)`` 真正生效；
7. **weight norm 用自实现**（``ronin_tcn.md`` §6）：新版 PyTorch 的 ``nn.utils.weight_norm``
   已弃用、``parametrizations.weight_norm`` 会改变参数名与注册顺序；:class:`WeightNormConv1d`
   保持官方的 ``bias, weight_g, weight_v`` 顺序，与夹具逐项一致；
8. **训练样本的速度上限过滤**（``max_velocity_norm=3.0``）尚未在协议层落地
   （``docs/ALGORITHMS.md`` §3），两个配方都不做该过滤，属已知差异；
9. 官方损失的“位置”没有乘 ``dt``（``ronin_lstm.md`` §10.10），数值是米制位置的 200 倍。
   :class:`GlobalPosLoss` 保持这一点——修正它会改变学习率等超参数的有效尺度。
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from ...nn.base import BaseModel, InputSpec
from ...nn.losses import masked_mean, mse, register_loss
from ...nn.registry import register_model


# --------------------------------------------------------------------------- 专用损失
def _invalid_counts(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """``cum[:, k]``：帧 ``0..k−1`` 中无效帧的个数（长 ``T+1``）；``mask`` 为 None 时返回 None。"""
    if mask is None:
        return None
    invalid = (~mask.bool()).to(torch.float32)
    return F.pad(torch.cumsum(invalid, dim=1), (1, 0))


@register_loss("ronin_global_pos")
class GlobalPosLoss:
    """RoNIN 的 latent velocity loss（``ronin_lstm.md`` §5、``ronin_tcn.md`` §5）。

    逐帧速度先去掉第 0 帧再累加成“位置”（**不乘 dt**，与官方一致）：

    * ``mode="full"``（RoNIN-LSTM）：``P_j = Σ_{f=1..j+1} v_f``，损失为
      ``mean_j (P̂_j − P_j)²``，``j = 0..T−2``；
    * ``mode="part"``（RoNIN-TCN，``history=253`` = 感受野）：``D_j = P_{j+history} − P_j``
      （等于帧 ``j+2 … j+history+1`` 的速度之和），损失为 ``mean_j (D̂_j − D_j)²``，
      ``j = 0..T−2−history``；``T=400``、``history=253`` 时共 146 项。

    掩码（DESIGN §3.1）：无效帧的误差先置零，再按“该项涉及的帧是否全部有效”给项加权；
    某个样本的全部项都无效时损失为 0 且保留计算图。视图只提供窗口级目标时（``(B, D)``）
    退化为 ``MSE(vel, target)``，并在 ``items`` 中标记 ``degraded=1``（结果中须注明）。
    """

    name = "ronin_global_pos"

    def __init__(self, mode: str = "full", history: int = 253) -> None:
        if mode not in ("full", "part"):
            raise ValueError(f"mode must be 'full' or 'part', got {mode!r}")
        self.mode = mode
        self.history = int(history)

    def terms(self, pred: torch.Tensor, target: torch.Tensor,
              mask: Optional[torch.Tensor] = None) -> tuple:
        """返回 ``(误差项 (B, J, D), 项掩码 (B, J) 或 None)``。"""
        frames = pred.shape[1]
        error = pred - target
        if mask is not None:
            error = torch.where(mask.bool().unsqueeze(-1), error, torch.zeros_like(error))
        prefix = torch.cumsum(error[:, 1:], dim=1)  # P̂_j − P_j，j = 0..T−2
        cum = _invalid_counts(mask)
        if self.mode == "full":
            if cum is None:
                return prefix, None
            ends = torch.arange(2, frames + 1, device=pred.device)
            valid = (cum[:, ends] - cum[:, 1:2]) == 0
            return prefix, valid
        if self.history >= frames - 1:
            raise ValueError(f"history={self.history} needs more than {self.history + 1} frames")
        terms = prefix[:, self.history:] - prefix[:, : -self.history]
        if cum is None:
            return terms, None
        j = torch.arange(terms.shape[1], device=pred.device)
        valid = (cum[:, j + self.history + 2] - cum[:, j + 2]) == 0
        return terms, valid

    def __call__(self, out: dict, target: torch.Tensor, epoch: int = 0,
                 mask: Optional[torch.Tensor] = None) -> tuple:
        pred = out["vel"]
        if target.ndim < 3:  # 视图只给窗口级目标：降级为窗口 MSE（结果中须标注）
            loss = mse(pred, target, mask)
            return loss, {"mse": loss.detach(), "degraded": torch.ones((), device=pred.device)}
        terms, valid = self.terms(pred, target, mask)
        loss = masked_mean(terms ** 2, None if valid is None
                           else valid.to(terms.dtype).unsqueeze(-1).expand_as(terms))
        return loss, {"global_pos": loss.detach(), "mse": mse(pred, target, mask).detach()}


# --------------------------------------------------------------------------- RoNIN-LSTM
class BilinearLSTMSeqNetwork(nn.Module):
    """``Bilinear(6,6,24) → concat → LSTM(30,100,3) → concat → Linear(130,10) → Linear(10,2)``。

    输入/输出都是 batch-first 的 ``(B, T, C)``（与官方模块一致）；两个线性层之间**没有非线性**
    （``ronin_lstm.md`` §10.3），等价于一个秩 ≤ 10 的线性映射，但参数按两层计。
    """

    def __init__(self, input_size: int = 6, out_size: int = 2, lstm_size: int = 100,
                 lstm_layers: int = 3, dropout: float = 0.0, mix_size: int = 24) -> None:
        super().__init__()
        self.bilinear = nn.Bilinear(input_size, input_size, mix_size)
        self.lstm = nn.LSTM(input_size + mix_size, lstm_size, lstm_layers, batch_first=True,
                            dropout=dropout)
        self.linear1 = nn.Linear(lstm_size + input_size + mix_size, out_size * 5)
        self.linear2 = nn.Linear(out_size * 5, out_size)

    def forward(self, x: torch.Tensor, state: Optional[tuple] = None) -> tuple:
        mix = torch.cat([x, self.bilinear(x, x)], dim=-1)
        out, state = self.lstm(mix, state)
        return self.linear2(self.linear1(torch.cat([mix, out], dim=-1))), state


# --------------------------------------------------------------------------- RoNIN-TCN
class WeightNormConv1d(nn.Module):
    """自实现的 weight-norm 卷积：参数注册顺序为 ``bias, weight_g, weight_v``（与夹具一致）。

    有效权重 ``W = g · v / ‖v‖``，范数按输出通道（``dim=0``）计算，与旧式
    ``torch.nn.utils.weight_norm(dim=0)`` 逐位等价；``g`` 与 ``v`` 由 ``nn.Conv1d`` 的默认
    初始化导出，因此 ``g = ‖v‖``、有效权重即 Conv1d 默认初始化（``ronin_tcn.md`` §4）。
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0,
                 dilation: int = 1) -> None:
        super().__init__()
        ref = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding,
                        dilation=dilation, bias=True)
        v = ref.weight.detach().clone()
        self.bias = nn.Parameter(ref.bias.detach().clone())
        self.weight_g = nn.Parameter(torch.linalg.vector_norm(v, dim=(1, 2), keepdim=True))
        self.weight_v = nn.Parameter(v)
        self.padding = int(padding)
        self.dilation = int(dilation)

    @property
    def weight(self) -> torch.Tensor:
        norm = torch.linalg.vector_norm(self.weight_v, dim=(1, 2), keepdim=True)
        return self.weight_g * self.weight_v / norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv1d(x, self.weight, self.bias, padding=self.padding, dilation=self.dilation)


class Chomp1d(nn.Module):
    """去掉时间轴最后 ``size`` 个样本，使膨胀卷积因果且长度不变。"""

    def __init__(self, size: int) -> None:
        super().__init__()
        self.size = int(size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., : -self.size] if self.size > 0 else x


class TemporalBlock(nn.Module):
    """TCN 残差块：两支 ``WN-Conv → Chomp → PReLU → Dropout``，捷径为 1×1 卷积（需要时）。"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int,
                 dropout: float = 0.2) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = WeightNormConv1d(in_channels, out_channels, kernel_size, padding, dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.PReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = WeightNormConv1d(out_channels, out_channels, kernel_size, padding, dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.PReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = (nn.Conv1d(in_channels, out_channels, 1)
                           if in_channels != out_channels else None)
        self.relu = nn.PReLU()
        if self.downsample is not None:
            nn.init.normal_(self.downsample.weight, 0.0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout1(self.relu1(self.chomp1(self.conv1(x))))
        out = self.dropout2(self.relu2(self.chomp2(self.conv2(out))))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """``network``：若干 :class:`TemporalBlock`，第 ``i`` 块的膨胀率为 ``2^i``（通道优先）。"""

    def __init__(self, input_channel: int, layer_channels: Sequence[int], kernel_size: int = 3,
                 dropout: float = 0.2) -> None:
        super().__init__()
        blocks = []
        in_channels = input_channel
        for i, channels in enumerate(layer_channels):
            blocks.append(TemporalBlock(in_channels, channels, kernel_size, 2 ** i, dropout))
            in_channels = channels
        self.network = nn.Sequential(*blocks)
        self.out_channels = in_channels
        self.kernel_size = int(kernel_size)
        self.num_blocks = len(layer_channels)

    @property
    def receptive_field(self) -> int:
        """``1 + 2·(k−1)·(2^L − 1)``：本结构（每块两个卷积、膨胀按 2^i 递增）下为 253。"""
        return 1 + 2 * (self.kernel_size - 1) * (2 ** self.num_blocks - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TCNSeqNetwork(nn.Module):
    """``6 个 TemporalBlock（膨胀 2^i）→ Dropout → Conv1d(k1)``；输入/输出为 ``(B, T, C)``。"""

    def __init__(self, input_channel: int = 6, output_channel: int = 2, kernel_size: int = 3,
                 layer_channels: Sequence[int] = (32, 64, 128, 256, 72, 36),
                 dropout: float = 0.2) -> None:
        super().__init__()
        self.tcn = TemporalConvNet(input_channel, layer_channels, kernel_size, dropout)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Conv1d(self.tcn.out_channels, output_channel, 1)
        nn.init.normal_(self.output_layer.weight, 0.0, 0.01)
        nn.init.normal_(self.output_layer.bias, 0.0, 0.001)

    @property
    def receptive_field(self) -> int:
        return self.tcn.receptive_field

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.output_layer(self.dropout(self.tcn(x.transpose(1, 2))))
        return out.transpose(1, 2)


# --------------------------------------------------------------------------- IPB 包装
class RoNINSeqModel(BaseModel):
    """RoNIN seq2seq 模型的公共包装：``(B, 6, T)`` ↔ 官方的 batch-first ``(B, T, C)``。

    ``output_layout == "frame"``（``target=frame_velocity``）时直接输出逐帧速度 ``(B, T, D)``；
    配成窗口级目标时按规格卡 §6 取 ``vel = mean_{k=0..T−2} y_k``（该均值恰好等于 IPB 的
    ``avg_velocity``，因为 ``y_k ≈ (p[k+1] − p[k])/dt``），逐帧输出仍放在 ``frame_vel`` 中。
    """

    default_loss = "ronin_global_pos"
    saved_outputs: tuple = ()  # 逐帧输出不逐窗口存盘（窗口级布局下体积过大）

    def module_forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - 抽象
        raise NotImplementedError

    def forward(self, imu: torch.Tensor) -> dict:
        self.check_input(imu)
        frame_vel = self.module_forward(imu.transpose(1, 2))
        if self.input_spec.output_layout == "frame":
            return {"vel": frame_vel}
        return {"vel": frame_vel[:, :-1].mean(dim=1), "frame_vel": frame_vel}

    @torch.no_grad()
    def forward_sequence(self, imu: torch.Tensor) -> torch.Tensor:
        """官方测试口径：整条序列 ``(B, 6, N)`` 一次前向，返回逐帧速度 ``(B, N, D)``。

        ``ronin_*@stream`` 的钩子（``ronin_lstm.md`` §6、``ronin_tcn.md`` §6）；LSTM 的隐藏
        状态贯穿全序列，TCN 因严格因果而与窗口模式在 ``t ≥ 252`` 上一致。
        """
        return self.module_forward(imu.transpose(1, 2))


@register_model("ronin_lstm")
class RoNINLSTM(RoNINSeqModel):
    """RoNIN-LSTM（官方 ``--type lstm_bi``）：216,620 参数，逐帧 2D 速度。"""

    def __init__(self, input_spec: InputSpec, lstm_size: int = 100, lstm_layers: int = 3,
                 dropout: float = 0.0, mix_size: int = 24) -> None:
        super().__init__(input_spec)
        self.net = BilinearLSTMSeqNetwork(input_spec.num_channels, input_spec.dims, lstm_size,
                                          lstm_layers, dropout, mix_size)

    def module_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)[0]


@register_model("ronin_tcn")
class RoNINTCN(RoNINSeqModel):
    """RoNIN-TCN：540,488 参数（官方 config 通道 ``[32,64,128,256,72,36]``），感受野 253 帧。"""

    def __init__(self, input_spec: InputSpec, kernel_size: int = 3,
                 layer_channels: Sequence[int] = (32, 64, 128, 256, 72, 36),
                 dropout: float = 0.2) -> None:
        super().__init__(input_spec)
        self.net = TCNSeqNetwork(input_spec.num_channels, input_spec.dims, kernel_size,
                                 layer_channels, dropout)

    @property
    def receptive_field(self) -> int:
        return self.net.receptive_field

    def module_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


__all__ = ["BilinearLSTMSeqNetwork", "Chomp1d", "GlobalPosLoss", "RoNINLSTM", "RoNINSeqModel",
           "RoNINTCN", "TCNSeqNetwork", "TemporalBlock", "TemporalConvNet", "WeightNormConv1d"]
