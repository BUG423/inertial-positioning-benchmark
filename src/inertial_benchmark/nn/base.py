"""模型基类与输入规格（DESIGN 第 4 节）。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional

import torch
from torch import nn

from ..data.views import CHANNELS, ViewConfig
from .losses import build_loss


@dataclass(frozen=True)
class InputSpec(ViewConfig):
    """模型输入/输出规格：任务视图字段 + 固定通道 ``[gyro_xyz, acc_xyz]``。

    输入形状由 ``history`` 决定：``(B, C, T)``，或声明历史子窗口时 ``(B, H_in, C, T)``。
    输出布局与各输出的时间偏移由 ``target`` 与 ``output_steps`` 决定（继承自 ``ViewConfig``）：

    * ``output_layout == "window"``：``(B, D)``，时间偏移为窗口中心（或末端）；
    * ``output_layout == "frame"``：``(B, T, D)``，第 ``i`` 行对应窗口内第 ``i`` 帧；
    * ``output_layout == "steps"``：``(B, H, D)``，第 ``h`` 行对应窗口第 ``h`` 段的中心。

    ``output_offsets``（秒，相对窗口首样本）与 ``output_scales``（换算为速度的比例）给出
    Predictor 把每个输出映射到时间轴所需的全部信息；``overlap`` 决定重叠预测的合并策略。
    """

    channels: tuple = CHANNELS

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "InputSpec":
        d = dict(d)
        d["channels"] = tuple(d.get("channels", CHANNELS))
        d["extra_inputs"] = tuple(d.get("extra_inputs", ()))
        return cls(**d)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["channels"] = list(self.channels)
        d["extra_inputs"] = list(self.extra_inputs)
        return d

    @property
    def num_channels(self) -> int:
        return len(self.channels)

    @property
    def input_shape(self) -> tuple:
        """单个样本的输入形状（不含批维）。"""
        if self.sub_windows > 0:
            return (self.sub_windows, self.num_channels, int(self.window))
        return (self.num_channels, int(self.window))


# 损失接口的 batch 结构：训练与验证都必须提供这些键（其余键可选，例如 mask / extra）
LOSS_BATCH_KEYS = ("target", "imu")


def check_loss_batch(batch: Mapping[str, Any]) -> None:
    """校验损失 batch 的结构，避免训练与验证传入不同的键。"""
    missing = [k for k in LOSS_BATCH_KEYS if k not in batch]
    if missing:
        raise KeyError(f"loss batch is missing {missing}; training and validation must pass the "
                       f"same keys (at least {list(LOSS_BATCH_KEYS)})")


class BaseModel(nn.Module):
    """所有模型的基类。

    ``forward(imu)`` 接收 ``(B, C, T)``（声明 ``history`` 时为 ``(B, H_in, C, T)``），返回 dict：
    必含 ``vel``（形状按 ``input_spec.output_shape``：``(B,D)`` / ``(B,T,D)`` / ``(B,H,D)``），
    可含 ``logstd``、``cov``、``aux`` 等。声明 ``extra_inputs`` 的模型签名为
    ``forward(imu, extra)``（``extra`` 为 ``{名称: 张量}``）。
    ``loss(out, batch, epoch)`` 默认按 ``loss_name`` 计算，可在子类覆盖。
    推理时 ``vel``/``logstd`` 之外首维为批大小的张量也会保存到预测文件（见 ``saved_outputs``）。

    ``batch`` 的结构在训练与验证中**一致**：至少含 ``target`` 与 ``imu``（``LOSS_BATCH_KEYS``），
    另有 ``mask``（逐帧/多步目标的有效掩码）与可选的 ``extra``（额外输入，见 DESIGN 第 3 节）。
    """

    default_loss = "mse"
    # 推理时额外保存的逐窗口输出键（``vel``/``logstd`` 之外）；None 表示保存全部逐窗口张量
    saved_outputs: Optional[tuple] = None

    def __init__(self, input_spec: InputSpec) -> None:
        super().__init__()
        self.input_spec = input_spec
        self.loss_name = self.default_loss
        self.loss_kwargs: dict = {}
        self.model_cfg: dict = {}

    def forward(self, imu: torch.Tensor) -> dict:  # pragma: no cover - 抽象方法
        raise NotImplementedError

    def set_loss(self, name: str, **kwargs: Any) -> None:
        build_loss(name, **kwargs)  # 立即校验名字
        self.loss_name = name
        self.loss_kwargs = dict(kwargs)

    def loss(self, out: dict, batch: dict, epoch: int = 0) -> tuple:
        """返回 ``(标量损失, {名称: 分离后的张量})``；``batch`` 必须含 ``LOSS_BATCH_KEYS``。"""
        check_loss_batch(batch)
        fn = build_loss(self.loss_name, **self.loss_kwargs)
        return fn(out, batch["target"], epoch, batch.get("mask"))

    def check_input(self, imu: torch.Tensor) -> None:
        spec = self.input_spec
        expected = spec.input_shape
        if imu.ndim != len(expected) + 1 or tuple(imu.shape[1:]) != expected:
            raise ValueError(f"expected input (B, {', '.join(str(v) for v in expected)}), "
                             f"got {tuple(imu.shape)}")

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class SequenceModel(BaseModel):
    """序列级 / 有状态模型接口（PDR、有状态递推、滤波类方法）。

    这类方法不能逐窗口独立运行（需要跨窗口的连续滤波、状态或整段标定），因此不实现
    ``forward``，而是实现 :meth:`predict_sequence`：

    ``predict_sequence(seq, view, starts) -> (times, velocities[, extras])``

    * ``starts``：Predictor 给出的窗口起点（由 ``eval_stride`` 决定），模型据此汇总到同一网格；
    * ``times``：``(K,)`` 秒，**必须**落在视图的窗口网格上，即
      ``view.target_times(starts)``；这样它们与逐窗口模型共用同一套轨迹重建
      （DESIGN 第 5 节）与同一套指标；
    * ``velocities``：``(K, dims)``，**视图坐标系**中的窗口速度（与 ``target`` 同一约定，
      由 Predictor 调用 ``view.to_world_velocity`` 转到世界系）；
    * ``extras``：可选的 ``{名称: (K, ...)}``，按逐窗口输出保存到预测文件。

    需要在 train 划分上拟合标定标量的方法实现 :meth:`calibrate`：Trainer 检测到序列级模型时
    不跑梯度循环，只调用一次 ``calibrate``（只传 train 划分的视图），再走同一套验证与结果写出。
    """

    def forward(self, imu: torch.Tensor) -> dict:  # pragma: no cover - 序列级模型不逐窗口前向
        raise NotImplementedError(
            f"{type(self).__name__} is a SequenceModel: use predict_sequence(seq, view)")

    def predict_sequence(self, seq: Any, view: Any,
                         starts: Any) -> tuple:  # pragma: no cover - 抽象方法
        raise NotImplementedError

    def calibrate(self, views: Any, split: str = "train") -> dict:
        """在 train 划分上拟合标定标量，返回写入结果的标定信息（缺省无需标定）。

        ``split`` 由 Trainer 显式传入，实现应在收到 ``train`` 以外的划分时报错（防止泄漏）。
        """
        return {}
