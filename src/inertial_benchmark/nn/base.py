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
    """模型输入规格：任务视图字段 + 固定通道 ``[gyro_xyz, acc_xyz]``。"""

    channels: tuple = CHANNELS

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "InputSpec":
        d = dict(d)
        d["channels"] = tuple(d.get("channels", CHANNELS))
        return cls(**d)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["channels"] = list(self.channels)
        return d

    @property
    def num_channels(self) -> int:
        return len(self.channels)


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

    ``forward(imu)`` 接收 ``(B, 6, T)``，返回 dict：必含 ``vel (B, dims)``，可含 ``logstd``、
    ``cov``、``aux`` 等。``loss(out, batch, epoch)`` 默认按 ``loss_name`` 计算，可在子类覆盖。
    推理时 ``vel``/``logstd`` 之外首维为批大小的张量也会保存到预测文件（见 ``saved_outputs``）。

    ``batch`` 的结构在训练与验证中**一致**：至少含 ``target`` 与 ``imu``（``LOSS_BATCH_KEYS``），
    需要时另有 ``mask``（逐帧/多步目标的有效掩码）与 ``extra``（额外输入，见 DESIGN 第 3 节）。
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
        return build_loss(self.loss_name, **self.loss_kwargs)(out, batch["target"], epoch)

    def check_input(self, imu: torch.Tensor) -> None:
        spec = self.input_spec
        if imu.ndim != 3 or imu.shape[1] != spec.num_channels or imu.shape[2] != spec.window:
            raise ValueError(f"expected input (B, {spec.num_channels}, {spec.window}), "
                             f"got {tuple(imu.shape)}")

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
