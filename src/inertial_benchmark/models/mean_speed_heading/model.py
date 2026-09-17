"""常速 + 航向基线（规格见 ``docs/algorithms/mean_speed_heading.md``）。

最简基线：速度**大小**取训练集平均水平速度，**方向**取设备（或参考）姿态给出的航向。与 ``pdr``
共用航向定义与 ``δ`` 标定。它给出“完全不看 IMU 运动信息、只知道朝向”的下界：与 ``pdr`` 对比可以
看出步检测/步长的增益，与学习方法对比可以看出速度大小估计的价值。

变体（``variant=``）：

* ``constant``（默认）：``ŝ_k = s̄``（含静止窗口的训练集平均速率）；
* ``gated``：用未滤波比力范数的样本标准差做运动检测，运动时 ``ŝ_k = s̄_move``，否则 0；
* ``zero_velocity``：``v_k = 0``（健全性检查，轨迹停在锚点）；
* ``train_mean_velocity``：``v_k = v̄``（向量平均，不使用航向；用于检查数据是否有全局方向偏好）。
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch

from ...nn.base import InputSpec, SequenceModel
from ...nn.registry import register_model
from ..pdr.heading import circular_mean, resolve_axes, window_headings

VARIANTS = ("constant", "gated", "zero_velocity", "train_mean_velocity")


@register_model("mean_speed_heading")
class MeanSpeedHeading(SequenceModel):
    """训练集平均速率 × 设备航向的常速基线（无可学习参数，只有标定标量）。

    Args:
        variant: 见模块文档。
        motion_threshold: ``gated`` 变体的运动检测阈值 ``σ_th``（m/s²，固定经验值）。
        horizontal_min: 前向轴水平投影阈值（低于时换备用轴）。
        body_axes: 补充的 ``body_frame → [前向轴, 备用轴]`` 映射。
        calibrate_delta: False 时固定 ``δ = 0``。
        delta_stride: 标定时的窗口步长（与训练视图一致）。
        min_speed: 标定 ``δ`` 时只用速度超过该值的窗口。
    """

    def __init__(
        self,
        input_spec: InputSpec,
        variant: str = "constant",
        motion_threshold: float = 0.5,
        horizontal_min: float = 0.2,
        body_axes: Optional[dict] = None,
        calibrate_delta: bool = True,
        delta_stride: int = 10,
        min_speed: float = 0.5,
    ) -> None:
        super().__init__(input_spec)
        if variant not in VARIANTS:
            raise ValueError(f"variant={variant!r} not in {VARIANTS}")
        if input_spec.frame != "gravity_world":
            raise ValueError("mean_speed_heading needs frame=gravity_world")
        if input_spec.output_layout != "window":
            raise ValueError("mean_speed_heading outputs one average velocity per window")
        self.variant = variant
        self.motion_threshold = float(motion_threshold)
        self.horizontal_min = float(horizontal_min)
        self.body_axes = dict(body_axes or {})
        self.calibrate_delta = bool(calibrate_delta)
        self.delta_stride = int(delta_stride)
        self.min_speed = float(min_speed)
        self.register_buffer("mean_speed", torch.zeros(()))
        self.register_buffer("mean_speed_moving", torch.zeros(()))
        self.register_buffer("mean_velocity", torch.zeros(int(input_spec.dims)))
        self.register_buffer("heading_bias", torch.zeros(()))

    # ------------------------------------------------------------------ 辅助
    def axes(self, view) -> tuple:
        return resolve_axes(str(view.seq.attrs.get("body_frame", "unknown")), self.body_axes)

    def moving_mask(self, view, starts: np.ndarray) -> np.ndarray:
        """运动检测：窗口内未滤波比力范数的样本标准差超过 ``σ_th``（``ddof=0``）。"""
        starts = np.asarray(starts, dtype=np.int64)
        norm = np.linalg.norm(view.imu[:, 3:6].astype(np.float64), axis=1)
        idx = starts[:, None] + np.arange(view.cfg.window)
        return norm[idx].std(axis=1) > self.motion_threshold

    def window_velocity(self, view, starts: np.ndarray) -> np.ndarray:
        """逐窗口速度 ``(K, dims)``（视图坐标系 = 重力对齐世界系）。"""
        cfg = view.cfg
        out = np.zeros((len(starts), cfg.dims))
        if self.variant == "zero_velocity":
            return out
        if self.variant == "train_mean_velocity":
            out[:] = self.mean_velocity.detach().cpu().numpy()[None, :]
            return out
        fwd, alt = self.axes(view)
        angles, _ = window_headings(view.q, starts, cfg.window, fwd, alt, self.horizontal_min)
        angles = angles + float(self.heading_bias)
        if self.variant == "gated":
            speed = np.where(self.moving_mask(view, starts), float(self.mean_speed_moving), 0.0)
        else:
            speed = np.full(len(starts), float(self.mean_speed))
        out[:, 0] = speed * np.cos(angles)
        out[:, 1] = speed * np.sin(angles)
        return out

    def predict_sequence(self, seq, view, starts: np.ndarray) -> tuple:
        starts = np.asarray(starts, dtype=np.int64)
        return view.target_times(starts), self.window_velocity(view, starts), {}

    # ------------------------------------------------------------------ 标定（只用 train）
    def calibrate(self, views: Sequence, split: str = "train") -> dict:
        """在 train 划分上拟合 ``s̄``、``s̄_move``、``v̄`` 与 ``δ``（其他划分报错）。"""
        if split != "train":
            raise ValueError("mean_speed_heading calibration may only use the train split, "
                             f"got {split!r}")
        speeds, moving_speeds, velocities, angles = [], [], [], []
        for view in views:
            starts = view.starts(self.delta_stride)
            if len(starts) == 0:
                continue
            target = view.targets_world(starts)[:, :view.cfg.dims]
            speed = np.linalg.norm(target[:, :2], axis=1)
            speeds.extend(speed)
            velocities.append(target)
            moving = self.moving_mask(view, starts)  # 用同一个检测器，测试时与标定一致
            moving_speeds.extend(speed[moving])
            if self.calibrate_delta:
                keep = speed > self.min_speed
                if keep.any():
                    fwd, alt = self.axes(view)
                    device, _ = window_headings(view.q, starts[keep], view.cfg.window, fwd, alt,
                                                self.horizontal_min)
                    truth = np.arctan2(target[keep, 1], target[keep, 0])
                    angles.extend(truth - device)
        mean_speed = float(np.mean(speeds)) if speeds else 0.0
        mean_moving = float(np.mean(moving_speeds)) if moving_speeds else mean_speed
        mean_velocity = (np.concatenate(velocities).mean(axis=0) if velocities
                         else np.zeros(self.input_spec.dims))
        delta, resultant = circular_mean(np.asarray(angles)) if angles else (0.0, 0.0)
        with torch.no_grad():
            self.mean_speed.fill_(mean_speed)
            self.mean_speed_moving.fill_(mean_moving)
            self.mean_velocity.copy_(torch.as_tensor(mean_velocity, dtype=torch.float32))
            self.heading_bias.fill_(delta)
        return {"variant": self.variant, "windows": int(len(speeds)),
                "mean_speed": mean_speed, "mean_speed_moving": mean_moving,
                "mean_velocity": [float(v) for v in np.asarray(mean_velocity).ravel()],
                "heading_bias_deg": float(np.degrees(delta)),
                "heading_resultant": float(resultant)}
