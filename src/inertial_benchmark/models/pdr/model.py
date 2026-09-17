"""PDR 基线：步检测 + Weinberg 步长 + 设备航向（规格见 ``docs/algorithms/pdr.md``）。

经典三段式行人航位推算，作为 benchmark 的**下界参照**：不学习也能做到多少，以及“设备朝向 ≠ 行走
方向”在各数据集上的影响。没有可学习参数，只有两个在 train 划分上拟合的标定标量：

* ``K``：Weinberg 步长系数 ``L_i = K · (a_max,i − a_min,i)^{1/4}``（AN-602）；
* ``δ``：常值航向偏置。

PDR 是**序列级**方法（步检测需要跨窗口的连续滤波与峰值检测），因此实现
:class:`~inertial_benchmark.nn.base.SequenceModel`：把检测到的步汇总到 IPB 的窗口网格上，再走
DESIGN 第 5 节统一的轨迹重建与第 6 节统一的指标。
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch

from ...nn.base import InputSpec, SequenceModel
from ...nn.registry import register_model
from ...utils import LOGGER
from .heading import circular_mean, resolve_axes, segment_heading, window_headings

G0 = 9.80665  # m/s²，标准重力（步检测信号去掉的常量）
SIGNALS = ("norm", "vertical")


def valid_segments(valid: np.ndarray, min_length: int) -> list:
    """``valid`` 中连续为真且长度 ≥ ``min_length`` 的区段 ``[(lo, hi), …]``（``hi`` 不含）。"""
    valid = np.asarray(valid, bool)
    edges = np.flatnonzero(np.diff(np.concatenate([[False], valid, [False]]).astype(np.int8)))
    return [(int(lo), int(hi)) for lo, hi in edges.reshape(-1, 2) if hi - lo >= min_length]


def lowpass(x: np.ndarray, rate: float, cutoff: float, order: int, causal: bool) -> np.ndarray:
    """Butterworth 低通；默认零相位 ``filtfilt``（离线评测允许非因果）。"""
    from scipy.signal import butter, filtfilt, lfilter

    b, a = butter(int(order), float(cutoff) / (0.5 * float(rate)), btype="low")
    if causal:
        return lfilter(b, a, x)
    return filtfilt(b, a, x, padlen=min(3 * max(len(a), len(b)), len(x) - 1))


@register_model("pdr")
class PDR(SequenceModel):
    """步检测 + Weinberg 步长 + 设备航向的行人航位推算基线。

    Args:
        signal: ``norm``（世界系比力模长减 g₀）或 ``vertical``（竖直分量减 g₀）。
        cutoff_hz / order: 低通截止频率与阶数。
        causal: True 时用 ``lfilter``（引入群延迟），默认 ``filtfilt``。
        min_height / min_prominence / min_interval_s: 峰值检测阈值与最小步间隔。
        amp_window_s: 振幅统计窗 ``W``。
        horizontal_min: 前向轴水平投影阈值（低于时换备用轴）。
        body_axes: 补充的 ``body_frame → [前向轴, 备用轴]`` 映射。
        calibrate_delta: False 时固定 ``δ = 0``。
        segment_s: 标定 ``K`` 时的分段长度。
        delta_stride: 标定 ``δ`` 时的窗口步长（与训练视图一致）。
        min_speed: 标定 ``δ`` 时只用速度超过该值的窗口。
    """

    def __init__(
        self,
        input_spec: InputSpec,
        signal: str = "norm",
        cutoff_hz: float = 3.0,
        order: int = 4,
        causal: bool = False,
        min_height: float = 1.0,
        min_prominence: float = 0.5,
        min_interval_s: float = 0.3,
        amp_window_s: float = 1.0,
        horizontal_min: float = 0.2,
        body_axes: Optional[dict] = None,
        calibrate_delta: bool = True,
        segment_s: float = 10.0,
        delta_stride: int = 10,
        min_speed: float = 0.5,
    ) -> None:
        super().__init__(input_spec)
        if signal not in SIGNALS:
            raise ValueError(f"signal={signal!r} not in {SIGNALS}")
        if input_spec.frame != "gravity_world":
            raise ValueError("pdr needs frame=gravity_world (the step signal and the heading are "
                             "defined in the gravity-aligned world frame)")
        if input_spec.output_layout != "window":
            raise ValueError("pdr outputs one average velocity per window "
                             "(target=avg_velocity / displacement / velocity_at_end)")
        self.signal = signal
        self.cutoff_hz = float(cutoff_hz)
        self.order = int(order)
        self.causal = bool(causal)
        self.min_height = float(min_height)
        self.min_prominence = float(min_prominence)
        self.min_interval_s = float(min_interval_s)
        self.amp_window_s = float(amp_window_s)
        self.horizontal_min = float(horizontal_min)
        self.body_axes = dict(body_axes or {})
        self.calibrate_delta = bool(calibrate_delta)
        self.segment_s = float(segment_s)
        self.delta_stride = int(delta_stride)
        self.min_speed = float(min_speed)
        # 标定标量随 checkpoint 保存（没有可学习参数）
        self.register_buffer("step_gain", torch.zeros(()))
        self.register_buffer("heading_bias", torch.zeros(()))

    # ------------------------------------------------------------------ 步检测
    def axes(self, view) -> tuple:
        return resolve_axes(str(view.seq.attrs.get("body_frame", "unknown")), self.body_axes)

    def scalar_signal(self, view) -> np.ndarray:
        """步检测用的标量信号（世界系比力，已减去 g₀）。"""
        acc = view.imu[:, 3:6].astype(np.float64)
        if self.signal == "vertical":
            return acc[:, 2] - G0
        return np.linalg.norm(acc, axis=1) - G0

    def detect_steps(self, view) -> dict:
        """整条序列的步：时间、振幅、（未加 δ 的）航向。"""
        from scipy.signal import find_peaks

        rate = float(view.cfg.rate)
        signal = self.scalar_signal(view)
        fwd, alt = self.axes(view)
        distance = max(int(round(self.min_interval_s * rate)), 1)
        amp_window = max(int(round(self.amp_window_s * rate)), 1)
        min_length = max(3 * max(2 * self.order + 1, 1) + 1, int(0.5 * rate))
        times, amps, headings, alt_used = [], [], [], []
        for lo, hi in valid_segments(view.valid_input, min_length):
            smooth = lowpass(signal[lo:hi], rate, self.cutoff_hz, self.order, self.causal)
            peaks, _ = find_peaks(smooth, height=self.min_height,
                                  prominence=self.min_prominence, distance=distance)
            previous = None
            for peak in peaks:
                # I_i = (peak_{i−1}, peak_i] ∩ [peak_i − W, peak_i]；首步左端为区段起点
                left = 0 if previous is None else previous + 1
                start = max(left, peak - amp_window)
                window = smooth[start:peak + 1]
                amps.append(float(window.max() - window.min()))
                times.append(float(view.seq.timestamp[lo + peak]))
                angle, used = segment_heading(view.q[lo + start:lo + peak + 1], fwd, alt,
                                              self.horizontal_min)
                headings.append(angle)
                alt_used.append(used)
                previous = peak
        order = np.argsort(times, kind="stable")
        return {"time": np.asarray(times)[order], "amplitude": np.asarray(amps)[order],
                "heading": np.asarray(headings)[order],
                "alt_axis": np.asarray(alt_used, bool)[order] if alt_used
                else np.zeros(0, bool)}

    # ------------------------------------------------------------------ 推理
    def step_displacements(self, steps: dict) -> np.ndarray:
        """各步的水平位移 ``L_i·[cos ψ_i, sin ψ_i]``（含标定的 K 与 δ）。"""
        gain = float(self.step_gain)
        delta = float(self.heading_bias)
        length = gain * np.power(np.maximum(steps["amplitude"], 0.0), 0.25)
        angle = steps["heading"] + delta
        return length[:, None] * np.stack([np.cos(angle), np.sin(angle)], axis=1)

    def predict_sequence(self, seq, view, starts: np.ndarray) -> tuple:
        """把步汇总到窗口网格：``v_k = Σ_{t_i ∈ (t_s, t_e]} L_i·u_i / ((T−1)·dt)``。"""
        cfg = view.cfg
        starts = np.asarray(starts, dtype=np.int64)
        steps = self.detect_steps(view)
        displacement = self.step_displacements(steps)
        cumulative = np.concatenate([np.zeros((1, 2)), np.cumsum(displacement, axis=0)])
        t_start = seq.timestamp[starts]
        t_end = seq.timestamp[starts + cfg.window - 1]
        # 时间正好等于 t_s 的步不计入该窗口，等于 t_e 的计入
        lo = np.searchsorted(steps["time"], t_start, side="right")
        hi = np.searchsorted(steps["time"], t_end, side="right")
        total = cumulative[hi] - cumulative[lo]
        velocity = total / ((cfg.window - 1) * cfg.dt)
        out = np.zeros((len(starts), cfg.dims))
        out[:, :2] = velocity
        extras = {"num_steps": (hi - lo).astype(np.float64)}
        return view.target_times(starts), out, extras

    # ------------------------------------------------------------------ 标定（只用 train）
    def calibrate(self, views: Sequence, split: str = "train") -> dict:
        """在 train 划分上拟合 ``K`` 与 ``δ``（传入其他划分时报错，防止泄漏）。"""
        if split != "train":
            raise ValueError(f"pdr calibration may only use the train split, got {split!r}")
        amp_sum, distance, angles = [], [], []
        for view in views:
            steps = self.detect_steps(view)
            a, d = self._segment_pairs(view, steps)
            amp_sum.extend(a)
            distance.extend(d)
            if self.calibrate_delta:
                angles.extend(self._heading_errors(view, steps))
        amp_sum = np.asarray(amp_sum, dtype=np.float64)
        distance = np.asarray(distance, dtype=np.float64)
        stats: dict = {"segments": int(len(amp_sum)), "windows": int(len(angles))}
        if amp_sum.size and float(np.sum(amp_sum**2)) > 0:
            gain = float(np.sum(amp_sum * distance) / np.sum(amp_sum**2))
            residual = distance - gain * amp_sum
            spread = float(np.sum((distance - distance.mean()) ** 2))
            stats["step_gain"] = gain
            stats["length_rmse"] = float(np.sqrt(np.mean(residual**2)))
            stats["r_squared"] = float(1.0 - np.sum(residual**2) / spread) if spread > 0 else \
                float("nan")
        else:
            gain = 0.0
            LOGGER.warning("pdr: no step was detected on the train split; K stays 0")
        delta, resultant = circular_mean(np.asarray(angles)) if angles else (0.0, 0.0)
        stats["heading_bias_deg"] = float(np.degrees(delta))
        stats["heading_resultant"] = float(resultant)
        with torch.no_grad():
            self.step_gain.fill_(gain)
            self.heading_bias.fill_(delta)
        return stats

    def _segment_pairs(self, view, steps: dict) -> tuple:
        """``K`` 标定的逐段 ``(Σ 振幅^{1/4}, 参考路径长)``：互不重叠、全部有效的 10 s 段。"""
        from ...metrics.trajectory import path_length

        cfg = view.cfg
        length = max(int(round(self.segment_s * cfg.rate)), 2)
        amp_quarter = np.power(np.maximum(steps["amplitude"], 0.0), 0.25)
        sums, distances = [], []
        n = len(view.seq)
        for lo in range(0, n - length + 1, length):
            hi = lo + length
            if not view.valid[lo:hi].all():
                continue
            mask = (steps["time"] >= view.seq.timestamp[lo]) & \
                   (steps["time"] <= view.seq.timestamp[hi - 1])
            total = float(amp_quarter[mask].sum())
            if total <= 0:
                continue
            sums.append(total)
            distances.append(path_length(view.seq.position[lo:hi], view.valid[lo:hi], cfg.rate,
                                         1.0, 2))
        return sums, distances

    def _heading_errors(self, view, steps: dict) -> list:
        """``δ`` 标定的逐窗口 ``ψ_gt − ψ_dev``（全部有效且真值速度超过阈值的窗口）。"""
        cfg = view.cfg
        starts = view.starts(self.delta_stride)
        if len(starts) == 0:
            return []
        target = view.targets_world(starts)[:, :2]
        speed = np.linalg.norm(target, axis=1)
        keep = speed > self.min_speed
        if not keep.any():
            return []
        starts = starts[keep]
        fwd, alt = self.axes(view)
        device, _ = window_headings(view.q, starts, cfg.window, fwd, alt, self.horizontal_min)
        truth = np.arctan2(target[keep, 1], target[keep, 0])
        return list(truth - device)
