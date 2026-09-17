"""任务视图（DESIGN 第 3 节）：从 ``Sequence`` 生成模型输入窗口与目标（不依赖 torch）。

输入通道顺序固定为 ``[gyro_xyz, acc_xyz]``，窗口形状 ``(6, T)``；目标与输入处于同一坐标系：

* ``gravity_world``：用姿态把 IMU 旋到重力对齐世界系，目标为世界系向量；
* ``gravity_yaw_local``：在上面的基础上再按窗口末端航向 ``ψ_e`` 旋转 ``Rz(-ψ_e)``；
  ``ψ_e`` 用 :func:`~inertial_benchmark.utils.geometry.heading_from_quat`（绕世界 z 轴的扭转分量）
  计算，在任意姿态下连续（机体 x 轴接近竖直时不会像 ZYX 偏航那样跳 180°）；
* ``body``：IMU 保持机体系，目标用窗口末端姿态旋到机体系（要求 ``dims=3``）。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping, Optional

import numpy as np

from ..utils.geometry import (
    GRAVITY,
    heading_from_quat,
    quat_conjugate,
    quat_from_yaw,
    quat_multiply,
    quat_normalize,
    quat_rotate,
    rotate_z,
)
from .format import Sequence

FRAMES = ("gravity_world", "body", "gravity_yaw_local")
ORIENTATIONS = ("reference", "device")
# 窗口级目标（每窗口一个）/ 逐帧目标（窗口内每帧一个）/ 多步位移（窗口等分为 output_steps 段）
WINDOW_TARGETS = ("avg_velocity", "displacement", "velocity_at_end")
FRAME_TARGETS = ("frame_velocity",)
STEP_TARGETS = ("multi_displacement",)
TARGETS = WINDOW_TARGETS + FRAME_TARGETS + STEP_TARGETS
LAYOUTS = ("window", "frame", "steps")
OVERLAPS = ("center", "mean")
CHANNELS = ("gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z")
# 额外输入（DESIGN 第 3 节）；``init_velocity`` 来自参考真值，属于**特权输入**
EXTRA_INPUTS = ("orientation", "gravity", "init_velocity")
PRIVILEGED_INPUTS = ("init_velocity",)


@dataclass(frozen=True)
class ViewConfig:
    """任务视图配置；字段与 ``default.yaml`` 同名。

    除窗口级目标外还支持**逐帧**目标（``frame_velocity``，输出布局 ``(T, dims)``）与**多步位移**
    （``multi_displacement``，把窗口等分为 ``output_steps`` 段，输出布局 ``(H, dims)``）；
    ``history``/``history_stride`` 给出历史子窗口个数与子窗口起点间隔（输入布局 ``(H, C, T)``）；
    ``extra_inputs`` 声明额外输入；``overlap`` 决定重叠预测的合并策略。
    """

    window: int = 200
    frame: str = "gravity_world"
    orientation: str = "reference"
    remove_gravity: bool = False
    target: str = "avg_velocity"
    dims: int = 2
    rate: float = 200.0
    output_steps: int = 0
    overlap: str = "mean"
    history: int = 0
    history_stride: int = 0
    extra_inputs: tuple = ()

    def __post_init__(self) -> None:
        if int(self.window) < 2:
            raise ValueError(f"window must be >= 2, got {self.window}")
        for name, value, allowed in (("frame", self.frame, FRAMES),
                                     ("orientation", self.orientation, ORIENTATIONS),
                                     ("target", self.target, TARGETS),
                                     ("overlap", self.overlap, OVERLAPS)):
            if value not in allowed:
                raise ValueError(f"{name}={value!r} not in {allowed}")
        if self.dims not in (2, 3):
            raise ValueError(f"dims must be 2 or 3, got {self.dims}")
        if self.frame == "body" and self.dims != 3:
            raise ValueError("frame=body expresses targets in the device frame and needs dims=3")
        if self.target in STEP_TARGETS and int(self.output_steps) < 1:
            raise ValueError(f"target={self.target} needs output_steps >= 1")
        if self.target in STEP_TARGETS and int(self.output_steps) > self.window - 1:
            raise ValueError(f"output_steps={self.output_steps} exceeds window-1={self.window - 1}")
        if int(self.history) < 0 or int(self.history_stride) < 0:
            raise ValueError("history and history_stride must be >= 0")
        if int(self.history) > 1 and int(self.history_stride) < 1:
            raise ValueError("history > 1 needs history_stride >= 1 (sub-window start spacing)")
        unknown = [name for name in self.extra_inputs if name not in EXTRA_INPUTS]
        if unknown:
            raise ValueError(f"unknown extra_inputs {unknown}; available: {list(EXTRA_INPUTS)}")

    @classmethod
    def from_cfg(cls, cfg: Any) -> "ViewConfig":
        """从 dict / 命名空间中挑出同名字段。"""
        kwargs = {}
        ints = ("window", "dims", "output_steps", "history", "history_stride")
        for f in fields(cls):
            value = cfg.get(f.name) if isinstance(cfg, Mapping) else getattr(cfg, f.name, None)
            if value is None:
                continue
            if f.name in ints:
                value = int(value)
            elif f.name == "extra_inputs":
                value = tuple(value)
            kwargs[f.name] = value
        return cls(**kwargs)

    def to_dict(self) -> dict:
        out = asdict(self)
        out["extra_inputs"] = list(self.extra_inputs)
        return out

    @property
    def dt(self) -> float:
        return 1.0 / self.rate

    @property
    def output_layout(self) -> str:
        """输出布局：``window`` ``(B,D)`` / ``frame`` ``(B,T,D)`` / ``steps`` ``(B,H,D)``。"""
        if self.target in FRAME_TARGETS:
            return "frame"
        return "steps" if self.target in STEP_TARGETS else "window"

    @property
    def num_outputs(self) -> int:
        """每个窗口的输出行数 ``R``（``window``=1，``frame``=T，``steps``=output_steps）。"""
        layout = self.output_layout
        if layout == "frame":
            return int(self.window)
        return int(self.output_steps) if layout == "steps" else 1

    @property
    def output_shape(self) -> tuple:
        """单个样本的输出形状（不含批维）：``(D,)`` / ``(T, D)`` / ``(H, D)``。"""
        return (self.dims,) if self.output_layout == "window" else (self.num_outputs, self.dims)

    @property
    def step_bounds(self) -> np.ndarray:
        """多步目标各段的 ``[lo, hi]`` 样本偏移（窗口内等分，边界四舍五入）。"""
        edges = np.round(np.linspace(0.0, self.window - 1, int(self.output_steps) + 1))
        return np.stack([edges[:-1], edges[1:]], axis=1).astype(np.int64)

    @property
    def output_offsets(self) -> np.ndarray:
        """各输出对应时刻相对窗口首样本的偏移（秒），长度 ``R``。

        窗口级目标取窗口中心（``velocity_at_end`` 取末端）；逐帧目标取该帧本身；
        多步位移取该段的中心。偏移一律是半个采样间隔的整数倍，重叠预测才能精确对齐合并。
        """
        layout = self.output_layout
        if layout == "frame":
            return np.arange(self.window) * self.dt
        if layout == "steps":
            lo, hi = self.step_bounds[:, 0], self.step_bounds[:, 1]
            return 0.5 * (lo + hi) * self.dt
        span = (self.window - 1) * self.dt
        return np.array([span if self.target == "velocity_at_end" else 0.5 * span])

    @property
    def output_scales(self) -> np.ndarray:
        """把各输出的目标数值换算为速度的比例（位移目标除以该输出的时间跨度）。"""
        if self.target == "displacement":
            return np.array([1.0 / ((self.window - 1) * self.dt)])
        if self.target in STEP_TARGETS:
            lo, hi = self.step_bounds[:, 0], self.step_bounds[:, 1]
            return 1.0 / ((hi - lo) * self.dt)
        return np.ones(self.num_outputs)

    @property
    def target_offset(self) -> float:
        """窗口级目标的时间偏移（秒）；多输出布局请用 :attr:`output_offsets`。"""
        return float(self.output_offsets[0] if self.output_layout == "window"
                     else 0.5 * (self.window - 1) * self.dt)

    @property
    def velocity_scale(self) -> float:
        """窗口级目标换算为速度的比例；多输出布局请用 :attr:`output_scales`。"""
        return float(self.output_scales[0]) if self.target != "multi_displacement" else 1.0

    @property
    def uses_orientation(self) -> bool:
        """视图构造输入时是否需要姿态（``body`` 且不去重力、且无姿态类额外输入时不需要）。"""
        if self.frame != "body" or self.remove_gravity:
            return True
        return bool(set(self.extra_inputs) & {"orientation", "gravity"})

    @property
    def sub_windows(self) -> int:
        """历史子窗口个数（``history=0`` 表示不使用历史，输入形状仍为 ``(C, T)``）。"""
        return int(self.history)

    @property
    def sub_window_stride(self) -> int:
        return int(self.history_stride) if self.history > 1 else 0

    @property
    def history_offset(self) -> int:
        """窗口起点之前还需要多少样本（子窗口向前延伸的长度）。"""
        return max(self.sub_windows - 1, 0) * self.sub_window_stride

    @property
    def input_span(self) -> int:
        """构造一个样本需要的连续样本数（历史 + 主窗口）。"""
        return self.history_offset + int(self.window)

    @property
    def privileged_inputs(self) -> tuple:
        """所用的特权输入（来自参考真值），必须在结果中显式标记。"""
        return tuple(name for name in self.extra_inputs if name in PRIVILEGED_INPUTS)


def yaw_offset(q_ref: np.ndarray, q_dev: np.ndarray) -> float:
    """``R_off = R_ref · R_dev^T`` 的航向分量：设备世界系 → 参考世界系。"""
    q_ref = np.asarray(q_ref, dtype=np.float64)
    q_dev = np.asarray(q_dev, dtype=np.float64)
    return float(heading_from_quat(quat_multiply(q_ref, quat_conjugate(q_dev))))


def first_valid_index(valid: np.ndarray) -> int:
    idx = np.flatnonzero(valid)
    return int(idx[0]) if len(idx) else 0


def device_orientation_valid(seq: Sequence) -> Optional[np.ndarray]:
    """设备姿态的有效掩码。

    格式里的 ``valid/device_orientation`` 是可选字段，字段不存在时返回 ``None``（视为全部有效）。
    """
    mask = getattr(seq, "valid_device_orientation", None)
    return None if mask is None else np.asarray(mask, bool)


def input_valid_mask(cfg: ViewConfig, valid_imu: np.ndarray, valid_pose: np.ndarray,
                     valid_device: Optional[np.ndarray] = None) -> np.ndarray:
    """**输入**有效的样本掩码：IMU 样本 + 视图构造输入所需的姿态。

    ``orientation=device`` 时只有设备姿态参与（设备姿态缺口仅在这种情况下影响窗口筛选）；
    ``orientation=reference`` 时输入旋转来自参考姿态，因此参考位姿缺口同样使输入无效；
    ``frame=body`` 且不去重力时完全不用姿态，输入只取决于 ``valid/imu``。
    """
    mask = np.asarray(valid_imu, bool)
    if not cfg.uses_orientation:
        return mask
    if cfg.orientation == "device":
        return mask if valid_device is None else (mask & valid_device)
    return mask & np.asarray(valid_pose, bool)


def device_yaw_offset(seq: Sequence) -> float:
    """设备世界系 → 参考世界系的常值偏航（首个 IMU/位姿/设备姿态均有效的样本处估计）。"""
    if seq.device_orientation is None:
        raise ValueError(f"{seq.sequence_id}: orientation=device requires imu/orientation")
    valid = seq.valid_imu & seq.valid_pose
    device_valid = device_orientation_valid(seq)
    if device_valid is not None:
        valid = valid & device_valid
    k = first_valid_index(valid)
    return yaw_offset(seq.orientation[k], seq.device_orientation[k])


def input_orientation(seq: Sequence, cfg: ViewConfig, yaw_offset: Optional[float] = None,
                      sl: slice = slice(None)) -> np.ndarray:
    """视图用于旋转的姿态（参考姿态，或偏航对齐后的设备姿态），float64。"""
    if cfg.orientation == "reference":
        return seq.orientation[sl].astype(np.float64)
    if yaw_offset is None:
        yaw_offset = device_yaw_offset(seq)
    q_dev = seq.device_orientation[sl].astype(np.float64)
    return quat_normalize(quat_multiply(quat_from_yaw(yaw_offset), q_dev))


def frame_imu(gyro: np.ndarray, acc: np.ndarray, q: np.ndarray, cfg: ViewConfig) -> np.ndarray:
    """把一段机体系 IMU 变到 ``gravity_world`` 或 ``body`` 系，返回 ``(T, 6)``。

    ``gravity_yaw_local`` 在这里返回世界系结果，逐窗口的偏航旋转由调用方完成。
    """
    gyro = np.asarray(gyro, dtype=np.float64)
    acc = np.asarray(acc, dtype=np.float64)
    if cfg.frame == "body":
        if cfg.remove_gravity:
            acc = acc - quat_rotate(quat_conjugate(q), np.array([0.0, 0.0, GRAVITY]))
        return np.concatenate([gyro, acc], axis=1)
    gw = quat_rotate(q, gyro)
    aw = quat_rotate(q, acc)
    if cfg.remove_gravity:
        aw[:, 2] -= GRAVITY
    return np.concatenate([gw, aw], axis=1)


def end_velocity(position: np.ndarray, velocity: Optional[np.ndarray], idx: np.ndarray,
                 rate: float) -> np.ndarray:
    """窗口末端速度：有 ``pose/velocity`` 时直接取，否则中心差分（序列末端退化为后向差分）。"""
    if velocity is not None:
        return np.asarray(velocity[idx], dtype=np.float64)
    n = len(position)
    lo = np.maximum(idx - 1, 0)
    hi = np.minimum(idx + 1, n - 1)
    return (position[hi] - position[lo]) * rate / np.maximum(hi - lo, 1)[:, None]


def world_to_frame(vec: np.ndarray, q_end: np.ndarray, yaw_end: np.ndarray,
                   cfg: ViewConfig) -> np.ndarray:
    """世界系 3D 向量 → 视图坐标系，截取 ``dims`` 维（支持任意前导维度）。"""
    if cfg.frame == "body":
        out = quat_rotate(quat_conjugate(q_end), vec)
    elif cfg.frame == "gravity_yaw_local":
        out = rotate_z(vec, -yaw_end)
    else:
        out = vec
    return out[..., : cfg.dims]


def frame_to_world(vec: np.ndarray, q_end: np.ndarray, yaw_end: np.ndarray,
                   cfg: ViewConfig) -> np.ndarray:
    """``world_to_frame`` 的逆变换；``body`` 返回 3D，其余返回 ``dims`` 维。"""
    vec = np.asarray(vec, dtype=np.float64)
    if cfg.frame == "body":
        return quat_rotate(q_end, vec)
    if cfg.frame == "gravity_yaw_local":
        return rotate_z(vec, yaw_end)
    return vec


def window_valid_mask(valid: np.ndarray, starts: np.ndarray, window: int) -> np.ndarray:
    """窗口 ``[s, s+T)`` 内全部样本有效。"""
    bad = np.concatenate([[0], np.cumsum(~np.asarray(valid, bool))])
    starts = np.asarray(starts, dtype=np.int64)
    return (bad[starts + window] - bad[starts]) == 0


class SequenceView:
    """一条序列上的向量化窗口视图（预先把整条序列旋到世界系/机体系）。

    有效性分两套掩码（DESIGN 第 3/5 节）：``valid_input``（模型输入可用）与
    ``valid_target``（参考位姿可用）。训练与窗口级指标要求两者同时成立；推理时预测只在输入无效处
    填补、目标只在位姿无效处填补，这样模型穿越位姿缺口的漂移才会被真实评测。
    """

    def __init__(self, seq: Sequence, cfg: ViewConfig,
                 yaw_offset: Optional[float] = None) -> None:
        if abs(seq.sample_rate - cfg.rate) > 1e-6:
            raise ValueError(f"{seq.sequence_id}: sample rate {seq.sample_rate} != {cfg.rate}")
        self.seq = seq
        self.cfg = cfg
        if yaw_offset is None and cfg.orientation == "device":
            yaw_offset = device_yaw_offset(seq)
        self.yaw_offset = yaw_offset
        self.q = input_orientation(seq, cfg, self.yaw_offset)
        self.imu = frame_imu(seq.gyroscope, seq.accelerometer, self.q, cfg).astype(np.float32)
        self.yaw = heading_from_quat(self.q) if cfg.frame == "gravity_yaw_local" else None
        self.valid_input = input_valid_mask(cfg, seq.valid_imu, seq.valid_pose,
                                            device_orientation_valid(seq))
        self.valid_target = np.asarray(seq.valid_pose, bool)
        self.valid = self.valid_input & self.valid_target

    def __len__(self) -> int:
        return len(self.seq)

    @property
    def sequence_id(self) -> str:
        return self.seq.sequence_id

    def starts(self, stride: int, require_valid: bool = True) -> np.ndarray:
        """``history_offset, +stride, …`` 中（可选）全部有效的窗口起点。

        起点从 ``history_offset`` 开始，保证历史子窗口不越过序列开头；有效性按整个输入跨度判断，
        因此子窗口在时间上连续、不跨越无效区。
        """
        cfg = self.cfg
        n_max = len(self.seq) - cfg.window
        if n_max < cfg.history_offset:
            return np.zeros(0, dtype=np.int64)
        starts = np.arange(cfg.history_offset, n_max + 1, int(stride), dtype=np.int64)
        return starts[self.window_valid(starts)] if require_valid else starts

    def _span_valid(self, mask: np.ndarray, starts: np.ndarray) -> np.ndarray:
        starts = np.asarray(starts, dtype=np.int64) - self.cfg.history_offset
        return window_valid_mask(mask, starts, self.cfg.input_span)

    def window_valid(self, starts: np.ndarray) -> np.ndarray:
        """整个输入跨度的输入与目标都有效（训练与窗口级指标用）。"""
        return self._span_valid(self.valid, starts)

    def window_valid_input(self, starts: np.ndarray) -> np.ndarray:
        """整个输入跨度（含历史子窗口）的模型输入有效（预测是否需要填补由它决定）。"""
        return self._span_valid(self.valid_input, starts)

    def window_valid_target(self, starts: np.ndarray) -> np.ndarray:
        """主窗口的参考位姿全部有效（目标是否需要填补由它决定）。"""
        return window_valid_mask(self.valid_target, starts, self.cfg.window)

    def target_mask(self, starts: np.ndarray) -> np.ndarray:
        """逐输出的目标有效掩码 ``(K, R)``：损失按它跳过无效的帧/步。"""
        starts = np.asarray(starts, dtype=np.int64)
        layout = self.cfg.output_layout
        if layout == "frame":
            idx = starts[:, None] + np.arange(self.cfg.window)
            return self.valid_target[idx]
        if layout == "steps":
            lo, hi = self.cfg.step_bounds[:, 0], self.cfg.step_bounds[:, 1]
            out = np.empty((len(starts), len(lo)), bool)
            for r, (a, b) in enumerate(zip(lo, hi)):
                out[:, r] = window_valid_mask(self.valid_target, starts + a, int(b - a) + 1)
            return out
        return self.window_valid_target(starts)[:, None]

    def _end_state(self, starts: np.ndarray):
        end = np.asarray(starts, dtype=np.int64) + self.cfg.window - 1
        yaw = self.yaw[end] if self.yaw is not None else np.zeros(len(end))
        return end, self.q[end], yaw

    def output_indices(self, starts: np.ndarray) -> np.ndarray:
        """各输出的参考样本索引 ``(K, R)``（``body`` 坐标系按该样本的姿态旋转）。"""
        starts = np.asarray(starts, dtype=np.int64)
        layout = self.cfg.output_layout
        if layout == "frame":
            return starts[:, None] + np.arange(self.cfg.window)
        if layout == "steps":
            lo, hi = self.cfg.step_bounds[:, 0], self.cfg.step_bounds[:, 1]
            return starts[:, None] + np.round(0.5 * (lo + hi)).astype(np.int64)
        return (starts + self.cfg.window - 1)[:, None]

    def _rotation_state(self, starts: np.ndarray):
        """逐输出的旋转参考：``body`` 用该输出自己的姿态，其余用窗口末端航向。"""
        idx = self.output_indices(starts)
        if self.cfg.frame == "body":
            return self.q[idx], np.zeros(idx.shape)
        _, _, yaw = self._end_state(starts)
        return self.q[idx], np.repeat(yaw[:, None], idx.shape[1], axis=1)

    def _rotate_windows(self, x: np.ndarray, yaw: np.ndarray) -> np.ndarray:
        """按窗口末端航向把 ``(..., T, 6)`` 的输入旋到 ``gravity_yaw_local``。"""
        return np.concatenate([rotate_z(x[..., :3], -yaw), rotate_z(x[..., 3:], -yaw)], axis=-1)

    def _span_index(self, starts: np.ndarray) -> np.ndarray:
        """输入索引：无历史时 ``(K, T)``，有历史时 ``(K, H, T)``（子窗口在时间上连续）。"""
        cfg = self.cfg
        starts = np.asarray(starts, dtype=np.int64)
        frames = np.arange(cfg.window)
        if cfg.sub_windows <= 0:
            return starts[:, None] + frames
        offsets = (np.arange(cfg.sub_windows) - (cfg.sub_windows - 1)) * cfg.sub_window_stride
        return starts[:, None, None] + offsets[None, :, None] + frames[None, None, :]

    def imu_windows(self, starts: np.ndarray) -> np.ndarray:
        """输入窗口：``(K, 6, T)``，声明 ``history`` 时为 ``(K, H, 6, T)``（float32）。"""
        starts = np.asarray(starts, dtype=np.int64)
        x = self.imu[self._span_index(starts)]  # (K[, H], T, 6)
        if self.yaw is not None:
            _, _, yaw = self._end_state(starts)
            x = self._rotate_windows(x, yaw.reshape((-1,) + (1,) * (x.ndim - 2)))
        return np.ascontiguousarray(np.moveaxis(x, -1, -2), dtype=np.float32)

    def extra_inputs(self, starts: np.ndarray) -> dict:
        """额外输入（DESIGN 第 3 节）：``orientation``、``gravity``、``init_velocity``。"""
        cfg = self.cfg
        if not cfg.extra_inputs:
            return {}
        starts = np.asarray(starts, dtype=np.int64)
        idx = self._span_index(starts)
        out: dict = {}
        if "orientation" in cfg.extra_inputs:
            q = self.q[idx]
            if self.yaw is not None:
                _, _, yaw = self._end_state(starts)
                shape = (-1,) + (1,) * (q.ndim - 2)
                q = quat_multiply(quat_from_yaw(-yaw.reshape(shape)), q)
            out["orientation"] = q.astype(np.float32)
        if "gravity" in cfg.extra_inputs:
            world = np.array([0.0, 0.0, GRAVITY])
            if cfg.frame == "body":
                gravity = quat_rotate(quat_conjugate(self.q[idx]), world)
            else:  # 世界系（含局部偏航系）下重力方向恒定
                gravity = np.broadcast_to(world, idx.shape + (3,))
            out["gravity"] = np.ascontiguousarray(gravity, dtype=np.float32)
        if "init_velocity" in cfg.extra_inputs:
            first = starts - cfg.history_offset
            world = end_velocity(self.seq.position, self.seq.velocity, first, cfg.rate)
            _, q_end, yaw = self._end_state(starts)
            out["init_velocity"] = world_to_frame(world, q_end, yaw, cfg).astype(np.float32)
        return out

    def targets_world(self, starts: np.ndarray) -> np.ndarray:
        """世界系 3D 目标：``window`` 布局 ``(K, 3)``，其余 ``(K, R, 3)``。"""
        return self._squeeze(self._targets_world_rows(starts))

    def _targets_world_rows(self, starts: np.ndarray) -> np.ndarray:
        """世界系 3D 目标 ``(K, R, 3)``（位移目标已是位移，速度目标为 m/s）。"""
        starts = np.asarray(starts, dtype=np.int64)
        cfg = self.cfg
        pos, vel = self.seq.position, self.seq.velocity
        layout = cfg.output_layout
        if layout == "frame":
            idx = starts[:, None] + np.arange(cfg.window)
            flat = end_velocity(pos, vel, idx.reshape(-1), cfg.rate)
            return flat.reshape(idx.shape + (3,))
        if layout == "steps":
            lo, hi = cfg.step_bounds[:, 0], cfg.step_bounds[:, 1]
            return pos[starts[:, None] + hi] - pos[starts[:, None] + lo]
        end = starts + cfg.window - 1
        if cfg.target == "velocity_at_end":
            return end_velocity(pos, vel, end, cfg.rate)[:, None, :]
        disp = (pos[end] - pos[starts])[:, None, :]
        return disp if cfg.target == "displacement" else disp / ((cfg.window - 1) * cfg.dt)

    def targets(self, starts: np.ndarray) -> np.ndarray:
        """视图坐标系中的目标：``window`` 布局 ``(K, dims)``，其余 ``(K, R, dims)``，float32。"""
        q_ref, yaw = self._rotation_state(starts)
        out = world_to_frame(self._targets_world_rows(starts), q_ref, yaw, self.cfg)
        return np.ascontiguousarray(self._squeeze(out), dtype=np.float32)

    def _squeeze(self, values: np.ndarray) -> np.ndarray:
        return values[:, 0] if self.cfg.output_layout == "window" else values

    def to_world_velocity(self, values: np.ndarray, starts: np.ndarray) -> np.ndarray:
        """把视图坐标系中的模型输出换算为世界系速度（``body`` 为 3D，其余为 ``dims`` 维）。

        接受 ``(K, dims)``（窗口级）与 ``(K, R, dims)``（逐帧/多步）两种形状，返回同样的布局。
        """
        values = np.asarray(values, dtype=np.float64)
        flat = values.ndim == 2 and self.cfg.output_layout == "window"
        vec = values[:, None, :] if flat else values
        q_ref, yaw = self._rotation_state(starts)
        scales = self.cfg.output_scales[: vec.shape[1]][None, :, None]
        out = frame_to_world(vec * scales, q_ref, yaw, self.cfg)
        return out[:, 0] if flat else out

    def output_times(self, starts: np.ndarray) -> np.ndarray:
        """各输出的时间戳 ``(K, R)``。"""
        t0 = self.seq.timestamp[np.asarray(starts, dtype=np.int64)]
        return t0[:, None] + self.cfg.output_offsets[None, :]

    def target_times(self, starts: np.ndarray) -> np.ndarray:
        """窗口级目标的时间戳 ``(K,)``；多输出布局请用 :meth:`output_times`。"""
        times = self.output_times(starts)
        return times[:, 0] if self.cfg.output_layout == "window" else times

    def body_to_frame(self, start: int, vec_body: np.ndarray) -> np.ndarray:
        """把机体系常向量（例如零偏）旋到输入跨度各样本的视图坐标系，返回 ``(span, 3)``。"""
        first = int(start) - self.cfg.history_offset
        sl = slice(first, first + self.cfg.input_span)
        if self.cfg.frame == "body":
            return np.broadcast_to(vec_body, (self.cfg.input_span, 3)).astype(np.float64)
        out = quat_rotate(self.q[sl], vec_body)
        if self.yaw is not None:
            out = rotate_z(out, -self.yaw[int(start) + self.cfg.window - 1])
        return out

    def windows(self, starts: np.ndarray) -> dict:
        out = {"imu": self.imu_windows(starts), "target": self.targets(starts),
               "mask": self.target_mask(starts)}
        extra = self.extra_inputs(starts)
        if extra:
            out["extra"] = extra
        return out


def read_window(handle: Any, start: int, cfg: ViewConfig, yaw_offset: Optional[float],
                has_velocity: bool) -> dict:
    """惰性模式：从打开的 h5 文件读取一个窗口（含历史子窗口、逐帧/多步目标与额外输入）。

    做法是把该样本需要的区间读成一条小 :class:`~inertial_benchmark.data.format.Sequence`，
    再用同一个 :class:`SequenceView` 做变换，因此与缓存模式逐位一致（末端/逐帧速度的中心差分会
    多读一个样本；区间已到序列末尾时与缓存模式同样退化为后向差分）。

    返回 ``{"imu", "target", "mask", "extra", "body_to_frame"}``。
    """
    n = int(handle["timestamp"].shape[0])
    lo = int(start) - cfg.history_offset
    hi = min(int(start) + cfg.window + 1, n)
    sl = slice(lo, hi)
    sub = Sequence(
        timestamp=np.asarray(handle["timestamp"][sl], dtype=np.float64),
        gyroscope=handle["imu/gyroscope"][sl],
        accelerometer=handle["imu/accelerometer"][sl],
        orientation=handle["pose/orientation"][sl],
        position=handle["pose/position"][sl],
        valid_imu=handle["valid/imu"][sl],
        valid_pose=handle["valid/pose"][sl],
        device_orientation=(handle["imu/orientation"][sl]
                            if "imu/orientation" in handle else None),
        velocity=handle["pose/velocity"][sl] if has_velocity else None,
        attrs={"sample_rate_hz": float(cfg.rate), "sequence_id": "lazy"},
    )
    view = SequenceView(sub, cfg, yaw_offset=yaw_offset)
    local = int(start) - lo
    out = view.windows(np.array([local], dtype=np.int64))
    return {
        "imu": out["imu"][0],
        "target": out["target"][0],
        "mask": out["mask"][0],
        "extra": {k: v[0] for k, v in out.get("extra", {}).items()},
        "body_to_frame": lambda v, w=view, s=local: w.body_to_frame(s, v),
    }
