"""任务视图（DESIGN 第 3 节）：从 ``Sequence`` 生成模型输入窗口与目标（不依赖 torch）。

输入通道顺序固定为 ``[gyro_xyz, acc_xyz]``，窗口形状 ``(6, T)``；目标与输入处于同一坐标系：

* ``gravity_world``：用姿态把 IMU 旋到重力对齐世界系，目标为世界系向量；
* ``gravity_yaw_local``：在上面的基础上再按窗口末端偏航 ``ψ_e`` 旋转 ``Rz(-ψ_e)``；
* ``body``：IMU 保持机体系，目标用窗口末端姿态旋到机体系（要求 ``dims=3``）。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping, Optional

import numpy as np

from ..utils.geometry import (
    GRAVITY,
    quat_conjugate,
    quat_from_yaw,
    quat_multiply,
    quat_normalize,
    quat_rotate,
    rotate_z,
    yaw_from_quat,
)
from .format import Sequence

FRAMES = ("gravity_world", "body", "gravity_yaw_local")
ORIENTATIONS = ("reference", "device")
TARGETS = ("avg_velocity", "displacement", "velocity_at_end")
CHANNELS = ("gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z")


@dataclass(frozen=True)
class ViewConfig:
    """任务视图配置；字段与 ``default.yaml`` 同名。"""

    window: int = 200
    frame: str = "gravity_world"
    orientation: str = "reference"
    remove_gravity: bool = False
    target: str = "avg_velocity"
    dims: int = 2
    rate: float = 200.0

    def __post_init__(self) -> None:
        if int(self.window) < 2:
            raise ValueError(f"window must be >= 2, got {self.window}")
        for name, value, allowed in (("frame", self.frame, FRAMES),
                                     ("orientation", self.orientation, ORIENTATIONS),
                                     ("target", self.target, TARGETS)):
            if value not in allowed:
                raise ValueError(f"{name}={value!r} not in {allowed}")
        if self.dims not in (2, 3):
            raise ValueError(f"dims must be 2 or 3, got {self.dims}")
        if self.frame == "body" and self.dims != 3:
            raise ValueError("frame=body expresses targets in the device frame and needs dims=3")

    @classmethod
    def from_cfg(cls, cfg: Any) -> "ViewConfig":
        """从 dict / 命名空间中挑出同名字段。"""
        kwargs = {}
        for f in fields(cls):
            value = cfg.get(f.name) if isinstance(cfg, Mapping) else getattr(cfg, f.name, None)
            if value is not None:
                kwargs[f.name] = int(value) if f.name in ("window", "dims") else value
        return cls(**kwargs)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def dt(self) -> float:
        return 1.0 / self.rate

    @property
    def target_offset(self) -> float:
        """目标对应时刻相对窗口首样本的偏移（秒）：中心或末端。"""
        span = (self.window - 1) * self.dt
        return span if self.target == "velocity_at_end" else 0.5 * span

    @property
    def velocity_scale(self) -> float:
        """把目标数值换算为速度的比例（位移目标除以窗口跨度）。"""
        return 1.0 / ((self.window - 1) * self.dt) if self.target == "displacement" else 1.0


def yaw_offset(q_ref: np.ndarray, q_dev: np.ndarray) -> float:
    """``R_off = R_ref · R_dev^T`` 的偏航分量：设备世界系 → 参考世界系。"""
    q_ref = np.asarray(q_ref, dtype=np.float64)
    q_dev = np.asarray(q_dev, dtype=np.float64)
    return float(yaw_from_quat(quat_multiply(q_ref, quat_conjugate(q_dev))))


def first_valid_index(valid: np.ndarray) -> int:
    idx = np.flatnonzero(valid)
    return int(idx[0]) if len(idx) else 0


def device_yaw_offset(seq: Sequence) -> float:
    """设备世界系 → 参考世界系的常值偏航（首个 IMU/位姿/设备姿态均有效的样本处估计）。"""
    if seq.device_orientation is None:
        raise ValueError(f"{seq.sequence_id}: orientation=device requires imu/orientation")
    k = first_valid_index(seq.valid_device)
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
    """世界系 3D 向量 → 视图坐标系，截取 ``dims`` 维。"""
    if cfg.frame == "body":
        out = quat_rotate(quat_conjugate(q_end), vec)
    elif cfg.frame == "gravity_yaw_local":
        out = rotate_z(vec, -yaw_end)
    else:
        out = vec
    return out[:, : cfg.dims]


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
    """一条序列上的向量化窗口视图（预先把整条序列旋到世界系/机体系）。"""

    def __init__(self, seq: Sequence, cfg: ViewConfig) -> None:
        if abs(seq.sample_rate - cfg.rate) > 1e-6:
            raise ValueError(f"{seq.sequence_id}: sample rate {seq.sample_rate} != {cfg.rate}")
        self.seq = seq
        self.cfg = cfg
        self.yaw_offset = device_yaw_offset(seq) if cfg.orientation == "device" else None
        self.q = input_orientation(seq, cfg, self.yaw_offset)
        self.imu = frame_imu(seq.gyroscope, seq.accelerometer, self.q, cfg).astype(np.float32)
        self.yaw = yaw_from_quat(self.q) if cfg.frame == "gravity_yaw_local" else None
        # orientation=device 时设备姿态也必须有效（valid/device_orientation）
        self.valid = seq.valid_device if cfg.orientation == "device" else seq.valid

    def __len__(self) -> int:
        return len(self.seq)

    @property
    def sequence_id(self) -> str:
        return self.seq.sequence_id

    def starts(self, stride: int, require_valid: bool = True) -> np.ndarray:
        """``0, stride, 2·stride, …`` 中（可选）全部有效的窗口起点。"""
        n_max = len(self.seq) - self.cfg.window
        if n_max < 0:
            return np.zeros(0, dtype=np.int64)
        starts = np.arange(0, n_max + 1, int(stride), dtype=np.int64)
        return starts[self.window_valid(starts)] if require_valid else starts

    def window_valid(self, starts: np.ndarray) -> np.ndarray:
        return window_valid_mask(self.valid, starts, self.cfg.window)

    def _end_state(self, starts: np.ndarray):
        end = np.asarray(starts, dtype=np.int64) + self.cfg.window - 1
        yaw = self.yaw[end] if self.yaw is not None else np.zeros(len(end))
        return end, self.q[end], yaw

    def imu_windows(self, starts: np.ndarray) -> np.ndarray:
        """``(K, 6, T)`` float32 输入窗口。"""
        starts = np.asarray(starts, dtype=np.int64)
        idx = starts[:, None] + np.arange(self.cfg.window)
        x = self.imu[idx]  # (K, T, 6)
        if self.yaw is not None:
            _, _, yaw = self._end_state(starts)
            x = np.concatenate([rotate_z(x[..., :3], -yaw[:, None]),
                                rotate_z(x[..., 3:], -yaw[:, None])], axis=-1)
        return np.ascontiguousarray(x.transpose(0, 2, 1), dtype=np.float32)

    def targets_world(self, starts: np.ndarray) -> np.ndarray:
        """世界系 3D 目标（位移目标已是位移，速度目标为 m/s）。"""
        starts = np.asarray(starts, dtype=np.int64)
        end = starts + self.cfg.window - 1
        pos = self.seq.position
        if self.cfg.target == "velocity_at_end":
            return end_velocity(pos, self.seq.velocity, end, self.cfg.rate)
        disp = pos[end] - pos[starts]
        if self.cfg.target == "displacement":
            return disp
        return disp / ((self.cfg.window - 1) * self.cfg.dt)

    def targets(self, starts: np.ndarray) -> np.ndarray:
        """视图坐标系中的目标 ``(K, dims)`` float32。"""
        end, q_end, yaw = self._end_state(starts)
        out = world_to_frame(self.targets_world(starts), q_end, yaw, self.cfg)
        return out.astype(np.float32)

    def to_world_velocity(self, values: np.ndarray, starts: np.ndarray) -> np.ndarray:
        """把视图坐标系中的模型输出换算为世界系速度（``body`` 为 3D，其余为 ``dims`` 维）。"""
        _, q_end, yaw = self._end_state(starts)
        vec = np.asarray(values, dtype=np.float64) * self.cfg.velocity_scale
        return frame_to_world(vec, q_end, yaw, self.cfg)

    def target_times(self, starts: np.ndarray) -> np.ndarray:
        return self.seq.timestamp[np.asarray(starts, dtype=np.int64)] + self.cfg.target_offset

    def body_to_frame(self, start: int, vec_body: np.ndarray) -> np.ndarray:
        """把机体系常向量（例如零偏）旋到窗口各样本的视图坐标系，返回 ``(T, 3)``。"""
        sl = slice(int(start), int(start) + self.cfg.window)
        if self.cfg.frame == "body":
            return np.broadcast_to(vec_body, (self.cfg.window, 3)).astype(np.float64)
        out = quat_rotate(self.q[sl], vec_body)
        if self.yaw is not None:
            out = rotate_z(out, -self.yaw[int(start) + self.cfg.window - 1])
        return out

    def windows(self, starts: np.ndarray) -> dict:
        return {"imu": self.imu_windows(starts), "target": self.targets(starts)}


def read_window(handle: Any, start: int, cfg: ViewConfig, yaw_offset: Optional[float],
                has_velocity: bool) -> dict:
    """惰性模式：从打开的 h5 文件读取一个窗口并做与 ``SequenceView`` 完全相同的变换。

    返回 ``{"imu": (6,T), "target": (dims,), "q": (T,4), "yaw_end": float}``。
    """
    t = cfg.window
    s, e = int(start), int(start) + t - 1
    n = handle["timestamp"].shape[0]
    gyro = handle["imu/gyroscope"][s:s + t]
    acc = handle["imu/accelerometer"][s:s + t]
    if cfg.orientation == "reference":
        q = handle["pose/orientation"][s:s + t].astype(np.float64)
    else:
        q_dev = handle["imu/orientation"][s:s + t].astype(np.float64)
        q = quat_normalize(quat_multiply(quat_from_yaw(yaw_offset), q_dev))
    imu = frame_imu(gyro, acc, q, cfg).astype(np.float32)
    yaw_end = float(yaw_from_quat(q[-1])) if cfg.frame == "gravity_yaw_local" else 0.0
    if cfg.frame == "gravity_yaw_local":
        imu = np.concatenate([rotate_z(imu[:, :3], -yaw_end), rotate_z(imu[:, 3:], -yaw_end)], 1)

    if cfg.target == "velocity_at_end":
        lo, hi = max(e - 1, 0), min(e + 1, n - 1)
        if has_velocity:
            world = handle["pose/velocity"][e].astype(np.float64)[None]
        else:
            pos = handle["pose/position"][lo:hi + 1]
            world = ((pos[-1] - pos[0]) * cfg.rate / max(hi - lo, 1))[None]
    else:
        p0 = handle["pose/position"][s]
        p1 = handle["pose/position"][e]
        world = (p1 - p0)[None]
        if cfg.target == "avg_velocity":
            world = world / ((t - 1) * cfg.dt)
    target = world_to_frame(world, q[-1:], np.array([yaw_end]), cfg)[0]
    return {"imu": np.ascontiguousarray(imu.T, dtype=np.float32),
            "target": target.astype(np.float32), "q": q, "yaw_end": yaw_end}
