"""合成数据生成器：解析轨迹 + 由轨迹反推的机体系 IMU（比力含重力）。

位置为若干低频正弦之和（行走尺度的平滑随机游走）叠加 1.8 Hz 竖直起伏；姿态为
``Rz(yaw) · Ry(pitch) · Rx(roll)``，各角为解析函数。角速度由四元数导数
``ω_body = 2 · Im(q* ⊗ q̇)`` 数值求得（中心差分，误差 ~1e-9），比力
``f_body = R(q)^T (a_world + [0, 0, g])``。
"""

from __future__ import annotations

from typing import Optional
from typing import Sequence as Seq

import numpy as np

from inertial_benchmark.data.converters.base import RawSequence
from inertial_benchmark.data.format import Sequence
from inertial_benchmark.utils.geometry import (
    GRAVITY,
    quat_conjugate,
    quat_from_euler_zyx,
    quat_from_yaw,
    quat_multiply,
    quat_rotate,
)


class Motion:
    """可解析求导的合成运动。"""

    def __init__(self, seed: int = 0, n_terms: int = 3) -> None:
        rng = np.random.default_rng(seed)
        self.amp = rng.uniform(1.5, 4.0, size=(2, n_terms))
        self.freq = rng.uniform(0.1, 0.25, size=(2, n_terms))
        self.phase = rng.uniform(0, 2 * np.pi, size=(2, n_terms))
        self.yaw0 = rng.uniform(-np.pi, np.pi)
        self.yaw_amp = rng.uniform(0.3, 1.0, size=n_terms)
        self.yaw_freq = rng.uniform(0.1, 0.5, size=n_terms)
        self.yaw_phase = rng.uniform(0, 2 * np.pi, size=n_terms)
        self.origin = rng.uniform(-5, 5, size=3)

    def _xy(self, t: np.ndarray, order: int) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)[:, None, None]
        arg = self.freq * t + self.phase
        w = self.freq ** order
        if order == 0:
            val = self.amp * np.sin(arg)
        elif order == 1:
            val = self.amp * w * np.cos(arg)
        else:
            val = -self.amp * w * np.sin(arg)
        return val.sum(axis=-1)

    def position(self, t: np.ndarray) -> np.ndarray:
        xy = self._xy(t, 0)
        z = 0.04 * np.sin(2 * np.pi * 1.8 * np.asarray(t))
        return np.column_stack([xy, z]) + self.origin

    def velocity(self, t: np.ndarray) -> np.ndarray:
        w = 2 * np.pi * 1.8
        return np.column_stack([self._xy(t, 1), 0.04 * w * np.cos(w * np.asarray(t))])

    def acceleration(self, t: np.ndarray) -> np.ndarray:
        w = 2 * np.pi * 1.8
        return np.column_stack([self._xy(t, 2), -0.04 * w * w * np.sin(w * np.asarray(t))])

    def orientation(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        yaw = self.yaw0 + (self.yaw_amp * np.sin(self.yaw_freq * t[:, None]
                                                  + self.yaw_phase)).sum(axis=1)
        pitch = 0.4 + 0.1 * np.sin(2 * np.pi * 0.9 * t)
        roll = 0.15 * np.sin(2 * np.pi * 0.5 * t + 1.0)
        return quat_from_euler_zyx(yaw, pitch, roll)

    def gyroscope(self, t: np.ndarray, h: float = 1e-5) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        q = self.orientation(t)
        dq = (self.orientation(t + h) - self.orientation(t - h)) / (2 * h)
        return 2.0 * quat_multiply(quat_conjugate(q), dq)[:, 1:]

    def accelerometer(self, t: np.ndarray) -> np.ndarray:
        q = self.orientation(t)
        a = self.acceleration(t) + np.array([0.0, 0.0, GRAVITY])
        return quat_rotate(quat_conjugate(q), a)


def raw_attrs(sequence_id: str, group_id: str = "g0", **extra) -> dict:
    attrs = {
        "subject_id": group_id,
        "device_id": "synthetic_phone",
        "placement": "handheld",
        "group_id": group_id,
        "position_source": "analytic",
        "orientation_source": "analytic",
        "device_orientation_source": "none",
        "body_frame": "synthetic_device",
        "source_files": [f"{sequence_id}.npz"],
    }
    attrs.update(extra)
    return attrs


def v1_attrs(sequence_id: str, dataset: str = "synthetic", group_id: str = "g0",
             rate: float = 200.0) -> dict:
    attrs = raw_attrs(sequence_id, group_id)
    attrs.update(
        schema_version="1.0",
        dataset=dataset,
        sequence_id=sequence_id,
        sample_rate_hz=rate,
        source_sample_rate_hz=rate,
        resampling="none (synthetic)",
        world_frame="gravity_aligned_z_up",
        timestamp_type="relative",
        start_time_unix=float("nan"),
        orientation_convention="body_to_world_wxyz",
        accelerometer_type="specific_force",
        source_license="generated",
        converter="synthetic@1",
    )
    return attrs


def make_sequence(
    duration: float = 20.0,
    rate: float = 200.0,
    seed: int = 0,
    sequence_id: Optional[str] = None,
    dataset: str = "synthetic",
    group_id: str = "g0",
    noise: float = 0.0,
    device_yaw_offset: Optional[float] = None,
) -> Sequence:
    """直接生成一条合规的 v1 序列（200 Hz）。"""
    motion = Motion(seed)
    n = int(round(duration * rate)) + 1
    t = np.arange(n) / rate
    rng = np.random.default_rng(seed + 1000)
    gyro = motion.gyroscope(t)
    acc = motion.accelerometer(t)
    if noise:
        gyro = gyro + rng.normal(0, 0.1 * noise, gyro.shape)
        acc = acc + rng.normal(0, noise, acc.shape)
    q = motion.orientation(t)
    device = None
    sid = sequence_id or f"seq{seed:03d}"
    attrs = v1_attrs(sid, dataset, group_id, rate)
    if device_yaw_offset is not None:
        device = quat_multiply(quat_from_yaw(np.full(n, device_yaw_offset)), q)
        attrs["device_orientation_source"] = "synthetic_game_rv"
    return Sequence(
        timestamp=t,
        gyroscope=gyro,
        accelerometer=acc,
        orientation=q,
        position=motion.position(t),
        valid_imu=np.ones(n, bool),
        valid_pose=np.ones(n, bool),
        device_orientation=device,
        velocity=motion.velocity(t),
        attrs=attrs,
    )


def make_raw_sequence(
    sequence_id: str = "raw000",
    duration: float = 20.0,
    imu_rate: float = 100.0,
    pose_rate: Optional[float] = None,
    seed: int = 0,
    group_id: str = "g0",
    jitter: float = 0.0,
    t0: float = 1000.0,
    imu_gaps: Seq[tuple] = (),
    pose_gaps: Seq[tuple] = (),
    duplicates: int = 0,
    with_device: bool = False,
) -> RawSequence:
    """生成原生时钟上的原始序列（可带时间抖动、缺口与重复时间戳）。"""
    motion = Motion(seed)
    rng = np.random.default_rng(seed + 7)
    pose_rate = pose_rate or imu_rate

    def clock(rate: float, gaps: Seq[tuple], start: float) -> np.ndarray:
        t = start + np.arange(int(duration * rate) + 1) / rate
        if jitter:
            t = t + rng.uniform(-jitter, jitter, len(t)) / rate
        keep = np.ones(len(t), bool)
        for a, b in gaps:  # 缺口以位姿起点 t0 为时间原点
            keep &= ~((t - t0 >= a) & (t - t0 < b))
        return t[keep]

    ti = clock(imu_rate, imu_gaps, t0 - 0.3)  # IMU 比位姿略早开始，考验重叠区计算
    tp = clock(pose_rate, pose_gaps, t0)
    if duplicates:
        idx = rng.choice(np.arange(1, len(ti) - 1), size=duplicates, replace=False)
        ti = np.sort(np.concatenate([ti, ti[idx]]))
    q = motion.orientation(tp - t0)
    device = None
    if with_device:
        device = quat_multiply(quat_from_yaw(np.full(len(ti), 0.7)), motion.orientation(ti - t0))
    return RawSequence(
        sequence_id=sequence_id,
        imu_time=ti,
        gyroscope=motion.gyroscope(ti - t0),
        accelerometer=motion.accelerometer(ti - t0),
        pose_time=tp,
        position=motion.position(tp - t0),
        orientation=q,
        velocity=motion.velocity(tp - t0),
        device_orientation=device,
        attrs=raw_attrs(sequence_id, group_id,
                        device_orientation_source="synthetic_game_rv" if with_device else "none"),
        notes=["synthetic: analytic trajectory"],
    )
