"""训练增强（不依赖 torch）。

配置写法（``augment`` 列表）::

    augment: [random_yaw, time_shift]
    augment: [{name: bias, gyro: 0.005, acc: 0.05}, {name: noise, gyro: 0.002, acc: 0.02}]
    augment: [{bias_shift: {gyro: 0.05, acc: 0.2}}, {gravity_perturb: {max_deg: 5}},
              random_yaw, {time_shift: {min_shift: 0, max_shift: 9}}]      # TLIO 官方管线

执行顺序固定：传感器级（``bias``/``noise``/``bias_shift``）先于坐标级（``gravity_perturb``、
``random_yaw``），同一级内保持列表顺序；``time_shift`` 由数据集在取窗口前处理。
每个增强接收 ``numpy.random.Generator``，保证可复现。
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable, Mapping, Optional

import numpy as np

from ..utils import LOGGER
from ..utils.geometry import rotate_z


class Augmentation:
    """增强基类：``sample`` 含 ``imu (6,T)``、``target (D,)`` 与 ``body_to_frame`` 回调。"""

    name = "base"
    stage = 0  # 0 = 传感器级，1 = 坐标级

    def __call__(self, sample: dict, rng: np.random.Generator) -> dict:
        raise NotImplementedError

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v}" for k, v in vars(self).items())
        return f"{type(self).__name__}({args})"


class RandomYaw(Augmentation):
    """在重力对齐系下绕 z 轴同步旋转 IMU（陀螺与加计）与目标。"""

    name = "random_yaw"
    stage = 1

    def __init__(self, max_angle: float = math.pi) -> None:
        self.max_angle = float(max_angle)

    def __call__(self, sample: dict, rng: np.random.Generator) -> dict:
        angle = rng.uniform(-self.max_angle, self.max_angle)
        imu = sample["imu"]
        imu[0:3] = rotate_z(imu[0:3].T, angle).T
        imu[3:6] = rotate_z(imu[3:6].T, angle).T
        target = sample["target"]
        target[:] = rotate_z(target[None], angle)[0]
        return sample


class Bias(Augmentation):
    """每个窗口注入机体系常值零偏 ``b ~ N(0, σ²)``，并随姿态旋到视图坐标系。"""

    name = "bias"

    def __init__(self, gyro: float = 0.0, acc: float = 0.0) -> None:
        self.gyro = float(gyro)
        self.acc = float(acc)

    def __call__(self, sample: dict, rng: np.random.Generator) -> dict:
        to_frame: Callable = sample["body_to_frame"]
        imu = sample["imu"]
        if self.gyro > 0:
            imu[0:3] += to_frame(rng.normal(0.0, self.gyro, 3)).T.astype(imu.dtype)
        if self.acc > 0:
            imu[3:6] += to_frame(rng.normal(0.0, self.acc, 3)).T.astype(imu.dtype)
        return sample


class Noise(Augmentation):
    """逐样本各向同性高斯白噪声（旋转不变，可直接加在任意坐标系）。"""

    name = "noise"

    def __init__(self, gyro: float = 0.0, acc: float = 0.0) -> None:
        self.gyro = float(gyro)
        self.acc = float(acc)

    def __call__(self, sample: dict, rng: np.random.Generator) -> dict:
        imu = sample["imu"]
        t = imu.shape[1]
        if self.gyro > 0:
            imu[0:3] += rng.normal(0.0, self.gyro, (3, t)).astype(imu.dtype)
        if self.acc > 0:
            imu[3:6] += rng.normal(0.0, self.acc, (3, t)).astype(imu.dtype)
        return sample


class BiasShift(Augmentation):
    """每个窗口在**视图坐标系**中加常值偏置 ``b ~ U[-r, r]``（逐轴独立）。

    对应 TLIO 官方管线：IMU 先旋到重力对齐系，再加陀螺 U[−0.05, 0.05] rad/s、
    加计 U[−0.2, 0.2] m/s² 的常值偏置（不是机体系偏置，机体系偏置见 :class:`Bias`）。
    """

    name = "bias_shift"

    def __init__(self, gyro: float = 0.0, acc: float = 0.0) -> None:
        self.gyro = float(gyro)
        self.acc = float(acc)

    def __call__(self, sample: dict, rng: np.random.Generator) -> dict:
        imu = sample["imu"]
        if self.gyro > 0:
            imu[0:3] += rng.uniform(-self.gyro, self.gyro, (3, 1)).astype(imu.dtype)
        if self.acc > 0:
            imu[3:6] += rng.uniform(-self.acc, self.acc, (3, 1)).astype(imu.dtype)
        return sample


def axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues 公式：绕单位轴 ``axis`` 旋转 ``angle`` 的 3×3 矩阵。"""
    x, y, z = axis
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


class GravityPerturb(Augmentation):
    """重力方向扰动：绕随机水平轴（方位 U[0, 2π)）把陀螺与加计旋转 U[0, max_deg]，**目标不变**。

    模拟姿态的横滚/俯仰误差（TLIO 官方 ``perturb_gravity``）；只在重力对齐坐标系下有意义。
    """

    name = "gravity_perturb"
    stage = 1

    def __init__(self, max_deg: float = 5.0) -> None:
        self.max_deg = float(max_deg)

    def __call__(self, sample: dict, rng: np.random.Generator) -> dict:
        azimuth = rng.uniform(0.0, 2.0 * math.pi)
        angle = math.radians(rng.uniform(0.0, self.max_deg))
        rot = axis_angle_matrix(np.array([math.cos(azimuth), math.sin(azimuth), 0.0]), angle)
        imu = sample["imu"]
        imu[0:3] = (rot @ imu[0:3].astype(np.float64)).astype(imu.dtype)
        imu[3:6] = (rot @ imu[3:6].astype(np.float64)).astype(imu.dtype)
        return sample


class TimeShift:
    """窗口起点随机平移 ``U{min_shift, …, max_shift}`` 个样本。

    缺省对称 ``[-max_shift, max_shift]``（RoNIN 的 random_shift，``max_shift`` 缺省为
    ``stride // 2``）；TLIO 官方为单向 ``min_shift=0, max_shift=9``。
    """

    name = "time_shift"

    def __init__(self, max_shift: Optional[int] = None, min_shift: Optional[int] = None) -> None:
        self.max_shift = max_shift
        self.min_shift = min_shift

    def resolve(self, stride: int) -> int:
        return int(self.max_shift) if self.max_shift is not None else max(int(stride) // 2, 0)

    def resolve_range(self, stride: int) -> tuple:
        """返回闭区间 ``(lo, hi)``。"""
        hi = self.resolve(stride)
        if self.min_shift is None:
            hi = max(hi, 0)
            return -hi, hi
        lo = int(self.min_shift)
        if lo > hi:
            raise ValueError(f"time_shift: min_shift={lo} > max_shift={hi}")
        return lo, hi

    def __repr__(self) -> str:
        return f"TimeShift(min_shift={self.min_shift}, max_shift={self.max_shift})"


REGISTRY = {"random_yaw": RandomYaw, "bias": Bias, "noise": Noise, "bias_shift": BiasShift,
            "gravity_perturb": GravityPerturb, "time_shift": TimeShift}


def _parse(item: Any) -> tuple:
    if isinstance(item, str):
        return item, {}
    if isinstance(item, Mapping):
        item = dict(item)
        if "name" in item:
            return str(item.pop("name")), item
        if len(item) == 1:
            (name, kwargs), = item.items()
            return str(name), dict(kwargs or {})
    raise ValueError(f"cannot parse augmentation spec {item!r}")


def build_augmentations(spec: Optional[Iterable[Any]], frame: str = "gravity_world") -> tuple:
    """解析 ``augment`` 配置，返回 ``(time_shift 或 None, [按阶段排序的增强])``。"""
    time_shift, augs = None, []
    for item in spec or []:
        name, kwargs = _parse(item)
        if name not in REGISTRY:
            raise ValueError(f"unknown augmentation {name!r}; available: {sorted(REGISTRY)}")
        obj = REGISTRY[name](**kwargs)
        if isinstance(obj, TimeShift):
            time_shift = obj
            continue
        if isinstance(obj, (RandomYaw, GravityPerturb)) and frame == "body":
            LOGGER.warning(f"augment: {obj.name} needs a gravity-aligned frame; "
                           "skipped for frame=body")
            continue
        augs.append(obj)
    augs.sort(key=lambda a: a.stage)
    return time_shift, augs
