"""训练增强（不依赖 torch）。

配置写法（``augment`` 列表）::

    augment: [random_yaw, time_shift]
    augment: [{name: bias, gyro: 0.005, acc: 0.05}, {name: noise, gyro: 0.002, acc: 0.02}]

执行顺序固定：传感器级（``bias``/``noise``，作用于原始量）先于坐标级（``random_yaw``），
``time_shift`` 由数据集在取窗口前处理。每个增强接收 ``numpy.random.Generator``，保证可复现。
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


class TimeShift:
    """窗口起点随机平移 ``[-max_shift, max_shift]`` 个样本（RoNIN 的 random_shift）。"""

    name = "time_shift"

    def __init__(self, max_shift: Optional[int] = None) -> None:
        self.max_shift = max_shift

    def resolve(self, stride: int) -> int:
        return int(self.max_shift) if self.max_shift is not None else max(int(stride) // 2, 0)

    def __repr__(self) -> str:
        return f"TimeShift(max_shift={self.max_shift})"


REGISTRY = {"random_yaw": RandomYaw, "bias": Bias, "noise": Noise, "time_shift": TimeShift}


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
        if isinstance(obj, RandomYaw) and frame == "body":
            LOGGER.warning("augment: random_yaw has no effect in frame=body; skipped")
            continue
        augs.append(obj)
    augs.sort(key=lambda a: a.stage)
    return time_shift, augs
