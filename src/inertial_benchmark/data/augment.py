"""训练增强（不依赖 torch）。

配置写法（``augment`` 列表）::

    augment: [random_yaw, time_shift]
    augment: [{name: bias, gyro: 0.005, acc: 0.05}, {name: noise, gyro: 0.002, acc: 0.02}]

执行顺序固定：传感器级（``bias``/``noise``，作用于原始量）先于坐标级（``random_yaw``），
``time_shift`` 由数据集在取窗口前处理。每个增强接收 ``numpy.random.Generator``，保证可复现。

外部模块可用 ``@register_augmentation(name)`` 注册 :class:`Augmentation` 子类；配置中出现未知名字时
会先加载 ``IPB_PLUGINS`` 列出的插件（见 ``utils/plugins.py``）再查找。
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable, Mapping, Optional

import numpy as np

from ..utils import LOGGER
from ..utils.geometry import quat_from_yaw, quat_multiply, rotate_z


def _align_span(values: np.ndarray, shape: tuple) -> np.ndarray:
    """把 ``(3, span)`` 的机体系量对齐到输入布局 ``(..., 3, T)``。

    无历史时 ``span == T`` 直接返回；有历史时按 ``(H, T)`` 切出各子窗口的片段
    （子窗口起点间隔由 ``span``、``T``、``H`` 反解，与视图一致）。
    """
    window, span = shape[-1], values.shape[-1]
    if len(shape) == 2 or span == window:
        return values[..., -window:]
    subs = shape[0]
    stride = (span - window) // max(subs - 1, 1)
    idx = np.arange(subs)[:, None] * stride + np.arange(window)[None, :]
    return values[:, idx].transpose(1, 0, 2)


class Augmentation:
    """增强基类。

    ``sample`` 含 ``imu``（``(6,T)``，声明 ``history`` 时为 ``(H,6,T)``）、``target``
    （``(D,)`` 或 ``(R,D)``）、``mask``、``extra``（额外输入）与 ``body_to_frame`` 回调。
    实现一律用 ``...`` 索引通道维，以同时支持有/无历史两种布局。
    """

    name = "base"
    stage = 0  # 0 = 传感器级，1 = 坐标级

    def __call__(self, sample: dict, rng: np.random.Generator) -> dict:
        raise NotImplementedError

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v}" for k, v in vars(self).items())
        return f"{type(self).__name__}({args})"


class RandomYaw(Augmentation):
    """在重力对齐系下绕 z 轴同步旋转 IMU（陀螺与加计）、目标与姿态类额外输入。"""

    name = "random_yaw"
    stage = 1

    def __init__(self, max_angle: float = math.pi) -> None:
        self.max_angle = float(max_angle)

    def __call__(self, sample: dict, rng: np.random.Generator) -> dict:
        angle = rng.uniform(-self.max_angle, self.max_angle)
        imu = sample["imu"]
        imu[..., 0:3, :] = np.moveaxis(rotate_z(np.moveaxis(imu[..., 0:3, :], -1, -2), angle),
                                       -1, -2)
        imu[..., 3:6, :] = np.moveaxis(rotate_z(np.moveaxis(imu[..., 3:6, :], -1, -2), angle),
                                       -1, -2)
        target = sample["target"]
        target[...] = rotate_z(target, angle)
        extra = sample.get("extra") or {}
        if "orientation" in extra:  # 姿态左乘 q_z(angle)
            extra["orientation"][...] = quat_multiply(quat_from_yaw(angle),
                                                      extra["orientation"].astype(np.float64))
        if "init_velocity" in extra:
            extra["init_velocity"][...] = rotate_z(extra["init_velocity"], angle)
        # gravity 沿世界 z 轴（或机体系中不随偏航变化），无需旋转
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
        for channels, sigma in ((slice(0, 3), self.gyro), (slice(3, 6), self.acc)):
            if sigma <= 0:
                continue
            # (span, 3) → (3, span)，再按输入布局切出各子窗口对应的片段
            bias = to_frame(rng.normal(0.0, sigma, 3)).T.astype(imu.dtype)
            imu[..., channels, :] += _align_span(bias, imu.shape)
        return sample


class Noise(Augmentation):
    """逐样本各向同性高斯白噪声（旋转不变，可直接加在任意坐标系）。"""

    name = "noise"

    def __init__(self, gyro: float = 0.0, acc: float = 0.0) -> None:
        self.gyro = float(gyro)
        self.acc = float(acc)

    def __call__(self, sample: dict, rng: np.random.Generator) -> dict:
        imu = sample["imu"]
        shape = imu.shape[:-2] + (3, imu.shape[-1])
        if self.gyro > 0:
            imu[..., 0:3, :] += rng.normal(0.0, self.gyro, shape).astype(imu.dtype)
        if self.acc > 0:
            imu[..., 3:6, :] += rng.normal(0.0, self.acc, shape).astype(imu.dtype)
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


def register_augmentation(*names: str):
    """类装饰器：以一个或多个名字注册增强（:class:`Augmentation` 子类，按 ``stage`` 排序执行）。"""

    def decorator(cls):
        if not (isinstance(cls, type) and issubclass(cls, Augmentation)):
            raise TypeError(f"{cls!r} must subclass Augmentation")
        for name in names:
            if name in REGISTRY and REGISTRY[name] is not cls:
                raise KeyError(f"augmentation {name!r} already registered by {REGISTRY[name]}")
            REGISTRY[name] = cls
        return cls

    return decorator


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
            from ..utils.plugins import load_plugins

            load_plugins()
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
