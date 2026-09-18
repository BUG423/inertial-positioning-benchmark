"""TartanIMU 的机体系增强（规格卡 ``docs/algorithms/tartanimu.md`` §5）。

框架自带的 :class:`~inertial_benchmark.data.augment.RandomYaw` 与
:class:`~inertial_benchmark.data.augment.GravityPerturb` 只在重力对齐坐标系下有意义，
``frame=body`` 时会被 ``build_augmentations`` 跳过。TartanIMU 官方**就是**在机体系里做这两种
旋转（``dataset_AirLab.py`` 的增强管线作用于 ``use_local_coord=True`` 的机体系 IMU），因此这里
注册两个机体系版本：

* ``body_yaw``：绕**机体 z 轴**随机偏航 ``θ ~ U[0, 2π)``，同步旋转陀螺、加计与目标的 xy 分量；
* ``body_tilt``：绕随机**水平机体轴**（方位 ``φ ~ U[0, 2π)``）旋转 ``U[0, max_deg]``，同时作用于
  陀螺、加计与**目标**（这一点与 TLIO/RNIN 的 ``gravity_perturb`` 不同——那两个不旋转目标）。

两者都属于“算法固有增强”（DESIGN §4），在 ``official`` 与 ``unified`` 两个配方中一致。
"""

from __future__ import annotations

import math

import numpy as np

from ...data.augment import Augmentation, axis_angle_matrix, register_augmentation
from ...utils.geometry import rotate_z


def _rotate_imu(imu: np.ndarray, rot: np.ndarray) -> None:
    """就地按 3×3 矩阵旋转 ``(..., 6, T)`` 的陀螺与加计（``...`` 覆盖可选的子窗口维）。"""
    for channels in (slice(0, 3), slice(3, 6)):
        rotated = np.einsum("ij,...jt->...it", rot, imu[..., channels, :].astype(np.float64))
        imu[..., channels, :] = rotated.astype(imu.dtype)


@register_augmentation("body_yaw")
class BodyYaw(Augmentation):
    """机体系随机偏航：绕机体 z 轴同步旋转陀螺、加计与目标（TartanIMU 官方增强 ①）。"""

    name = "body_yaw"
    stage = 1

    def __init__(self, max_angle: float = math.pi) -> None:
        self.max_angle = float(max_angle)

    def __call__(self, sample: dict, rng: np.random.Generator) -> dict:
        angle = rng.uniform(-self.max_angle, self.max_angle)
        imu = sample["imu"]
        for channels in (slice(0, 3), slice(3, 6)):
            block = np.moveaxis(imu[..., channels, :], -1, -2)
            imu[..., channels, :] = np.moveaxis(rotate_z(block, angle), -1, -2)
        sample["target"][...] = rotate_z(sample["target"], angle)
        return sample


@register_augmentation("body_tilt")
class BodyTilt(Augmentation):
    """机体系倾斜扰动：绕随机水平机体轴旋转 ``U[0, max_deg]``，**目标一起旋转**（官方增强 ③）。"""

    name = "body_tilt"
    stage = 1

    def __init__(self, max_deg: float = 5.0) -> None:
        self.max_deg = float(max_deg)

    def __call__(self, sample: dict, rng: np.random.Generator) -> dict:
        azimuth = rng.uniform(0.0, 2.0 * math.pi)
        angle = math.radians(rng.uniform(0.0, self.max_deg))
        rot = axis_angle_matrix(np.array([math.cos(azimuth), math.sin(azimuth), 0.0]), angle)
        _rotate_imu(sample["imu"], rot)
        target = sample["target"]
        if target.shape[-1] == 3:  # 机体系目标是 3 维，可以整体旋转
            target[...] = np.einsum("ij,...j->...i", rot, target.astype(np.float64)).astype(
                target.dtype)
        return sample
