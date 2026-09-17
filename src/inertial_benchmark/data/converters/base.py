"""转换器契约：原始数据集只负责“解析”，统一流水线负责“重采样、校验、写出”。

每个数据集模块（例如 ``converters/ronin.py``）必须提供：

``NAME: str``
    规范数据集名（小写，如 ``"ronin"``）。
``VERSION: str``
    转换器版本；解析逻辑任何改变都要提升版本号。
``LICENSE: str``
    原始数据许可证标识或条款链接。
``official_splits(source: Path) -> dict[str, list[str]]``
    官方划分，键为 ``train`` / ``val`` / ``test`` 以及可选官方子集（``test_seen`` 等），
    值为 ``sequence_id`` 列表。没有官方 val 时不要返回 ``val``，由统一流水线分组生成。
``iter_raw_sequences(source: Path, only: Collection[str] | None = None) -> Iterator[RawSequence]``
    逐条产出原生时钟上的数据；``only`` 非空时只解析这些 ``sequence_id``。
    无法解析的序列应产出 ``RawSequence`` 并在 ``rejected`` 中写明原因，而不是静默跳过。

解析阶段的职责边界（见 ``docs/DESIGN.md`` 第 2 节）：

* 统一单位：秒、rad/s、m/s²（比力，含重力）、米；
* 统一四元数：``wxyz``、``body_to_world``；参考世界系必须重力对齐且 z 轴向上；
* 应用数据集公开定义的标定（零偏、比例因子、轴映射），并写入 ``notes``；
* **不做**重采样、平滑或跨缺口插值——这些由统一流水线完成。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# 每条序列必须给出的属性；未知时写 "unknown"。统一流水线会补齐采样率、重采样与转换器字段。
RAW_REQUIRED_ATTRS = (
    "subject_id",
    "device_id",
    "placement",
    "group_id",
    "position_source",
    "orientation_source",
    "device_orientation_source",
    "body_frame",
    "source_files",
)


@dataclass
class RawSequence:
    """一条原生时钟上的序列。IMU 与参考位姿可以有各自的时间戳，但必须共享同一时钟。"""

    sequence_id: str
    imu_time: np.ndarray  # (M,) float64 秒
    gyroscope: np.ndarray  # (M,3) rad/s，机体系
    accelerometer: np.ndarray  # (M,3) m/s²，机体系比力
    pose_time: np.ndarray  # (K,) float64 秒，与 imu_time 同一时钟
    position: np.ndarray  # (K,3) 米，重力对齐 z 轴向上的参考世界系
    orientation: np.ndarray  # (K,4) wxyz，body_to_world
    velocity: Optional[np.ndarray] = None  # (K,3) m/s，仅当来源直接提供
    device_orientation: Optional[np.ndarray] = None  # (M,4) wxyz，定义在 imu_time 上
    imu_valid: Optional[np.ndarray] = None  # (M,) bool
    pose_valid: Optional[np.ndarray] = None  # (K,) bool
    attrs: dict = field(default_factory=dict)  # 至少包含 RAW_REQUIRED_ATTRS；可含 start_time_unix
    notes: list = field(default_factory=list)  # 已应用的修正与警告（人类可读）
    rejected: Optional[str] = None  # 非空表示整条序列被拒收及原因

    def check_shapes(self) -> list:
        """返回形状/数值问题列表（空列表表示通过），供转换器单元测试使用。"""

        problems = []
        m, k = len(self.imu_time), len(self.pose_time)
        expected = {
            "gyroscope": (self.gyroscope, (m, 3)),
            "accelerometer": (self.accelerometer, (m, 3)),
            "position": (self.position, (k, 3)),
            "orientation": (self.orientation, (k, 4)),
        }
        if self.velocity is not None:
            expected["velocity"] = (self.velocity, (k, 3))
        if self.device_orientation is not None:
            expected["device_orientation"] = (self.device_orientation, (m, 4))
        for name, (value, shape) in expected.items():
            if value.shape != shape:
                problems.append(f"{name}: expected {shape}, got {value.shape}")
        missing = [key for key in RAW_REQUIRED_ATTRS if key not in self.attrs]
        if missing:
            problems.append("attrs missing: " + ", ".join(missing))
        return problems
