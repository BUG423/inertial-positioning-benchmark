"""统一序列的读取、校验，以及与模型框架无关的滑窗视图。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Mapping, Optional, Union

import h5py
import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.floating]
PathLike = Union[str, Path]

# 统一维护必需属性，避免各数据集适配器产生不兼容的元数据。
REQUIRED_ATTRIBUTES = {
    "schema_version",
    "dataset",
    "sequence_id",
    "world_frame",
    "timestamp_type",
    "orientation_convention",
    "accelerometer_type",
    "position_source",
    "orientation_source",
    "subject_id",
    "device_id",
    "source_license",
}


class SequenceValidationError(ValueError):
    """当一条序列不满足统一数据契约时抛出。"""


@dataclass(frozen=True)
class CanonicalSequence:
    """统一表示下的一条时间对齐轨迹。

    陀螺仪和加速度计均保留在机体系；姿态为 ``wxyz`` 顺序的 body-to-world
    单位四元数；位置和可选速度位于 ``world_frame`` 声明的世界坐标系中。
    """

    timestamp: Array
    gyroscope: Array
    accelerometer: Array
    orientation: Array
    position: Array
    velocity: Optional[Array] = None
    valid_imu: Optional[NDArray[np.bool_]] = None
    valid_orientation: Optional[NDArray[np.bool_]] = None
    valid_position: Optional[NDArray[np.bool_]] = None
    attributes: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def from_hdf5(cls, path: PathLike, *, validate: bool = True) -> "CanonicalSequence":
        """从 HDF5 文件读取一条统一序列，并默认立即校验。"""

        with h5py.File(path, "r") as handle:
            optional = lambda name: np.asarray(handle[name]) if name in handle else None
            sequence = cls(
                timestamp=np.asarray(handle["timestamp"], dtype=np.float64),
                gyroscope=np.asarray(handle["imu/gyroscope"]),
                accelerometer=np.asarray(handle["imu/accelerometer"]),
                orientation=np.asarray(handle["pose/orientation"]),
                position=np.asarray(handle["pose/position"]),
                velocity=optional("pose/velocity"),
                valid_imu=optional("valid/imu"),
                valid_orientation=optional("valid/orientation"),
                valid_position=optional("valid/position"),
                attributes={key: _decode(value) for key, value in handle.attrs.items()},
            )
        if validate:
            sequence.validate()
        return sequence

    def validate(self, *, quaternion_atol: float = 1e-3) -> None:
        """检查维度、数值、时间戳、四元数、元数据和坐标约定。"""

        errors = []
        n = len(self.timestamp)
        arrays = {
            "timestamp": (self.timestamp, (n,)),
            "imu/gyroscope": (self.gyroscope, (n, 3)),
            "imu/accelerometer": (self.accelerometer, (n, 3)),
            "pose/orientation": (self.orientation, (n, 4)),
            "pose/position": (self.position, (n, 3)),
        }
        if self.velocity is not None:
            arrays["pose/velocity"] = (self.velocity, (n, 3))
        for name, (value, expected) in arrays.items():
            if value.shape != expected:
                errors.append(f"{name}: expected shape {expected}, got {value.shape}")
            elif not np.isfinite(value).all():
                errors.append(f"{name}: contains NaN or Inf")

        if n < 2:
            errors.append("timestamp: at least two samples are required")
        elif not np.all(np.diff(self.timestamp) > 0):
            errors.append("timestamp: values must be strictly increasing")

        # 先确认形状和有限性，再检查范数，避免次生异常掩盖真正的数据问题。
        if self.orientation.shape == (n, 4) and np.isfinite(self.orientation).all():
            norm = np.linalg.norm(self.orientation, axis=1)
            if not np.allclose(norm, 1.0, atol=quaternion_atol, rtol=0.0):
                errors.append("pose/orientation: quaternions are not normalized")

        for name, mask in (
            ("valid/imu", self.valid_imu),
            ("valid/orientation", self.valid_orientation),
            ("valid/position", self.valid_position),
        ):
            if mask is not None:
                if mask.shape != (n,):
                    errors.append(f"{name}: expected shape {(n,)}, got {mask.shape}")
                elif not np.issubdtype(mask.dtype, np.bool_):
                    errors.append(f"{name}: expected boolean dtype, got {mask.dtype}")

        missing = sorted(REQUIRED_ATTRIBUTES - set(self.attributes))
        if missing:
            errors.append("attributes: missing " + ", ".join(missing))
        if self.attributes.get("orientation_convention") != "body_to_world_wxyz":
            errors.append("orientation_convention must be 'body_to_world_wxyz'")
        if errors:
            raise SequenceValidationError("Invalid canonical sequence:\n- " + "\n- ".join(errors))


@dataclass(frozen=True)
class WindowSample:
    """可直接交给模型的 IMU 窗口及其端点目标。"""

    features: Array
    target: Array
    sequence_id: str
    start_index: int
    end_index: int
    duration: float


class WindowDataset:
    """统一序列之上的惰性滑窗视图。

    该视图不假设固定采样率。目标为窗口首尾位置的位移，或位移除以真实时间差
    得到的平均速度；输出维度可选 2D 或 3D。
    """

    def __init__(
        self,
        sequences: Union[CanonicalSequence, list[CanonicalSequence]],
        *,
        window_size: int,
        stride: int = 1,
        target: str = "velocity",
        dimensions: int = 3,
        require_valid: bool = True,
    ) -> None:
        self.sequences = sequences if isinstance(sequences, list) else [sequences]
        if window_size < 2 or stride < 1:
            raise ValueError("window_size must be >= 2 and stride must be >= 1")
        if target not in {"velocity", "displacement"}:
            raise ValueError("target must be 'velocity' or 'displacement'")
        if dimensions not in {2, 3}:
            raise ValueError("dimensions must be 2 or 3")
        self.window_size = window_size
        self.stride = stride
        self.target = target
        self.dimensions = dimensions
        self.require_valid = require_valid
        self._index = self._build_index()

    def _build_index(self) -> list[tuple[int, int]]:
        """只建立轻量索引，不复制窗口数据。"""

        index = []
        for sequence_index, sequence in enumerate(self.sequences):
            sequence.validate()
            for start in range(0, len(sequence.timestamp) - self.window_size + 1, self.stride):
                end = start + self.window_size
                if not self.require_valid or _window_is_valid(sequence, start, end):
                    index.append((sequence_index, start))
        return index

    def __len__(self) -> int:
        return len(self._index)

    def __iter__(self) -> Iterator[WindowSample]:
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, index: int) -> WindowSample:
        sequence_index, start = self._index[index]
        sequence = self.sequences[sequence_index]
        stop = start + self.window_size
        end = stop - 1
        # 统一通道顺序：[gyro_xyz, accelerometer_xyz]，输出形状为 (6, T)。
        features = np.concatenate(
            (sequence.gyroscope[start:stop], sequence.accelerometer[start:stop]), axis=1
        ).T.astype(np.float32, copy=False)
        # 目标使用窗口首、末样本，因此 duration 对应 T-1 个采样间隔。
        displacement = sequence.position[end] - sequence.position[start]
        duration = float(sequence.timestamp[end] - sequence.timestamp[start])
        value = displacement / duration if self.target == "velocity" else displacement
        return WindowSample(
            features=features,
            target=np.asarray(value[: self.dimensions], dtype=np.float32),
            sequence_id=str(sequence.attributes["sequence_id"]),
            start_index=start,
            end_index=end,
            duration=duration,
        )


def _window_is_valid(sequence: CanonicalSequence, start: int, stop: int) -> bool:
    """任一已提供的有效掩码为 False 时，丢弃整个窗口。"""

    for mask in (sequence.valid_imu, sequence.valid_orientation, sequence.valid_position):
        if mask is not None and not np.asarray(mask[start:stop], dtype=bool).all():
            return False
    return True


def _decode(value: object) -> object:
    return value.decode("utf-8") if isinstance(value, bytes) else value
