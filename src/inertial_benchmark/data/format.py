"""IPB v1 序列格式（DESIGN 2.2）：``Sequence`` 数据类、HDF5 读写与校验。

``load_sequence`` 兼容读取 v0.1 文件：按 DESIGN 2.3 在内存中重采样为 v1 对象；写出永远是 v1。
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional, Union

import h5py
import numpy as np

from ..utils.geometry import GRAVITY, quat_conjugate, quat_rotate
from .resample import DEFAULT_RATE, resample_streams

PathLike = Union[str, Path]
SCHEMA_VERSION = "1.0"
WORLD_FRAME = "gravity_aligned_z_up"
ORIENTATION_CONVENTION = "body_to_world_wxyz"
ACCELEROMETER_TYPE = "specific_force"
PLACEMENTS = ("handheld", "pocket", "bag", "trolley", "head", "body", "mixed", "unknown")

REQUIRED_ATTRS = (
    "schema_version",
    "dataset",
    "sequence_id",
    "sample_rate_hz",
    "source_sample_rate_hz",
    "resampling",
    "world_frame",
    "timestamp_type",
    "start_time_unix",
    "orientation_convention",
    "accelerometer_type",
    "body_frame",
    "position_source",
    "orientation_source",
    "device_orientation_source",
    "subject_id",
    "device_id",
    "placement",
    "group_id",
    "source_license",
    "source_files",
    "converter",
)

# 固定取值的属性：校验时逐一比对
FIXED_ATTRS = {
    "schema_version": SCHEMA_VERSION,
    "world_frame": WORLD_FRAME,
    "timestamp_type": "relative",
    "orientation_convention": ORIENTATION_CONVENTION,
    "accelerometer_type": ACCELEROMETER_TYPE,
}


class SequenceError(ValueError):
    """序列不满足 IPB v1 契约。"""


@dataclass
class Sequence:
    """一条 IPB v1 序列：统一 200 Hz 时间轴上的 IMU、参考位姿与有效掩码。"""

    timestamp: np.ndarray  # (N,) f8，从 0 开始、严格均匀
    gyroscope: np.ndarray  # (N,3) f4，rad/s，机体系
    accelerometer: np.ndarray  # (N,3) f4，m/s²，机体系比力（含重力）
    orientation: np.ndarray  # (N,4) f4，参考姿态 body_to_world_wxyz
    position: np.ndarray  # (N,3) f8，m，参考世界系
    valid_imu: np.ndarray  # (N,) bool
    valid_pose: np.ndarray  # (N,) bool
    device_orientation: Optional[np.ndarray] = None  # (N,4) f4，设备自身估计姿态
    velocity: Optional[np.ndarray] = None  # (N,3) f4，仅当来源提供
    attrs: dict = field(default_factory=dict)
    valid_device_orientation: Optional[np.ndarray] = None  # (N,) bool，仅当有设备姿态时

    def __post_init__(self) -> None:
        self.timestamp = np.asarray(self.timestamp, dtype=np.float64)
        self.gyroscope = np.asarray(self.gyroscope, dtype=np.float32)
        self.accelerometer = np.asarray(self.accelerometer, dtype=np.float32)
        self.orientation = np.asarray(self.orientation, dtype=np.float32)
        self.position = np.asarray(self.position, dtype=np.float64)
        self.valid_imu = np.asarray(self.valid_imu, dtype=bool)
        self.valid_pose = np.asarray(self.valid_pose, dtype=bool)
        if self.device_orientation is not None:
            self.device_orientation = np.asarray(self.device_orientation, dtype=np.float32)
        if self.velocity is not None:
            self.velocity = np.asarray(self.velocity, dtype=np.float32)
        if self.valid_device_orientation is not None:
            self.valid_device_orientation = np.asarray(self.valid_device_orientation, dtype=bool)
        self.attrs = dict(self.attrs)

    def __len__(self) -> int:
        return len(self.timestamp)

    def __repr__(self) -> str:
        return (f"Sequence(id={self.sequence_id!r}, dataset={self.dataset!r}, n={len(self)}, "
                f"duration={self.duration:.1f}s, rate={self.sample_rate:g}Hz)")

    @property
    def sequence_id(self) -> str:
        return str(self.attrs.get("sequence_id", "unknown"))

    @property
    def dataset(self) -> str:
        return str(self.attrs.get("dataset", "unknown"))

    @property
    def group_id(self) -> str:
        return str(self.attrs.get("group_id", self.sequence_id))

    @property
    def sample_rate(self) -> float:
        rate = self.attrs.get("sample_rate_hz")
        if rate is None and len(self) > 1:
            return float(1.0 / np.median(np.diff(self.timestamp)))
        return float(rate if rate is not None else DEFAULT_RATE)

    @property
    def duration(self) -> float:
        return float(self.timestamp[-1] - self.timestamp[0]) if len(self) > 1 else 0.0

    @property
    def valid(self) -> np.ndarray:
        """IMU 与参考位姿同时有效的样本。"""
        return self.valid_imu & self.valid_pose

    @property
    def valid_device(self) -> np.ndarray:
        """IMU、参考位姿与设备姿态同时有效的样本（``orientation=device`` 的任务视图使用）。

        设备姿态的缺口只影响本掩码，不影响 ``valid``：使用参考姿态的模型不应因此丢窗口。
        """
        both = self.valid
        if self.valid_device_orientation is None:
            return both
        return both & self.valid_device_orientation

    def slice(self, start: int, stop: int) -> "Sequence":
        """返回 ``[start, stop)`` 的子序列（时间戳不重置）。"""
        s = slice(start, stop)
        return replace(
            self,
            timestamp=self.timestamp[s],
            gyroscope=self.gyroscope[s],
            accelerometer=self.accelerometer[s],
            orientation=self.orientation[s],
            position=self.position[s],
            valid_imu=self.valid_imu[s],
            valid_pose=self.valid_pose[s],
            device_orientation=None if self.device_orientation is None
            else self.device_orientation[s],
            velocity=None if self.velocity is None else self.velocity[s],
            attrs=dict(self.attrs),
            valid_device_orientation=None if self.valid_device_orientation is None
            else self.valid_device_orientation[s],
        )

    def distance(self, resolution: float = 1.0, dims: int = 2) -> float:
        """参考轨迹路径长度（米，默认水平面、1 s 分辨率，见 METRICS.md）。"""
        from ..metrics.trajectory import path_length

        return path_length(self.position, self.valid_pose, self.sample_rate, resolution, dims)


# ----------------------------------------------------------------------------- 属性编解码


def _encode_attr(value: Any) -> Any:
    if value is None:
        return "none"
    if isinstance(value, (str, bool, int, float, np.integer, np.floating, np.bool_)):
        return value
    if isinstance(value, (list, tuple)):
        if all(isinstance(v, str) for v in value):
            return np.array(list(value), dtype=h5py.string_dtype())
        return np.asarray(value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _decode_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "OSU":
            return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in value.tolist()]
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def read_attrs(handle: h5py.File) -> dict:
    return {key: _decode_attr(value) for key, value in handle.attrs.items()}


# ----------------------------------------------------------------------------- 读写


def load_sequence(
    path: PathLike, *, validate: bool = False, rate: float = DEFAULT_RATE
) -> Sequence:
    """读取一条序列。v1 直接读取；v0.1 在内存中重采样到 ``rate`` Hz。

    ``validate=True`` 时执行完整校验（含重力检查），失败抛出 :class:`SequenceError`。
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        attrs = read_attrs(f)
        version = str(attrs.get("schema_version", "0.1"))
        if version.split(".")[0] == "1":
            seq = _read_v1(f, attrs)
        else:
            seq = _read_v01(f, attrs, rate)
    if validate:
        check_sequence(seq)
    return seq


def _read_v1(f: h5py.File, attrs: dict) -> Sequence:
    def opt(name: str) -> Optional[np.ndarray]:
        return np.asarray(f[name]) if name in f else None

    return Sequence(
        timestamp=np.asarray(f["timestamp"]),
        gyroscope=np.asarray(f["imu/gyroscope"]),
        accelerometer=np.asarray(f["imu/accelerometer"]),
        orientation=np.asarray(f["pose/orientation"]),
        position=np.asarray(f["pose/position"]),
        valid_imu=np.asarray(f["valid/imu"]),
        valid_pose=np.asarray(f["valid/pose"]),
        device_orientation=opt("imu/orientation"),
        velocity=opt("pose/velocity"),
        attrs=attrs,
        valid_device_orientation=opt("valid/device_orientation"),
    )


def _read_v01(f: h5py.File, attrs: dict, rate: float) -> Sequence:
    """v0.1：单时钟、可能非均匀；按 DESIGN 2.3 重采样为 v1（不修改原文件）。"""
    t = np.asarray(f["timestamp"], dtype=np.float64)
    n = len(t)

    def mask(name: str) -> np.ndarray:
        return np.asarray(f[name], dtype=bool) if name in f else np.ones(n, dtype=bool)

    pose_valid = mask("valid/orientation") & mask("valid/position")
    result = resample_streams(
        t,
        np.asarray(f["imu/gyroscope"]),
        np.asarray(f["imu/accelerometer"]),
        t,
        np.asarray(f["pose/position"]),
        np.asarray(f["pose/orientation"]),
        velocity=np.asarray(f["pose/velocity"]) if "pose/velocity" in f else None,
        imu_valid=mask("valid/imu"),
        pose_valid=pose_valid,
        rate=rate,
    )
    finite_t = t[np.isfinite(t)]
    is_unix = attrs.get("timestamp_type") == "unix" or (len(finite_t) and finite_t[0] > 1e8)
    new_attrs = dict(attrs)
    new_attrs.update(
        schema_version=SCHEMA_VERSION,
        legacy_schema_version=str(attrs.get("schema_version", "0.1")),
        sample_rate_hz=float(rate),
        source_sample_rate_hz=float(result.info["imu"]["source_rate_hz"]),
        resampling="in-memory from v0.1: " + result.description,
        timestamp_type="relative",
        start_time_unix=float(result.start_time) if is_unix else float("nan"),
    )
    # v0.1 的 world_frame 原样保留（不静默改写），由 validate() 报告是否符合 v1
    for key in REQUIRED_ATTRS:
        new_attrs.setdefault(key, "unknown")
    if new_attrs["group_id"] == "unknown":
        new_attrs["group_id"] = str(new_attrs.get("subject_id", "unknown"))
    if new_attrs["device_orientation_source"] == "unknown":
        new_attrs["device_orientation_source"] = "none"
    arrays = result.arrays
    return Sequence(
        timestamp=result.timestamp,
        gyroscope=arrays["gyroscope"],
        accelerometer=arrays["accelerometer"],
        orientation=arrays["orientation"],
        position=arrays["position"],
        valid_imu=result.valid_imu,
        valid_pose=result.valid_pose,
        velocity=arrays.get("velocity"),
        attrs=new_attrs,
    )


def save_sequence(
    path: PathLike,
    seq: Sequence,
    *,
    compression: str = "gzip",
    check: bool = True,
) -> Path:
    """写出 v1 HDF5（先写临时文件再原子替换）。

    ``compression`` 取 ``gzip``（默认，字节级可复现）/ ``lzf`` / ``none``。注意 h5py 自带的
    LZF 不初始化哈希表，相同数据两次写出的字节可能不同，会破坏基于 sha256 的数据集指纹。
    ``check=True`` 时先做结构校验（不含重力检查）。
    """
    if check:
        check_sequence(seq, check_gravity=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict = {}
    if compression == "gzip":
        kwargs = {"compression": "gzip", "compression_opts": 4, "shuffle": True}
    elif compression == "lzf":
        kwargs = {"compression": "lzf", "shuffle": True}
    elif compression not in ("none", None):
        raise ValueError(f"unsupported compression {compression!r}")

    tmp = path.with_name(path.name + ".tmp")
    with h5py.File(tmp, "w") as f:
        def put(name: str, data: np.ndarray, dtype: Any) -> None:
            arr = np.ascontiguousarray(data, dtype=dtype)
            opts = kwargs if arr.size >= 64 else {}
            f.create_dataset(name, data=arr, **opts)

        put("timestamp", seq.timestamp, np.float64)
        put("imu/gyroscope", seq.gyroscope, np.float32)
        put("imu/accelerometer", seq.accelerometer, np.float32)
        if seq.device_orientation is not None:
            put("imu/orientation", seq.device_orientation, np.float32)
        put("pose/orientation", seq.orientation, np.float32)
        put("pose/position", seq.position, np.float64)
        if seq.velocity is not None:
            put("pose/velocity", seq.velocity, np.float32)
        put("valid/imu", seq.valid_imu, bool)
        put("valid/pose", seq.valid_pose, bool)
        if seq.device_orientation is not None and seq.valid_device_orientation is not None:
            put("valid/device_orientation", seq.valid_device_orientation, bool)
        attrs = dict(seq.attrs)
        attrs["schema_version"] = SCHEMA_VERSION
        for key in sorted(attrs):
            f.attrs[key] = _encode_attr(attrs[key])
    os.replace(tmp, path)
    return path


# ----------------------------------------------------------------------------- 校验


@dataclass
class ValidationReport:
    """校验结果：``errors`` 非空表示不合格；``info`` 含重力检查等诊断数值。"""

    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    info: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_failed(self, name: str = "sequence") -> None:
        if self.errors:
            raise SequenceError(f"{name} failed IPB v1 validation:\n- " + "\n- ".join(self.errors))

    def __str__(self) -> str:
        lines = [f"ok={self.ok}"]
        lines += [f"error: {e}" for e in self.errors]
        lines += [f"warning: {w}" for w in self.warnings]
        return "\n".join(lines)


def validate(
    seq: Sequence,
    *,
    sample_rate: Optional[float] = DEFAULT_RATE,
    quat_tol: float = 1e-3,
    check_gravity: bool = True,
    gravity_tol_deg: float = 5.0,
    gravity_norm_tol: float = 0.6,
) -> ValidationReport:
    """按 DESIGN 2.2 校验：形状、有限值、严格均匀时间、四元数、必需属性、重力方向。"""
    rep = ValidationReport()
    n = len(seq.timestamp)
    if seq.timestamp.ndim != 1 or n < 2:
        rep.errors.append("timestamp: expected a 1-D array with at least 2 samples")
        return rep

    shapes = {
        "imu/gyroscope": (seq.gyroscope, (n, 3)),
        "imu/accelerometer": (seq.accelerometer, (n, 3)),
        "pose/orientation": (seq.orientation, (n, 4)),
        "pose/position": (seq.position, (n, 3)),
        "valid/imu": (seq.valid_imu, (n,)),
        "valid/pose": (seq.valid_pose, (n,)),
    }
    if seq.device_orientation is not None:
        shapes["imu/orientation"] = (seq.device_orientation, (n, 4))
    if seq.velocity is not None:
        shapes["pose/velocity"] = (seq.velocity, (n, 3))
    if seq.valid_device_orientation is not None:
        shapes["valid/device_orientation"] = (seq.valid_device_orientation, (n,))
    shape_ok = True
    for name, (arr, shape) in shapes.items():
        if arr.shape != shape:
            rep.errors.append(f"{name}: expected shape {shape}, got {arr.shape}")
            shape_ok = False
        elif arr.dtype != bool and not np.isfinite(arr).all():
            rep.errors.append(f"{name}: contains NaN/Inf (fill invalid samples and mask them)")
    if not np.isfinite(seq.timestamp).all():
        rep.errors.append("timestamp: contains NaN/Inf")
        return rep

    # 时间：从 0 开始、严格均匀
    rate = _attr_float(seq.attrs.get("sample_rate_hz"))
    if rate is None or not rate > 0:
        rep.errors.append("attrs: sample_rate_hz missing or not positive")
        rate = float(1.0 / np.median(np.diff(seq.timestamp)))
    if sample_rate is not None and abs(rate - sample_rate) > 1e-6:
        rep.errors.append(f"sample_rate_hz={rate:g}, expected {sample_rate:g}")
    if abs(seq.timestamp[0]) > 1e-9:
        rep.errors.append(f"timestamp: must start at 0, got {seq.timestamp[0]:.6f}")
    ideal = seq.timestamp[0] + np.arange(n) / rate
    dev = float(np.max(np.abs(seq.timestamp - ideal)))
    if dev > 1e-6:
        rep.errors.append(f"timestamp: not uniform at {rate:g} Hz (max deviation {dev:.2e} s)")
    rep.info["duration_s"] = float(seq.timestamp[-1] - seq.timestamp[0])

    # 四元数：单位范数、符号连续
    for name, q in (("pose/orientation", seq.orientation), ("imu/orientation",
                                                            seq.device_orientation)):
        if q is None or q.shape != (n, 4) or not np.isfinite(q).all():
            continue
        norm_err = float(np.max(np.abs(np.linalg.norm(q.astype(np.float64), axis=1) - 1.0)))
        if norm_err > quat_tol:
            rep.errors.append(f"{name}: not unit quaternions (max |‖q‖-1| = {norm_err:.2e})")
        flips = int(np.sum(np.einsum("ij,ij->i", q[1:].astype(np.float64), q[:-1]) < 0))
        if flips:
            rep.errors.append(f"{name}: {flips} sign discontinuities (q and -q alternate)")

    # 掩码
    masks = [("valid/imu", seq.valid_imu), ("valid/pose", seq.valid_pose)]
    if seq.valid_device_orientation is not None:
        masks.append(("valid/device_orientation", seq.valid_device_orientation))
        if seq.device_orientation is None:
            rep.errors.append("valid/device_orientation: present although imu/orientation is not")
    for name, mask in masks:
        if mask.dtype != bool:
            rep.errors.append(f"{name}: expected bool dtype, got {mask.dtype}")
    both = seq.valid_imu & seq.valid_pose if shape_ok else np.zeros(n, bool)
    frac = float(both.mean())
    rep.info["valid_fraction"] = frac
    if shape_ok and not both.any():
        rep.errors.append("valid: no sample has both valid IMU and valid pose")
    elif frac < 0.5:
        rep.warnings.append(f"valid: only {frac:.1%} of samples have valid IMU and pose")

    # 属性
    missing = [k for k in REQUIRED_ATTRS if k not in seq.attrs]
    if missing:
        rep.errors.append("attrs: missing " + ", ".join(missing))
    for key, expected in FIXED_ATTRS.items():
        value = seq.attrs.get(key)
        if value is not None and str(value) != expected:
            rep.errors.append(f"attrs: {key}={value!r}, expected {expected!r}")
    placement = seq.attrs.get("placement")
    if placement is not None and placement not in PLACEMENTS:
        rep.warnings.append(f"attrs: placement={placement!r} not in {PLACEMENTS}")
    if seq.device_orientation is None and seq.attrs.get("device_orientation_source",
                                                         "none") != "none":
        rep.warnings.append("attrs: device_orientation_source set but imu/orientation is absent")
    if seq.valid_device_orientation is not None and shape_ok:
        frac_dev = float(seq.valid_device.mean())
        rep.info["valid_fraction_device"] = frac_dev
        if frac_dev < frac:
            rep.warnings.append(f"valid/device_orientation: {frac - frac_dev:.1%} of the samples "
                                "have a valid reference pose but no valid device orientation")

    if check_gravity and shape_ok and both.sum() >= rate:
        _check_gravity(seq, both, rate, rep, gravity_tol_deg, gravity_norm_tol)
    return rep


def check_sequence(seq: Sequence, **kwargs: Any) -> ValidationReport:
    """``validate`` 并在失败时抛出 :class:`SequenceError`。"""
    rep = validate(seq, **kwargs)
    rep.raise_if_failed(seq.sequence_id)
    return rep


def _attr_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _tilt(vec: np.ndarray) -> float:
    """向量与 +z 的夹角（度）。"""
    norm = float(np.linalg.norm(vec))
    return float(np.degrees(np.arccos(np.clip(vec[2] / norm, -1.0, 1.0)))) if norm > 0 else 180.0


def _static_mask(seq: Sequence, mask: np.ndarray, rate: float) -> np.ndarray:
    """静止样本：角速度小、比力模长接近 g 且 0.5 s 滑窗内模长波动小。"""
    acc_norm = np.linalg.norm(seq.accelerometer.astype(np.float64), axis=1)
    gyro_norm = np.linalg.norm(seq.gyroscope.astype(np.float64), axis=1)
    w = max(int(0.5 * rate), 1)
    kernel = np.ones(w) / w
    mean = np.convolve(acc_norm, kernel, mode="same")
    var = np.convolve(acc_norm**2, kernel, mode="same") - mean**2
    return mask & (gyro_norm < 0.1) & (np.abs(acc_norm - GRAVITY) < 0.3) & (var < 0.05**2)


def _static_segments(static: np.ndarray, rate: float) -> dict:
    """静止样本的分布：数量、覆盖的时间区间与片段数（用于判断该段是否具有代表性）。"""
    idx = np.flatnonzero(static)
    if not len(idx):
        return {"samples": 0}
    return {
        "samples": int(len(idx)),
        "first_s": float(idx[0] / rate),
        "last_s": float(idx[-1] / rate),
        "segments": int(np.sum(np.diff(idx) > 1) + 1),
    }


def _check_gravity(seq: Sequence, mask: np.ndarray, rate: float, rep: ValidationReport,
                   tol_deg: float, norm_tol: float) -> None:
    """把有效比力旋到参考世界系：均值应接近 ``[0, 0, +9.81]``。

    同时计算两种统计量：静止段均值（不受运动加速度影响）与全部有效样本的均值
    （不受局部姿态误差影响，例如 SLAM 初始化阶段的倾斜）。**任一统计量通过即通过**：
    约定性错误（四元数顺序/方向、单位、符号）会让两者同时失败；两者不一致时记警告。
    """
    q = seq.orientation[mask].astype(np.float64)
    acc = seq.accelerometer[mask].astype(np.float64)
    world = quat_rotate(q, acc)
    mean_all = world.mean(axis=0)
    static = _static_mask(seq, mask, rate)
    spread = _static_segments(static, rate)
    mean_static = None
    if static.sum() >= rate:
        mean_static = quat_rotate(seq.orientation[static].astype(np.float64),
                                  seq.accelerometer[static].astype(np.float64)).mean(axis=0)

    def passes(vec: np.ndarray) -> bool:
        return _tilt(vec) <= tol_deg and abs(float(np.linalg.norm(vec)) - GRAVITY) <= norm_tol

    candidates = [("static", mean_static), ("all_valid", mean_all)]
    ok = [name for name, vec in candidates if vec is not None and passes(vec)]
    used = ok[0] if ok else ("static" if mean_static is not None else "all_valid")
    mean = mean_static if used == "static" else mean_all
    tilt, norm_err = _tilt(mean), float(np.linalg.norm(mean) - GRAVITY)

    # 30 s 分块，诊断姿态漂移
    block = int(30 * rate)
    idx = np.flatnonzero(mask)
    block_tilts = []
    for k in range(0, len(idx), block):
        if len(idx[k:k + block]) >= block // 2:
            block_tilts.append(_tilt(world[k:k + block].mean(axis=0)))
    rep.info["gravity"] = {
        "mean_world_acc": mean.tolist(),
        "mean_world_acc_all": mean_all.tolist(),
        "mean_world_acc_static": None if mean_static is None else mean_static.tolist(),
        "tilt_deg": tilt,
        "norm_error": norm_err,
        "static_samples": int(static.sum()),
        "static_span": spread,
        "used": used,
        "passed": ok,
        "max_block_tilt_deg": max(block_tilts) if block_tilts else math.nan,
    }
    if ok:
        if len(ok) == 1 and mean_static is not None:
            other = mean_all if used == "static" else mean_static
            rep.warnings.append(
                f"gravity: {used} mean passes (tilt {tilt:.2f} deg) but the "
                f"{'all_valid' if used == 'static' else 'static'} mean does not "
                f"({np.round(other, 3).tolist()}, tilt {_tilt(other):.2f} deg; static samples "
                f"{spread.get('samples', 0)} in {spread.get('first_s', float('nan')):.0f}-"
                f"{spread.get('last_s', float('nan')):.0f} s): the reference orientation may be "
                "locally inconsistent with gravity (e.g. SLAM initialisation)")
        if tilt > tol_deg / 2 or abs(norm_err) > norm_tol / 2:
            rep.warnings.append(
                f"gravity: marginal (tilt {tilt:.2f} deg, |g| error {norm_err:+.3f})")
        if block_tilts and max(block_tilts) > 2 * tol_deg:
            rep.warnings.append(f"gravity: 30 s block tilt up to {max(block_tilts):.1f} deg "
                                "(orientation drift?)")
        return

    hints = []
    norm = float(np.linalg.norm(mean))
    if abs(norm - 1.0) < 0.2:
        hints.append("|mean| ≈ 1: accelerometer may be in g instead of m/s²")
    elif norm < 3.0:
        hints.append("|mean| ≪ g: accelerometer may have gravity removed (not specific force)")
    if mean[2] < 0 and tilt > 135:
        hints.append("mean points to -z: world z axis may point down or acc sign is flipped")
    inv = quat_rotate(quat_conjugate(q), acc).mean(axis=0)
    if _tilt(inv) <= tol_deg:
        hints.append("conjugated orientation passes: quaternion may be world_to_body")
    q_from_xyzw = np.concatenate([q[:, 3:], q[:, :3]], axis=1)  # 假设存储顺序实为 xyzw
    if _tilt(quat_rotate(q_from_xyzw, acc).mean(axis=0)) <= tol_deg:
        hints.append("reordered quaternion passes: stored order may be xyzw")
    msg = (f"gravity: mean world specific force {np.round(mean, 3).tolist()} ({used}; "
           f"tilt {tilt:.2f} deg > {tol_deg} or |g| error {norm_err:+.3f} > {norm_tol})")
    if hints:
        msg += "; hints: " + "; ".join(hints)
    rep.errors.append(msg)
