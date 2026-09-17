"""TLIO golden 数据集（v1.5，``golden-new-format-cc-by-nc-with-imus-v1.5``）解析器。

原始布局（官方 README，https://github.com/CathIAS/TLIO）::

    tlio_golden/
    ├── <seq_id>/
    │   ├── imu0_resampled.npy               # (N,17) float64，200 Hz
    │   ├── imu0_resampled_description.json  # 列名、行数、频率
    │   ├── calibration.json                 # 离线 IMU 标定（零偏、整流矩阵、T_Device_Imu）
    │   └── imu_samples_0.csv                # 约 1 kHz 原始 IMU（仅 212/354 条；本转换器不使用）
    ├── train_list.txt / val_list.txt / test_list.txt / all_ids.txt

``imu0_resampled.npy`` 的列（描述文件原文）：``ts_us(1)``、
``gyr_compensated_rotated_in_World(3)``、``acc_compensated_rotated_in_World(3)``、
``qxyzw_World_Device(4)``、``pos_World_Device(3)``、``vel_World(3)``。

已核实的约定（官方 dataloader ``src/dataloader/sequences_dataset.py`` + 实测）：

* IMU 是**世界系**且已补偿（零偏/比例），不是机体系。本转换器用同一行的
  ``q_World_Device`` 逆旋转回机体系：``ω_B = R_WDᵀ ω_W``、``f_B = R_WDᵀ f_W``。
* 逆旋转得到的机体系数据与 ``imu_samples_0.csv``（IMU 系 S）按 ``calibration.json`` 标定后的数据一致
  （陀螺 RMS 差约 0.01 rad/s，主要是 1 kHz→200 Hz 重采样差异），且**不需要** ``T_Device_Imu``；
  即 “Device” 系就是原始 IMU 系。
* “compensated” 与离线标定之间存在缓慢变化的差值（加速度约 0.02–0.09 m/s²），说明补偿量来自
  VIO 状态估计（在线零偏），属于参考系统的信息；这里如实标注，不做撤销。
* 世界系重力对齐、z 轴向上（``R_WD f_B`` 的全段均值 ≈ [0, 0, 9.81]）；四元数 ``xyzw`` → ``wxyz``。
* 时间戳为设备时钟微秒（非 Unix），严格 5000 µs 间隔。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Collection, Iterator, Optional

import numpy as np

from . import _rig_utils as rig
from .base import RawSequence

NAME = "tlio"
VERSION = "1.0.0"
LICENSE = "CC-BY-NC-4.0 (TLIO golden dataset v1.5, https://github.com/CathIAS/TLIO)"

EXPECTED_COLUMNS = [
    "ts_us(1)",
    "gyr_compensated_rotated_in_World(3)",
    "acc_compensated_rotated_in_World(3)",
    "qxyzw_World_Device(4)",
    "pos_World_Device(3)",
    "vel_World(3)",
]
SPLIT_FILES = {"train": "train_list.txt", "val": "val_list.txt", "test": "test_list.txt"}


def _root(source) -> Path:
    """接受 ``tlio_golden`` 目录本身或其父目录。"""

    source = Path(source)
    if (source / "train_list.txt").exists():
        return source
    if (source / "tlio_golden" / "train_list.txt").exists():
        return source / "tlio_golden"
    raise FileNotFoundError(f"TLIO golden root not found under {source}")


def _read_list(path: Path) -> list:
    with open(path) as handle:
        return [line.strip() for line in handle if line.strip()]


def official_splits(source) -> dict:
    root = _root(source)
    return {split: _read_list(root / name) for split, name in SPLIT_FILES.items()}


def list_sequences(source) -> list:
    """全部可转换的 ``sequence_id``：含 ``imu0_resampled.npy`` 的子目录名（发布版 354 条，全部在官方列表中）。"""

    root = _root(source)
    return sorted(p.name for p in root.iterdir() if p.is_dir() and (p / "imu0_resampled.npy").exists())


all_sequence_ids = list_sequences  # 兼容别名


def headset_fingerprint(calibration: dict) -> str:
    """同一物理头显的离线标定相同；用标定参数的哈希作为匿名设备编号。"""

    key = {
        "acc_bias": [round(x, 6) for x in calibration["Accelerometer"]["Bias"]["Offset"]],
        "gyr_bias": [round(x, 6) for x in calibration["Gyroscope"]["Bias"]["Offset"]],
        "t_device_imu": [round(x, 6) for x in calibration["T_Device_Imu"]["Translation"]],
    }
    digest = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:8]
    return f"headset_{digest}"


def parse_sequence(seq_dir: Path, sequence_id: Optional[str] = None) -> RawSequence:
    """解析单个序列目录（不含物理自检）。"""

    seq_dir = Path(seq_dir)
    sequence_id = sequence_id or seq_dir.name
    notes = []
    with open(seq_dir / "imu0_resampled_description.json") as handle:
        desc = json.load(handle)
    calibration = {}
    if (seq_dir / "calibration.json").exists():
        with open(seq_dir / "calibration.json") as handle:
            calibration = json.load(handle)
    device_id = headset_fingerprint(calibration) if calibration else "unknown"
    source_files = ["imu0_resampled.npy", "imu0_resampled_description.json"]
    if calibration:
        source_files.append("calibration.json")
    attrs = {
        "subject_id": "unknown",
        "device_id": device_id,
        "placement": "head",
        "group_id": device_id,
        "position_source": "VIO (headset MSCKF, TLIO golden)",
        "orientation_source": "VIO (headset MSCKF, TLIO golden)",
        "device_orientation_source": "none",
        "body_frame": "tlio_headset_imu",
        "source_files": ",".join(f"{sequence_id}/{name}" for name in source_files),
        "source_license": LICENSE,
        "imu_calibration": "source-compensated (VIO-estimated bias/scale, applied by the dataset authors)",
        "start_time_unix": float("nan"),
    }
    data = np.load(seq_dir / "imu0_resampled.npy")
    if desc.get("columns_name(width)") != EXPECTED_COLUMNS or data.ndim != 2 or data.shape[1] != 17:
        return rig.rejected_sequence(
            sequence_id, f"unexpected column layout {desc.get('columns_name(width)')} / shape {data.shape}", attrs, notes
        )
    time = data[:, 0] * 1e-6
    q_wxyz = rig.xyzw_to_wxyz(data[:, 7:11])
    finite = np.isfinite(data).all(axis=1) & (np.linalg.norm(q_wxyz, axis=1) > 0.5)
    q_safe = np.where(finite[:, None], q_wxyz, [1.0, 0.0, 0.0, 0.0])
    q_safe = rig.normalize_quaternions(q_safe)
    rot = rig.as_rotation(q_safe)
    gyro_world = np.where(finite[:, None], data[:, 1:4], 0.0)
    acc_world = np.where(finite[:, None], data[:, 4:7], 0.0)
    gyro_body = rot.inv().apply(gyro_world)
    acc_body = rot.inv().apply(acc_world)
    notes.append(
        "IMU in imu0_resampled.npy is world-frame (gyr/acc_compensated_rotated_in_World); rotated back to the "
        "body (IMU/Device) frame with the same row's q_World_Device: omega_B = R_WD^T omega_W, f_B = R_WD^T f_W"
    )
    notes.append(
        "IMU is bias/scale compensated by the source using VIO-state estimates (differs from calibration.json "
        "offline values by slowly varying offsets); compensation kept as provided, calibration.json not re-applied"
    )
    notes.append("quaternion reordered xyzw -> wxyz (q_World_Device = body_to_world); timestamps us -> s (device clock)")
    notes.append("imu_samples_0.csv (raw ~1 kHz IMU) not used")
    if not finite.all():
        notes.append(f"{int((~finite).sum())} rows with non-finite values or invalid quaternion marked invalid")
    return RawSequence(
        sequence_id=sequence_id,
        imu_time=time,
        gyroscope=gyro_body,
        accelerometer=acc_body,
        pose_time=time.copy(),
        position=np.where(finite[:, None], data[:, 11:14], np.nan),
        orientation=q_safe,
        velocity=np.where(finite[:, None], data[:, 14:17], np.nan),
        imu_valid=finite.copy(),
        pose_valid=finite.copy(),
        attrs=attrs,
        notes=notes,
    )


def iter_raw_sequences(source, only: Optional[Collection[str]] = None) -> Iterator[RawSequence]:
    root = _root(source)
    wanted = None if not only else set(only)
    for sequence_id in list_sequences(root):
        if wanted is not None and sequence_id not in wanted:
            continue
        try:
            raw = parse_sequence(root / sequence_id, sequence_id)
        except Exception as exc:  # 解析失败也要产出记录，而不是静默跳过
            yield rig.rejected_sequence(sequence_id, f"parse error: {exc!r}", {"source_license": LICENSE})
            continue
        if raw.rejected is None:
            stats = rig.physical_checks(raw)
            failures, warnings = rig.evaluate_checks(stats)
            rig.apply_checks(raw, stats, failures, warnings)
        yield raw
