"""RNIN-VIO 自采数据集（SenseINS）解析器。

原始布局（官方仓库 https://github.com/zju3dv/rnin-vio 与 ``Sense_INS_Data.md``）::

    data/
    ├── data_train/<k>/SenseINS.csv    # 241 条
    ├── data_val/<k>/SenseINS.csv      # 51 条
    ├── data_test/<k>/SenseINS.csv     # 9 条
    └── Sense_INS_Data.md

``SenseINS.csv`` 列：``times``（秒，设备时钟）、``gyro_*``/``acce_*``（Android
``TYPE_*_UNCALIBRATED``，机体系，rad/s 与 m/s²，含重力）、``vio_gyro_bias_*``/``vio_acce_bias_*``
（VIO 估计的零偏）、``vio_p_*``/``vio_q_[wxyz]``/``vio_v_*``（BVIO 位姿与速度）、
``gt_p_*``/``gt_q_[wxyz]``（“VICON 或其他设备” 的真值，缺失时填默认值）、``gv_*``（game rotation
vector，wxyz）、``rv_*``、磁力计与气压（缺失填 0）。

已核实的约定（官方 ``dataloader/dataset.py`` + 实测）：

* 四元数 ``wxyz``、``body_to_world``（``R f`` 的全段均值 ≈ [0, 0, 9.8]），世界系 z 轴向上；
* 官方判定“无 gt”：``(gt_q_w[0]==1 且 gt_q_w[100]==1) 或 gt_q_w[0]==gt_q_w[-1]``；本转换器沿用该判据
  逐序列选择参考来源（``attrs['reference_type']`` 为 ``gt`` 或 ``vio``）；
* 官方训练输入用 VIO 姿态旋转 IMU，并减去**最后一行**的 VIO 零偏——这是参考系统信息向输入的泄漏。
  本转换器**不**补偿零偏，只把 VIO 零偏写入 ``attrs['vio_bias']`` 与 ``notes``；
* 采样率并非文档所说的 200 Hz：实测 250 Hz（160 条）、约 200 Hz（127 条）、约 400 Hz（14 条），
  且时间戳不均匀；
* gt 世界系与 VIO（重力对齐）世界系之间存在最多约 3° 的倾斜，
  本转换器用 VIO 的重力方向把 gt 世界系调平（只做倾斜校正，保留 gt 偏航），角度写入 ``notes``；
* 发布数据里 ``data_train/178`` 与 ``data_train/179`` 的 ``SenseINS.csv`` 逐字节相同，后者被拒收
  （见 ``KNOWN_DUPLICATES``）。
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Collection, Iterator, Optional

import numpy as np
import pandas as pd

from . import _rig_utils as rig
from .base import RawSequence

NAME = "rnin"
VERSION = "1.1.0"
LICENSE = (
    "RNIN-VIO SenseINS data (https://github.com/zju3dv/rnin-vio); repository licensed Apache-2.0, "
    "no separate data license stated; IP belongs to SenseTime Group Ltd."
)

SPLIT_DIRS = {"train": "data_train", "val": "data_val", "test": "data_test"}
GYRO = ["gyro_x", "gyro_y", "gyro_z"]
ACCE = ["acce_x", "acce_y", "acce_z"]
VIO_BG = ["vio_gyro_bias_x", "vio_gyro_bias_y", "vio_gyro_bias_z"]
VIO_BA = ["vio_acce_bias_x", "vio_acce_bias_y", "vio_acce_bias_z"]
VIO_P = ["vio_p_x", "vio_p_y", "vio_p_z"]
VIO_Q = ["vio_q_w", "vio_q_x", "vio_q_y", "vio_q_z"]
VIO_V = ["vio_v_x", "vio_v_y", "vio_v_z"]
GT_P = ["gt_p_x", "gt_p_y", "gt_p_z"]
GT_Q = ["gt_q_w", "gt_q_x", "gt_q_y", "gt_q_z"]
GV_Q = ["gv_w", "gv_x", "gv_y", "gv_z"]
REQUIRED = ["times"] + GYRO + ACCE + VIO_BG + VIO_BA + VIO_P + VIO_Q + VIO_V + GT_P + GT_Q + GV_Q

SESSION_GAP_S = 3600.0  # 同一设备时钟上相邻序列间隔不超过 1 小时视为同一采集会话

# 发布数据中逐字节相同的序列：键被拒收，值为保留的副本及其 SenseINS.csv 的 md5。
# data_train/178 与 data_train/179 完全相同（两者都在官方 train 中，保留编号较小的）。
# tests/data/test_raw_rnin.py 扫描全部 CSV 确认此表完整。
KNOWN_DUPLICATES = {"train_179": ("train_178", "ddafa5fbc7aef4a3e8f21e47487ee31c")}


def _root(source) -> Path:
    source = Path(source)
    for cand in (source, source / "data"):
        if (cand / "data_train").is_dir():
            return cand
    raise FileNotFoundError(f"RNIN data root (with data_train/) not found under {source}")


def _dir_sort_key(name: str):
    return (0, int(name)) if name.isdigit() else (1, name)


def _listing(root: Path) -> list:
    """``[(sequence_id, split, relative_csv_path)]``，按划分与目录编号排序。"""

    out = []
    for split, dirname in SPLIT_DIRS.items():
        base = root / dirname
        if not base.is_dir():
            continue
        for name in sorted(os.listdir(base), key=_dir_sort_key):
            if (base / name / "SenseINS.csv").exists():
                out.append((f"{split}_{name}", split, f"{dirname}/{name}/SenseINS.csv"))
    return out


def official_splits(source) -> dict:
    splits = {split: [] for split in SPLIT_DIRS}
    for sequence_id, split, _ in _listing(_root(source)):
        splits[split].append(sequence_id)
    return splits


def list_sequences(source) -> list:
    """全部可转换的 ``sequence_id``（``<split>_<目录号>``）；三个官方目录之外没有其他序列。"""

    return [sequence_id for sequence_id, _, _ in _listing(_root(source))]


all_sequence_ids = list_sequences  # 兼容别名


def _first_last_time(path: Path) -> tuple:
    with open(path, "rb") as handle:
        handle.readline()
        first = float(handle.readline().split(b",", 1)[0])
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - 16384))
        lines = [ln for ln in handle.read().splitlines() if ln.strip()]
        last = float(lines[-1].split(b",", 1)[0])
        # 采样率类别：用开头若干行的中位间隔
        handle.seek(0)
        handle.readline()
        times = [float(handle.readline().split(b",", 1)[0]) for _ in range(400)]
    rate = 1.0 / float(np.median(np.diff(times)))
    return first, last, rate


@lru_cache(maxsize=4)
def session_table(root_str: str) -> dict:
    """按“采样率类别 + 设备时钟连续性”把序列聚成采集会话，返回 ``{sequence_id: session_id}``。

    SenseINS 的 ``times`` 是设备开机时钟；同一台手机连续采集的序列时间戳相近、采样率相同。
    这是启发式分组（数据集未提供受试者/设备/会话标签），用于防泄漏检查与分组划分。
    """

    root = Path(root_str)
    rows = []
    for sequence_id, _, rel in _listing(root):
        t0, t1, rate = _first_last_time(root / rel)
        rows.append((int(round(rate)), t0, t1, sequence_id))
    rows.sort()
    table, session, prev = {}, -1, None
    for rate, t0, t1, sequence_id in rows:
        if (
            prev is None
            or rate != prev[0]
            or t0 - prev[2] > SESSION_GAP_S
            or t0 < prev[1] - SESSION_GAP_S
        ):
            session += 1
            prev = (rate, t0, t1)
        else:
            prev = (rate, prev[1], max(prev[2], t1))
        table[sequence_id] = f"session{session:02d}_{rate}hz"
    return table


def file_md5(path: Path, chunk: int = 1 << 20) -> str:
    import hashlib

    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def duplicate_of(sequence_id: str, root: Path) -> Optional[str]:
    """``sequence_id`` 在重复表中、且本地文件确实与保留副本相同时，返回保留副本的名字。"""

    if sequence_id not in KNOWN_DUPLICATES:
        return None
    keep, _ = KNOWN_DUPLICATES[sequence_id]  # 表中 md5 供真实数据测试核对
    paths = {sid: rel for sid, _, rel in _listing(root)}
    if keep not in paths or sequence_id not in paths:
        return None
    return keep if file_md5(root / paths[sequence_id]) == file_md5(root / paths[keep]) else None


def _identity_rows(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """缺失数据的默认填充：单位四元数 + 零位置，或全零四元数。"""

    default = (q[:, 0] == 1.0) & (q[:, 1:] == 0.0).all(1) & (p == 0.0).all(1)
    return default | (q == 0.0).all(1)


def official_has_gt(gt_qw: np.ndarray) -> bool:
    """官方 ``SenseINSSequence.load`` 的判据（原样复刻）。"""

    k = min(100, len(gt_qw) - 1)
    return not ((gt_qw[0] == 1.0 and gt_qw[k] == 1.0) or gt_qw[0] == gt_qw[-1])


def _nonzero_median(values: np.ndarray) -> Optional[np.ndarray]:
    rows = ~(values == 0.0).all(1)
    if not rows.any():
        return None
    return np.median(values[rows], axis=0)


def _fill_invalid_quaternions(q: np.ndarray) -> tuple:
    bad = ~np.isfinite(q).all(1) | (np.linalg.norm(np.nan_to_num(q), axis=1) < 0.5)
    if bad.all():
        return None, int(bad.sum())
    if bad.any():
        good = np.flatnonzero(~bad)
        nearest = good[np.clip(np.searchsorted(good, np.arange(len(q))), 0, len(good) - 1)]
        prev = good[np.clip(np.searchsorted(good, np.arange(len(q))) - 1, 0, len(good) - 1)]
        pick = np.where(
            np.abs(prev - np.arange(len(q))) < np.abs(nearest - np.arange(len(q))), prev, nearest
        )
        q = q.copy()
        q[bad] = q[pick[bad]]
    return rig.normalize_quaternions(q), int(bad.sum())


def parse_csv(
    path: Path,
    sequence_id: str,
    split: str = "unknown",
    group_id: str = "unknown",
    rel_path: Optional[str] = None,
) -> RawSequence:
    """解析一个 ``SenseINS.csv``（不含物理自检）。"""

    notes = []
    attrs = {
        "subject_id": "unknown",
        "device_id": "unknown",
        "placement": "unknown",
        "group_id": group_id,
        "body_frame": "android_device",
        "device_orientation_source": "android_game_rotation_vector",
        "source_files": rel_path or str(path),
        "source_license": LICENSE,
        "official_split": split,
        "start_time_unix": float("nan"),
        "imu_calibration": (
            "none (Android TYPE_*_UNCALIBRATED; VIO bias estimates in attrs['vio_bias'], "
            "not applied)"
        ),
    }
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        attrs.update(
            position_source="unknown", orientation_source="unknown", reference_type="unknown"
        )
        return rig.rejected_sequence(sequence_id, f"missing columns {missing}", attrs, notes)
    time = df["times"].to_numpy(np.float64)
    if not np.all(np.diff(time) > 0):
        # 先稳定排序，再丢弃重复时间戳，数量写入 notes（发布数据中未出现，属防御性处理）
        order = np.argsort(time, kind="stable")
        df = df.iloc[order]
        time = time[order]
        keep = np.concatenate([[True], np.diff(time) > 0])
        notes.append(
            f"timestamps not strictly increasing: sorted and dropped "
            f"{int((~keep).sum())} duplicate rows"
        )
        df = df.iloc[np.flatnonzero(keep)]
        time = time[keep]
    gyro = df[GYRO].to_numpy(np.float64)
    acc = df[ACCE].to_numpy(np.float64)
    imu_valid = (
        np.isfinite(gyro).all(1) & np.isfinite(acc).all(1) & (np.linalg.norm(acc, axis=1) > 0)
    )
    if (~imu_valid).any():
        notes.append(
            f"{int((~imu_valid).sum())} IMU rows are missing (all-zero/non-finite) "
            "and marked invalid"
        )

    gt_q = df[GT_Q].to_numpy(np.float64)
    gt_p = df[GT_P].to_numpy(np.float64)
    vio_q = df[VIO_Q].to_numpy(np.float64)
    vio_p = df[VIO_P].to_numpy(np.float64)
    vio_default = (
        _identity_rows(vio_q, vio_p) | ~np.isfinite(vio_q).all(1) | ~np.isfinite(vio_p).all(1)
    )
    has_gt = official_has_gt(gt_q[:, 0])
    gt_default = _identity_rows(gt_q, gt_p) | ~np.isfinite(gt_q).all(1) | ~np.isfinite(gt_p).all(1)
    if not has_gt and (~gt_default).mean() > 0.5:
        notes.append(
            "gt_* columns contain non-default values but fail the official gt criterion "
            "(gt_q_w[0] == gt_q_w[-1]); following the official loader, VIO is used as reference"
        )

    # VIO 零偏：只记录，不补偿
    bg_all = df[VIO_BG].to_numpy(np.float64)
    ba_all = df[VIO_BA].to_numpy(np.float64)
    bg_med, ba_med = _nonzero_median(bg_all), _nonzero_median(ba_all)
    vio_bias = {
        "gyro_last": bg_all[-1].round(6).tolist(),
        "acce_last": ba_all[-1].round(6).tolist(),
        "gyro_median_nonzero": None if bg_med is None else bg_med.round(6).tolist(),
        "acce_median_nonzero": None if ba_med is None else ba_med.round(6).tolist(),
        "rows_with_bias": int((~(bg_all == 0.0).all(1)).sum()),
    }
    attrs["vio_bias"] = json.dumps(vio_bias)
    notes.append(
        "IMU kept uncompensated (raw Android uncalibrated sensors); VIO bias estimates recorded in "
        "attrs['vio_bias'] only (the official RNIN loader subtracts the last-row VIO bias = "
        "reference leak)"
    )

    velocity = None
    if has_gt:
        pose_valid = ~gt_default
        q_ref = np.where(pose_valid[:, None], gt_q, [1.0, 0.0, 0.0, 0.0])
        q_ref = rig.normalize_quaternions(q_ref)
        position = np.where(pose_valid[:, None], gt_p, np.nan)
        both = pose_valid & ~vio_default
        level, tilt, spread = rig.estimate_world_tilt(q_ref, vio_q, both)
        if level is not None:
            q_ref = rig.from_rotation(level * rig.as_rotation(q_ref))
            position = level.apply(np.nan_to_num(position))
            position[~pose_valid] = np.nan
            notes.append(
                f"gt world frame leveled with the VIO gravity direction: tilt {tilt:.2f} deg "
                f"(per-sample spread median {spread:.2f} deg); gt yaw kept"
            )
        else:
            notes.append(
                "gt world frame could not be leveled (no overlapping VIO); used as provided"
            )
        attrs.update(
            reference_type="gt",
            position_source="external_gt (SenseINS gt_p: VICON motion capture or other device)",
            orientation_source="external_gt (SenseINS gt_q: VICON motion capture or other device)",
        )
        if (~pose_valid).any():
            notes.append(
                f"{int((~pose_valid).sum())} gt rows hold default values and are marked invalid"
            )
    else:
        pose_valid = ~vio_default
        q_ref = rig.normalize_quaternions(
            np.where(pose_valid[:, None], vio_q, [1.0, 0.0, 0.0, 0.0])
        )
        position = np.where(pose_valid[:, None], vio_p, np.nan)
        velocity = np.where(pose_valid[:, None], df[VIO_V].to_numpy(np.float64), np.nan)
        attrs.update(
            reference_type="vio",
            position_source="VIO (SenseINS vio_p: BVIO, gravity aligned)",
            orientation_source="VIO (SenseINS vio_q: BVIO, gravity aligned)",
        )
        if (~pose_valid).any():
            notes.append(
                f"{int((~pose_valid).sum())} VIO rows hold default values and are marked invalid"
            )

    device_q, n_bad = _fill_invalid_quaternions(df[GV_Q].to_numpy(np.float64))
    if device_q is None:
        attrs["device_orientation_source"] = "none"
        notes.append("game rotation vector missing in all rows; device_orientation omitted")
    elif n_bad:
        notes.append(
            f"game rotation vector missing in {n_bad} rows; filled with the nearest valid sample"
        )
    notes.append(
        "quaternions are wxyz body_to_world; timestamps in seconds on the device clock (not unix)"
    )

    return RawSequence(
        sequence_id=sequence_id,
        imu_time=time,
        gyroscope=gyro,
        accelerometer=acc,
        pose_time=time.copy(),
        position=position,
        orientation=q_ref,
        velocity=velocity,
        device_orientation=device_q,
        imu_valid=imu_valid,
        pose_valid=pose_valid,
        attrs=attrs,
        notes=notes,
    )


def check_sequence(raw: RawSequence) -> None:
    """物理自检 + RNIN 专属诊断（VIO 零偏只用于解释，不写回 IMU）。"""

    stats = rig.physical_checks(raw)
    # 外部 gt 与手机 IMU 分属不同时钟，可能残留时延；VIO 与 IMU 同源同行，不做此项
    if raw.attrs.get("reference_type") == "gt" and rig.correct_time_offset_if_confirmed(
        raw, stats
    ):
        stats = rig.physical_checks(raw)
    failures, warnings = rig.evaluate_checks(stats)
    gravity_failures = [f for f in failures if f.startswith("gravity check")]
    if gravity_failures:
        bias = json.loads(raw.attrs["vio_bias"])["acce_median_nonzero"]
        if bias is not None:
            mean, n = rig.gravity_mean(
                raw.imu_time,
                raw.accelerometer - np.asarray(bias),
                raw.pose_time,
                raw.orientation,
                imu_valid=raw.imu_valid,
                pose_valid=raw.pose_valid,
            )
            err = (
                float(np.linalg.norm(mean - [0.0, 0.0, rig.STANDARD_GRAVITY]))
                if n
                else float("nan")
            )
            stats["gravity_error_vio_bias_removed"] = err
            if np.isfinite(err) and err <= rig.LIMITS["gravity_error_max"]:
                failures = [f for f in failures if f not in gravity_failures]
                warnings.append(
                    gravity_failures[0]
                    + "; explained by the VIO-estimated accelerometer bias "
                    f"(error {err:.3f} m/s^2 after removing it for this check only)"
                )
    rig.apply_checks(raw, stats, failures, warnings)


def iter_raw_sequences(source, only: Optional[Collection[str]] = None) -> Iterator[RawSequence]:
    root = _root(source)
    wanted = None if not only else set(only)
    sessions = None
    for sequence_id, split, rel in _listing(root):
        if wanted is not None and sequence_id not in wanted:
            continue
        if sessions is None:
            sessions = session_table(str(root))
        try:
            raw = parse_csv(
                root / rel, sequence_id, split, sessions.get(sequence_id, "unknown"), rel
            )
        except Exception as exc:
            yield rig.rejected_sequence(
                sequence_id,
                f"parse error: {exc!r}",
                {"source_license": LICENSE, "source_files": rel},
            )
            continue
        duplicate = duplicate_of(sequence_id, root)
        if duplicate and raw.rejected is None:
            raw.rejected = (f"byte-identical duplicate of {duplicate} (SenseINS.csv md5 match); "
                            "the lower-numbered copy is kept")
        if raw.rejected is None:
            check_sequence(raw)
        yield raw
