"""IDOL 数据集（Zenodo 4484093，CC BY 4.0）解析器。需要可选依赖 ``pyarrow``（读取 feather）。

原始布局（Zenodo README）::

    building{1,2,3}/{known,unknown[,train]}/<k>.feather + metadata.json

每个 feather 文件的列（按名称访问；列顺序在不同文件中不同，
``building3/known/12`` 多一列 ``level_0``）：
``timestamp``（Unix 秒，100 Hz，含缺口）、``orient[WXYZ]``（Kaarta Stencil 真值姿态）、
``processedPos[XYZ]``（Stencil 真值位置，已低通平滑）、``iphoneOrient[WXYZ]``（CoreMotion 姿态）、
``iphoneAcc*``（原始加速度，单位 G）、``iphoneGyro*``（rad/s）、``iphoneMag*``、
``stencilAcc*``/``stencilGyro*``（Xsens）。

已核实的约定（README + 物理自检）：

* ``orient`` 是 **Stencil 机体**的 ``body_to_world``（wxyz）；Stencil IMU 与真值同系（README）。
* iPhone 与 Stencil 轴向大致为 ``x→−x, y→−y, z→z``（README）；由陀螺与真值角速度做常值旋转对齐
  （Wahba，121/130 条序列的稳健平均）得到 ``R_stencil_iphone``，与先验 ``Rz(180°)`` 相差 1.4°。
  本转换器**保持 IMU 在 iPhone 机体系**，
  把参考姿态换到 iPhone 机体：``q_ref = q_stencil ⊗ q_stencil_iphone``。
* iOS 原始加速度 = **−比力 / g**（静止平放时读数为 −1 G）：``f = −9.80665 · iphoneAcc``；
  陀螺符号与右手系一致，不变号。
* ``iphoneOrient`` 是 CoreMotion 的 ``body_to_world``（z 轴向上，偏航参考不同），
  作为 ``device_orientation``。
* 真值世界系（全局地图对齐后）相对重力倾斜 0.3°–8.4°（中位 1.7°）：Stencil 自带 IMU 与 iPhone IMU
  给出一致的水平残差，且地面平面拟合给出同样的倾斜。本转换器用 **Stencil 自带 IMU** 的平均比力方向
  把真值世界系调平（只校正倾斜，位置与姿态同步旋转），角度写入 ``notes``。
* 位置单位是米（水平速度中位数约 1 m/s）；时间缺口（最大 15.8 s）原样保留，不插值。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Collection, Iterator, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from . import _rig_utils as rig
from .base import RawSequence

NAME = "idol"
VERSION = "1.0.0"
LICENSE = "CC-BY-4.0 (IDOL dataset, https://zenodo.org/record/4484093)"

BUILDINGS = ("building1", "building2", "building3")
SUBSETS = ("train", "known", "unknown")
# iPhone 机体系 → Stencil 机体系的安装旋转（旋转向量，rad）；见模块文档与 docs/datasets/idol.md
R_STENCIL_IPHONE = Rotation.from_rotvec([-0.01853, 0.02801, 3.12916])
R_STENCIL_IPHONE_PRIOR = Rotation.from_euler("z", 180.0, degrees=True)

COLUMNS = {
    "time": "timestamp",
    "q": ["orientW", "orientX", "orientY", "orientZ"],
    "p": ["processedPosX", "processedPosY", "processedPosZ"],
    "q_ios": ["iphoneOrientW", "iphoneOrientX", "iphoneOrientY", "iphoneOrientZ"],
    "acc": ["iphoneAccX", "iphoneAccY", "iphoneAccZ"],
    "gyr": ["iphoneGyroX", "iphoneGyroY", "iphoneGyroZ"],
    "stencil_acc": ["stencilAccX", "stencilAccY", "stencilAccZ"],
}


def _root(source) -> Path:
    source = Path(source)
    for cand in (source, source / "IDOL"):
        if (cand / "building1").is_dir():
            return cand
    raise FileNotFoundError(f"IDOL root (with building1/) not found under {source}")


def _key(path: Path) -> tuple:
    stem = path.stem
    return (0, int(stem)) if stem.isdigit() else (1, stem)


def _listing(root: Path) -> list:
    """``[(sequence_id, building, subset, feather_path)]``。"""

    out = []
    for building in BUILDINGS:
        for subset in SUBSETS:
            base = root / building / subset
            if not base.is_dir():
                continue
            for path in sorted(base.glob("*.feather"), key=_key):
                out.append((f"{building}_{subset}_{path.stem}", building, subset, path))
    return out


def official_splits(source) -> dict:
    """IDOL 论文的划分：``building1/train`` 训练；各楼栋 ``known``/``unknown`` 为测试。

    ``test_known`` / ``test_unknown`` 汇总三栋楼；``test`` 为二者并集；
    ``test_known_building1`` / ``test_unknown_building1`` 对应论文 Table 2 的跨受试者设置。
    发布数据中 building2/3 没有训练子集，因此在本划分下它们是“未见楼栋”测试。
    """

    splits = {"train": [], "test_known": [], "test_unknown": [], "test_known_building1": [],
              "test_unknown_building1": []}
    for sequence_id, building, subset, _ in _listing(_root(source)):
        if subset == "train":
            splits["train"].append(sequence_id)
            continue
        splits[f"test_{subset}"].append(sequence_id)
        if building == "building1":
            splits[f"test_{subset}_building1"].append(sequence_id)
    splits["test"] = splits["test_known"] + splits["test_unknown"]
    return splits


def list_sequences(source) -> list:
    """全部可转换的 ``sequence_id``（``building<b>_<subset>_<k>``）；
    每条都属于 train/known/unknown 之一，因而都在官方划分中。
    """

    return [sequence_id for sequence_id, *_ in _listing(_root(source))]


all_sequence_ids = list_sequences  # 兼容别名


def _load_metadata(folder: Path) -> dict:
    path = folder / "metadata.json"
    if not path.exists():
        return {}
    with open(path) as handle:
        return json.load(handle)


def _read_feather(path: Path):
    try:
        import pandas as pd

        return pd.read_feather(path)
    except ImportError as exc:  # pragma: no cover - 依赖缺失时给出明确提示
        raise ImportError(
            "reading IDOL .feather files requires the optional dependency 'pyarrow'"
        ) from exc


def level_rotation(
    orientation_wxyz: np.ndarray, specific_force: np.ndarray, valid: np.ndarray
) -> tuple:
    """用与真值同系的 IMU 比力估计世界系倾斜，返回 ``(leveling_rotation, tilt_deg)``。"""

    rot = rig.as_rotation(orientation_wxyz[valid])
    up = rot.apply(specific_force[valid]).mean(axis=0)
    up /= np.linalg.norm(up)
    axis = np.cross(up, [0.0, 0.0, 1.0])
    angle = float(np.arctan2(np.linalg.norm(axis), up[2]))
    if np.linalg.norm(axis) < 1e-12:
        return Rotation.identity(), 0.0
    return Rotation.from_rotvec(axis / np.linalg.norm(axis) * angle), float(np.degrees(angle))


def floor_plane_tilt(position: np.ndarray) -> float:
    """位置点云的最小二乘平面 ``z = ax + by + c`` 的倾角（度），仅作独立诊断。"""

    p = position[np.isfinite(position).all(1)][::10]
    if len(p) < 10:
        return float("nan")
    design = np.c_[p[:, 0], p[:, 1], np.ones(len(p))]
    sol, *_ = np.linalg.lstsq(design, p[:, 2], rcond=None)
    return float(np.degrees(np.arctan(np.hypot(sol[0], sol[1]))))


def parse_file(path: Path, sequence_id: str, building: str = "unknown", subset: str = "unknown",
               meta: Optional[dict] = None, rel_path: Optional[str] = None) -> RawSequence:
    """解析一个 IDOL feather 文件（不含物理自检）。"""

    meta = meta or {}
    notes = []
    subject = meta.get("subjectID")
    subject_id = f"subject{int(subject):02d}" if subject is not None else "unknown"
    attrs = {
        "subject_id": subject_id,
        "device_id": "iphone8",
        "placement": "handheld",
        "group_id": subject_id,
        "position_source": (
            "LiDAR-visual-inertial SLAM (Kaarta Stencil; processedPos, low-pass smoothed)"
        ),
        "orientation_source": (
            "LiDAR-visual-inertial SLAM (Kaarta Stencil orient, rotated into the iPhone frame)"
        ),
        "device_orientation_source": "ios_coremotion",
        "body_frame": "ios_device",
        "source_files": rel_path or str(path),
        "source_license": LICENSE,
        "building": building,
        "official_subset": subset,
        "calibration_motion": str(meta.get("calibration", "unknown")),
        "imu_calibration": "none (raw iPhone 8 CoreMotion accelerometer/gyroscope)",
    }
    df = _read_feather(path)
    needed = (
        [COLUMNS["time"]]
        + COLUMNS["q"]
        + COLUMNS["p"]
        + COLUMNS["q_ios"]
        + COLUMNS["acc"]
        + COLUMNS["gyr"]
        + COLUMNS["stencil_acc"]
    )
    missing = [c for c in needed if c not in df.columns]
    if missing:
        return rig.rejected_sequence(sequence_id, f"missing columns {missing}", attrs, notes)
    extra = [c for c in df.columns if c in ("level_0",)]
    if extra:
        notes.append(f"ignored extra column(s) {extra} (pandas index artefact)")
    time = df[COLUMNS["time"]].to_numpy(np.float64)
    if not np.all(np.diff(time) > 0):
        order = np.argsort(time, kind="stable")
        df = df.iloc[order]
        time = time[order]
        keep = np.concatenate([[True], np.diff(time) > 0])
        df = df.iloc[np.flatnonzero(keep)]
        time = time[keep]
        notes.append(
            f"timestamps not strictly increasing: sorted and dropped {int((~keep).sum())} rows"
        )
    attrs["start_time_unix"] = float(time[0])
    gaps = np.diff(time)
    if (gaps > 0.05).any():
        notes.append(
            f"{int((gaps > 0.05).sum())} timestamp gap(s) > 0.05 s kept as-is "
            f"(max {gaps.max():.3f} s); no interpolation across gaps"
        )

    gyro = df[COLUMNS["gyr"]].to_numpy(np.float64)
    acc = -rig.STANDARD_GRAVITY * df[COLUMNS["acc"]].to_numpy(np.float64)
    notes.append(
        "iPhone accelerometer converted from G with iOS sign: f = -9.80665 * iphoneAcc "
        "(specific force)"
    )
    imu_valid = np.isfinite(gyro).all(1) & np.isfinite(acc).all(1)

    q_st = df[COLUMNS["q"]].to_numpy(np.float64)
    pos = df[COLUMNS["p"]].to_numpy(np.float64)
    pose_valid = (
        np.isfinite(q_st).all(1)
        & np.isfinite(pos).all(1)
        & (np.linalg.norm(np.nan_to_num(q_st), axis=1) > 0.5)
    )
    q_st = rig.normalize_quaternions(np.where(pose_valid[:, None], q_st, [1.0, 0.0, 0.0, 0.0]))

    # 1) 用 Stencil 自带 IMU（与真值同系）调平真值世界系
    s_acc = df[COLUMNS["stencil_acc"]].to_numpy(np.float64)
    lvl_mask = pose_valid & np.isfinite(s_acc).all(1)
    tilt_plane_before = floor_plane_tilt(np.where(pose_valid[:, None], pos, np.nan))
    level, tilt = level_rotation(q_st, s_acc, lvl_mask)
    q_st = rig.from_rotation(level * rig.as_rotation(q_st))
    pos = level.apply(np.nan_to_num(pos))
    pos[~pose_valid] = np.nan
    tilt_plane_after = floor_plane_tilt(pos)
    notes.append(
        f"reference world frame leveled with the Stencil IMU mean specific force: "
        f"tilt {tilt:.2f} deg "
        f"(independent floor-plane fit tilt {tilt_plane_before:.2f} -> {tilt_plane_after:.2f} deg)"
    )
    # 2) 参考姿态从 Stencil 机体换到 iPhone 机体
    q_ref = rig.from_rotation(rig.as_rotation(q_st) * R_STENCIL_IPHONE)
    rv = np.round(R_STENCIL_IPHONE.as_rotvec(), 5).tolist()
    notes.append(
        f"orientation expressed in the iPhone body frame: "
        f"q_ref = q_stencil * R_stencil_iphone, rotvec {rv} rad "
        f"({np.degrees((R_STENCIL_IPHONE_PRIOR.inv() * R_STENCIL_IPHONE).magnitude()):.2f} "
        "deg from the README axis map "
        "x->-x, y->-y, z->z; estimated by gyro/reference angular-rate alignment over the dataset)"
    )

    q_ios = df[COLUMNS["q_ios"]].to_numpy(np.float64)
    ios_ok = np.isfinite(q_ios).all(1) & (np.linalg.norm(np.nan_to_num(q_ios), axis=1) > 0.5)
    device_q = None
    if ios_ok.all():
        device_q = rig.normalize_quaternions(q_ios)
    else:
        attrs["device_orientation_source"] = "none"
        notes.append(
            f"CoreMotion orientation invalid in {int((~ios_ok).sum())} rows; "
            "device_orientation omitted"
        )
    if not imu_valid.all():
        notes.append(f"{int((~imu_valid).sum())} IMU rows non-finite and marked invalid")
    if not pose_valid.all():
        notes.append(f"{int((~pose_valid).sum())} reference rows non-finite and marked invalid")
    notes.append(
        "position in metres (verified by walking speed); building-level xy rotation offsets "
        "from the README are not applied (they only rotate about z)"
    )

    return RawSequence(
        sequence_id=sequence_id,
        imu_time=time,
        gyroscope=gyro,
        accelerometer=acc,
        pose_time=time.copy(),
        position=pos,
        orientation=q_ref,
        device_orientation=device_q,
        imu_valid=imu_valid,
        pose_valid=pose_valid,
        attrs=attrs,
        notes=notes,
    )


def iter_raw_sequences(source, only: Optional[Collection[str]] = None) -> Iterator[RawSequence]:
    root = _root(source)
    wanted = None if not only else set(only)
    meta_cache = {}
    for sequence_id, building, subset, path in _listing(root):
        if wanted is not None and sequence_id not in wanted:
            continue
        folder = path.parent
        if folder not in meta_cache:
            meta_cache[folder] = _load_metadata(folder)
        meta = meta_cache[folder].get(path.stem)
        rel = str(path.relative_to(root))
        try:
            raw = parse_file(path, sequence_id, building, subset, meta, rel)
        except Exception as exc:
            yield rig.rejected_sequence(
                sequence_id,
                f"parse error: {exc!r}",
                {"source_license": LICENSE, "source_files": rel},
            )
            continue
        if raw.rejected is None:
            if meta is None:
                raw.notes.append("warning: no metadata.json entry (subject unknown)")
            check_sequence(raw)
        yield raw


def check_sequence(raw: RawSequence) -> None:
    """物理自检；iPhone 与 Stencil 的残余时延用“角速度互相关 + 陀螺短窗误差”双证据确认后修正。"""

    stats = rig.physical_checks(raw)
    if rig.correct_time_offset_if_confirmed(raw, stats, confirm="gyro"):
        stats = rig.physical_checks(raw)
    failures, warnings = rig.evaluate_checks(stats, skip=("acceleration_consistency",))
    rig.apply_checks(raw, stats, failures, warnings)
