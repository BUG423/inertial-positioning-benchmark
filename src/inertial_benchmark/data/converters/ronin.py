"""RoNIN（Herath, Yan, Furukawa, ICRA 2020）原始数据解析。

数据约定依据数据集 README.txt 与官方仓库 https://github.com/Sachini/ronin
（``source/data_glob_speed.py``、``source/data_utils.py``，提交 805b7f0）独立实现：

* 每条序列一个目录 ``<subject>_<k>/``，含 ``data.hdf5`` 与 ``info.json``；
* ``synced/*`` 是手机（IMU 设备）数据，200 Hz，时间 ``synced/time`` 为手机系统时间（秒）；
  ``pose/*`` 与 ``synced/time`` 逐行对应（README："timestamp for data is synced/time"）；
* 陀螺：``synced/gyro_uncalib − imu_init_gyro_bias``；加速度：``imu_acce_scale ⊙ (synced/acce − imu_acce_bias)``；
* ``pose/tango_pos`` 是**胸前 Tango 设备**（另一个刚体）的 VIO 位置，世界系为 Tango 起始系（重力对齐，z 向上）；
  ``pose/tango_ori`` 是 Tango 设备姿态，**不是手机姿态**，因此不能作为 IMU 机体系的参考姿态；
* 手机在 Tango 世界系中的姿态由官方公式得到：
  ``q_wb(t) = tango_ori[0] ⊗ start_calibration ⊗ src[0]* ⊗ src(t)``，
  其中 ``src`` 为所选姿态源（game_rv / EKF / 陀螺积分），``start_calibration`` 是起始校准段内手机→Tango 的旋转；
* 姿态源选择与官方训练模式一致（``select_orientation_source``，``max_ori_error=20``，``grv_only=False``）：
  game_rv 末端对齐误差 < 20° 时用 game_rv，否则取 ``[gyro_integration, game_rv, ekf]`` 中误差最小者；
  官方测试模式只用 game_rv，对应 IPB 的 ``imu/orientation``（原始 game_rv，与参考世界系差常值偏航）；
* ``start_frame`` 之前是预校准与同步段，丢弃（官方同样从 ``start_frame`` 开始使用）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Collection, Dict, Iterator, List, Optional

import numpy as np

from . import _phone_utils as pu
from .base import RawSequence

NAME = "ronin"
VERSION = "1.0"
LICENSE = "RoNIN Data License (non-commercial scientific research only; see LICENSE.txt, https://doi.org/10.20383/102.0543)"

MAX_GRV_ERROR_DEG = 20.0  # 官方 --max_ori_error 默认值

# 官方仓库 lists/（提交 805b7f0）。数据只公开约 50%，本地不存在的序列在 official_splits 中被过滤。
_OFFICIAL_LISTS = {
    "train": """
        a001_1 a001_3 a002_1 a002_2 a003_1 a003_2 a004_2 a005_1 a007_2 a007_3 a009_3 a010_1 a010_3 a011_1
        a011_3 a012_1 a012_3 a013_1 a013_3 a014_1 a015_1 a015_3 a016_3 a017_1 a017_3 a018_1 a018_2 a018_3
        a020_1 a020_3 a021_1 a022_1 a022_3 a023_2 a023_3 a025_1 a025_3 a026_3 a027_1 a027_3 a028_3 a030_1
        a031_1 a031_3 a033_1 a033_3 a034_1 a034_2 a034_3 a035_1 a036_1 a036_3 a037_3 a038_3 a040_2 a040_3
        a043_1 a043_3 a044_1 a044_3 a045_1 a045_3 a046_1 a046_3 a047_3 a000_1 a000_10 a000_2 a000_3 a000_4
        a000_5 a059_1 a059_3
    """,
    "val": """
        a000_6 a000_8 a000_9 a009_2 a014_3 a015_2 a021_3 a026_1 a028_1 a038_1 a039_1 a039_2 a045_2 a047_1
        a056_1 a056_3
    """,
    "test_seen": """
        a001_2 a003_3 a004_3 a005_3 a009_1 a010_2 a011_2 a012_2 a013_2 a014_2 a016_1 a017_2 a020_2 a021_2
        a022_2 a023_1 a025_2 a026_2 a027_2 a030_3 a031_2 a033_2 a035_3 a036_2 a037_1 a038_2 a044_2 a046_2
        a047_2 a000_7 a000_11 a059_2
    """,
    "test_unseen": """
        a006_2 a019_3 a024_1 a024_3 a029_1 a029_2 a032_1 a032_3 a042_2 a049_1 a049_2 a049_3 a050_1 a050_3
        a051_1 a051_2 a051_3 a052_2 a053_1 a053_2 a053_3 a054_1 a054_2 a054_3 a055_2 a055_3 a057_1 a057_2
        a057_3 a058_1 a058_2 a058_3
    """,
}


def _official_lists() -> Dict[str, List[str]]:
    lists = {key: value.split() for key, value in _OFFICIAL_LISTS.items()}
    lists["test"] = lists["test_seen"] + lists["test_unseen"]
    return lists


def _discover(source: Path) -> Dict[str, Path]:
    """查找含 ``info.json`` 与 ``data.hdf5`` 的序列目录（``source/<seq>`` 或 ``source/<子集>/<seq>``）。"""

    source = Path(source)
    found: Dict[str, Path] = {}
    candidates = [source] + sorted(p for p in source.iterdir() if p.is_dir()) if source.is_dir() else []
    for folder in candidates:
        for seq in sorted(p for p in folder.iterdir() if p.is_dir()):
            if (seq / "info.json").is_file() and (seq / "data.hdf5").is_file():
                if seq.name in found and found[seq.name] != seq:
                    raise ValueError(f"duplicate RoNIN sequence {seq.name}: {found[seq.name]} and {seq}")
                found[seq.name] = seq
    return found


def list_sequences(source: Path) -> List[str]:
    """契约可选成员：本地可转换的全部 sequence_id（排序，含不在官方划分中的序列；本地 152 条均在官方列表中）。"""

    return sorted(_discover(source))


def official_splits(source: Path) -> Dict[str, List[str]]:
    """官方 train/val/test（= test_seen + test_unseen），过滤为本地存在的序列。"""

    available = set(_discover(source))
    return {key: [name for name in names if name in available] for key, names in _official_lists().items()}


def iter_raw_sequences(source: Path, only: Optional[Collection[str]] = None) -> Iterator[RawSequence]:
    found = _discover(source)
    names = sorted(found) if only is None else list(only)
    for name in names:
        if name not in found:
            yield pu.rejected_sequence(name, f"sequence folder not found under {source}")
            continue
        try:
            raw = load_sequence(found[name], root=Path(source))
        except (OSError, KeyError, ValueError) as exc:
            yield pu.rejected_sequence(name, f"parse error: {type(exc).__name__}: {exc}")
            continue
        yield pu.finalize_sequence(raw)


def select_orientation_source(info: dict, available: Dict[str, np.ndarray]) -> tuple:
    """官方训练模式的姿态源选择，返回 ``(名称, 误差度数)``。"""

    errors = {
        "gyro_integration": float(info["gyro_integration_error"]),
        "game_rv": float(info["grv_ori_error"]),
        "ekf": float(info["ekf_ori_error"]),
    }
    if errors["game_rv"] < MAX_GRV_ERROR_DEG:
        return "game_rv", errors["game_rv"]
    names = ["gyro_integration", "game_rv"] + (["ekf"] if "ekf" in available else [])
    best = min(names, key=lambda key: errors[key])  # 并列时取靠前者，与 np.argmin 一致
    return best, errors[best]


def load_sequence(folder: Path, root: Optional[Path] = None) -> RawSequence:
    """解析一个 RoNIN 序列目录（不做物理自检）。"""

    import h5py

    folder = Path(folder)
    name = folder.name
    with open(folder / "info.json", "r", encoding="utf-8") as handle:
        info = json.load(handle)
    with h5py.File(folder / "data.hdf5", "r") as handle:
        time = np.asarray(handle["synced/time"], dtype=np.float64)
        gyro_uncalib = np.asarray(handle["synced/gyro_uncalib"], dtype=np.float64)
        acce = np.asarray(handle["synced/acce"], dtype=np.float64)
        game_rv = np.asarray(handle["synced/game_rv"], dtype=np.float64)
        tango_pos = np.asarray(handle["pose/tango_pos"], dtype=np.float64)
        tango_ori = np.asarray(handle["pose/tango_ori"], dtype=np.float64)
        ekf = np.asarray(handle["pose/ekf_ori"], dtype=np.float64) if "pose/ekf_ori" in handle else None
    n = len(time)
    for label, arr in [("gyro_uncalib", gyro_uncalib), ("acce", acce), ("game_rv", game_rv),
                       ("tango_pos", tango_pos), ("tango_ori", tango_ori)]:
        if len(arr) != n:
            raise ValueError(f"{label} has {len(arr)} rows, synced/time has {n}")

    gyro_bias = np.asarray(info["imu_init_gyro_bias"], dtype=np.float64)
    acce_bias = np.asarray(info["imu_acce_bias"], dtype=np.float64)
    acce_scale = np.asarray(info["imu_acce_scale"], dtype=np.float64)
    gyro = gyro_uncalib - gyro_bias
    acc = acce_scale * (acce - acce_bias)

    game_rv = pu.quat_normalize(game_rv)
    available = {"game_rv": game_rv}
    if ekf is not None:
        available["ekf"] = ekf
    source_name, source_error = select_orientation_source(info, available)
    if source_name == "gyro_integration":
        # 官方从第 0 帧、以 game_rv[0] 为初值积分去零偏陀螺；此处用零阶保持指数映射积分（与官方欧拉积分差异可忽略）
        src = pu.quat_mul(game_rv[0], pu.gyro_relative_rotations(time, gyro))
    else:
        src = pu.quat_normalize(available[source_name])

    start_calib = pu.quat_normalize(np.asarray(info["start_calibration"], dtype=np.float64))
    init_rotor = pu.quat_mul(pu.quat_mul(pu.quat_normalize(tango_ori[0]), start_calib), pu.quat_conj(src[0]))
    reference = pu.quat_mul(init_rotor, src)
    grv_rotor = pu.quat_mul(pu.quat_mul(pu.quat_normalize(tango_ori[0]), start_calib), pu.quat_conj(game_rv[0]))

    start = int(info.get("start_frame", 0))
    if not 0 <= start < n - 1:
        raise ValueError(f"start_frame {start} outside [0, {n - 1})")
    sl = slice(start, n)

    subject = name.split("_")[0]
    notes = [
        f"gyroscope = synced/gyro_uncalib - imu_init_gyro_bias {np.round(gyro_bias, 6).tolist()} (official)",
        f"accelerometer = imu_acce_scale {np.round(acce_scale, 6).tolist()} * (synced/acce - imu_acce_bias "
        f"{np.round(acce_bias, 6).tolist()}) (official)",
        f"dropped {start} pre-calibration frames before start_frame (official)",
        "pose/orientation is the PHONE attitude, not pose/tango_ori (tango_ori belongs to the body-mounted Tango device)",
        f"reference orientation source = {source_name} (official selection rule, max_ori_error={MAX_GRV_ERROR_DEG:g} deg; "
        f"end-alignment errors: game_rv={info['grv_ori_error']:.2f}, ekf={info['ekf_ori_error']:.2f}, "
        f"gyro_integration={info['gyro_integration_error']:.2f} deg)",
        "reference orientation aligned to Tango world at frame 0: tango_ori[0] * start_calibration * src[0]^-1 * src(t)",
        f"game_rv alignment rotor tilt {np.degrees(pu.tilt_of_quat(grv_rotor)):.3f} deg "
        "(device_orientation = raw game_rv; differs from reference world by this ~yaw-only rotor)",
        "position = pose/tango_pos (Tango VIO of the body-mounted device, not the phone position)",
    ]
    if source_name == "gyro_integration":
        notes.append(
            "WARNING: gyro_integration has no gravity correction, so its tilt drifts; the physics check decides "
            "whether this reference is usable (game_rv and ekf both exceed the official end-alignment tolerance)"
        )
    attrs = {
        "subject_id": subject,
        "device_id": str(info.get("device", "unknown")),
        "placement": "mixed",
        "group_id": subject,
        "position_source": "tango_vio_body_mounted",
        "orientation_source": f"{source_name}_aligned_to_tango_start",
        "device_orientation_source": "android_game_rotation_vector",
        "body_frame": "android_device",
        "source_files": ";".join(pu.relative_source(folder / fn, root) for fn in ("data.hdf5", "info.json")),
        "start_time_unix": float("nan"),
        "ronin_date": str(info.get("date", "unknown")),
        "ronin_orientation_error_deg": float(source_error),
        "ronin_grv_error_deg": float(info["grv_ori_error"]),
    }
    return RawSequence(
        sequence_id=name,
        imu_time=time[sl],
        gyroscope=gyro[sl],
        accelerometer=acc[sl],
        pose_time=time[sl].copy(),
        position=tango_pos[sl],
        orientation=pu.quat_make_continuous(reference[sl]),
        device_orientation=pu.quat_make_continuous(game_rv[sl]),
        attrs=attrs,
        notes=notes,
    )
