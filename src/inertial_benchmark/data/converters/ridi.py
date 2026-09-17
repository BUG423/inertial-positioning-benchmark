"""RIDI（Yan, Shan, Furukawa, ECCV 2018）原始数据解析（data_publish_v2）。

数据约定依据官方仓库 https://github.com/higerra/ridi_imu （``python/gen_dataset.py``）
与采集 App https://github.com/higerra/TangoIMURecorder
（``MainActivity.java`` / ``PoseIMURecorder.java``）独立实现：

* 每条序列一个目录，官方预处理结果在 ``processed/data.csv``（pandas 写出，首列为空名索引）；
* ``time`` 是 Tango 位姿时间戳（纳秒）；官方把 IMU
  （Android ``TYPE_GYROSCOPE`` / ``TYPE_ACCELEROMETER`` 等）线性插值到位姿时间戳上，
  并去掉首尾各 800 个位姿（约 4 s），约 200 Hz；
* ``pos_*``、``ori_*``：同一台 Tango 手机 ``START_OF_SERVICE → DEVICE`` 的 VIO 位姿；
  Tango 起始系重力对齐、z 向上，``DEVICE`` 系与 Android 传感器坐标系一致；
  原始 ``pose.txt`` 四元数为 xyzw，官方已换成 wxyz；
* ``rv_*``：``TYPE_GAME_ROTATION_VECTOR``
  （原始 ``orientation.txt`` 为 xyzw，官方已换成 wxyz、SLERP 到位姿时间）；
* IMU 与位姿来自**同一台设备**：物理自检中陀螺与 ``ori`` 的最佳常值旋转 < 1°，
  因此 ``ori`` 直接作为参考姿态；
* 数据集没有提供 IMU 标定参数；Android 陀螺已由系统做零偏补偿，加速度计为原始比力。
"""

from __future__ import annotations

from pathlib import Path
from typing import Collection, Dict, Iterator, List, Optional

from . import _phone_utils as pu
from .base import RawSequence

NAME = "ridi"
VERSION = "1.1"
LICENSE = "unspecified (RIDI data_publish_v2, public download; cite Yan et al., ECCV 2018)"

_SPLIT_FILES = {"train": "list_train_publish_v2.txt", "test": "list_test_publish_v2.txt"}
# RIDI 的放置名 → IPB placement 枚举
_PLACEMENTS = {"handheld": "handheld", "leg": "pocket", "bag": "bag", "body": "body"}


def _root(source: Path) -> Path:
    """允许传入 ``data_publish_v2`` 本身或其父目录。"""

    source = Path(source)
    if not (source / _SPLIT_FILES["train"]).is_file() and (source / "data_publish_v2").is_dir():
        return source / "data_publish_v2"
    return source


def _discover(source: Path) -> Dict[str, Path]:
    root = _root(source)
    if not root.is_dir():
        return {}
    return {p.name: p for p in sorted(root.iterdir()) if (p / "processed" / "data.csv").is_file()}


def _read_list(path: Path) -> List[tuple]:
    rows = []
    if not path.is_file():
        return rows
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [part.strip() for part in line.split(",")]
            rows.append((parts[0], parts[1] if len(parts) > 1 else ""))
    return rows


def list_sequences(source: Path) -> List[str]:
    """契约可选成员：本地全部含 ``processed/data.csv`` 的序列（排序）。

    包括 22 条不在官方 publish 列表中的序列。
    """

    return sorted(_discover(source))


def official_splits(source: Path) -> Dict[str, List[str]]:
    """官方 ``list_train_publish_v2`` / ``list_test_publish_v2``，过滤为本地存在的序列。

    无官方 val。
    """

    root = _root(source)
    available = set(_discover(source))
    return {
        key: [name for name, _ in _read_list(root / fn) if name in available]
        for key, fn in _SPLIT_FILES.items()
    }


EXTRA_SPLIT_NOTES = {
    "test_unseen_subject": (
        "local sequences outside list_train/list_test_publish_v2 whose subject never appears in "
        "those lists (ruixuan, shali); superset of list_crosssubject.txt; never merged into train"
    ),
    "test_unlisted_seen_subject": (
        "local sequences outside list_train/list_test_publish_v2 whose subject does appear in "
        "those lists (dan, hang, hao, huayi); seen-subject setting like the official test; "
        "never merged into train"
    ),
}


def extra_splits(source: Path) -> Dict[str, List[str]]:
    """契约可选成员：不在官方 publish 列表中的本地序列。

    按受试者是否出现在官方列表中分成两个附加测试子集（见 ``EXTRA_SPLIT_NOTES``）。
    """

    root = _root(source)
    listed = {name for fn in _SPLIT_FILES.values() for name, _ in _read_list(root / fn)}
    seen_subjects = {name.split("_")[0] for name in listed}
    out: Dict[str, List[str]] = {key: [] for key in EXTRA_SPLIT_NOTES}
    for name in sorted(_discover(source)):
        if name in listed:
            continue
        key = ("test_unlisted_seen_subject" if name.split("_")[0] in seen_subjects
               else "test_unseen_subject")
        out[key].append(name)
    return out


def placement_of(name: str, listed: Dict[str, str]) -> str:
    """优先用官方列表第二列，否则从序列名中的放置关键词解析；返回 RIDI 原始放置名或空串。"""

    if listed.get(name):
        return listed[name]
    for token in name.split("_")[1:]:
        token = token.rstrip("0123456789")
        if token in _PLACEMENTS:
            return token
    return ""


def iter_raw_sequences(
    source: Path, only: Optional[Collection[str]] = None
) -> Iterator[RawSequence]:
    root = _root(source)
    found = _discover(source)
    listed: Dict[str, str] = {}
    for fn in _SPLIT_FILES.values():
        listed.update(dict(_read_list(root / fn)))
    names = sorted(found) if only is None else list(only)
    for name in names:
        if name not in found:
            yield pu.rejected_sequence(name, f"sequence folder not found under {root}")
            continue
        try:
            raw = load_sequence(found[name], placement=placement_of(name, listed), root=root)
        except (OSError, KeyError, ValueError) as exc:
            yield pu.rejected_sequence(name, f"parse error: {type(exc).__name__}: {exc}")
            continue
        yield pu.finalize_sequence(raw)


def load_sequence(folder: Path, placement: str = "", root: Optional[Path] = None) -> RawSequence:
    """解析一个 RIDI 序列目录（不做物理自检）。"""

    folder = Path(folder)
    name = folder.name
    csv_path = folder / "processed" / "data.csv"
    data = pu.read_ridi_processed_csv(csv_path)
    time = data["time"] / 1e9
    keep = pu.monotonic_mask(time)
    notes = [
        "source = processed/data.csv (official gen_dataset.py: IMU linearly interpolated onto "
        "Tango pose timestamps)",
        "time: nanoseconds -> seconds (Android/Tango boot clock, not unix time)",
        "position/orientation = Tango VIO pose of the same device (START_OF_SERVICE, gravity "
        "aligned, z up); "
        "ori columns already wxyz",
        "device_orientation = rv columns (Android game rotation vector, wxyz)",
        "no IMU calibration published; gyroscope is Android-calibrated TYPE_GYROSCOPE, "
        "accelerometer is raw specific force",
    ]
    dropped = int((~keep).sum())
    if dropped:
        notes.append(f"dropped {dropped} duplicate/non-increasing timestamps")
    time = time[keep]
    subject = name.split("_")[0]
    attrs = {
        "subject_id": subject,
        "device_id": "tango_phone",
        "placement": _PLACEMENTS.get(placement, "unknown"),
        "group_id": subject,
        "position_source": "tango_vio_same_device",
        "orientation_source": "tango_vio_same_device",
        "device_orientation_source": "android_game_rotation_vector",
        "body_frame": "android_device",
        "source_files": pu.relative_source(csv_path, root),
        "start_time_unix": float("nan"),
        "ridi_placement": placement or "unknown",
    }
    return RawSequence(
        sequence_id=name,
        imu_time=time,
        gyroscope=data["gyro"][keep],
        accelerometer=data["acce"][keep],
        pose_time=time.copy(),
        position=data["pos"][keep],
        orientation=pu.quat_make_continuous(pu.quat_normalize(data["ori"][keep])),
        device_orientation=pu.quat_make_continuous(pu.quat_normalize(data["rv"][keep])),
        attrs=attrs,
        notes=notes,
    )
