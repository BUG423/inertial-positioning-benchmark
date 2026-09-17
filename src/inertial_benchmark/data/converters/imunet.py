"""IMUNet 数据集（Zeinali et al., IEEE TIM 2024）原始数据解析。

数据约定依据官方仓库 https://github.com/BehnamZeinali/IMUNet
（``Datasets/proposed/read_data_s10.py``、``read_data_tango.py``）
与采集 App https://github.com/BehnamZeinali/IMUNet_Android （``MainActivity.java``）
独立实现，并用物理自检逐设备确认：

* 目录名 ``<Indoor|Outdoor>_Subject_<k>_<设备>_<n>``（部分目录拼成 ``Subjetc``）；
  设备为 ``Tango``（RIDI 方法）或 ``S10`` / ``S21`` / ``Xiaomi``（ARCore 方法）；
  数据在 ``processed/data.csv``，格式同 RIDI；
* ``time`` 为位姿时间戳（纳秒）；IMU 被官方线性插值到位姿时间上；
* **Tango 设备**：与 RIDI 相同，``ori`` 是同一台设备 ``START_OF_SERVICE → DEVICE``
  的 VIO 姿态（z 向上），直接作为参考；
* **ARCore 设备**：App 记录 ``camera.getPose()``
  （``Pose.getRotationQuaternion()`` 为 xyzw，官方已换成 wxyz）。
  该姿态是**相机系 → ARCore 世界系（y 向上）**。
  官方只把位置转到 z 向上 ``(x, y, z) → (x, −z, y)``，四元数未转换。
  本转换器使用 ``q_wb = R_x(+90°) ⊗ ori ⊗ R_z(+90°)``：
  左乘把世界系转成 z 向上（与官方位置变换一致）；
  右乘是相机系与 Android 传感器系之间的常值旋转（后摄、图像读出方向横屏），
  由陀螺/姿态手眼对齐在全部 ARCore 序列上估计得到（残差 < 2°）。
  四元数**不是**共轭关系（共轭假设下手眼残差大一个数量级，重力落到 −x 轴）；
* ARCore 位姿原始约 30 Hz，官方用样条时间戳 + SLERP/线性插值上采样到约 200 Hz；
* ``rv_*`` 为 Android game rotation vector（wxyz），作为 ``device_orientation``；
  数据集未提供 IMU 标定参数。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Collection, Dict, Iterator, List, Optional

import numpy as np

from . import _phone_utils as pu
from .base import RawSequence

NAME = "imunet"
VERSION = "1.0"
LICENSE = (
    "unspecified (IMUNet_dataset public Google Drive release; cite Zeinali et al., IEEE TIM 2024)"
)

ARCORE_DEVICES = ("S10", "S21", "Xiaomi")
DEVICE_IDS = {
    "Tango": "tango_phone",
    "S10": "samsung_galaxy_s10",
    "S21": "samsung_galaxy_s21",
    "Xiaomi": "xiaomi",
}
# 相机系（ARCore Camera.getPose）→ Android 传感器系的常值旋转：q_sb，满足 ω_cam = R_sb ω_body
Q_CAMERA_FROM_BODY = pu.quat_from_axis_angle([0.0, 0.0, 1.0], np.pi / 2)
MAX_EXTRINSIC_RESIDUAL_DEG = 5.0  # 应用固定外参后，手眼对齐残余旋转的上限
MAX_ACCEL_SCALE_DEVIATION = 0.05  # 原始加速度计与 Android 融合 (grav + linacce) 的比例偏差上限

# 发布数据中内容逐字节相同的序列：键被拒收，值为保留的副本及其 data.csv 的 md5。
# Outdoor_Subjetc_1_S10_13 在 list_train、Outdoor_Subjetc_1_S10_16 在 list_test，
# 保留 test 副本以免 train/test 泄漏。
# tests/data/test_raw_imunet.py 会扫描全部文件确认此表完整。
KNOWN_DUPLICATES = {
    "Outdoor_Subjetc_1_S10_13": ("Outdoor_Subjetc_1_S10_16", "8e2e9f096ca302f31edf12d57052ad49"),
}

_NAME_RE = re.compile(
    r"^(?P<env>Indoor|Outdoor)_Subje(?:ct|tc)_(?P<subject>\d+)_(?P<device>[A-Za-z0-9]+)_(?P<index>\d+)$"
)
_SPLIT_FILES = {"train": "list_train.txt", "test": "list_test.txt"}


def _root(source: Path) -> Path:
    source = Path(source)
    if not (source / _SPLIT_FILES["train"]).is_file() and (source / "IMUNet_dataset").is_dir():
        return source / "IMUNet_dataset"
    return source


def _discover(source: Path) -> Dict[str, Path]:
    root = _root(source)
    if not root.is_dir():
        return {}
    return {p.name: p for p in sorted(root.iterdir()) if (p / "processed" / "data.csv").is_file()}


def parse_name(name: str) -> dict:
    """解析目录名，返回 ``env/subject/device/index``；不匹配时抛 ``ValueError``。"""

    match = _NAME_RE.match(name)
    if not match:
        raise ValueError(f"unrecognised IMUNet sequence name {name!r}")
    info = match.groupdict()
    if info["device"] not in DEVICE_IDS:
        raise ValueError(f"unknown IMUNet device {info['device']!r} in {name!r}")
    return info


def list_sequences(source: Path) -> List[str]:
    """契约可选成员：本地全部含 ``processed/data.csv`` 的序列（排序）。

    发布数据的 126 条均在官方列表中。
    """

    return sorted(_discover(source))


def official_splits(source: Path) -> Dict[str, List[str]]:
    """数据包自带的 ``list_train.txt`` / ``list_test.txt``（90/36），过滤为本地存在的序列。

    无官方 val。

    注意：GitHub 仓库 ``Datasets/proposed`` 下的同名列表是旧命名（``behnam_*``），与发布数据不对应，
    不使用。
    """

    root = _root(source)
    available = set(_discover(source))
    out = {}
    for key, fn in _SPLIT_FILES.items():
        path = root / fn
        names = []
        if path.is_file():
            with open(path, "r", encoding="utf-8") as handle:
                names = [
                    line.strip() for line in handle if line.strip() and not line.startswith("#")
                ]
        out[key] = [name for name in names if name in available]
    return out


def iter_raw_sequences(
    source: Path, only: Optional[Collection[str]] = None
) -> Iterator[RawSequence]:
    root = _root(source)
    found = _discover(source)
    names = sorted(found) if only is None else list(only)
    for name in names:
        if name not in found:
            yield pu.rejected_sequence(name, f"sequence folder not found under {root}")
            continue
        duplicate = duplicate_of(name, found)
        try:
            raw = load_sequence(found[name], root=root)
        except (OSError, KeyError, ValueError) as exc:
            yield pu.rejected_sequence(name, f"parse error: {type(exc).__name__}: {exc}")
            continue
        if duplicate:
            raw.rejected = (
                f"byte-identical duplicate of {duplicate} (data.csv md5 match); the copy listed "
                "in the official "
                "test split is kept to avoid train/test leakage"
            )
        yield pu.finalize_sequence(raw)


def file_md5(path: Path, chunk: int = 1 << 20) -> str:
    import hashlib

    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def duplicate_of(name: str, found: Dict[str, Path]) -> Optional[str]:
    """若 ``name`` 在已知重复表中且本地文件确实与保留副本相同，返回保留副本名。"""

    if name not in KNOWN_DUPLICATES:
        return None
    keep, _ = KNOWN_DUPLICATES[name]  # 表中 md5 供真实数据测试核对
    if keep not in found:
        return None
    this = file_md5(found[name] / "processed" / "data.csv")
    other = file_md5(found[keep] / "processed" / "data.csv")
    return keep if this == other else None


def reference_orientation(device: str, ori_wxyz: np.ndarray) -> np.ndarray:
    """把 data.csv 的 ``ori`` 转成 Android 机体系 → z 向上世界系（wxyz）。"""

    ori = pu.quat_normalize(ori_wxyz)
    if device in ARCORE_DEVICES:
        return pu.quat_mul(pu.quat_mul(pu.Q_ZUP_FROM_YUP, ori), Q_CAMERA_FROM_BODY)
    return ori


def load_sequence(folder: Path, root: Optional[Path] = None) -> RawSequence:
    """解析一个 IMUNet 序列目录（不做物理自检，但会检查外参与加速度计比例并在必要时拒收）。"""

    folder = Path(folder)
    name = folder.name
    info = parse_name(name)
    device = info["device"]
    csv_path = folder / "processed" / "data.csv"
    data = pu.read_ridi_processed_csv(csv_path)
    time = data["time"] / 1e9
    keep = pu.monotonic_mask(time)
    time = time[keep]
    gyro, acc = data["gyro"][keep], data["acce"][keep]
    orientation = pu.quat_make_continuous(reference_orientation(device, data["ori"][keep]))
    arcore = device in ARCORE_DEVICES

    notes = [
        "source = processed/data.csv (official preprocessing: IMU linearly interpolated onto pose "
        "timestamps)",
        "time: nanoseconds -> seconds (Android boot clock, not unix time)",
        "device_orientation = rv columns (Android game rotation vector, wxyz)",
        "no IMU calibration published; gyroscope is Android-calibrated TYPE_GYROSCOPE, "
        "accelerometer is raw specific force",
    ]
    if arcore:
        notes += [
            "reference = ARCore camera.getPose() (~30 Hz, upsampled by the official script with "
            "spline timestamps + SLERP)",
            "orientation = Rx(+90deg) * ori * Rz(+90deg): y-up ARCore world -> z-up (same map the "
            "official script applied "
            "to positions: (x, y, z) -> (x, -z, y)); camera frame -> Android sensor frame "
            "(constant, estimated by hand-eye "
            "alignment over all ARCore sequences); quaternions are NOT conjugated",
            "position = official pos columns (already z-up); camera optical centre, lever arm to "
            "IMU ignored",
        ]
    else:
        notes += [
            "reference = Tango VIO pose of the same device (START_OF_SERVICE, gravity aligned, z "
            "up); ori used as-is",
        ]
    dropped = int((~keep).sum())
    if dropped:
        notes.append(f"dropped {dropped} duplicate/non-increasing timestamps")

    rejected = None
    # 固定外参是否与数据一致（约定错误的早期信号）
    q_res, rms, count = pu.estimate_body_extrinsic(time, gyro, time, orientation)
    residual = float(np.degrees(pu.quat_angle(q_res)))
    notes.append(
        f"hand-eye residual rotation after convention {residual:.2f} deg (median rate residual "
        f"{rms:.3f} rad/s, n={count})"
    )
    if not residual <= MAX_EXTRINSIC_RESIDUAL_DEG:
        rejected = f"gyro/reference frame mismatch: residual rotation {residual:.1f} deg"
    # 原始加速度计相对 Android 融合输出 (grav + linacce) 的比例，独立于参考姿态
    fused = data["grav"][keep] + data["linacce"][keep]
    scale = float(np.sum(acc * fused) / np.sum(fused * fused))
    notes.append(f"accelerometer / (android gravity + linear_acceleration) scale = {scale:.4f}")
    if abs(scale - 1.0) > MAX_ACCEL_SCALE_DEVIATION and rejected is None:
        rejected = (
            f"accelerometer scale error: raw TYPE_ACCELEROMETER is {scale:.3f} x Android's own "
            "gravity+linear "
            "acceleration (no official calibration to correct it)"
        )

    subject = f"subject{int(info['subject'])}"
    attrs = {
        "subject_id": subject,
        "device_id": DEVICE_IDS[device],
        "placement": "unknown",
        "group_id": subject,
        "position_source": "arcore_vio_same_device" if arcore else "tango_vio_same_device",
        "orientation_source": "arcore_camera_pose_to_android_body"
        if arcore
        else "tango_vio_same_device",
        "device_orientation_source": "android_game_rotation_vector",
        "body_frame": "android_device",
        "source_files": pu.relative_source(csv_path, root),
        "start_time_unix": float("nan"),
        "imunet_environment": info["env"].lower(),
    }
    return RawSequence(
        sequence_id=name,
        imu_time=time,
        gyroscope=gyro,
        accelerometer=acc,
        pose_time=time.copy(),
        position=data["pos"][keep],
        orientation=orientation,
        device_orientation=pu.quat_make_continuous(pu.quat_normalize(data["rv"][keep])),
        attrs=attrs,
        notes=notes,
        rejected=rejected,
    )
