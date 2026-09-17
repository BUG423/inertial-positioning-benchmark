"""OxIOD（Chen et al., "OxIOD: The Dataset for Deep Inertial Odometry", 2018）原始数据解析。

依据数据集 ``ReadMe.txt`` 与各场景 ``Train.txt`` / ``Test.txt`` 独立实现，约定均由物理自检确认：

* 只使用每个场景下的 ``raw/`` 目录；
  ``syn/`` 的时间戳被导出成科学计数法（如 ``1.50E+11``），已损坏，不使用；
* ``imu<k>.csv``（iPhone，CoreMotion，约 100 Hz）16 列：
  时间（unix 秒，0.01 s 分辨率）、attitude roll/pitch/yaw（rad）、
  rotation rate xyz（rad/s，CoreMotion 已去零偏）、gravity xyz（G）、
  user acceleration xyz（G）、磁场 xyz（μT）；
* iOS 加速度符号与 Android 相反：静止屏幕朝上时 gravity ≈ (0, 0, −1)。
  本转换器的比力为 ``−(gravity + user_acc) · 9.80665``
  （物理自检：取 + 号时世界系重力均值为 −9.8，取 − 号为 +9.8）；
* 设备姿态 ``q = R_z(yaw) ⊗ R_x(pitch) ⊗ R_y(roll)``（CoreMotion 参考系 z 竖直向上、偏航任意），
  与 iOS gravity 向量方向误差 0°、与陀螺手眼残差 < 2°；
* ``vi<k>.csv``（部分目录名为 ``hand<k>.csv``）为 Vicon：
  时间（unix 纳秒）、帧号、位置 xyz（m）、四元数 xyzw；
  Vicon 世界系 z 向上（物理自检确认）；
  Vicon 刚体系与手机机体系之间有**未知常值旋转**
  （标记点每次重新粘贴，各序列约绕 z 轴 170–180° 且有几度倾斜），
  本转换器逐序列做陀螺/Vicon 角速度手眼对齐估计 ``q_sb`` 并应用；
* 手机时钟与 Vicon 时钟不同步，用线性时钟模型（偏移 + 可选速率差）统一到手机时钟：
  先用角速度模长互相关（±5 s）粗估，再在外参下用三轴互相关（±0.3 s）细化，
  最后按 60 s 分段细化并线性拟合速率差（见 :func:`synchronise`）；
* Vicon 野值（标记识别导致的 180° 单帧翻转、短时抖动、位置跳变）
  按隐含角速度 > 600°/s 或速度 > 10 m/s 检出，前后 0.1 s 标为 ``pose_valid = False``；
* IMU 行偶有相邻两行顺序颠倒（时间戳 ``.37, .39, .38, .40``），按时间稳定排序后去掉重复时间戳；
* ``multi devices/nexus 5`` 为 Android 长表格式
  （毫秒时间、传感器类型码 1=加速度计/4=陀螺/2=磁力计、xyz），
  加速度计已是比力；陀螺按时间线性插值到加速度计时间戳；无设备姿态；
* ``large scale``（Tango 真值、IMU 与真值分目录）与 ``test/`` 目录（无 raw/syn 区分）
  不在本版本转换范围内。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Collection, Dict, Iterator, List, Optional, Tuple

import numpy as np

from . import _phone_utils as pu
from .base import RawSequence

NAME = "oxiod"
VERSION = "1.0"
LICENSE = "OxIOD (University of Oxford) academic/non-commercial use; cite Chen et al. 2018, http://deepio.cs.ox.ac.uk/"

G_IOS = 9.80665  # CoreMotion 的 1 G 取标准重力
ROOT_NAME = "Oxford Inertial Odometry Dataset"
COARSE_MAX_LAG = 5.0  # s
FINE_MAX_LAG = 0.3  # s
MIN_SYNC_CORRELATION = 0.3  # 粗同步互相关峰值下限
SKEW_SEGMENT = 60.0  # 时钟速率差估计的分段长度，s
SKEW_MIN_CORRELATION = 0.4  # 参与速率差拟合的分段三轴相关系数下限
SKEW_MIN_DRIFT = 0.005  # 全序列累计漂移超过该值（s）才启用速率差
MAX_EXTRINSIC_RESIDUAL = 0.3  # 手眼对齐后角速度残差中位数上限，rad/s
GLITCH_MAX_RATE_DEG = 600.0
GLITCH_MAX_SPEED = 10.0
GLITCH_PAD = 0.1

# 场景目录 → (序列名前缀, placement, 设备, 受试者)。设备/受试者依据 OxIOD 论文与目录命名
# （multi users 为 user2–5，推断主场景为 user1），见 docs/datasets/oxiod.md。
_SCENES = {
    "handheld": ("handheld", "handheld", "iphone7plus", "user1"),
    "pocket": ("pocket", "pocket", "iphone7plus", "user1"),
    "handbag": ("handbag", "bag", "iphone7plus", "user1"),
    "trolley": ("trolley", "trolley", "iphone7plus", "user1"),
    "slow walking": ("slow_walking", "handheld", "iphone7plus", "user1"),
    "running": ("running", "handheld", "iphone7plus", "user1"),
    "multi users": ("multi_users", "unknown", "iphone7plus", None),
    "multi devices": ("multi_devices", "unknown", None, "unknown"),
}
_OFFICIAL_SCENES = ("handheld", "pocket", "handbag", "trolley", "slow walking", "running")

# 场景目录名不含放置方式的会话：逐条放置表 [(起, 止, placement)] 与来源。
# "readme"：syn/Readme.txt 原文（user3："1-2: handheld, 3-5: pocket, 6-7: handbag"；
# user5："1-3 handheld, 4-6 pocket, 7-11 handbag"），且与机体系重力方向证据一致；
# "gravity_direction"：无文档，仅在“屏幕朝上”证据明确
# （加速度 z 分量 > 0.5 g 的时间占比 ≈ 100%）时标 handheld，
# 其余直立持握（口袋/包无法可靠区分）标 unknown。证据统计见 docs/datasets/oxiod.md。
_PLACEMENT_TABLE = {
    ("multi users", "user2"): ([(1, 3, "handheld")], "gravity_direction"),
    ("multi users", "user3"): ([(1, 2, "handheld"), (3, 5, "pocket"), (6, 7, "bag")], "readme"),
    ("multi users", "user4"): ([(1, 3, "handheld")], "gravity_direction"),
    ("multi users", "user5"): ([(1, 3, "handheld"), (4, 6, "pocket"), (7, 11, "bag")], "readme"),
    ("multi devices", "iPhone 5"): ([(1, 3, "handheld")], "gravity_direction"),
    ("multi devices", "iPhone 6"): ([(1, 3, "handheld")], "gravity_direction"),
    ("multi devices", "nexus 5"): ([(1, 8, "handheld")], "gravity_direction"),
}


def placement_of(scene: str, session: str, index: int) -> Tuple[str, str]:
    """返回 (placement, 来源)。"""

    if (scene, session) in _PLACEMENT_TABLE:
        ranges, source = _PLACEMENT_TABLE[(scene, session)]
        for lo, hi, placement in ranges:
            if lo <= index <= hi:
                return placement, source
        return "unknown", source
    placement = _SCENES[scene][1]
    source = (
        "scene_folder"
        if scene in ("handheld", "pocket", "handbag", "trolley")
        else "gravity_direction"
    )
    return placement, source


_DEVICE_SLUG = {"iPhone 5": "iphone5", "iPhone 6": "iphone6", "nexus 5": "nexus5"}


def _root(source: Path) -> Path:
    source = Path(source)
    if (source / ROOT_NAME).is_dir():
        return source / ROOT_NAME
    return source


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _entries(source: Path) -> Dict[str, dict]:
    """枚举全部 raw 序列：sequence_id → 元信息（文件路径、场景、会话等）。"""

    root = _root(source)
    out: Dict[str, dict] = {}
    if not root.is_dir():
        return out
    for scene, (prefix, placement, device, subject) in _SCENES.items():
        scene_dir = root / scene
        if not scene_dir.is_dir():
            continue
        for session_dir in sorted(p for p in scene_dir.iterdir() if (p / "raw").is_dir()):
            raw_dir = session_dir / "raw"
            session = session_dir.name
            session_slug = _DEVICE_SLUG.get(session, _slug(session))
            for imu_path in sorted(
                raw_dir.glob("imu*.csv"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0)
            ):
                k = re.sub(r"\D", "", imu_path.stem)
                if not k:
                    continue
                ref = raw_dir / f"vi{k}.csv"
                if not ref.is_file():
                    ref = raw_dir / f"hand{k}.csv"
                sequence_id = f"{prefix}_{session_slug}_seq{k}"
                placement, placement_source = placement_of(scene, session, int(k))
                out[sequence_id] = {
                    "scene": scene,
                    "session": session,
                    "index": int(k),
                    "imu": imu_path,
                    "vicon": ref if ref.is_file() else None,
                    "placement": placement,
                    "placement_source": placement_source,
                    "device": device or session_slug,
                    "subject": subject or session_slug,
                    "group": f"{prefix}_{session_slug}",
                    "android": session == "nexus 5",
                }
    return out


def list_sequences(source: Path) -> List[str]:
    """契约可选成员：全部 raw 序列（排序），含无官方划分的 multi users/devices（57 条）。"""

    return sorted(_entries(source))


def official_splits(source: Path) -> Dict[str, List[str]]:
    """各放置场景目录下的 ``Train.txt`` / ``Test.txt``（条目为 ``dataN`` 或 ``dataN/imuK.csv``）。

    ``multi users``、``multi devices`` 没有官方划分，不出现在返回值中（无官方 val）。
    """

    root = _root(source)
    entries = _entries(source)
    splits = {"train": [], "test": []}
    for scene in _OFFICIAL_SCENES:
        prefix = _SCENES[scene][0]
        for key, fn in (("train", "Train.txt"), ("test", "Test.txt")):
            path = root / scene / fn
            if not path.is_file():
                continue
            for item in path.read_text(encoding="utf-8", errors="replace").split():
                item = item.strip().strip("/")
                if not item:
                    continue
                parts = item.split("/")
                session = _slug(parts[0])
                if len(parts) == 1:
                    ids = sorted(
                        (
                            sid
                            for sid, e in entries.items()
                            if e["scene"] == scene and _slug(e["session"]) == session
                        ),
                        key=lambda sid: entries[sid]["index"],
                    )
                else:
                    k = re.sub(r"\D", "", Path(parts[1]).stem)
                    ids = [f"{prefix}_{session}_seq{k}"]
                splits[key].extend(sid for sid in ids if sid in entries and sid not in splits[key])
    return splits


def iter_raw_sequences(
    source: Path, only: Optional[Collection[str]] = None
) -> Iterator[RawSequence]:
    root = _root(source)
    entries = _entries(source)
    names = sorted(entries) if only is None else list(only)
    for name in names:
        if name not in entries:
            yield pu.rejected_sequence(name, f"sequence not found under {root}")
            continue
        entry = entries[name]
        try:
            raw = load_sequence(name, entry, root=root)
        except (OSError, KeyError, ValueError) as exc:
            yield pu.rejected_sequence(
                name, f"parse error: {type(exc).__name__}: {exc}", _attrs(entry, root)
            )
            continue
        yield pu.finalize_sequence(raw)


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------


def _attrs(entry: dict, root: Optional[Path]) -> dict:
    files = [pu.relative_source(entry["imu"], root)]
    if entry["vicon"] is not None:
        files.append(pu.relative_source(entry["vicon"], root))
    return {
        "subject_id": entry["subject"],
        "device_id": entry["device"],
        "placement": entry["placement"],
        "group_id": entry["group"],
        "position_source": "vicon",
        "orientation_source": "vicon_with_estimated_body_extrinsic",
        "device_orientation_source": "none" if entry["android"] else "ios_coremotion_attitude",
        "body_frame": "android_device" if entry["android"] else "ios_device",
        "source_files": ";".join(files),
        "oxiod_scene": entry["scene"],
        "oxiod_session": entry["session"],
        "oxiod_placement_source": entry["placement_source"],
    }


def _first_field_is_scientific(path: Path) -> bool:
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for _ in range(20):
            line = handle.readline()
            if not line:
                break
            if re.search(r"[eE][+-]?\d", line.split(",")[0]):
                return True
    return False


def read_ios_imu(path: Path) -> Tuple[np.ndarray, dict]:
    """读取 iPhone ``imu<k>.csv``，返回 (按时间排序、去重后的 16 列数组, 统计)。"""

    data = np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=2)
    if data.shape[1] < 13:
        raise ValueError(f"{path}: expected >= 13 columns, got {data.shape[1]}")
    stats = {"rows": len(data), "out_of_order": int((np.diff(data[:, 0]) < 0).sum())}
    data = data[np.argsort(data[:, 0], kind="stable")]
    keep = pu.monotonic_mask(data[:, 0])
    stats["duplicates"] = int((~keep).sum())
    return data[keep], stats


def read_android_imu(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """读取 Nexus 5 长表（毫秒, 类型, x, y, z），返回 (t 秒, gyro, acc, 统计)。

    陀螺按时间线性插值到加速度计时间戳。
    """

    data = np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=2)
    acc = data[data[:, 1] == 1]
    gyr = data[data[:, 1] == 4]
    acc = acc[np.argsort(acc[:, 0], kind="stable")]
    gyr = gyr[np.argsort(gyr[:, 0], kind="stable")]
    acc = acc[pu.monotonic_mask(acc[:, 0])]
    gyr = gyr[pu.monotonic_mask(gyr[:, 0])]
    if len(acc) < 2 or len(gyr) < 2:
        raise ValueError(f"{path}: missing accelerometer or gyroscope rows")
    t = acc[:, 0] / 1e3
    tg = gyr[:, 0] / 1e3
    inside = (t >= tg[0]) & (t <= tg[-1])
    t, acc = t[inside], acc[inside]
    gyro = np.stack([np.interp(t, tg, gyr[:, 2 + k]) for k in range(3)], axis=1)
    stats = {"accel_rows": int(len(acc)), "gyro_rows": int(len(gyr))}
    return t, gyro, acc[:, 2:5], stats


def read_vicon(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """读取 Vicon ``vi<k>.csv``：返回 (t 秒, position, q_wxyz, 统计)。"""

    data = np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=2)
    if data.shape[1] < 9:
        raise ValueError(f"{path}: expected 9 columns, got {data.shape[1]}")
    t = data[:, 0] / 1e9
    order = np.argsort(t, kind="stable")
    stats = {"rows": len(data), "out_of_order": int((np.diff(t) < 0).sum())}
    data, t = data[order], t[order]
    keep = pu.monotonic_mask(t)
    stats["duplicates"] = int((~keep).sum())
    data, t = data[keep], t[keep]
    q = pu.xyzw_to_wxyz(data[:, 5:9])
    norm = np.linalg.norm(q, axis=1)
    valid = (
        np.isfinite(data[:, 2:9]).all(1)
        & (np.abs(norm - 1.0) < 0.1)
        & ~np.all(data[:, 2:5] == 0, axis=1)
    )
    q = np.where(
        valid[:, None], q / np.where(norm > 0, norm, 1.0)[:, None], np.array([1.0, 0, 0, 0])
    )
    stats["invalid_rows"] = int((~valid).sum())
    return t, data[:, 2:5], pu.quat_make_continuous(q), dict(stats, valid=valid)


def ios_attitude(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """CoreMotion 欧拉角 → body→world 四元数：``R_z(yaw) ⊗ R_x(pitch) ⊗ R_y(roll)``。"""

    def axis_q(axis: int, angle: np.ndarray) -> np.ndarray:
        vec = np.zeros((len(angle), 3))
        vec[:, axis] = angle
        return pu.quat_from_rotvec(vec)

    return pu.quat_mul(pu.quat_mul(axis_q(2, yaw), axis_q(0, pitch)), axis_q(1, roll))


class ClockModel:
    """IMU 时钟到 Vicon 时钟的线性映射：``τ = t + offset + skew · (t − t_ref)``。"""

    def __init__(self, offset: float, skew: float = 0.0, t_ref: float = 0.0):
        self.offset, self.skew, self.t_ref = float(offset), float(skew), float(t_ref)

    def to_vicon(self, t_imu: np.ndarray) -> np.ndarray:
        return t_imu + self.offset + self.skew * (t_imu - self.t_ref)

    def to_imu(self, t_vicon: np.ndarray) -> np.ndarray:
        return (t_vicon - self.offset + self.skew * self.t_ref) / (1.0 + self.skew)


def _fit_skew(t_imu, gyro, t_ref, q_ref, q_sb, offset) -> Tuple[float, float, int, float]:
    """分段细化偏移并线性拟合，返回 (斜率, 参考时刻, 可用段数, 拟合残差 RMS)。"""

    centers, offsets = [], []
    for start in np.arange(t_imu[0], t_imu[-1] - SKEW_SEGMENT / 2, SKEW_SEGMENT):
        sel = (t_imu >= start) & (t_imu < start + SKEW_SEGMENT)
        if sel.sum() < 0.5 * SKEW_SEGMENT * 90:
            continue
        off, corr = pu.refine_time_offset(
            t_imu[sel], gyro[sel], t_ref, q_ref, q_sb, center=offset, max_lag=FINE_MAX_LAG
        )
        if np.isfinite(off) and corr > SKEW_MIN_CORRELATION:
            centers.append(float(np.mean(t_imu[sel])))
            offsets.append(off)
    t_mid = float(0.5 * (t_imu[0] + t_imu[-1]))
    if len(centers) < 3:
        return 0.0, t_mid, len(centers), float("nan")
    x = np.asarray(centers) - t_mid
    slope, intercept = np.polyfit(x, np.asarray(offsets), 1)
    rms = float(np.sqrt(np.mean((np.asarray(offsets) - (slope * x + intercept)) ** 2)))
    return float(slope), t_mid, len(centers), rms


def synchronise(t_imu: np.ndarray, gyro: np.ndarray, t_ref: np.ndarray, q_ref: np.ndarray) -> dict:
    """估计 IMU→Vicon 时钟模型（偏移 + 可选速率差）与刚体外参 ``q_sb``。

    1. 角速度模长互相关（±5 s）粗估偏移；2. 在粗外参下三轴互相关（±0.3 s）细化；
    3. 每 60 s 一段重复细化并线性拟合，累计漂移超过 SKEW_MIN_DRIFT 时启用速率差；
    4. 在最终时钟下重估外参。
    约定：``ω_imu(t) ≈ ω_vicon(clock.to_vicon(t))``。
    """

    t_mid, omega = pu.body_rates_from_orientation(t_ref, q_ref)
    offset0, corr0 = pu.estimate_time_offset(
        t_imu,
        np.linalg.norm(gyro, axis=1),
        t_mid,
        np.linalg.norm(omega, axis=1),
        max_lag=COARSE_MAX_LAG,
        rate=50.0,
    )
    if not np.isfinite(offset0):
        return {"clock": None, "coarse_offset": offset0, "coarse_corr": corr0}
    q_sb, _, _ = pu.estimate_body_extrinsic(t_imu + offset0, gyro, t_ref, q_ref)
    offset, corr = pu.refine_time_offset(
        t_imu, gyro, t_ref, q_ref, q_sb, center=offset0, max_lag=FINE_MAX_LAG
    )
    if not np.isfinite(offset):
        offset, corr = offset0, float("nan")
    q_sb, _, _ = pu.estimate_body_extrinsic(t_imu + offset, gyro, t_ref, q_ref)
    slope, t_center, segments, fit_rms = _fit_skew(t_imu, gyro, t_ref, q_ref, q_sb, offset)
    drift = abs(slope) * (t_imu[-1] - t_imu[0])
    clock = ClockModel(offset)
    if drift > SKEW_MIN_DRIFT:
        # 以序列中点为参考重算截距，使 offset 表示中点处的偏移
        clock = ClockModel(offset, slope, t_center)
        off_mid, _ = pu.refine_time_offset(
            t_center + (t_imu - t_center) * (1.0 + slope),
            gyro,
            t_ref,
            q_ref,
            q_sb,
            center=offset,
            max_lag=FINE_MAX_LAG,
        )
        if np.isfinite(off_mid):
            clock = ClockModel(off_mid, slope, t_center)
    q_sb, resid, count = pu.estimate_body_extrinsic(clock.to_vicon(t_imu), gyro, t_ref, q_ref)
    return {
        "clock": clock,
        "corr": float(corr),
        "coarse_offset": float(offset0),
        "coarse_corr": float(corr0),
        "skew_segments": segments,
        "skew_fit_rms": fit_rms,
        "skew_drift": float(drift),
        "q_sb": q_sb,
        "residual": float(resid),
        "count": int(count),
    }


def load_sequence(name: str, entry: dict, root: Optional[Path] = None) -> RawSequence:
    """解析一条 OxIOD raw 序列（含同步与外参估计，不含物理自检）。"""

    attrs = _attrs(entry, root)
    notes: List[str] = []
    if entry["vicon"] is None:
        return pu.rejected_sequence(
            name, "no Vicon file (vi<k>.csv / hand<k>.csv) for this IMU file", attrs
        )
    if _first_field_is_scientific(entry["vicon"]) or _first_field_is_scientific(entry["imu"]):
        return pu.rejected_sequence(
            name,
            "timestamps corrupted (exported in scientific notation, sub-second information lost)",
            attrs,
        )

    device_orientation = None
    if entry["android"]:
        t_imu, gyro, acc, stats = read_android_imu(entry["imu"])
        notes.append(
            "Nexus 5 long-format log: accelerometer (type 1, m/s^2, specific force) timestamps "
            "used as IMU clock; "
            f"gyroscope (type 4, rad/s) linearly interpolated onto them ({stats})"
        )
    else:
        imu, stats = read_ios_imu(entry["imu"])
        if (
            len(imu) < 2
            or np.mean(np.diff(imu[:, 0]) > 0) < 0.5
            or np.median(np.diff(imu[:, 0])) > 0.1
        ):
            return pu.rejected_sequence(
                name, "IMU timestamps corrupted (rounded to whole seconds)", attrs
            )
        t_imu = imu[:, 0]
        gyro = imu[:, 4:7]
        acc = -(imu[:, 7:10] + imu[:, 10:13]) * G_IOS
        device_orientation = pu.quat_make_continuous(ios_attitude(imu[:, 1], imu[:, 2], imu[:, 3]))
        notes += [
            f"IMU rows re-sorted by timestamp ({stats['out_of_order']} out-of-order steps), "
            f"{stats['duplicates']} duplicate timestamps dropped",
            f"accelerometer = -(gravity + user_acc) * {G_IOS} (iOS sign convention -> specific "
            "force)",
            "gyroscope = CoreMotion rotation rate (bias-compensated by iOS)",
            "device_orientation = Rz(yaw) * Rx(pitch) * Ry(roll) from CoreMotion attitude (z-up "
            "reference, arbitrary yaw)",
        ]
    if len(t_imu) < 2:
        return pu.rejected_sequence(name, "too few IMU samples", attrs)

    t_ref, position, q_ref, vstats = read_vicon(entry["vicon"])
    glitch_ok = pu.pose_glitch_mask(
        t_ref, q_ref, position, GLITCH_MAX_RATE_DEG, GLITCH_MAX_SPEED, GLITCH_PAD
    )
    pose_valid = vstats.pop("valid") & glitch_ok
    notes.append(
        f"Vicon: {vstats['out_of_order']} out-of-order / {vstats['duplicates']} duplicate / "
        f"{vstats['invalid_rows']} invalid rows; {int((~glitch_ok).sum())} samples masked around "
        "glitches "
        f"(> {GLITCH_MAX_RATE_DEG:g} deg/s or > {GLITCH_MAX_SPEED:g} m/s, pad {GLITCH_PAD:g} s)"
    )
    if pose_valid.sum() < 100:
        return pu.rejected_sequence(name, "fewer than 100 valid Vicon samples", attrs)

    sync = synchronise(t_imu, gyro, t_ref[pose_valid], q_ref[pose_valid])
    clock = sync["clock"]
    if clock is None:
        return pu.rejected_sequence(
            name, "IMU and Vicon time ranges do not overlap enough to synchronise", attrs
        )
    q_sb = sync["q_sb"]
    notes += [
        f"clock model: vicon_time = imu_time + {clock.offset:.4f} s + {clock.skew * 1e6:.1f} ppm * "
        f"(imu_time - {clock.t_ref:.3f}) (coarse offset {sync['coarse_offset']:.3f} s, |w| corr "
        f"{sync['coarse_corr']:.2f}; refined 3-axis corr {sync['corr']:.2f}; skew from "
        f"{sync['skew_segments']} "
        f"60 s segments, drift {sync['skew_drift'] * 1e3:.1f} ms, applied only if > "
        f"{SKEW_MIN_DRIFT * 1e3:g} ms); "
        "pose_time = inverse clock model applied to vicon_time",
        f"Vicon body -> IMU body extrinsic q_sb (wxyz) = {np.round(q_sb, 5).tolist()} "
        f"({np.degrees(pu.quat_angle(q_sb)):.2f} deg), estimated by gyro/Vicon angular-rate "
        "alignment; "
        f"median residual {sync['residual']:.3f} rad/s over {sync['count']} 20 Hz bins",
        "orientation = q_vicon * q_sb (IMU body -> Vicon world, z up); position = Vicon "
        "marker-body origin",
    ]
    rejected = None
    if not sync["coarse_corr"] >= MIN_SYNC_CORRELATION:
        rejected = f"cannot synchronise IMU and Vicon (|w| correlation {sync['coarse_corr']:.2f})"
    elif not sync["residual"] <= MAX_EXTRINSIC_RESIDUAL:
        rejected = (
            "Vicon body not rigidly aligned with IMU (angular-rate residual "
            f"{sync['residual']:.3f} rad/s)"
        )

    attrs["start_time_unix"] = float(t_imu[0])
    attrs["oxiod_clock_offset_s"] = clock.offset
    attrs["oxiod_clock_skew_ppm"] = clock.skew * 1e6
    attrs["oxiod_extrinsic_wxyz"] = ",".join(f"{v:.6f}" for v in q_sb)
    return RawSequence(
        sequence_id=name,
        imu_time=t_imu,
        gyroscope=gyro,
        accelerometer=acc,
        pose_time=clock.to_imu(t_ref),
        position=position,
        orientation=pu.quat_make_continuous(pu.quat_mul(q_ref, q_sb)),
        device_orientation=device_orientation,
        pose_valid=pose_valid,
        attrs=attrs,
        notes=notes,
        rejected=rejected,
    )
