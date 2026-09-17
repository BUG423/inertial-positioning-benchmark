"""PedLocData（武汉大学，Zenodo 18149105，CC BY 4.0）解析器。

原始文件（Zenodo 说明）：

* ``YT_server_200Hz.h5``（约 15.1 GB）：``train``/``valid``/``test``
  三个组，分别含 1479/422/213 条切片；
* ``SimpleDemo.h5``（约 0.55 GB）：同格式的小样本，32/9/6 条切片；
* 每条切片是一个组，含 ``acc``（m/s²）、``gyr``（rad/s）、``ts``（Unix 秒）、``pos``（m，真值）、
  ``qua``（真值四元数，device→world，wxyz）、``acc_bias``/``gyr_bias``/``mag``/``motion_qua``
  （发布数据中全为 0）；
* ``para/mean``、``para/std``：导航系下加速度/陀螺的全数据集均值与标准差（归一化用）。

命名规律：YT 为 ``<受试者缩写>_<楼层>_server_<录制号>_<切片号>``，
Demo 为 ``mid360_<日期>_<时间>_<切片号>``。

已核实的约定（Zenodo 说明 + 物理自检）：机体系比力含重力；``qua`` 为 ``body_to_world`` wxyz；
世界系 z 轴向上；200 Hz；真值来源未在公开资料中注明
（Demo 文件名 ``mid360`` 暗示 Livox Mid-360 激光方案）。

**划分泄漏**：官方划分按切片随机分配，同一次录制的相邻切片（间隔 10–600 s，位置连续）
分散在 train/valid/test 中；27 名受试者中 26 人出现在全部三个划分。
``official_splits`` 仍按原样返回，
另提供按受试者（Demo 按录制）分组的无泄漏划分 ``grouped_splits``，
审计细节见 ``audit_official_splits``
与 ``docs/datasets/pedlocdata.md``。本模块声明 ``OFFICIAL_SPLITS_LEAK = True``，统一流水线据此把
``grouped_splits`` 写为默认 ``train/val/test``，官方划分另存为 ``official_*.txt``（DESIGN 2.4）。
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Collection, Iterator, Optional

import numpy as np

from . import _rig_utils as rig
from .base import RawSequence

NAME = "pedlocdata"
VERSION = "1.1.0"
LICENSE = "CC-BY-4.0 (PedLocData, https://zenodo.org/records/18149105)"
# DESIGN 2.4：官方划分存在录制级泄漏，默认划分改用 grouped_splits，官方划分以 official_* 保留
OFFICIAL_SPLITS_LEAK = True
OFFICIAL_SPLITS_LEAK_REASON = (
    "official train/valid/test assign slices of one recording at random: 329 of 474 YT recordings "
    "and 10 of 13 Demo recordings span several splits (same subject, device, floor and world "
    "frame; Demo slices are contiguous), and 26 of 27 subjects appear in all three splits; "
    "default splits are subject-grouped (YT) / recording-grouped (Demo) from grouped_splits()"
)

FILES = {"yt": "YT_server_200Hz.h5", "demo": "SimpleDemo.h5"}
SPLIT_GROUPS = {"train": "train", "valid": "val", "test": "test"}
YT_NAME = re.compile(r"^(?P<subject>[a-z]+)_(?P<floor>[A-Z]\d)_server_(?P<rec>\d+)_(?P<slice>\d+)$")
DEMO_NAME = re.compile(
    r"^(?P<rig>mid360)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{6})_(?P<slice>\d+)$"
)
GROUPED_SEED = 0
GROUPED_FRACTIONS = {"test": 0.10, "val": 0.20}  # 按时长；与官方划分的比例（约 69/20/11）一致


def _root(source) -> Path:
    source = Path(source)
    for cand in (source, source / "PedLocData"):
        if any((cand / name).exists() for name in FILES.values()):
            return cand
    raise FileNotFoundError(
        f"PedLocData files ({', '.join(FILES.values())}) not found under {source}"
    )


def _h5():
    import h5py

    return h5py


def parse_name(prefix: str, name: str) -> dict:
    """从切片名解析受试者/楼层/录制/切片号；不符合命名规律时返回 ``unknown``。"""

    if prefix == "yt":
        m = YT_NAME.match(name)
        if m:
            return {
                "subject": m["subject"],
                "floor": m["floor"],
                "recording": f"{m['subject']}_{m['floor']}_server_{m['rec']}",
                "slice": int(m["slice"]),
                "group": f"yt_{m['subject']}",
            }
    else:
        m = DEMO_NAME.match(name)
        if m:
            rec = f"mid360_{m['date']}_{m['time']}"
            return {
                "subject": "unknown",
                "floor": "unknown",
                "recording": rec,
                "slice": int(m["slice"]),
                "group": f"demo_{m['date']}_{m['time']}",
            }
    return {
        "subject": "unknown",
        "floor": "unknown",
        "recording": name,
        "slice": -1,
        "group": f"{prefix}_{name}",
    }


def _listing(root: Path) -> list:
    """``[(sequence_id, prefix, h5_group, name, split)]``，按文件、划分、名称排序。"""

    h5py = _h5()
    out = []
    for prefix, filename in FILES.items():
        path = root / filename
        if not path.exists():
            continue
        with h5py.File(path, "r") as handle:
            for group, split in SPLIT_GROUPS.items():
                if group not in handle:
                    continue
                for name in sorted(handle[group].keys()):
                    out.append((f"{prefix}_{name}", prefix, group, name, split))
    return out


def list_sequences(source) -> list:
    """全部可转换的 ``sequence_id``（``yt_*`` / ``demo_*``）；
    两个文件的切片都位于 train/valid/test 组中。
    """

    return [row[0] for row in _listing(_root(source))]


all_sequence_ids = list_sequences  # 兼容别名


def official_splits(source) -> dict:
    """官方划分（``valid`` → ``val``），另附 ``test_yt`` / ``test_demo`` 子集。
    **存在会话级泄漏**，见模块文档。
    """

    splits = {"train": [], "val": [], "test": [], "test_yt": [], "test_demo": []}
    for sequence_id, prefix, _, _, split in _listing(_root(source)):
        splits[split].append(sequence_id)
        if split == "test":
            splits[f"test_{prefix}"].append(sequence_id)
    return splits


def slice_table(source) -> list:
    """每条切片的元数据（只读首尾样本）：时间、位置、长度、官方划分与命名解析结果。"""

    h5py = _h5()
    root = _root(source)
    rows = []
    for prefix, filename in FILES.items():
        path = root / filename
        if not path.exists():
            continue
        with h5py.File(path, "r") as handle:
            for group, split in SPLIT_GROUPS.items():
                if group not in handle:
                    continue
                for name, node in handle[group].items():
                    ts = node["ts"]
                    pos = node["pos"]
                    row = {
                        "sequence_id": f"{prefix}_{name}",
                        "prefix": prefix,
                        "split": split,
                        "n": int(ts.shape[0]),
                        "t0": float(ts[0]),
                        "t1": float(ts[-1]),
                        "p0": pos[0].tolist(),
                        "p1": pos[-1].tolist(),
                    }
                    row.update(parse_name(prefix, name))
                    rows.append(row)
    return rows


def audit_official_splits(source, rows: Optional[list] = None) -> dict:
    """审计官方划分的泄漏：同一录制/受试者跨划分、相邻切片的时间与位置连续性。"""

    rows = rows if rows is not None else slice_table(source)
    report = {}
    for prefix in sorted({r["prefix"] for r in rows}):
        sub = [r for r in rows if r["prefix"] == prefix]
        by_rec = defaultdict(list)
        for r in sub:
            by_rec[r["recording"]].append(r)
        gaps, jumps, cross_pairs = [], [], 0
        for items in by_rec.values():
            items.sort(key=lambda r: r["slice"])
            for a, b in zip(items[:-1], items[1:]):
                if b["slice"] != a["slice"] + 1:
                    continue
                gaps.append(b["t0"] - a["t1"])
                jumps.append(float(np.linalg.norm(np.subtract(b["p0"], a["p1"]))))
                cross_pairs += a["split"] != b["split"]
        per_split = {}
        for split in ("train", "val", "test"):
            items = [r for r in sub if r["split"] == split]
            shared = [r for r in items if len({x["split"] for x in by_rec[r["recording"]]}) > 1]
            adjacent = [
                r
                for r in items
                if any(
                    x["split"] != split and abs(x["slice"] - r["slice"]) == 1
                    for x in by_rec[r["recording"]]
                )
            ]
            per_split[split] = {
                "slices": len(items),
                "hours": round(sum(r["n"] for r in items) / 200.0 / 3600.0, 3),
                "share_recording_with_other_split": len(shared),
                "adjacent_slice_in_other_split": len(adjacent),
                "subjects": len({r["subject"] for r in items}),
            }
        subjects_by_split = defaultdict(set)
        for r in sub:
            subjects_by_split[r["subject"]].add(r["split"])
        report[prefix] = {
            "slices": len(sub),
            "recordings": len(by_rec),
            "recordings_with_multiple_slices": sum(len(v) > 1 for v in by_rec.values()),
            "recordings_spanning_splits": sum(
                len({x["split"] for x in v}) > 1 for v in by_rec.values()
            ),
            "consecutive_slice_pairs": len(gaps),
            "consecutive_pairs_in_different_splits": cross_pairs,
            "gap_s_percentiles_0_50_100": (
                [round(float(x), 3) for x in np.percentile(gaps, [0, 50, 100])] if gaps else []
            ),
            "jump_m_percentiles_0_50_100": (
                [round(float(x), 3) for x in np.percentile(jumps, [0, 50, 100])] if jumps else []
            ),
            "subjects": len(subjects_by_split),
            "subjects_in_all_three_splits": sum(len(v) == 3 for v in subjects_by_split.values()),
            "per_split": per_split,
        }
    return report


def grouped_splits(source, rows: Optional[list] = None, seed: int = GROUPED_SEED,
                   fractions: Optional[dict] = None) -> dict:
    """无泄漏的分组划分（可选，非官方）。

    分组键 = ``group_id``：YT 为受试者（同一受试者的所有录制与切片同组），Demo 为录制会话。
    每个文件内部分别处理：固定种子打乱分组后，按时长依次填满 ``test``（约 10%）
    与 ``val``（约 20%），其余为 ``train``。
    返回的键与官方划分相同，另含 ``test_yt`` / ``test_demo``。
    """

    rows = rows if rows is not None else slice_table(source)
    fractions = dict(GROUPED_FRACTIONS if fractions is None else fractions)
    rng = np.random.default_rng(seed)
    splits = {"train": [], "val": [], "test": [], "test_yt": [], "test_demo": []}
    for prefix in sorted({r["prefix"] for r in rows}):
        sub = [r for r in rows if r["prefix"] == prefix]
        groups = defaultdict(list)
        for r in sub:
            groups[r["group"]].append(r)
        names = sorted(groups)
        order = [names[i] for i in rng.permutation(len(names))]
        total = sum(r["n"] for r in sub)
        filled = {"test": 0, "val": 0}
        assign = {}
        for name in order:
            size = sum(r["n"] for r in groups[name])
            for split in ("test", "val"):
                target = fractions[split] * total
                # 只有加入后更接近目标时长才加入，避免大组造成明显超额
                if abs(filled[split] + size - target) < abs(filled[split] - target):
                    assign[name] = split
                    filled[split] += size
                    break
            else:
                assign[name] = "train"
        for name in names:
            ids = sorted(r["sequence_id"] for r in groups[name])
            splits[assign[name]].extend(ids)
            if assign[name] == "test":
                splits[f"test_{prefix}"].extend(ids)
    for key in splits:
        splits[key].sort()
    return splits


def read_slice(
    node, prefix: str, name: str, split: str, source_file: str, group: str = "unknown"
) -> RawSequence:
    """把一个 h5 切片组解析为 ``RawSequence``（不含物理自检）。"""

    sequence_id = f"{prefix}_{name}"
    info = parse_name(prefix, name)
    notes = []
    attrs = {
        "subject_id": info["subject"] if prefix == "yt" else "unknown",
        "device_id": "unknown",
        "placement": "unknown",
        "group_id": info["group"],
        "position_source": (
            "undocumented reference trajectory (PedLocData 'Ground truth position'; demo files are "
            "named mid360, suggesting Livox Mid-360 LiDAR-based SLAM)"
        ),
        "orientation_source": (
            "undocumented reference (PedLocData 'Ground truth quaternion device to world')"
        ),
        "device_orientation_source": "none",
        "body_frame": "pedlocdata_device",
        "source_files": f"{source_file}:/{group}/{name}",
        "source_license": LICENSE,
        "official_split": split,
        "recording_id": f"{prefix}_{info['recording']}",
        "slice_index": int(info["slice"]),
        "floor": info["floor"],
        "imu_calibration": (
            "unknown (acc_bias/gyr_bias fields are all zero in the release; no calibration applied)"
        ),
    }
    required = ("acc", "gyr", "ts", "pos", "qua")
    missing = [k for k in required if k not in node]
    if missing:
        return rig.rejected_sequence(sequence_id, f"missing datasets {missing}", attrs, notes)
    time = node["ts"][()].astype(np.float64)
    acc = node["acc"][()].astype(np.float64)
    gyro = node["gyr"][()].astype(np.float64)
    pos = node["pos"][()].astype(np.float64)
    quat = node["qua"][()].astype(np.float64)
    lengths = {len(time), len(acc), len(gyro), len(pos), len(quat)}
    if len(lengths) != 1:
        return rig.rejected_sequence(
            sequence_id, f"inconsistent array lengths {sorted(lengths)}", attrs, notes
        )
    for key in ("acc_bias", "gyr_bias", "mag", "motion_qua"):
        if key in node:
            values = node[key][()]
            if np.any(values != 0):
                notes.append(
                    f"warning: '{key}' contains non-zero values "
                    "(expected all-zero placeholder); ignored"
                )
    notes.append(
        "acc_bias/gyr_bias/mag/motion_qua are all-zero placeholders in the release and are ignored"
    )
    attrs["start_time_unix"] = float(time[0])
    if not np.all(np.diff(time) > 0):
        order = np.argsort(time, kind="stable")
        time, acc, gyro, pos, quat = time[order], acc[order], gyro[order], pos[order], quat[order]
        keep = np.concatenate([[True], np.diff(time) > 0])
        time, acc, gyro, pos, quat = time[keep], acc[keep], gyro[keep], pos[keep], quat[keep]
        notes.append(
            f"timestamps not strictly increasing: sorted and dropped {int((~keep).sum())} rows"
        )
    gaps = np.diff(time)
    if (gaps > 0.05).any():
        notes.append(
            f"{int((gaps > 0.05).sum())} timestamp gap(s) > 0.05 s kept as-is "
            f"(max {gaps.max():.3f} s)"
        )
    imu_valid = np.isfinite(acc).all(1) & np.isfinite(gyro).all(1)
    norms = np.linalg.norm(np.nan_to_num(quat), axis=1)
    pose_valid = np.isfinite(quat).all(1) & np.isfinite(pos).all(1) & (norms > 0.5)
    quat = rig.normalize_quaternions(np.where(pose_valid[:, None], quat, [1.0, 0.0, 0.0, 0.0]))
    if pose_valid.any() and np.abs(norms[pose_valid] - 1).max() > 1e-3:
        notes.append(
            f"quaternions renormalised (max norm deviation "
            f"{np.abs(norms[pose_valid] - 1).max():.2e})"
        )
    if not imu_valid.all():
        notes.append(f"{int((~imu_valid).sum())} IMU rows non-finite and marked invalid")
    if not pose_valid.all():
        notes.append(
            f"{int((~pose_valid).sum())} reference rows invalid (non-finite or zero quaternion)"
        )
    notes.append(
        "qua is wxyz device_to_world (= body_to_world); ts is unix seconds; "
        "IMU is body-frame specific force"
    )
    notes.append(f"official split '{split}' has recording-level leakage (slices of one recording "
                 "spread over train/valid/test); IPB default splits are subject-grouped "
                 "(grouped_splits), the official split is kept as official_*.txt")
    return RawSequence(
        sequence_id=sequence_id,
        imu_time=time,
        gyroscope=gyro,
        accelerometer=acc,
        pose_time=time.copy(),
        position=np.where(pose_valid[:, None], pos, np.nan),
        orientation=quat,
        imu_valid=imu_valid,
        pose_valid=pose_valid,
        attrs=attrs,
        notes=notes,
    )


def iter_raw_sequences(source, only: Optional[Collection[str]] = None) -> Iterator[RawSequence]:
    h5py = _h5()
    root = _root(source)
    wanted = None if not only else set(only)
    for prefix, filename in FILES.items():
        path = root / filename
        if not path.exists():
            continue
        with h5py.File(path, "r") as handle:
            for group, split in SPLIT_GROUPS.items():
                if group not in handle:
                    continue
                for name in sorted(handle[group].keys()):
                    sequence_id = f"{prefix}_{name}"
                    if wanted is not None and sequence_id not in wanted:
                        continue
                    try:
                        raw = read_slice(handle[group][name], prefix, name, split, filename, group)
                    except Exception as exc:
                        yield rig.rejected_sequence(
                            sequence_id,
                            f"parse error: {exc!r}",
                            {
                                "source_license": LICENSE,
                                "source_files": f"{filename}:/{group}/{name}",
                            },
                        )
                        continue
                    if raw.rejected is None:
                        check_sequence(raw)
                    yield raw


def check_sequence(raw: RawSequence) -> None:
    """物理自检；参考系统与 IMU 的残余时延经“角速度互相关 + 加速度一致性”双证据确认后修正。"""

    stats = rig.physical_checks(raw)
    if rig.correct_time_offset_if_confirmed(raw, stats, confirm="acceleration"):
        stats = rig.physical_checks(raw)
    failures, warnings = rig.evaluate_checks(stats)
    rig.apply_checks(raw, stats, failures, warnings)
