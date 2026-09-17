"""统一转换流水线：``converters.<name>`` 解析 → 重采样 → 掩码 → 校验 → 写出 → 划分与清单。

用法::

    from inertial_benchmark.data.convert import convert_dataset
    convert_dataset("ronin", "/raw/ronin", "/data/ipb/ronin", workers=8)

转换器除 ``base.py`` 规定的成员外，可选提供 ``list_sequences(source) -> list[str]``，
用于多进程时枚举全部序列；未提供时用官方划分的并集。
"""

from __future__ import annotations

import datetime as _dt
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import ModuleType
from typing import Collection, Iterable, Optional, Union

import numpy as np

from .. import __version__
from ..utils import LOGGER, datasets_dir, json_load, json_save
from .converters import converter_ref, load_converter, resolve_converter_ref
from .converters.base import RawSequence
from .format import (
    ACCELEROMETER_TYPE,
    ORIENTATION_CONVENTION,
    REQUIRED_ATTRS,
    SCHEMA_VERSION,
    WORLD_FRAME,
    Sequence,
    save_sequence,
    validate,
)
from .manifest import MANIFEST, REPORT, fingerprint, sha256_file
from .resample import DEFAULT_GAP_THRESHOLD, DEFAULT_RATE, resample_streams
from .splits import check_leakage, resolve_splits, write_split

PathLike = Union[str, Path]


def raw_to_sequence(raw: RawSequence, dataset: str, converter: str, license_: str,
                    opts: dict) -> tuple:
    """把一条 ``RawSequence`` 变为 v1 ``Sequence``；返回 ``(sequence 或 None, record)``。"""
    record: dict = {"sequence_id": raw.sequence_id, "notes": list(raw.notes)}
    if raw.rejected:
        record.update(status="rejected", reason=f"converter: {raw.rejected}")
        return None, record
    problems = raw.check_shapes()
    if problems:
        record.update(status="rejected", reason="raw shape check failed", errors=problems)
        return None, record
    try:
        res = resample_streams(
            raw.imu_time, raw.gyroscope, raw.accelerometer, raw.pose_time, raw.position,
            raw.orientation, velocity=raw.velocity, device_orientation=raw.device_orientation,
            imu_valid=raw.imu_valid, pose_valid=raw.pose_valid, rate=opts["rate"],
            gap_threshold=opts["gap_threshold"],
        )
    except ValueError as exc:
        record.update(status="rejected", reason=f"resampling failed: {exc}")
        return None, record

    duration = float(res.timestamp[-1])
    if duration < opts["min_duration"]:
        record.update(status="rejected",
                      reason=f"overlap {duration:.2f}s shorter than {opts['min_duration']}s")
        return None, record

    attrs = {k: v for k, v in raw.attrs.items()}
    imu_t0 = float(np.asarray(raw.imu_time)[np.isfinite(raw.imu_time)][0])
    if "start_time_unix" in attrs and attrs["start_time_unix"] is not None:
        # 约定：raw attrs 的 start_time_unix 对应原始 IMU 时钟的第一个样本
        start_unix = float(attrs["start_time_unix"]) + (res.start_time - imu_t0)
    elif imu_t0 > 1e8:
        start_unix = res.start_time  # 原始时钟本身就是 Unix 秒
    else:
        start_unix = math.nan
    dropped = res.info["dropped_timestamps"]
    notes = list(raw.notes)
    if dropped["imu"] or dropped["pose"]:
        notes.append(f"dropped duplicate/non-monotonic timestamps: imu={dropped['imu']}, "
                     f"pose={dropped['pose']}")
    attrs.update(
        schema_version=SCHEMA_VERSION,
        dataset=dataset,
        sequence_id=raw.sequence_id,
        sample_rate_hz=float(opts["rate"]),
        source_sample_rate_hz=float(res.info["imu"]["source_rate_hz"]),
        source_pose_rate_hz=float(res.info["pose_rate_hz"]),
        resampling=res.description,
        world_frame=WORLD_FRAME,
        timestamp_type="relative",
        start_time_unix=float(start_unix),
        orientation_convention=ORIENTATION_CONVENTION,
        accelerometer_type=ACCELEROMETER_TYPE,
        source_license=str(attrs.get("source_license", license_)),
        converter=converter,
        conversion_notes=notes if notes else ["none"],
    )
    for key in REQUIRED_ATTRS:
        attrs.setdefault(key, "unknown")
    arrays = res.arrays
    seq = Sequence(
        timestamp=res.timestamp,
        gyroscope=arrays["gyroscope"],
        accelerometer=arrays["accelerometer"],
        orientation=arrays["orientation"],
        position=arrays["position"],
        valid_imu=res.valid_imu,
        valid_pose=res.valid_pose,
        device_orientation=arrays.get("device_orientation"),
        velocity=arrays.get("velocity"),
        attrs=attrs,
    )
    rep = validate(seq, sample_rate=opts["rate"])
    record.update(notes=notes, warnings=list(rep.warnings), gravity=rep.info.get("gravity"))
    if not rep.ok:
        record.update(status="rejected", reason="validation failed", errors=list(rep.errors))
        return None, record
    record.update(
        status="accepted",
        num_samples=len(seq),
        duration_s=seq.duration,
        distance_m=seq.distance(),
        valid_fraction=float(seq.valid.mean()),
        group_id=str(attrs["group_id"]),
        subject_id=str(attrs["subject_id"]),
        placement=str(attrs["placement"]),
        source_sample_rate_hz=attrs["source_sample_rate_hz"],
    )
    return seq, record


def _write(seq: Sequence, record: dict, output: Path, compression: str) -> dict:
    rel = Path("sequences") / f"{seq.sequence_id}.h5"
    path = save_sequence(output / rel, seq, compression=compression)
    record.update(file=rel.as_posix(), sha256=sha256_file(path),
                  size_bytes=path.stat().st_size)
    return record


def _process(raw: RawSequence, meta: dict, output: Path, opts: dict) -> dict:
    try:
        seq, record = raw_to_sequence(raw, meta["dataset"], meta["converter"], meta["license"],
                                      opts)
        if seq is not None:
            record = _write(seq, record, output, opts["compression"])
    except Exception as exc:  # noqa: BLE001 - 单条失败不应中断整个数据集
        record = {"sequence_id": raw.sequence_id, "status": "rejected",
                  "reason": f"{type(exc).__name__}: {exc}"}
    return record


def _worker(ref: tuple, source: str, sequence_id: str, meta: dict, output: str,
            opts: dict) -> list:
    module = resolve_converter_ref(ref)
    records = []
    try:
        for raw in module.iter_raw_sequences(Path(source), only=[sequence_id]):
            records.append(_process(raw, meta, Path(output), opts))
    except Exception as exc:  # noqa: BLE001 - 转换器解析异常只影响这一条
        records.append({"sequence_id": sequence_id, "status": "rejected",
                        "reason": f"converter raised {type(exc).__name__}: {exc}"})
    if not records:
        records.append({"sequence_id": sequence_id, "status": "rejected",
                        "reason": "converter produced no data for this id"})
    return records


def _list_ids(module: ModuleType, source: Path, official: dict) -> list:
    if hasattr(module, "list_sequences"):
        return list(module.list_sequences(source))
    return sorted({i for ids in official.values() for i in ids})


def convert_dataset(
    name: str,
    source: PathLike,
    output: Optional[PathLike] = None,
    only: Optional[Collection[str]] = None,
    workers: int = 1,
    *,
    converter: Union[str, Path, ModuleType, None] = None,
    rate: float = DEFAULT_RATE,
    gap_threshold: float = DEFAULT_GAP_THRESHOLD,
    compression: str = "gzip",
    min_duration: float = 2.0,
    val_fraction: float = 0.1,
    seed: int = 0,
    overwrite: bool = True,
) -> dict:
    """转换整个数据集，写出 ``sequences/``、``splits/``、``dataset.json`` 与
    ``conversion_report.json``，返回 ``dataset.json`` 内容。

    ``converter`` 默认按 ``name`` 加载 ``converters.<name>``，也可传入模块对象或文件路径
    （测试注入）。``overwrite=False`` 时跳过已有文件的序列（沿用上次报告中的记录）。
    """
    module = load_converter(converter if converter is not None else name)
    source = Path(source).expanduser()
    output = Path(output).expanduser() if output else datasets_dir() / name
    output.mkdir(parents=True, exist_ok=True)
    conv_id = f"{module.NAME}@{module.VERSION}"
    meta = {"dataset": name, "converter": conv_id, "license": str(module.LICENSE)}
    opts = dict(rate=rate, gap_threshold=gap_threshold, compression=compression,
                min_duration=min_duration)

    official = {k: list(v) for k, v in module.official_splits(source).items()}
    previous = _previous_records(output)
    wanted = set(only) if only else None
    ids: Optional[list] = None
    if workers > 1 or not overwrite:
        ids = _list_ids(module, source, official)
        if wanted is not None:
            ids = [i for i in ids if i in wanted]
        if not overwrite:
            ids = [i for i in ids if not (previous.get(i, {}).get("status") == "accepted"
                                          and (output / previous[i]["file"]).exists())]

    LOGGER.info(f"convert {name}: {conv_id}, source={source}, output={output}, "
                f"workers={workers}")
    records: list = []
    if ids is not None and workers > 1:
        ref = converter_ref(module)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_worker, ref, str(source), i, meta, str(output), opts)
                       for i in ids]
            for fut in as_completed(futures):
                for rec in fut.result():
                    records.append(rec)
                    _log_record(rec, len(records), len(ids))
    else:
        only_arg = ids if ids is not None else (sorted(wanted) if wanted else None)
        if only_arg is None or only_arg:
            for raw in module.iter_raw_sequences(source, only=only_arg):
                rec = _process(raw, meta, output, opts)
                records.append(rec)
                _log_record(rec, len(records), len(only_arg) if only_arg else None)

    # 合并本次未处理的历史记录（文件仍在时）
    done = {r["sequence_id"] for r in records}
    for sid, rec in previous.items():
        if sid in done:
            continue
        if rec.get("status") != "accepted" or (output / rec.get("file", "")).exists():
            records.append(rec)
    return write_dataset_files(output, name, meta, official, records, opts,
                               val_fraction=val_fraction, seed=seed, source=source)


def _log_record(rec: dict, k: int, total: Optional[int]) -> None:
    prefix = f"[{k}/{total}]" if total else f"[{k}]"
    if rec["status"] == "accepted":
        LOGGER.info(f"{prefix} {rec['sequence_id']}: {rec['duration_s']:.1f}s, "
                    f"{rec['distance_m']:.1f}m, valid {rec['valid_fraction']:.1%}")
    else:
        LOGGER.warning(f"{prefix} {rec['sequence_id']}: REJECTED ({rec['reason']})")


def _previous_records(output: Path) -> dict:
    path = output / REPORT
    if not path.exists():
        return {}
    try:
        return {r["sequence_id"]: r for r in json_load(path).get("sequences", [])}
    except (ValueError, KeyError):
        return {}


def _stats(entries: Iterable[dict]) -> dict:
    entries = list(entries)
    return {
        "num_sequences": len(entries),
        "num_groups": len({e["group_id"] for e in entries}),
        "total_duration_s": float(sum(e["duration_s"] for e in entries)),
        "total_duration_h": float(sum(e["duration_s"] for e in entries) / 3600.0),
        "total_distance_m": float(sum(e["distance_m"] for e in entries)),
    }


def write_dataset_files(output: Path, name: str, meta: dict, official: dict, records: list,
                        opts: dict, *, val_fraction: float, seed: int,
                        source: Optional[Path] = None) -> dict:
    """根据转换记录写出划分文件、``dataset.json`` 与 ``conversion_report.json``。"""
    records = sorted(records, key=lambda r: r["sequence_id"])
    accepted = {r["sequence_id"]: r for r in records if r["status"] == "accepted"}
    rejected = [r for r in records if r["status"] != "accepted"]
    groups = {sid: r["group_id"] for sid, r in accepted.items()}
    weights = {sid: r["duration_s"] for sid, r in accepted.items()}
    splits, method, missing = resolve_splits(official, accepted, groups, fraction=val_fraction,
                                             seed=seed, weights=weights)
    split_dir = output / "splits"
    if split_dir.is_dir():
        for old in split_dir.glob("*.txt"):
            if old.stem not in splits:
                old.unlink()
    for split, ids in splits.items():
        write_split(split_dir / f"{split}.txt", ids)
    leakage = check_leakage(splits, groups)
    assigned = {i for ids in splits.values() for i in ids}
    unassigned = sorted(set(accepted) - assigned)

    sequences = {
        sid: {k: r[k] for k in ("file", "sha256", "num_samples", "duration_s", "distance_m",
                                "group_id", "subject_id", "placement", "valid_fraction",
                                "source_sample_rate_hz")}
        for sid, r in accepted.items()
    }
    by_split = {s: _stats(accepted[i] for i in ids) for s, ids in splits.items()}
    now = _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": name,
        "converter": meta["converter"],
        "source_license": meta["license"],
        "ipb_version": __version__,
        "created_utc": now,
        "sample_rate_hz": opts["rate"],
        "conversion": {"gap_threshold_s": opts["gap_threshold"],
                       "compression": opts["compression"],
                       "min_duration_s": opts["min_duration"],
                       "resampling": "DESIGN 2.3 (see per-sequence 'resampling' attribute)"},
        "split_method": method,
        "splits": {s: len(ids) for s, ids in splits.items()},
        "statistics": {**_stats(accepted.values()), "by_split": by_split},
        "fingerprint": fingerprint({k: v["sha256"] for k, v in sequences.items()}, splits),
        "sequences": sequences,
    }
    json_save(output / MANIFEST, manifest)
    report = {
        "dataset": name,
        "converter": meta["converter"],
        "created_utc": now,
        "source": str(source) if source else None,
        "accepted": len(accepted),
        "rejected": len(rejected),
        "rejected_ids": [r["sequence_id"] for r in rejected],
        "missing_from_splits": missing,
        "unassigned": unassigned,
        "leakage": leakage,
        "warnings_total": int(sum(len(r.get("warnings", [])) for r in records)),
        "sequences": records,
    }
    json_save(output / REPORT, report)
    level = LOGGER.info if leakage["ok"] else LOGGER.warning
    level(f"convert {name}: accepted {len(accepted)}, rejected {len(rejected)}, "
          f"splits {manifest['splits']}, {manifest['statistics']['total_duration_h']:.2f} h"
          + ("" if leakage["ok"] else f", LEAKAGE: {leakage['leaks']}"))
    if unassigned:
        LOGGER.warning(f"convert {name}: {len(unassigned)} accepted sequences are in no split")
    return manifest


def default_workers() -> int:
    return max(1, min(8, (os.cpu_count() or 2) // 2))
