"""统一转换流水线：``converters.<name>`` 解析 → 重采样 → 掩码 → 校验 → 写出 → 划分与清单。

用法::

    from inertial_benchmark.data.convert import convert_dataset
    convert_dataset("ronin", "/raw/ronin", "/data/ipb/ronin", workers=8)

转换器除 ``base.py`` 规定的成员外，可选提供（见 ``converters/base.py``）：

* ``list_sequences(source) -> list[str]``：多进程时枚举全部序列；未提供时用各划分的并集；
* ``extra_splits(source) -> dict[str, list[str]]``（与可选的 ``EXTRA_SPLIT_NOTES``）：
  不在官方划分中的序列的附加子集（例如 ``test_unseen_subject``），原样写出，绝不并入 train；
* ``OFFICIAL_SPLITS_LEAK = True`` + ``grouped_splits(source)``（与可选的
  ``OFFICIAL_SPLITS_LEAK_REASON``）：官方划分存在泄漏时，默认 ``train/val/test`` 取分组划分，
  官方划分另存为 ``official_*.txt``（DESIGN 2.4）。

已转换但不属于任何默认划分的序列不会被静默丢弃或并入 train：它们保留在 ``sequences/`` 中，
并列在 ``dataset.json`` 与 ``conversion_report.json`` 的 ``unassigned`` 里，``ipb check`` 会告警。
"""

from __future__ import annotations

import datetime as _dt
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from types import ModuleType
from typing import Collection, Iterable, Optional, Union

import numpy as np

from .. import __version__
from ..utils import LOGGER, datasets_dir, json_load, json_save
from .converters import ConverterError, converter_ref, load_converter, resolve_converter_ref
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
from .manifest import (
    MANIFEST,
    REPORT,
    find_duplicates,
    fingerprint,
    imu_content_hash,
    sha256_file,
)
from .resample import (
    DEFAULT_EDGE_MIN_RUN,
    DEFAULT_GAP_THRESHOLD,
    DEFAULT_RATE,
    resample_streams,
    trim_result,
    valid_extent,
)
from .splits import OFFICIAL_PREFIX, check_leakage, resolve_splits, split_family, write_split

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

    rate = float(opts["rate"])
    edge_min_run = float(opts.get("edge_min_run", DEFAULT_EDGE_MIN_RUN))
    start, stop = valid_extent(res.valid_imu & res.valid_pose, rate, edge_min_run)
    trimmed = None
    if (start, stop) != (0, len(res.timestamp)):
        trimmed = {"start_s": start / rate, "end_s": (len(res.timestamp) - stop) / rate}
        res = trim_result(res, start, stop)

    duration = float(res.timestamp[-1]) if len(res.timestamp) else 0.0
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
    if trimmed:
        notes.append(f"trimmed {trimmed['start_s']:.3f} s at the start and "
                     f"{trimmed['end_s']:.3f} s at the end: samples without valid IMU+pose, and "
                     f"edge fragments shorter than {edge_min_run:g} s separated by invalid "
                     "samples (DESIGN 2.3)")
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
        valid_device_orientation=res.valid_device,
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
        valid_fraction_device=float(seq.valid_device.mean()),
        group_id=str(attrs["group_id"]),
        subject_id=str(attrs["subject_id"]),
        placement=str(attrs["placement"]),
        source_sample_rate_hz=attrs["source_sample_rate_hz"],
        imu_sha256=imu_content_hash(seq.gyroscope, seq.accelerometer),
    )
    if trimmed:
        record["trimmed_s"] = [round(trimmed["start_s"], 6), round(trimmed["end_s"], 6)]
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


def _rejected(sequence_id: str, reason: str) -> dict:
    return {"sequence_id": sequence_id, "status": "rejected", "reason": reason}


def convert_ids(module: ModuleType, source: Path, ids: Iterable[str], meta: dict, output: Path,
                opts: dict, on_record=None) -> list:
    """转换给定的 ``sequence_id``，单条失败不影响其余（串行与子进程共用）。

    转换器的生成器可能在中途抛出异常（损坏文件、内存不足…），此时生成器无法继续：
    把下一条未产出的序列记为 rejected，再用剩余的 id 重新调用，直到全部有结论。
    没有抛异常但也没有产出数据的 id 记为 ``converter produced no data for this id``。
    """
    records: list = []
    remaining = list(dict.fromkeys(ids))
    while remaining:
        produced, error = set(), None
        try:
            for raw in module.iter_raw_sequences(source, only=list(remaining)):
                record = _process(raw, meta, output, opts)
                produced.add(record["sequence_id"])
                records.append(record)
                if on_record:
                    on_record(record)
        except Exception as exc:  # noqa: BLE001 - 解析异常只影响这一条与生成器的继续
            error = exc
        pending = [i for i in remaining if i not in produced]
        if error is None:
            for sequence_id in pending:
                records.append(_rejected(sequence_id, "converter produced no data for this id"))
                if on_record:
                    on_record(records[-1])
            remaining = []
        else:
            # 无法确定生成器停在哪一条时，按剩余顺序的第一条归因，保证每轮都有进展
            sequence_id = pending[0] if pending else remaining[0]
            records.append(_rejected(sequence_id,
                                     f"converter raised {type(error).__name__}: {error}"))
            if on_record:
                on_record(records[-1])
            remaining = [i for i in pending if i != sequence_id]
    return records


def _worker(ref: tuple, source: str, sequence_id: str, meta: dict, output: str,
            opts: dict) -> list:
    module = resolve_converter_ref(ref)
    return convert_ids(module, Path(source), [sequence_id], meta, Path(output), opts)


def _list_ids(module: ModuleType, source: Path, *split_sets: dict) -> list:
    """要转换的 ``sequence_id``：优先用转换器的 ``list_sequences``，否则取各划分的并集。"""
    if hasattr(module, "list_sequences"):
        return list(module.list_sequences(source))
    LOGGER.warning(f"converter {getattr(module, 'NAME', module.__name__)} has no "
                   "list_sequences(): only sequences listed in a split will be converted")
    return sorted({i for splits in split_sets if splits for ids in splits.values() for i in ids})


def split_sources(module: ModuleType, source: Path) -> dict:
    """收集转换器给出的划分来源：``official``、``grouped``（泄漏时）、``extra`` 与 ``policy``。

    约定（DESIGN 2.4）：``OFFICIAL_SPLITS_LEAK = True`` 时必须提供 ``grouped_splits(source)``；
    ``extra_splits(source)`` 的子集名不得与官方/分组划分重名，也不得包含官方划分中的序列。
    """
    name = getattr(module, "NAME", module.__name__)
    official = {k: list(v) for k, v in module.official_splits(source).items()}
    leak = bool(getattr(module, "OFFICIAL_SPLITS_LEAK", False))
    policy: dict = {"default": "official", "official_splits_leak": leak}
    grouped = None
    if leak:
        if not callable(getattr(module, "grouped_splits", None)):
            raise ConverterError(f"converter {name} declares OFFICIAL_SPLITS_LEAK but has no "
                                 "grouped_splits(source)")
        grouped = {k: list(v) for k, v in module.grouped_splits(source).items()}
        policy.update(
            default="grouped",
            reason=str(getattr(module, "OFFICIAL_SPLITS_LEAK_REASON",
                               "declared by the converter (OFFICIAL_SPLITS_LEAK = True)")),
            official_prefix=OFFICIAL_PREFIX,
            official_splits=sorted(OFFICIAL_PREFIX + k for k in official),
        )
    extra = {}
    if callable(getattr(module, "extra_splits", None)):
        extra = {k: list(v) for k, v in module.extra_splits(source).items()}
        base = grouped if grouped is not None else official
        clash = sorted(set(extra) & set(base))
        if clash:
            raise ConverterError(f"converter {name}: extra_splits reuse split names {clash}")
        listed = {i for ids in official.values() for i in ids}
        overlap = sorted({i for ids in extra.values() for i in ids} & listed)
        if overlap:
            raise ConverterError(f"converter {name}: extra_splits contain {len(overlap)} sequences "
                                 f"that are in the official splits (e.g. {overlap[0]})")
        notes = dict(getattr(module, "EXTRA_SPLIT_NOTES", {}) or {})
        if extra:
            policy["extra_splits"] = {k: notes.get(k, "extra subset declared by the converter")
                                      for k in sorted(extra)}
    for splits in (official, grouped or {}, extra):
        bad = sorted(k for k in splits if split_family(k))
        if bad:
            raise ConverterError(f"converter {name}: split names must not start with "
                                 f"{OFFICIAL_PREFIX!r}: {bad}")
    return {"official": official, "grouped": grouped, "extra": extra, "policy": policy}


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

    单条序列的解析或写出失败只会让该序列变成 ``rejected``（原因写入转换报告），无论串行还是多进程：
    清单与转换报告总会写出，每个 ``sequence_id`` 都有一个结论。
    """
    module = load_converter(converter if converter is not None else name)
    source = Path(source).expanduser()
    output = Path(output).expanduser() if output else datasets_dir() / name
    output.mkdir(parents=True, exist_ok=True)
    conv_id = f"{module.NAME}@{module.VERSION}"
    meta = {"dataset": name, "converter": conv_id, "license": str(module.LICENSE)}
    opts = dict(rate=rate, gap_threshold=gap_threshold, compression=compression,
                min_duration=min_duration, edge_min_run=DEFAULT_EDGE_MIN_RUN)

    sources = split_sources(module, source)
    previous = _previous_records(output)
    ids = _list_ids(module, source, sources["official"], sources["grouped"], sources["extra"])
    if only:
        wanted = set(only)
        missing = sorted(wanted - set(ids))
        ids = [i for i in ids if i in wanted]
        if missing and hasattr(module, "list_sequences"):
            LOGGER.warning(f"convert {name}: {len(missing)} requested ids are unknown to the "
                           f"converter: {missing[:5]}")
        elif missing:  # 无 list_sequences 时不能判断 id 是否存在，交给转换器
            ids = sorted(wanted)
    if not overwrite:
        ids = [i for i in ids if not (previous.get(i, {}).get("status") == "accepted"
                                      and (output / previous[i]["file"]).exists())]

    LOGGER.info(f"convert {name}: {conv_id}, source={source}, output={output}, "
                f"workers={workers}")
    records: list = []
    total = len(ids)
    counter = {"n": 0}

    def log(record: dict) -> None:
        counter["n"] += 1
        _log_record(record, counter["n"], total)

    if workers > 1 and total > 1:
        ref = converter_ref(module)
        done: set = set()
        try:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_worker, ref, str(source), i, meta, str(output), opts): i
                           for i in ids}
                for fut in as_completed(futures):
                    sequence_id = futures[fut]
                    try:
                        batch = fut.result()
                    except Exception as exc:  # noqa: BLE001 - 子进程崩溃（含 BrokenProcessPool）
                        batch = [_rejected(sequence_id,
                                           f"worker failed: {type(exc).__name__}: {exc}")]
                    for record in batch:
                        done.add(record["sequence_id"])
                        records.append(record)
                        log(record)
        except BrokenProcessPool as exc:  # pragma: no cover - 池整体损坏时兜底
            LOGGER.error(f"convert {name}: worker pool broke ({exc})")
        for sequence_id in ids:  # 池损坏时仍要给每条 id 一个结论
            if sequence_id not in done:
                records.append(_rejected(sequence_id, "worker did not return a record "
                                                      "(process pool broken?)"))
    else:
        records = convert_ids(module, source, ids, meta, output, opts, on_record=log)

    # 合并本次未处理的历史记录（文件仍在时）
    done = {r["sequence_id"] for r in records}
    for sid, rec in previous.items():
        if sid in done:
            continue
        if rec.get("status") != "accepted" or (output / rec.get("file", "")).exists():
            records.append(rec)
    return write_dataset_files(output, name, meta, sources["official"], records, opts,
                               val_fraction=val_fraction, seed=seed, source=source,
                               grouped=sources["grouped"], extra=sources["extra"],
                               policy=sources["policy"])


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


def _resolve_all(official: dict, grouped: Optional[dict], extra: dict, accepted: dict,
                 groups: dict, weights: dict, val_fraction: float, seed: int) -> tuple:
    """按划分策略合并各来源，返回 ``(splits, method, missing)``（官方泄漏族带前缀）。"""
    kw = dict(fraction=val_fraction, seed=seed, weights=weights)
    if grouped is None:
        splits, method, missing = resolve_splits(official, accepted, groups, **kw)
    else:
        splits, method, missing = resolve_splits(
            grouped, accepted, groups, label="grouped (converter grouped_splits)", **kw)
        off, off_method, off_missing = resolve_splits(
            official, accepted, groups,
            label="official (leaks; kept for literature comparison only)", **kw)
        for key, ids in off.items():
            splits[OFFICIAL_PREFIX + key] = ids
            method[OFFICIAL_PREFIX + key] = off_method[key]
        missing.update({OFFICIAL_PREFIX + k: v for k, v in off_missing.items()})
    if extra:
        ext, ext_method, ext_missing = resolve_splits(
            extra, accepted, groups, label="extra (converter extra_splits; not in official splits)",
            **{**kw, "fraction": 0.0})
        ext.pop("val", None)  # extra 子集从不生成 val（也不应含 train）
        ext_method.pop("val", None)
        splits.update(ext)
        method.update(ext_method)
        missing.update(ext_missing)
    return splits, method, missing


def write_dataset_files(output: Path, name: str, meta: dict, official: dict, records: list,
                        opts: dict, *, val_fraction: float, seed: int,
                        source: Optional[Path] = None, grouped: Optional[dict] = None,
                        extra: Optional[dict] = None, policy: Optional[dict] = None) -> dict:
    """根据转换记录写出划分文件、``dataset.json`` 与 ``conversion_report.json``。

    ``grouped`` 非空表示官方划分泄漏：默认划分取 ``grouped``，官方划分以 ``official_`` 前缀写出；
    ``extra`` 为转换器声明的附加子集（原样写出）。
    """
    output = Path(output)
    records = sorted(records, key=lambda r: r["sequence_id"])
    accepted = {r["sequence_id"]: r for r in records if r["status"] == "accepted"}
    rejected = [r for r in records if r["status"] != "accepted"]
    groups = {sid: r["group_id"] for sid, r in accepted.items()}
    weights = {sid: r["duration_s"] for sid, r in accepted.items()}
    extra = dict(extra or {})
    if "train" in extra:
        raise ConverterError("extra_splits must not define 'train'")
    policy = dict(policy or {"default": "official", "official_splits_leak": grouped is not None})
    splits, method, missing = _resolve_all(official, grouped, extra, accepted, groups, weights,
                                           val_fraction, seed)
    split_dir = output / "splits"
    if split_dir.is_dir():
        for old in split_dir.glob("*.txt"):
            if old.stem not in splits:
                old.unlink()
    for split, ids in splits.items():
        write_split(split_dir / f"{split}.txt", ids)
    leakage = check_leakage(splits, groups)
    if "official" in leakage:
        policy["official_leaks"] = leakage["official"]["leaks"]
    assigned = {i for key, ids in splits.items() if not split_family(key) for i in ids}
    unassigned = sorted(set(accepted) - assigned)
    duplicates = find_duplicates({sid: r.get("imu_sha256") for sid, r in accepted.items()})

    keys = ("file", "sha256", "imu_sha256", "num_samples", "duration_s", "distance_m", "group_id",
            "subject_id", "placement", "valid_fraction", "valid_fraction_device",
            "source_sample_rate_hz")
    sequences = {sid: {k: r[k] for k in keys if k in r} for sid, r in accepted.items()}
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
                       "edge_min_run_s": opts.get("edge_min_run", DEFAULT_EDGE_MIN_RUN),
                       "resampling": "DESIGN 2.3 (see per-sequence 'resampling' attribute)"},
        "split_policy": policy,
        "split_method": method,
        "splits": {s: len(ids) for s, ids in splits.items()},
        "unassigned": unassigned,
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
        "duplicates": duplicates,
        "leakage": leakage,
        "warnings_total": int(sum(len(r.get("warnings", [])) for r in records)),
        "sequences": records,
    }
    json_save(output / REPORT, report)
    level = LOGGER.info if leakage["ok"] and not duplicates else LOGGER.warning
    level(f"convert {name}: accepted {len(accepted)}, rejected {len(rejected)}, "
          f"splits {manifest['splits']}, {manifest['statistics']['total_duration_h']:.2f} h"
          + ("" if leakage["ok"] else f", LEAKAGE: {leakage['leaks']}"))
    if "official" in leakage:
        LOGGER.info(f"convert {name}: default splits are grouped ({policy.get('reason', '')}); "
                    f"official splits kept as {OFFICIAL_PREFIX}*.txt, leaks: "
                    f"{leakage['official']['leaks']}")
    if duplicates:
        LOGGER.warning(f"convert {name}: sequences with identical IMU content: {duplicates}")
    if unassigned:
        LOGGER.warning(f"convert {name}: {len(unassigned)} accepted sequences are in no split "
                       f"(listed as 'unassigned' in {MANIFEST}): {unassigned[:5]}")
    return manifest


def default_workers() -> int:
    return max(1, min(8, (os.cpu_count() or 2) // 2))
