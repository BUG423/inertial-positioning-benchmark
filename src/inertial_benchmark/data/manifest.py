"""数据集级信息（DESIGN 2.1）：数据集定位、``dataset.json`` 读写、指纹与完整性检查。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import numpy as np

from ..utils import CFG_DIR, LOGGER, datasets_dir, json_load, yaml_load
from .splits import MAIN_SPLITS, check_leakage, load_splits, read_split, split_family

PathLike = Union[str, Path]
MANIFEST = "dataset.json"
REPORT = "conversion_report.json"


def sha256_file(path: PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def imu_content_hash(gyroscope: np.ndarray, accelerometer: np.ndarray) -> str:
    """IMU 数组内容的 sha256（float32 小端字节 + 形状），与文件属性和压缩无关。

    同一段录制被重复发布（例如换了目录名）时，转换结果的 IMU 逐位相同，据此发现重复。
    """
    h = hashlib.sha256()
    for arr in (gyroscope, accelerometer):
        a = np.ascontiguousarray(arr, dtype="<f4")
        h.update(repr(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def find_duplicates(hashes: Mapping[str, str]) -> list:
    """按内容哈希分组，返回包含多条序列的组（每组为排序后的 ``sequence_id`` 列表）。"""
    by_hash: dict = {}
    for sid, digest in hashes.items():
        if digest:
            by_hash.setdefault(digest, []).append(sid)
    return sorted(sorted(ids) for ids in by_hash.values() if len(ids) > 1)


def fingerprint(sequences: Mapping[str, str], splits: Mapping[str, list]) -> str:
    """数据集指纹：序列文件哈希与划分内容的联合 sha256（与文件修改时间无关）。"""
    payload = json.dumps({"sequences": dict(sorted(sequences.items())),
                          "splits": {k: list(v) for k, v in sorted(splits.items())}},
                         sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass
class DatasetSpec:
    """一个已转换数据集的定位信息与划分映射（来自 ``cfg/datasets/<name>.yaml`` 或目录）。"""

    name: str
    root: Path
    splits: dict = field(default_factory=lambda: {s: s for s in MAIN_SPLITS})
    test_splits: list = field(default_factory=lambda: ["test"])
    notes: str = ""
    strata: Optional[list] = None  # 划分分层/条件维度声明（DESIGN 2.4），None 表示用默认维度

    def split_file(self, split: str) -> Path:
        """逻辑划分名 → ``splits/<file>.txt``；未映射的名字按原名查找。"""
        return self.root / "splits" / f"{self.splits.get(split, split)}.txt"

    def has_split(self, split: str) -> bool:
        return self.split_file(split).exists()

    def sequence_ids(self, split: str) -> list:
        path = self.split_file(split)
        if not path.exists():
            raise FileNotFoundError(f"split {split!r} not found for dataset {self.name!r}: {path}")
        return read_split(path)

    def sequence_path(self, sequence_id: str) -> Path:
        return self.root / "sequences" / f"{sequence_id}.h5"

    def sequence_paths(self, split: str) -> list:
        return [self.sequence_path(i) for i in self.sequence_ids(split)]

    def manifest(self) -> dict:
        path = self.root / MANIFEST
        return json_load(path) if path.exists() else {}

    def fingerprint(self) -> Optional[str]:
        return self.manifest().get("fingerprint")

    def to_dict(self) -> dict:
        return {"name": self.name, "root": str(self.root), "splits": dict(self.splits),
                "test_splits": list(self.test_splits), "strata": self.strata,
                "notes": self.notes}


def dataset_yaml_names() -> list:
    return sorted(p.stem for p in (CFG_DIR / "datasets").glob("*.yaml"))


def dataset_strata(name: str) -> Optional[list]:
    """数据卡 ``cfg/datasets/<name>.yaml`` 声明的条件/分层维度；没有数据卡时返回 ``None``。

    返回 ``None`` 表示由 ``splits.normalise_strata`` 用 ``DEFAULT_STRATA``。
    """
    path = CFG_DIR / "datasets" / f"{name}.yaml"
    if not path.exists():
        return None
    raw = yaml_load(path).get("strata")
    return list(raw) if raw else None


def _spec_from_dict(d: Mapping[str, Any], base: Optional[Path] = None) -> DatasetSpec:
    name = str(d.get("name", "dataset"))
    raw = d.get("path", name)
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = (base or datasets_dir()) / path
    splits = {s: s for s in MAIN_SPLITS}
    splits.update({str(k): str(v) for k, v in (d.get("splits") or {}).items()})
    return DatasetSpec(
        name=name,
        root=path,
        splits=splits,
        test_splits=list(d.get("test_splits") or ["test"]),
        strata=list(d["strata"]) if d.get("strata") else None,
        notes=str(d.get("notes", "")),
    )


def resolve_dataset(data: Union[str, Path, Mapping, DatasetSpec]) -> DatasetSpec:
    """解析 ``data=`` 参数：数据集名、数据集 YAML、已转换目录或 dict。"""
    if isinstance(data, DatasetSpec):
        return data
    if isinstance(data, Mapping):
        return _spec_from_dict(data)
    text = str(data)
    path = Path(text).expanduser()
    if text.endswith((".yaml", ".yml")):
        if not path.exists():
            candidate = CFG_DIR / "datasets" / path.name
            path = candidate if candidate.exists() else path
        return _spec_from_dict(yaml_load(path))
    if path.is_dir() and ((path / MANIFEST).exists() or (path / "sequences").is_dir()):
        spec = DatasetSpec(name=path.name, root=path.resolve())
        manifest = spec.manifest()
        spec.name = manifest.get("dataset", spec.name)
        extra = [s for s in load_splits(path) if s.startswith("test") and s != "test"]
        spec.test_splits = ["test"] + extra if spec.has_split("test") else extra
        return spec
    yaml_path = CFG_DIR / "datasets" / f"{text}.yaml"
    if yaml_path.exists():
        return _spec_from_dict(yaml_load(yaml_path))
    candidate = datasets_dir() / text
    if candidate.is_dir():
        return resolve_dataset(candidate)
    raise FileNotFoundError(
        f"cannot resolve dataset {text!r}: not a YAML, a converted directory, or one of "
        f"{dataset_yaml_names()} (IPB_DATASETS={datasets_dir()})"
    )


def check_dataset(
    data: Union[str, Path, Mapping, DatasetSpec],
    *,
    full: bool = False,
    verify_hash: bool = False,
) -> dict:
    """检查已转换数据集：清单、文件、划分泄漏、重复内容。

    ``full`` 逐条校验并重算 IMU 内容哈希，``verify_hash`` 重算文件 sha256。

    错误（``ok=False``）：文件缺失、默认划分泄漏、IMU 内容重复、逐条校验失败、哈希不符。
    警告：默认划分之间的 ``group_id`` 重叠（例如 seen-subject 设定）、官方泄漏划分
    （``official_*``）的泄漏明细、不在任何默认划分中的序列。
    """
    from .format import load_sequence, validate

    spec = resolve_dataset(data)
    errors, warnings = [], []
    manifest = spec.manifest()
    if not manifest:
        errors.append(f"{spec.root / MANIFEST} missing")
    entries = manifest.get("sequences", {})
    policy = manifest.get("split_policy", {})
    splits = load_splits(spec.root)
    if not splits:
        errors.append("no split files under splits/")
    on_disk = {p.stem for p in (spec.root / "sequences").glob("*.h5")}
    for sid in sorted(set(entries) - on_disk):
        errors.append(f"{sid}: listed in {MANIFEST} but file is missing")
    for sid in sorted(on_disk - set(entries)):
        warnings.append(f"{sid}: file exists but is not listed in {MANIFEST}")
    for name, ids in splits.items():
        absent = [i for i in ids if i not in on_disk]
        if absent:
            errors.append(f"split {name}: {len(absent)} sequences have no file (e.g. {absent[0]})")
    for logical in list(MAIN_SPLITS) + list(spec.test_splits):
        if not spec.has_split(logical):
            warnings.append(f"split {logical!r} ({spec.split_file(logical).name}) not found")

    groups = {sid: e.get("group_id", sid) for sid, e in entries.items()}
    leakage = check_leakage(splits, groups)
    errors += [f"leakage: {m}" for m in leakage["leaks"]]
    for pair in leakage["pairs"]:
        if pair["group_overlap"] and not pair["sequence_overlap"]:
            a, b = pair["splits"]
            warnings.append(f"{a}/{b} share {len(pair['group_overlap'])} group_id values")
    official = leakage.get("official")
    if official is not None:
        if not policy.get("official_splits_leak"):
            warnings.append("official_* split files exist but dataset.json does not declare "
                            "official_splits_leak")
        for m in official["leaks"]:
            warnings.append(f"official split leakage (declared; official_* files are for "
                            f"literature comparison only): {m}")
        if official["ok"]:
            warnings.append("official splits are declared leaky but share no group_id values")
    elif policy.get("official_splits_leak"):
        errors.append("dataset.json declares leaky official splits but no official_*.txt exists")

    assigned = {i for name, ids in splits.items() if not split_family(name) for i in ids}
    unassigned = sorted((set(entries) & on_disk) - assigned)
    if unassigned:
        warnings.append(f"{len(unassigned)} sequences are in no default split "
                        f"(e.g. {unassigned[0]}); see dataset.json 'unassigned'")

    hashes = {sid: e.get("imu_sha256") for sid, e in entries.items()}
    if verify_hash:
        for sid, e in entries.items():
            path = spec.sequence_path(sid)
            if path.exists() and e.get("sha256") and sha256_file(path) != e["sha256"]:
                errors.append(f"{sid}: sha256 mismatch")
    if full:
        for sid in sorted(on_disk):
            try:
                seq = load_sequence(spec.sequence_path(sid))
                rep = validate(seq)
            except Exception as exc:  # noqa: BLE001 - 报告任何读取错误
                errors.append(f"{sid}: cannot load ({exc})")
                continue
            errors += [f"{sid}: {e}" for e in rep.errors]
            warnings += [f"{sid}: {w}" for w in rep.warnings]
            digest = imu_content_hash(seq.gyroscope, seq.accelerometer)
            if hashes.get(sid) and hashes[sid] != digest:
                errors.append(f"{sid}: IMU content hash differs from {MANIFEST}")
            hashes[sid] = digest
    duplicates = find_duplicates(hashes)
    for ids in duplicates:
        where = [f"{i} ({', '.join(n for n, v in splits.items() if i in v) or 'no split'})"
                 for i in ids]
        errors.append("duplicate IMU content: " + " == ".join(where))

    report = {
        "dataset": spec.name,
        "root": str(spec.root),
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "splits": {k: len(v) for k, v in splits.items()},
        "split_policy": policy,
        "statistics": manifest.get("statistics", {}),
        "fingerprint": manifest.get("fingerprint"),
        "leakage": leakage,
        "unassigned": unassigned,
        "duplicates": duplicates,
    }
    level = LOGGER.info if report["ok"] else LOGGER.warning
    level(f"check {spec.name}: {'OK' if report['ok'] else 'FAILED'} "
          f"({len(errors)} errors, {len(warnings)} warnings)")
    return report
