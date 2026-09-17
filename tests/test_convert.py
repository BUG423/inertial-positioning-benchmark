import json
from pathlib import Path

import numpy as np
import pytest
from synthetic import Motion

from inertial_benchmark.data.convert import convert_dataset
from inertial_benchmark.data.converters import (
    ConverterError,
    available_converters,
    load_converter,
)
from inertial_benchmark.data.format import load_sequence, validate
from inertial_benchmark.data.manifest import check_dataset, resolve_dataset, sha256_file
from inertial_benchmark.data.splits import (
    check_leakage,
    group_holdout,
    read_split,
    resolve_splits,
)

FAKE = Path(__file__).with_name("fake_converter.py")


def write_spec(source: Path, entries: list) -> Path:
    source.mkdir(parents=True, exist_ok=True)
    (source / "spec.json").write_text(json.dumps({"sequences": entries}))
    return source


def default_entries() -> list:
    entries = []
    for k in range(8):
        entries.append({"id": f"tr{k}", "seed": k, "group": f"subj{k % 4}", "split": "train",
                        "imu_rate": 100.0 if k % 2 else 250.0, "pose_rate": 100.0})
    entries += [
        {"id": "te0", "seed": 20, "group": "subj9", "split": "test+test_unseen",
         "imu_rate": 200.0, "jitter": 0.1, "duplicates": 5, "with_device": True},
        {"id": "te1", "seed": 21, "group": "subj0", "split": "test+test_seen",
         "imu_gaps": [[3.0, 3.5]]},
        {"id": "bad_rej", "kind": "reject", "split": "train"},
        {"id": "bad_grav", "kind": "bad_gravity", "seed": 3, "split": "train"},
        {"id": "bad_short", "kind": "short", "seed": 4, "split": "test"},
        {"id": "nan_ok", "kind": "nan", "seed": 5, "group": "subj1", "split": "train"},
    ]
    return entries


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    root = tmp_path_factory.mktemp("convert")
    source = write_spec(root / "raw", default_entries())
    out = root / "ipb" / "fake"
    manifest = convert_dataset("fake", source, out, converter=FAKE, workers=1)
    return source, out, manifest


def test_manifest_and_files(converted):
    source, out, manifest = converted
    assert manifest["dataset"] == "fake"
    assert manifest["converter"] == "fake@0.1"
    assert set(manifest["sequences"]) == {f"tr{k}" for k in range(8)} | {"te0", "te1", "nan_ok"}
    for entry in manifest["sequences"].values():
        path = out / entry["file"]
        assert path.exists()
        assert sha256_file(path) == entry["sha256"]
        assert entry["duration_s"] > 10
        assert entry["distance_m"] > 1
    stats = manifest["statistics"]
    assert stats["num_sequences"] == 11
    assert stats["by_split"]["test"]["num_sequences"] == 2
    assert manifest["split_method"]["val"]["method"] == "group_holdout"
    assert manifest["split_method"]["test"] == "official"
    assert len(manifest["fingerprint"]) == 64


def test_splits_and_report(converted):
    _, out, manifest = converted
    splits = {p.stem: read_split(p) for p in (out / "splits").glob("*.txt")}
    assert set(splits) == {"train", "val", "test", "test_seen", "test_unseen"}
    assert splits["test"] == ["te0", "te1"]
    assert splits["val"], "val must be generated from train"
    groups = {k: v["group_id"] for k, v in manifest["sequences"].items()}
    train_groups = {groups[i] for i in splits["train"]}
    val_groups = {groups[i] for i in splits["val"]}
    assert not train_groups & val_groups
    report = json.loads((out / "conversion_report.json").read_text())
    assert report["accepted"] == 11 and report["rejected"] == 3
    reasons = {r["sequence_id"]: r for r in report["sequences"] if r["status"] == "rejected"}
    assert "missing pose file" in reasons["bad_rej"]["reason"]
    assert "shorter" in reasons["bad_short"]["reason"]
    assert any("-z" in e for e in reasons["bad_grav"]["errors"])
    assert report["missing_from_splits"]["test"] == ["bad_short"]
    assert report["leakage"]["ok"]
    te0 = next(r for r in report["sequences"] if r["sequence_id"] == "te0")
    assert any("dropped duplicate" in n for n in te0["notes"])


def test_converted_sequences_are_faithful(converted):
    _, out, _ = converted
    seq = load_sequence(out / "sequences" / "tr1.h5", validate=True)
    assert seq.attrs["converter"] == "fake@0.1"
    assert seq.attrs["source_sample_rate_hz"] == pytest.approx(100.0)
    assert "resample_poly" in seq.attrs["resampling"]
    assert seq.attrs["source_files"] == ["tr1.npz"]
    motion = Motion(1)
    # 原始时钟起点为 1000 s（位姿），网格起点即位姿起点
    t = seq.timestamp
    inner = (t > 0.5) & (t < t[-1] - 0.5)
    np.testing.assert_allclose(seq.position[inner], motion.position(t[inner]), atol=1e-4)
    np.testing.assert_allclose(seq.accelerometer[inner], motion.accelerometer(t[inner]),
                               atol=0.05)
    np.testing.assert_allclose(seq.gyroscope[inner], motion.gyroscope(t[inner]), atol=0.01)

    te0 = load_sequence(out / "sequences" / "te0.h5", validate=True)
    assert te0.device_orientation is not None
    assert "linear interpolation" in te0.attrs["resampling"]
    te1 = load_sequence(out / "sequences" / "te1.h5")
    gap = (te1.timestamp > 3.02) & (te1.timestamp < 3.48)
    assert not te1.valid_imu[gap].any()
    assert te1.valid_pose.all()
    nan_ok = load_sequence(out / "sequences" / "nan_ok.h5")
    assert not nan_ok.valid_imu.all()
    assert np.isfinite(nan_ok.gyroscope).all()
    assert validate(nan_ok).ok


def test_check_dataset(converted, tmp_path):
    _, out, _ = converted
    rep = check_dataset(out, full=True, verify_hash=True)
    assert rep["ok"], rep["errors"]
    assert any("group_id" in w for w in rep["warnings"])  # test_seen 与 train 共享受试者
    spec = resolve_dataset(out)
    assert spec.name == "fake"
    assert spec.test_splits == ["test", "test_seen", "test_unseen"]
    assert len(spec.sequence_paths("train")) == len(read_split(out / "splits" / "train.txt"))


def test_check_dataset_detects_problems(converted, tmp_path):
    import shutil

    _, out, _ = converted
    copy = tmp_path / "fake"
    shutil.copytree(out, copy)
    (copy / "sequences" / "tr0.h5").unlink()
    with open(copy / "splits" / "test_unseen.txt", "a") as f:
        f.write("tr2\n")
    rep = check_dataset(copy)
    text = "\n".join(rep["errors"])
    assert "tr0: listed" in text
    assert "test_unseen" in text


def test_parallel_matches_sequential(converted, tmp_path):
    source, out, manifest = converted
    out2 = tmp_path / "par"
    manifest2 = convert_dataset("fake", source, out2, converter=FAKE, workers=3)
    assert manifest2["fingerprint"] == manifest["fingerprint"]
    for sid, entry in manifest["sequences"].items():
        assert manifest2["sequences"][sid]["sha256"] == entry["sha256"]


def test_partial_and_incremental_conversion(tmp_path):
    source = write_spec(tmp_path / "raw", default_entries()[:4])
    out = tmp_path / "out"
    m1 = convert_dataset("fake", source, out, converter=FAKE, only=["tr0", "tr1"])
    assert set(m1["sequences"]) == {"tr0", "tr1"}
    m2 = convert_dataset("fake", source, out, converter=FAKE, only=["tr2"])
    assert set(m2["sequences"]) == {"tr0", "tr1", "tr2"}
    before = (out / "sequences" / "tr0.h5").stat().st_mtime_ns
    m3 = convert_dataset("fake", source, out, converter=FAKE, overwrite=False)
    assert set(m3["sequences"]) == {"tr0", "tr1", "tr2", "tr3"}
    assert (out / "sequences" / "tr0.h5").stat().st_mtime_ns == before


def test_converter_exception_is_isolated_in_workers(tmp_path):
    entries = default_entries()[:3] + [{"id": "boom", "kind": "raise", "split": "train"}]
    source = write_spec(tmp_path / "raw", entries)
    manifest = convert_dataset("fake", source, tmp_path / "out", converter=FAKE, workers=2)
    assert "boom" not in manifest["sequences"]
    report = json.loads((tmp_path / "out" / "conversion_report.json").read_text())
    boom = next(r for r in report["sequences"] if r["sequence_id"] == "boom")
    assert "cannot parse" in boom["reason"]


def test_load_converter_errors(tmp_path):
    with pytest.raises(ConverterError, match="unknown converter"):
        load_converter("definitely_not_a_dataset")
    bad = tmp_path / "bad_conv.py"
    bad.write_text("NAME = 'x'\n")
    with pytest.raises(ConverterError, match="lacks"):
        load_converter(bad)
    assert "base" not in available_converters()


def test_group_holdout_properties():
    ids = [f"s{i}" for i in range(40)]
    groups = {s: f"g{i % 10}" for i, s in enumerate(ids)}
    train, val, info = group_holdout(ids, groups, 0.1, seed=0)
    assert set(train) | set(val) == set(ids) and not set(train) & set(val)
    assert {groups[i] for i in val}.isdisjoint({groups[i] for i in train})
    assert len(val) == 4
    again = group_holdout(ids, groups, 0.1, seed=0)
    assert again[1] == val
    other = group_holdout(ids, groups, 0.1, seed=1)
    assert other[1] != val
    # 只有一个组：退化并警告
    _, val1, info1 = group_holdout(ids, {s: "same" for s in ids}, 0.1)
    assert "warning" in info1 and 1 <= len(val1) <= 5
    # 一个大组不应被整体放进 val
    big = {s: ("big" if i < 30 else f"g{i}") for i, s in enumerate(ids)}
    _, val2, _ = group_holdout(ids, big, 0.1, seed=0)
    assert len(val2) <= 6


def test_check_leakage_and_resolve():
    splits = {"train": ["a", "b"], "val": ["c"], "test": ["d", "b"], "test_unseen": ["d"],
              "test_seen": ["e"]}
    groups = {"a": "g1", "b": "g1", "c": "g2", "d": "g1", "e": "g1"}
    rep = check_leakage(splits, groups)
    assert not rep["ok"]
    assert any("train/test share 1 sequences" in m for m in rep["leaks"])
    assert any("test_unseen" in m for m in rep["leaks"])
    assert not any("test_seen" in m for m in rep["leaks"])

    official = {"train": ["a", "b", "x"], "test": ["d"]}
    out, method, missing = resolve_splits(official, ["a", "b", "d"], {"a": "1", "b": "2"})
    assert missing == {"train": ["x"]}
    assert sorted(out["train"] + out["val"]) == ["a", "b"] and len(out["val"]) == 1
    assert method["val"]["method"] == "group_holdout"


def test_read_split_ignores_comments(tmp_path):
    p = tmp_path / "s.txt"
    p.write_text("# header\na\n\nb  # trailing\n")
    assert read_split(p) == ["a", "b"]
