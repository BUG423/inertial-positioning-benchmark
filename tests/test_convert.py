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
from inertial_benchmark.data.manifest import (
    MANIFEST,
    check_dataset,
    resolve_dataset,
    sha256_file,
)
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


@pytest.mark.parametrize("workers", [1, 2])
def test_converter_exception_is_isolated(tmp_path, workers):
    """转换器抛异常只影响该序列：其余序列照常转换，清单与报告照常写出。"""
    entries = default_entries()[:3] + [{"id": "boom", "kind": "raise", "split": "train"},
                                       {"id": "tr9", "seed": 9, "group": "subj9",
                                        "split": "test"}]
    source = write_spec(tmp_path / "raw", entries)
    out = tmp_path / f"out{workers}"
    manifest = convert_dataset("fake", source, out, converter=FAKE, workers=workers)
    assert set(manifest["sequences"]) == {"tr0", "tr1", "tr2", "tr9"}
    assert (out / MANIFEST).exists() and (out / "conversion_report.json").exists()
    report = json.loads((out / "conversion_report.json").read_text())
    assert report["rejected"] == 1
    boom = next(r for r in report["sequences"] if r["sequence_id"] == "boom")
    assert "cannot parse" in boom["reason"]
    assert {r["sequence_id"] for r in report["sequences"]} == set(manifest["sequences"]) | {"boom"}


def test_worker_crash_still_writes_the_manifest(tmp_path):
    """子进程被杀（模拟 OOM）后，剩余 id 记为 rejected，清单与报告仍然写出。"""
    entries = default_entries()[:3] + [{"id": "kaboom", "kind": "crash", "split": "train"}]
    source = write_spec(tmp_path / "raw", entries)
    out = tmp_path / "out"
    manifest = convert_dataset("fake", source, out, converter=FAKE, workers=2)
    assert (out / MANIFEST).exists()
    report = json.loads((out / "conversion_report.json").read_text())
    assert {r["sequence_id"] for r in report["sequences"]} == {"tr0", "tr1", "tr2", "kaboom"}
    kaboom = next(r for r in report["sequences"] if r["sequence_id"] == "kaboom")
    assert kaboom["status"] == "rejected"
    assert "worker failed" in kaboom["reason"] or "did not return" in kaboom["reason"]
    assert "kaboom" not in manifest["sequences"]


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


# ---------------------------------------------------------------------------
# 划分策略（DESIGN 2.4）：泄漏官方划分、附加子集、未分配序列、重复内容
# ---------------------------------------------------------------------------

LEAKY_WRAPPER = '''
import sys
sys.path.insert(0, {tests!r})
from fake_converter import *  # noqa: F401,F403
from fake_converter import _spec_all

NAME = "fakeleak"
OFFICIAL_SPLITS_LEAK = True
OFFICIAL_SPLITS_LEAK_REASON = "synthetic: slices of one recording spread over several splits"
{extra}
'''


def make_leaky_converter(tmp_path: Path, with_grouped: bool = True) -> Path:
    extra = ("def grouped_splits(source):\n    return dict(_spec_all(source)['grouped'])\n"
             if with_grouped else "")
    path = tmp_path / ("leaky_conv.py" if with_grouped else "leaky_nogroup_conv.py")
    path.write_text(LEAKY_WRAPPER.format(tests=str(FAKE.parent), extra=extra))
    return path


def leaky_source(root: Path) -> Path:
    # 两个受试者 × 两次录制，每次录制切成 3 片；官方划分按切片随机分配（跨划分泄漏）
    entries, official = [], ["train", "val", "test"]
    k = 0
    for subj in ("s1", "s2", "s3", "s4"):
        for rec in range(2):
            for piece in range(3):
                entries.append({"id": f"{subj}_r{rec}_{piece}", "seed": k, "group": subj,
                                "split": official[(k + rec) % 3]})
                k += 1
    grouped = {
        "train": [e["id"] for e in entries if e["group"] in ("s1", "s2")],
        "val": [e["id"] for e in entries if e["group"] == "s3"],
        "test": [e["id"] for e in entries if e["group"] == "s4"],
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "spec.json").write_text(json.dumps({"sequences": entries, "grouped": grouped}))
    return root


def test_leaky_official_splits_are_demoted(tmp_path):
    conv = make_leaky_converter(tmp_path)
    source = leaky_source(tmp_path / "raw")
    out = tmp_path / "out"
    manifest = convert_dataset("fakeleak", source, out, converter=conv, workers=2)
    files = {p.stem: read_split(p) for p in (out / "splits").glob("*.txt")}
    assert set(files) == {"train", "val", "test", "official_train", "official_val",
                          "official_test"}
    assert files["test"] == sorted(f"s4_r{r}_{i}" for r in range(2) for i in range(3))
    assert len(files["official_train"]) + len(files["official_val"]) \
        + len(files["official_test"]) == 24
    policy = manifest["split_policy"]
    assert policy["default"] == "grouped" and policy["official_splits_leak"] is True
    assert "several splits" in policy["reason"]
    assert policy["official_splits"] == ["official_test", "official_train", "official_val"]
    assert any("official_train/official_test share" in m for m in policy["official_leaks"])
    assert manifest["split_method"]["train"].startswith("grouped")
    assert manifest["split_method"]["official_test"].startswith("official (leaks")
    assert manifest["statistics"]["by_split"]["official_test"]["num_sequences"] == 8
    assert manifest["unassigned"] == []
    report = json.loads((out / "conversion_report.json").read_text())
    assert report["leakage"]["ok"] and not report["leakage"]["official"]["ok"]

    rep = check_dataset(out, full=True)
    assert rep["ok"], rep["errors"]
    assert rep["split_policy"]["default"] == "grouped"
    assert any(w.startswith("official split leakage") for w in rep["warnings"])
    # 默认族与官方族之间的重叠是预期的，不报告
    assert not any("official_train/train" in w or "train/official_train" in w
                   for w in rep["warnings"])


def test_declared_leak_requires_grouped_splits(tmp_path):
    conv = make_leaky_converter(tmp_path, with_grouped=False)
    source = leaky_source(tmp_path / "raw")
    with pytest.raises(ConverterError, match="grouped_splits"):
        convert_dataset("fakeleak", source, tmp_path / "out", converter=conv)


def test_check_flags_undeclared_official_files(tmp_path):
    import shutil

    source = write_spec(tmp_path / "raw", default_entries()[:4])
    out = tmp_path / "out"
    convert_dataset("fake", source, out, converter=FAKE)
    shutil.copy(out / "splits" / "train.txt", out / "splits" / "official_train.txt")
    rep = check_dataset(out)
    assert rep["ok"]
    assert any("does not declare" in w for w in rep["warnings"])


def test_extra_splits_and_unassigned_sequences(tmp_path):
    entries = default_entries()[:6] + [
        {"id": "ux0", "seed": 30, "group": "subjX", "split": ""},
        {"id": "ux1", "seed": 31, "group": "subjX", "split": ""},
        {"id": "orphan", "seed": 32, "group": "subj0", "split": ""},
    ]
    source = write_spec(tmp_path / "raw", entries)
    spec = json.loads((source / "spec.json").read_text())
    spec["extra"] = {"test_unseen_subject": ["ux0", "ux1", "ux_missing"]}
    (source / "spec.json").write_text(json.dumps(spec))
    out = tmp_path / "out"
    manifest = convert_dataset("fake", source, out, converter=FAKE, workers=2)
    splits = {p.stem: read_split(p) for p in (out / "splits").glob("*.txt")}
    assert splits["test_unseen_subject"] == ["ux0", "ux1"]
    for name in ("train", "val", "test"):
        assert not {"ux0", "ux1", "orphan"} & set(splits.get(name, []))
    assert manifest["unassigned"] == ["orphan"]
    assert "orphan" in manifest["sequences"]  # 保留文件，不静默丢弃
    assert manifest["split_method"]["test_unseen_subject"].startswith("extra")
    assert "test_unseen_subject" in manifest["split_policy"]["extra_splits"]
    report = json.loads((out / "conversion_report.json").read_text())
    assert report["missing_from_splits"]["test_unseen_subject"] == ["ux_missing"]
    rep = check_dataset(out)
    assert rep["ok"], rep["errors"]
    assert rep["unassigned"] == ["orphan"]
    assert any("in no default split" in w for w in rep["warnings"])


def test_extra_splits_must_not_touch_official_sequences(tmp_path):
    source = write_spec(tmp_path / "raw", default_entries()[:3])
    spec = json.loads((source / "spec.json").read_text())
    spec["extra"] = {"test_unseen_subject": ["tr0"]}
    (source / "spec.json").write_text(json.dumps(spec))
    with pytest.raises(ConverterError, match="official splits"):
        convert_dataset("fake", source, tmp_path / "out", converter=FAKE)
    spec["extra"] = {"train": ["not_listed"]}
    (source / "spec.json").write_text(json.dumps(spec))
    with pytest.raises(ConverterError, match="reuse split names"):
        convert_dataset("fake", source, tmp_path / "out", converter=FAKE)


def test_duplicate_content_is_reported(tmp_path):
    entries = default_entries()[:4] + [
        {"id": "copy_of_tr1", "kind": "duplicate_of:tr1", "split": "test"}]
    source = write_spec(tmp_path / "raw", entries)
    out = tmp_path / "out"
    manifest = convert_dataset("fake", source, out, converter=FAKE)
    seqs = manifest["sequences"]
    assert seqs["copy_of_tr1"]["imu_sha256"] == seqs["tr1"]["imu_sha256"]
    assert seqs["tr0"]["imu_sha256"] != seqs["tr1"]["imu_sha256"]
    report = json.loads((out / "conversion_report.json").read_text())
    assert report["duplicates"] == [["copy_of_tr1", "tr1"]]
    for full in (False, True):
        rep = check_dataset(out, full=full)
        assert not rep["ok"]
        assert any(e.startswith("duplicate IMU content") and "copy_of_tr1 (test)" in e
                   for e in rep["errors"])


def test_pose_clock_offset_is_respected():
    from synthetic import make_raw_sequence

    from inertial_benchmark.data.convert import raw_to_sequence

    opts = dict(rate=200.0, gap_threshold=0.05, min_duration=2.0)
    offset = 0.07  # 位姿时钟比 IMU 晚 70 ms（如转换器做完时延修正后）
    raw = make_raw_sequence("off", duration=12.0, imu_rate=250.0, pose_rate=100.0, seed=4,
                            pose_offset=offset)
    assert not np.array_equal(raw.imu_time[:5], raw.pose_time[:5])
    seq, record = raw_to_sequence(raw, "fake", "fake@0.1", "CC0", opts)
    assert seq is not None, record
    # 网格起点 = 位姿起点（IMU 早开始 0.3 s），终点 = IMU 终点（位姿晚结束 0.37 s）
    t0 = 1000.0
    start = t0 + offset
    end = t0 - 0.3 + 12.0
    assert len(seq) == int(np.floor((end - start) * 200 + 1e-6)) + 1
    assert seq.valid.all()
    motion = Motion(4)
    tm = seq.timestamp + offset  # 运动时间原点为 t0
    inner = (seq.timestamp > 0.3) & (seq.timestamp < seq.timestamp[-1] - 0.3)
    np.testing.assert_allclose(seq.position, motion.position(tm), atol=1e-3)
    np.testing.assert_allclose(seq.gyroscope[inner], motion.gyroscope(tm[inner]), atol=0.01)
    np.testing.assert_allclose(seq.accelerometer[inner], motion.accelerometer(tm[inner]),
                               atol=0.05)
    assert np.isnan(seq.attrs["start_time_unix"])  # 原始时钟不是 Unix 秒


def test_edge_fragments_are_trimmed_and_noted():
    from synthetic import make_raw_sequence

    from inertial_benchmark.data.convert import raw_to_sequence

    opts = dict(rate=200.0, gap_threshold=0.05, min_duration=2.0)
    # 位姿从 t0 开始；0.4–2.4 s 的 IMU 缺口把开头 0.4 s 变成孤立片段；
    # 11.0–11.5 s 的位姿缺口把结尾 11.5–11.7 s（IMU 终点）变成孤立片段
    raw = make_raw_sequence("edge", duration=12.0, imu_rate=200.0, seed=6,
                            imu_gaps=[(0.4, 2.4)], pose_gaps=[(11.0, 11.5)])
    seq, record = raw_to_sequence(raw, "fake", "fake@0.1", "CC0", opts)
    assert seq is not None, record
    head, tail = record["trimmed_s"]
    assert 2.35 < head < 2.45
    assert 0.65 < tail < 0.75
    assert any(n.startswith("trimmed") for n in record["notes"])
    assert seq.valid[0] and seq.valid[-1] and seq.valid.all()
    motion = Motion(6)
    np.testing.assert_allclose(seq.position, motion.position(seq.timestamp + head), atol=1e-3)
    # 内部缺口不裁剪
    raw = make_raw_sequence("mid", duration=12.0, imu_rate=200.0, seed=6, imu_gaps=[(5.0, 5.5)])
    seq, record = raw_to_sequence(raw, "fake", "fake@0.1", "CC0", opts)
    assert "trimmed_s" not in record and not seq.valid.all()


def test_check_leakage_families():
    splits = {"train": ["a", "b"], "val": ["c"], "test": ["d"],
              "official_train": ["a", "c"], "official_val": ["b"], "official_test": ["d"]}
    groups = {"a": "g1", "b": "g1", "c": "g2", "d": "g3"}
    rep = check_leakage(splits, groups)
    assert rep["ok"], rep["leaks"]  # 默认族无泄漏；两族之间的重叠不检查
    assert all(not (p["splits"][0].startswith("official_") ^ p["splits"][1].startswith("official_"))
               for p in rep["pairs"] + rep["official"]["pairs"])
    assert not rep["official"]["ok"]
    assert rep["official"]["leaks"] == ["official_train/official_val share 1 groups"]


def test_device_orientation_gap_is_masked_separately(tmp_path):
    entries = [{"id": "dev0", "seed": 12, "group": "subj0", "split": "train",
                "with_device": True, "device_gaps": [[4.0, 5.0]], "imu_rate": 200.0},
               {"id": "dev1", "seed": 13, "group": "subj1", "split": "test",
                "with_device": True, "imu_rate": 200.0}]
    source = write_spec(tmp_path / "raw", entries)
    out = tmp_path / "out"
    manifest = convert_dataset("fake", source, out, converter=FAKE)
    seq = load_sequence(out / "sequences" / "dev0.h5", validate=True)
    assert seq.valid_imu.all() and seq.valid_pose.all()  # 设备姿态缺口不拉低 valid/imu
    gap = (seq.timestamp > 4.05) & (seq.timestamp < 4.95)
    assert not seq.valid_device_orientation[gap].any()
    assert seq.valid_device_orientation[seq.timestamp < 3.9].all()
    assert manifest["sequences"]["dev0"]["valid_fraction"] == 1.0
    assert manifest["sequences"]["dev0"]["valid_fraction_device"] < 1.0
    assert manifest["sequences"]["dev1"]["valid_fraction_device"] == 1.0
    report = json.loads((out / "conversion_report.json").read_text())
    dev0 = next(r for r in report["sequences"] if r["sequence_id"] == "dev0")
    assert any("device orientation" in w for w in dev0["warnings"])
