"""PedLocData 解析器、泄漏审计与分组划分的合成数据单元测试。"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from inertial_benchmark.data.converters import _rig_utils as rig
from inertial_benchmark.data.converters import pedlocdata as pl
from inertial_benchmark.data.converters.base import RAW_REQUIRED_ATTRS

T0 = 1.72e9

# (文件前缀, 官方组, 切片名, 种子, 起始时间偏移 s, 是否把四元数错存为 xyzw)
YT_SLICES = [
    ("train", "aaa_F1_server_0_0", 1, 0.0, False),
    ("valid", "aaa_F1_server_0_1", 2, 45.0, False),
    ("test", "aaa_F1_server_0_2", 3, 90.0, False),
    ("train", "bbb_B1_server_1_0", 4, 1000.0, False),
    ("train", "bbb_B1_server_1_1", 5, 1040.0, False),
    ("valid", "ccc_F2_server_2_0", 6, 2000.0, False),
    ("test", "ddd_F3_server_0_0", 7, 3000.0, False),
    ("train", "eee_B3_server_0_0", 8, 4000.0, True),
]
DEMO_SLICES = [
    ("train", "mid360_2025-11-01_142720_00", 11, 0.0, False),
    ("test", "mid360_2025-11-01_142720_01", 12, 30.0, False),
    ("valid", "mid360_2025-11-02_191202_00", 13, 5000.0, False),
]


def write_file(path, slices, duration=30.0):
    with h5py.File(path, "w") as handle:
        para = handle.create_group("para")
        para["mean"] = np.array([0, 0, 9.8, 0, 0, 0.0])
        para["std"] = np.ones(6)
        for group in ("train", "valid", "test"):
            handle.create_group(group)
        for group, name, seed, offset, wrong_order in slices:
            m = rig.simulate_rig_motion(duration, rate=200.0, seed=seed)
            node = handle[group].create_group(name)
            n = len(m["time"])
            node["ts"] = T0 + offset + m["time"]
            node["acc"] = m["accelerometer"]
            node["gyr"] = m["gyroscope"]
            node["pos"] = m["position"] + np.array([offset * 0.1, 0.0, 0.0])
            node["qua"] = rig.wxyz_to_xyzw(m["orientation"]) if wrong_order else m["orientation"]
            for key, width in (("acc_bias", 3), ("gyr_bias", 3), ("mag", 3), ("motion_qua", 4)):
                node[key] = np.zeros((n, width))


@pytest.fixture(scope="module")
def ped_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("ped") / "PedLocData"
    root.mkdir()
    write_file(root / pl.FILES["yt"], YT_SLICES)
    write_file(root / pl.FILES["demo"], DEMO_SLICES)
    return root


def test_official_splits_keep_the_published_assignment(ped_root):
    splits = pl.official_splits(ped_root.parent)
    assert sorted(splits["train"]) == sorted(
        ["yt_aaa_F1_server_0_0", "yt_bbb_B1_server_1_0", "yt_bbb_B1_server_1_1", "yt_eee_B3_server_0_0",
         "demo_mid360_2025-11-01_142720_00"])
    assert sorted(splits["val"]) == ["demo_mid360_2025-11-02_191202_00", "yt_aaa_F1_server_0_1", "yt_ccc_F2_server_2_0"]
    assert splits["test_yt"] == ["yt_aaa_F1_server_0_2", "yt_ddd_F3_server_0_0"]
    assert splits["test_demo"] == ["demo_mid360_2025-11-01_142720_01"]
    listed = pl.list_sequences(ped_root)
    assert len(listed) == len(YT_SLICES) + len(DEMO_SLICES)
    assert set(listed) == set(splits["train"]) | set(splits["val"]) | set(splits["test"])


def test_slices_parse_with_grouping_attributes(ped_root):
    raws = {r.sequence_id: r for r in pl.iter_raw_sequences(ped_root, only=["yt_aaa_F1_server_0_1",
                                                                             "demo_mid360_2025-11-01_142720_01"])}
    raw = raws["yt_aaa_F1_server_0_1"]
    assert raw.rejected is None, raw.rejected
    assert raw.check_shapes() == [] and set(RAW_REQUIRED_ATTRS) <= set(raw.attrs)
    m = rig.simulate_rig_motion(30.0, rate=200.0, seed=2)
    np.testing.assert_allclose(raw.accelerometer, m["accelerometer"])
    np.testing.assert_allclose(raw.orientation, m["orientation"], atol=1e-12)
    assert raw.attrs["subject_id"] == "aaa" and raw.attrs["group_id"] == "yt_aaa"
    assert raw.attrs["recording_id"] == "yt_aaa_F1_server_0" and raw.attrs["slice_index"] == 1
    assert raw.attrs["floor"] == "F1" and raw.attrs["official_split"] == "val"
    assert raw.attrs["start_time_unix"] == pytest.approx(T0 + 45.0)
    assert "/valid/aaa_F1_server_0_1" in raw.attrs["source_files"]
    stats = rig.parse_stats_note(raw.notes)
    # 30 s 片段首尾速度不同，水平加速度均值不为零，重力误差允许 0.15
    assert stats["gravity_error"] < 0.15 and stats["gyro_window_error_median_deg"] < 0.1
    demo = raws["demo_mid360_2025-11-01_142720_01"]
    assert demo.rejected is None and demo.attrs["group_id"] == "demo_2025-11-01_142720"
    assert demo.attrs["subject_id"] == "unknown"


def test_wrong_quaternion_order_is_rejected(ped_root):
    raw = next(pl.iter_raw_sequences(ped_root, only=["yt_eee_B3_server_0_0"]))
    assert raw.rejected and "physical check failed" in raw.rejected


def test_nonzero_placeholder_is_reported(tmp_path):
    path = tmp_path / pl.FILES["yt"]
    write_file(path, [("train", "fff_F1_server_0_0", 9, 0.0, False)], duration=15.0)
    with h5py.File(path, "r+") as handle:
        handle["train/fff_F1_server_0_0/gyr_bias"][0, 0] = 1.0
    raw = next(pl.iter_raw_sequences(tmp_path))
    assert any("gyr_bias" in n and "non-zero" in n for n in raw.notes)


def test_audit_detects_recording_level_leakage(ped_root):
    report = pl.audit_official_splits(ped_root)
    yt = report["yt"]
    assert yt["recordings"] == 5 and yt["recordings_spanning_splits"] == 1
    assert yt["consecutive_slice_pairs"] == 3 and yt["consecutive_pairs_in_different_splits"] == 2
    assert yt["per_split"]["test"]["adjacent_slice_in_other_split"] == 1
    assert yt["gap_s_percentiles_0_50_100"][0] == pytest.approx(10.005, abs=1e-3)
    assert report["demo"]["recordings_spanning_splits"] == 1


def test_grouped_splits_are_leak_free_and_deterministic(ped_root):
    rows = pl.slice_table(ped_root)
    a = pl.grouped_splits(ped_root, rows=rows)
    b = pl.grouped_splits(ped_root, rows=rows)
    assert a == b
    group_of = {r["sequence_id"]: r["group"] for r in rows}
    seen = {}
    for split in ("train", "val", "test"):
        for sid in a[split]:
            assert seen.setdefault(group_of[sid], split) == split, f"group {group_of[sid]} leaks"
    all_ids = sorted(a["train"] + a["val"] + a["test"])
    assert all_ids == sorted(group_of)
    assert sorted(a["test_yt"] + a["test_demo"]) == sorted(a["test"])
    assert a["test"] and a["val"]
