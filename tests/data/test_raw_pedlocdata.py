"""PedLocData 真实数据抽样检查与泄漏审计；数据不存在时跳过。路径可用环境变量 ``IPB_RAW_PEDLOCDATA`` 覆盖。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from inertial_benchmark.data.converters import _rig_utils as rig
from inertial_benchmark.data.converters import pedlocdata as pl

SOURCE = Path(os.environ.get("IPB_RAW_PEDLOCDATA", "/workspace/webCodex/datasets/imu_odometry/raw/PedLocData"))
pytestmark = pytest.mark.skipif(not (SOURCE / pl.FILES["yt"]).exists(), reason="PedLocData raw data not available")


@pytest.fixture(scope="module")
def rows():
    return pl.slice_table(SOURCE)


def test_official_split_sizes():
    splits = pl.official_splits(SOURCE)
    sizes = {k: len(v) for k, v in splits.items()}
    assert sizes == {"train": 1479 + 32, "val": 422 + 9, "test": 213 + 6, "test_yt": 213, "test_demo": 6}
    listed = pl.list_sequences(SOURCE)
    assert len(listed) == 2161 and set(listed) == set(splits["train"]) | set(splits["val"]) | set(splits["test"])


def test_audit_reports_session_level_leakage(rows):
    report = pl.audit_official_splits(SOURCE, rows)
    yt = report["yt"]
    assert yt["slices"] == 2114 and yt["recordings"] == 474
    assert yt["recordings_spanning_splits"] == 329
    assert yt["subjects"] == 27 and yt["subjects_in_all_three_splits"] == 26
    assert yt["per_split"]["test"]["adjacent_slice_in_other_split"] == 201
    assert report["demo"]["gap_s_percentiles_0_50_100"][2] < 0.01  # Demo 切片首尾相接


def test_grouped_splits_have_no_group_overlap(rows):
    splits = pl.grouped_splits(SOURCE, rows=rows)
    group_of = {r["sequence_id"]: r["group"] for r in rows}
    groups = {k: {group_of[i] for i in splits[k]} for k in ("train", "val", "test")}
    assert not groups["train"] & groups["val"] and not groups["train"] & groups["test"]
    assert not groups["val"] & groups["test"]
    assert sum(len(splits[k]) for k in ("train", "val", "test")) == len(rows)


def test_sampled_slices():
    raws = {r.sequence_id: r for r in pl.iter_raw_sequences(
        SOURCE, only=["yt_dzw_F1_server_0_0", "yt_zxc_F1_server_1_0", "demo_mid360_2025-10-27_221831_00"])}
    good = raws["yt_dzw_F1_server_0_0"]
    assert good.rejected is None, good.rejected
    assert good.check_shapes() == [] and good.attrs["group_id"] == "yt_dzw"
    stats = rig.parse_stats_note(good.notes)
    assert stats["gravity_error"] < 0.5 and 0.8 < stats["acc_scale"] < 1.2
    bad = raws["yt_zxc_F1_server_1_0"]
    assert bad.rejected and "gyro/reference" in bad.rejected
    assert raws["demo_mid360_2025-10-27_221831_00"].rejected is None
