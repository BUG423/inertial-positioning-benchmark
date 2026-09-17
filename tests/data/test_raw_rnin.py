"""RNIN（SenseINS）真实数据抽样检查；数据不存在时跳过。路径可用环境变量 ``IPB_RAW_RNIN`` 覆盖。"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from inertial_benchmark.data.converters import _rig_utils as rig
from inertial_benchmark.data.converters import rnin

SOURCE = Path(os.environ.get("IPB_RAW_RNIN", "/workspace/webCodex/datasets/imu_odometry/raw/_staging/RNIN"))
pytestmark = pytest.mark.skipif(not (SOURCE / "data" / "data_train").is_dir() and not (SOURCE / "data_train").is_dir(),
                                reason="RNIN raw data not available")


def load(sequence_id):
    return next(rnin.iter_raw_sequences(SOURCE, only=[sequence_id]))


def test_official_split_sizes():
    splits = rnin.official_splits(SOURCE)
    assert {k: len(v) for k, v in splits.items()} == {"train": 241, "val": 51, "test": 9}
    assert set(rnin.list_sequences(SOURCE)) == set().union(*splits.values())


def test_session_table_covers_all_sequences():
    table = rnin.session_table(str(rnin._root(SOURCE)))
    assert len(table) == 301
    assert len(set(table.values())) > 10


def test_vio_sequence_keeps_uncompensated_imu():
    raw = load("train_0")
    assert raw.rejected is None, raw.rejected
    assert raw.attrs["reference_type"] == "vio" and raw.velocity is not None
    assert "vio_bias" in raw.attrs and json.loads(raw.attrs["vio_bias"])["rows_with_bias"] >= 0
    assert raw.check_shapes() == []
    stats = rig.parse_stats_note(raw.notes)
    assert stats["gravity_error"] < 0.5 and stats["gyro_window_error_median_deg_debiased"] < 5.0


def test_gt_sequence_is_leveled_and_time_aligned():
    raw = load("train_115")
    assert raw.rejected is None, raw.rejected
    assert raw.attrs["reference_type"] == "gt" and raw.velocity is None
    assert any("leveled" in n for n in raw.notes)
    assert any(n.startswith("time offset corrected") for n in raw.notes)


def test_frozen_gt_falls_back_to_vio_like_the_official_loader():
    raw = load("train_122")
    assert raw.rejected is None
    assert raw.attrs["reference_type"] == "vio"
    assert any("official gt criterion" in n for n in raw.notes)
