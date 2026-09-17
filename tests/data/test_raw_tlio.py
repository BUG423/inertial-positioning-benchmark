"""TLIO 真实数据抽样检查；数据不存在时跳过。路径可用环境变量 ``IPB_RAW_TLIO`` 覆盖。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from inertial_benchmark.data.converters import _rig_utils as rig
from inertial_benchmark.data.converters import tlio

SOURCE = Path(os.environ.get("IPB_RAW_TLIO", "/workspace/webCodex/datasets/imu_odometry/raw/_staging/TLIO"))
pytestmark = pytest.mark.skipif(not (SOURCE / "tlio_golden" / "train_list.txt").exists()
                                and not (SOURCE / "train_list.txt").exists(), reason="TLIO raw data not available")


def test_official_split_sizes_and_disjointness():
    splits = tlio.official_splits(SOURCE)
    assert {k: len(v) for k, v in splits.items()} == {"train": 283, "val": 35, "test": 36}
    union = set().union(*splits.values())
    assert len(union) == 354 and union == set(tlio.list_sequences(SOURCE))


@pytest.mark.parametrize("kind", ["train", "val", "test"])
def test_sampled_sequences_pass_physical_checks(kind):
    sequence_id = tlio.official_splits(SOURCE)[kind][0]
    raw = next(tlio.iter_raw_sequences(SOURCE, only=[sequence_id]))
    assert raw.rejected is None, raw.rejected
    assert raw.check_shapes() == []
    stats = rig.parse_stats_note(raw.notes)
    assert stats["gravity_error"] < 0.1
    assert stats["gyro_window_error_median_deg"] < 1.0
    assert 0.95 < stats["acc_scale"] < 1.05  # 位置单位为米
    assert raw.velocity is not None and raw.attrs["placement"] == "head"


def test_fast_loop_sequence_is_kept_with_warning():
    raw = next(tlio.iter_raw_sequences(SOURCE, only=["253159831907410"]))
    assert raw.rejected is None
    assert any("fast motion" in n for n in raw.notes)
