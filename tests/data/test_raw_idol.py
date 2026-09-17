"""IDOL 真实数据抽样检查；数据或 pyarrow 不存在时跳过。路径可用环境变量 ``IPB_RAW_IDOL`` 覆盖。"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")

from inertial_benchmark.data.converters import _rig_utils as rig  # noqa: E402
from inertial_benchmark.data.converters import idol  # noqa: E402

SOURCE = Path(
    os.environ.get("IPB_RAW_IDOL", "/workspace/webCodex/datasets/imu_odometry/raw/_staging/IDOL")
)
pytestmark = pytest.mark.skipif(
    not (SOURCE / "building1").is_dir(), reason="IDOL raw data not available"
)


def load(sequence_id):
    return next(idol.iter_raw_sequences(SOURCE, only=[sequence_id]))


def test_official_split_sizes_and_subject_disjointness():
    splits = idol.official_splits(SOURCE)
    sizes = {k: len(v) for k, v in splits.items()}
    assert sizes["train"] == 43 and sizes["test_known"] == 51 and sizes["test_unknown"] == 36
    assert (
        sizes["test"] == 87
        and sizes["test_known_building1"] == 15
        and sizes["test_unknown_building1"] == 14
    )
    assert set(idol.list_sequences(SOURCE)) == set(splits["train"]) | set(splits["test"])
    subject_of = {}
    for sequence_id, _, _, path in idol._listing(idol._root(SOURCE)):
        subject_of[sequence_id] = idol._load_metadata(path.parent)[path.stem]["subjectID"]
    subjects = {
        key: {subject_of[i] for i in splits[key]} for key in ("train", "test_known", "test_unknown")
    }
    assert not subjects["test_unknown"] & (subjects["train"] | subjects["test_known"])
    assert subjects["test_known"] <= subjects["train"]


def test_sampled_sequence_units_and_frames():
    raw = load("building1_train_0")
    assert raw.rejected is None, raw.rejected
    assert raw.check_shapes() == []
    assert 9.0 < np.median(np.linalg.norm(raw.accelerometer, axis=1)) < 10.5  # G → m/s²
    stats = rig.parse_stats_note(raw.notes)
    assert stats["gravity_error"] < 0.5  # iOS 符号与调平后
    assert stats["gyro_window_error_median_deg_debiased"] < 5.0
    assert 0.3 < stats["speed_median"] < 2.0  # 位置单位为米
    assert raw.device_orientation is not None
    assert any("leveled" in n for n in raw.notes)


def test_known_file_quirks():
    raw = load("building3_known_12")
    assert raw.rejected is None and any("level_0" in n for n in raw.notes)
    raw = load("building3_known_14")
    assert np.diff(raw.imu_time).max() == pytest.approx(15.845, abs=0.01)  # 缺口原样保留
