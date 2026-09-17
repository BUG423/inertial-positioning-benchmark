"""TLIO 解析器的合成数据单元测试（不依赖真实数据）。"""

from __future__ import annotations

import json

import numpy as np
import pytest

from inertial_benchmark.data.converters import _rig_utils as rig
from inertial_benchmark.data.converters import tlio
from inertial_benchmark.data.converters.base import RAW_REQUIRED_ATTRS

CALIBRATION = {
    "Accelerometer": {"Bias": {"Name": "Constant", "Offset": [0.1, -0.2, 0.05]},
                      "Model": {"Name": "Linear", "RectificationMatrix": np.eye(3).tolist()}},
    "Calibrated": True,
    "Gyroscope": {"Bias": {"Name": "Constant", "Offset": [0.001, 0.002, 0.003]},
                  "Model": {"Name": "Linear", "RectificationMatrix": np.eye(3).tolist()}},
    "Label": "unlabeled_imu_0",
    "SerialNumber": "rift://",
    "T_Device_Imu": {"Translation": [0.01, 0.02, 0.03], "UnitQuaternion": [1.0, [0.0, 0.0, 0.0]]},
}


def write_sequence(root, seq_id, motion, *, world_frame_imu=True, columns=None):
    folder = root / seq_id
    folder.mkdir(parents=True)
    rot = rig.as_rotation(motion["orientation"])
    gyr = rot.apply(motion["gyroscope"]) if world_frame_imu else motion["gyroscope"]
    acc = rot.apply(motion["accelerometer"]) if world_frame_imu else motion["accelerometer"]
    ts_us = np.round((motion["time"] + 1000.0) * 1e6)
    data = np.column_stack([ts_us, gyr, acc, rig.wxyz_to_xyzw(motion["orientation"]), motion["position"],
                            motion["velocity"]])
    np.save(folder / "imu0_resampled.npy", data)
    desc = {"columns_name(width)": columns or tlio.EXPECTED_COLUMNS, "num_rows": len(data),
            "approximate_frequency_hz": 200.0, "t_start_us": float(ts_us[0]), "t_end_us": float(ts_us[-1])}
    (folder / "imu0_resampled_description.json").write_text(json.dumps(desc))
    (folder / "calibration.json").write_text(json.dumps(CALIBRATION))
    return data


@pytest.fixture()
def tlio_root(tmp_path):
    root = tmp_path / "TLIO" / "tlio_golden"
    root.mkdir(parents=True)
    write_sequence(root, "111", rig.simulate_rig_motion(40.0, seed=1))
    write_sequence(root, "222", rig.simulate_rig_motion(40.0, seed=2))
    write_sequence(root, "333", rig.simulate_rig_motion(40.0, seed=3), world_frame_imu=False)
    (root / "train_list.txt").write_text("111\n333\n")
    (root / "val_list.txt").write_text("222\n")
    (root / "test_list.txt").write_text("\n")
    (root / "all_ids.txt").write_text("111\n222\n333\n")
    return root


def test_official_splits_accept_parent_or_golden_dir(tlio_root):
    for source in (tlio_root, tlio_root.parent):
        assert tlio.official_splits(source) == {"train": ["111", "333"], "val": ["222"], "test": []}


def test_list_sequences_includes_directories_outside_official_lists(tlio_root):
    write_sequence(tlio_root, "999", rig.simulate_rig_motion(20.0, seed=9))
    (tlio_root / "not_a_sequence").mkdir()
    assert tlio.list_sequences(tlio_root) == ["111", "222", "333", "999"]
    assert "999" not in set().union(*tlio.official_splits(tlio_root).values())
    assert [r.sequence_id for r in tlio.iter_raw_sequences(tlio_root, only=["999"])] == ["999"]


def test_world_frame_imu_is_rotated_back_to_body(tlio_root):
    motion = rig.simulate_rig_motion(40.0, seed=1)
    raw = next(tlio.iter_raw_sequences(tlio_root, only=["111"]))
    assert raw.rejected is None, raw.rejected
    assert raw.check_shapes() == []
    assert set(RAW_REQUIRED_ATTRS) <= set(raw.attrs)
    np.testing.assert_allclose(raw.gyroscope, motion["gyroscope"], atol=1e-9)
    np.testing.assert_allclose(raw.accelerometer, motion["accelerometer"], atol=1e-9)
    # xyzw → wxyz，且保持 body_to_world
    np.testing.assert_allclose(np.abs(np.sum(raw.orientation * motion["orientation"], axis=1)), 1.0, atol=1e-9)
    np.testing.assert_allclose(raw.imu_time - raw.imu_time[0], motion["time"], atol=1e-6)
    assert raw.imu_time[0] == pytest.approx(1000.0)
    np.testing.assert_allclose(raw.velocity, motion["velocity"])
    assert raw.attrs["placement"] == "head"
    assert raw.attrs["group_id"] == raw.attrs["device_id"] == tlio.headset_fingerprint(CALIBRATION)
    stats = rig.parse_stats_note(raw.notes)
    assert stats["gravity_error"] < 0.05 and stats["gyro_window_error_median_deg"] < 0.1


def test_body_frame_data_mislabelled_as_world_is_rejected(tlio_root):
    raw = next(tlio.iter_raw_sequences(tlio_root, only=["333"]))
    assert raw.rejected is not None and "physical check failed" in raw.rejected


def test_only_filter_and_unexpected_layout(tlio_root):
    ids = [r.sequence_id for r in tlio.iter_raw_sequences(tlio_root, only={"222"})]
    assert ids == ["222"]
    write_sequence(tlio_root, "444", rig.simulate_rig_motion(20.0), columns=["ts_us(1)", "other(16)"])
    raw = next(tlio.iter_raw_sequences(tlio_root, only=["444"]))
    assert raw.rejected and "unexpected column layout" in raw.rejected
    assert set(RAW_REQUIRED_ATTRS) <= set(raw.attrs)


def test_non_finite_rows_are_marked_invalid(tlio_root):
    path = tlio_root / "111" / "imu0_resampled.npy"
    data = np.load(path)
    data[100:110, 4] = np.nan
    np.save(path, data)
    raw = tlio.parse_sequence(tlio_root / "111")
    assert (~raw.imu_valid).sum() == 10 and (~raw.pose_valid).sum() == 10
    assert np.isfinite(raw.accelerometer).all()
