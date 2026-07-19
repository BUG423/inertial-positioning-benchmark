from pathlib import Path

import numpy as np
import pytest

from inertial_benchmark import CanonicalSequence, SequenceValidationError, WindowDataset


def make_sequence(n: int = 11) -> CanonicalSequence:
    """构造速度恒为 [2, 1, 0] m/s 的解析轨迹，避免依赖真实数据。"""

    timestamp = np.arange(n, dtype=np.float64) * 0.1
    position = np.column_stack((2.0 * timestamp, timestamp, np.zeros(n)))
    attributes = {
        "schema_version": "0.1",
        "dataset": "synthetic",
        "sequence_id": "constant_velocity",
        "world_frame": "gravity_aligned_local",
        "timestamp_type": "relative",
        "orientation_convention": "body_to_world_wxyz",
        "accelerometer_type": "specific_force",
        "position_source": "analytic",
        "orientation_source": "analytic",
        "subject_id": "synthetic",
        "device_id": "synthetic",
        "source_license": "generated",
    }
    return CanonicalSequence(
        timestamp=timestamp,
        gyroscope=np.zeros((n, 3), dtype=np.float32),
        accelerometer=np.tile([0.0, 0.0, 9.81], (n, 1)).astype(np.float32),
        orientation=np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)).astype(np.float32),
        position=position,
        attributes=attributes,
    )


def test_hdf5_round_trip(tmp_path: Path):
    original = make_sequence()
    output = tmp_path / "sequence.h5"
    original.to_hdf5(output)
    loaded = CanonicalSequence.from_hdf5(output)
    np.testing.assert_allclose(loaded.position, original.position)
    assert loaded.attributes["sequence_id"] == "constant_velocity"


def test_velocity_window_shape_and_value():
    dataset = WindowDataset(make_sequence(), window_size=5, stride=2, target="velocity")
    assert len(dataset) == 4
    sample = dataset[0]
    assert sample.features.shape == (6, 5)
    np.testing.assert_allclose(sample.target, [2.0, 1.0, 0.0])
    # 5 帧窗口覆盖 4 个采样间隔，端点索引应为 0 和 4。
    assert sample.start_index == 0 and sample.end_index == 4


def test_invalid_timestamp_is_rejected():
    sequence = make_sequence()
    timestamp = sequence.timestamp.copy()
    timestamp[3] = timestamp[2]
    invalid = CanonicalSequence(**{**sequence.__dict__, "timestamp": timestamp})
    with pytest.raises(SequenceValidationError, match="strictly increasing"):
        invalid.validate()


def test_invalid_mask_removes_windows():
    sequence = make_sequence()
    mask = np.ones(len(sequence.timestamp), dtype=bool)
    mask[4] = False
    sequence = CanonicalSequence(**{**sequence.__dict__, "valid_imu": mask})
    dataset = WindowDataset(sequence, window_size=5, stride=1)
    assert all(not (sample.start_index <= 4 <= sample.end_index) for sample in dataset)
