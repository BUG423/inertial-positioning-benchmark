from dataclasses import replace

import numpy as np
import pytest
from synthetic import Motion, make_sequence

from inertial_benchmark.data.augment import Bias, Noise, RandomYaw, build_augmentations
from inertial_benchmark.data.views import (
    SequenceView,
    ViewConfig,
    window_valid_mask,
)
from inertial_benchmark.utils.geometry import (
    GRAVITY,
    quat_from_yaw,
    quat_multiply,
    quat_rotate,
    rotate_z,
)


def rotated_world(seq, alpha):
    """把参考世界系整体绕 z 轴转 ``alpha``（机体系 IMU 不变）。"""
    q = quat_multiply(quat_from_yaw(np.full(len(seq), alpha)), seq.orientation)
    return replace(seq, orientation=q, position=rotate_z(seq.position, alpha),
                   velocity=rotate_z(seq.velocity, alpha), attrs=dict(seq.attrs))


@pytest.fixture(scope="module")
def seq():
    return make_sequence(duration=12.0, seed=2, device_yaw_offset=0.3)


def test_view_config_validation():
    with pytest.raises(ValueError, match="dims=3"):
        ViewConfig(frame="body", dims=2)
    with pytest.raises(ValueError, match="frame"):
        ViewConfig(frame="enu")
    with pytest.raises(ValueError, match="target"):
        ViewConfig(target="speed")
    cfg = ViewConfig.from_cfg({"window": "100", "dims": 3, "frame": "body", "extra": 1})
    assert cfg.window == 100 and cfg.dims == 3
    assert ViewConfig(window=201).target_offset == pytest.approx(0.5)
    assert ViewConfig(window=201, target="velocity_at_end").target_offset == pytest.approx(1.0)


def test_gravity_world_inputs_and_targets(seq):
    cfg = ViewConfig(window=200, dims=3)
    view = SequenceView(seq, cfg)
    starts = view.starts(50)
    assert starts.tolist() == list(range(0, len(seq) - 199, 50))
    x = view.imu_windows(starts)
    assert x.shape == (len(starts), 6, 200) and x.dtype == np.float32
    s = starts[3]
    q = seq.orientation[s:s + 200].astype(np.float64)
    np.testing.assert_allclose(x[3, :3].T, quat_rotate(q, seq.gyroscope[s:s + 200]), atol=1e-5)
    np.testing.assert_allclose(x[3, 3:].T, quat_rotate(q, seq.accelerometer[s:s + 200]),
                               atol=1e-4)
    y = view.targets(starts)
    motion = Motion(2)
    t0, t1 = seq.timestamp[s], seq.timestamp[s + 199]
    expected = (motion.position(np.array([t1])) - motion.position(np.array([t0])))[0] / (t1 - t0)
    np.testing.assert_allclose(y[3], expected, atol=1e-5)
    np.testing.assert_allclose(view.target_times(starts)[3], t0 + 199 / 400)


def test_remove_gravity(seq):
    with_g = SequenceView(seq, ViewConfig(dims=3)).imu_windows(np.array([0]))
    without = SequenceView(seq, ViewConfig(dims=3, remove_gravity=True)).imu_windows(
        np.array([0]))
    np.testing.assert_allclose(with_g[0, 5] - without[0, 5], GRAVITY, atol=1e-4)
    np.testing.assert_allclose(with_g[0, :5], without[0, :5])
    body = SequenceView(seq, ViewConfig(frame="body", dims=3, remove_gravity=True))
    acc_b = body.imu_windows(np.array([0]))[0, 3:].T
    world = quat_rotate(seq.orientation[:200].astype(np.float64), acc_b)
    motion = Motion(2)
    np.testing.assert_allclose(world, motion.acceleration(seq.timestamp[:200]), atol=1e-4)


def test_yaw_local_is_heading_invariant(seq):
    cfg = ViewConfig(frame="gravity_yaw_local", dims=2)
    a = SequenceView(seq, cfg)
    b = SequenceView(rotated_world(seq, 1.1), cfg)
    starts = a.starts(100)
    np.testing.assert_allclose(a.imu_windows(starts), b.imu_windows(starts), atol=2e-4)
    np.testing.assert_allclose(a.targets(starts), b.targets(starts), atol=1e-5)
    # 视图目标旋回世界系即世界系目标
    world = a.to_world_velocity(a.targets(starts), starts)
    np.testing.assert_allclose(world, a.targets_world(starts)[:, :2], atol=1e-5)
    # gravity_world 不具备该不变性
    g = ViewConfig(dims=2)
    assert not np.allclose(SequenceView(seq, g).targets(starts),
                           SequenceView(rotated_world(seq, 1.1), g).targets(starts), atol=1e-3)


def test_body_frame_roundtrip(seq):
    cfg = ViewConfig(frame="body", dims=3)
    view = SequenceView(seq, cfg)
    starts = view.starts(100)
    x = view.imu_windows(starts)
    np.testing.assert_array_equal(x[1, :3].T, seq.gyroscope[100:300])
    world = view.to_world_velocity(view.targets(starts), starts)
    np.testing.assert_allclose(world, view.targets_world(starts), atol=1e-5)


def test_device_orientation_is_yaw_aligned(seq):
    ref = SequenceView(seq, ViewConfig(dims=2))
    dev = SequenceView(seq, ViewConfig(dims=2, orientation="device"))
    assert dev.yaw_offset == pytest.approx(-0.3)
    starts = ref.starts(100)
    np.testing.assert_allclose(dev.imu_windows(starts), ref.imu_windows(starts), atol=2e-4)
    no_device = make_sequence(duration=3.0)
    with pytest.raises(ValueError, match="imu/orientation"):
        SequenceView(no_device, ViewConfig(orientation="device"))


def test_displacement_and_end_velocity_targets(seq):
    starts = np.array([0, 400, 1000])
    disp = SequenceView(seq, ViewConfig(dims=3, target="displacement"))
    avg = SequenceView(seq, ViewConfig(dims=3))
    np.testing.assert_allclose(disp.targets(starts) * disp.cfg.velocity_scale,
                               avg.targets(starts), atol=1e-5)
    np.testing.assert_allclose(disp.to_world_velocity(disp.targets(starts), starts),
                               avg.targets_world(starts), atol=1e-5)
    end_cfg = ViewConfig(dims=3, target="velocity_at_end")
    with_vel = SequenceView(seq, end_cfg).targets(starts)
    no_vel = SequenceView(replace(seq, velocity=None, attrs=dict(seq.attrs)), end_cfg)
    np.testing.assert_allclose(no_vel.targets(starts), with_vel, atol=2e-4)
    motion = Motion(2)
    np.testing.assert_allclose(with_vel, motion.velocity(seq.timestamp[starts + 199]), atol=1e-5)


def test_invalid_windows_are_skipped(seq):
    s2 = replace(seq, attrs=dict(seq.attrs))
    s2.valid_pose = s2.valid_pose.copy()
    s2.valid_pose[450] = False
    view = SequenceView(s2, ViewConfig())
    starts = view.starts(10)
    assert not np.any((starts <= 450) & (starts + 200 > 450))
    assert len(view.starts(10, require_valid=False)) == len(starts) + 20
    assert window_valid_mask(np.array([1, 1, 0, 1, 1], bool), np.array([0, 1, 3]), 2).tolist() \
        == [True, False, True]


def test_random_yaw_matches_rotated_world(seq):
    cfg = ViewConfig(dims=2)
    view = SequenceView(seq, cfg)
    start = np.array([600])
    sample = {"imu": view.imu_windows(start)[0].copy(), "target": view.targets(start)[0].copy()}
    rng = np.random.default_rng(5)
    angle = np.random.default_rng(5).uniform(-np.pi, np.pi)
    RandomYaw()(sample, rng)
    rotated = SequenceView(rotated_world(seq, angle), cfg)
    np.testing.assert_allclose(sample["imu"], rotated.imu_windows(start)[0], atol=2e-4)
    np.testing.assert_allclose(sample["target"], rotated.targets(start)[0], atol=1e-5)


def test_bias_and_noise(seq):
    view = SequenceView(seq, ViewConfig(dims=2))
    start = 300
    base = view.imu_windows(np.array([start]))[0]
    sample = {"imu": base.copy(), "target": np.zeros(2, np.float32),
              "body_to_frame": lambda v: view.body_to_frame(start, v)}
    Bias(gyro=0.1, acc=0.5)(sample, np.random.default_rng(0))
    rng = np.random.default_rng(0)
    bg, ba = rng.normal(0, 0.1, 3), rng.normal(0, 0.5, 3)
    q = seq.orientation[start:start + 200].astype(np.float64)
    np.testing.assert_allclose(sample["imu"][:3] - base[:3], quat_rotate(q, bg).T, atol=1e-5)
    np.testing.assert_allclose(sample["imu"][3:] - base[3:], quat_rotate(q, ba).T, atol=1e-5)
    sample2 = {"imu": base.copy(), "target": np.zeros(2, np.float32)}
    Noise(gyro=0.01, acc=0.1)(sample2, np.random.default_rng(1))
    diff = sample2["imu"] - base
    assert 0.005 < diff[:3].std() < 0.02 and 0.05 < diff[3:].std() < 0.2


def test_build_augmentations_order_and_errors():
    shift, augs = build_augmentations(
        ["random_yaw", {"name": "noise", "acc": 0.1}, {"bias": {"gyro": 0.01}}, "time_shift"])
    assert [a.name for a in augs] == ["noise", "bias", "random_yaw"]
    assert shift.resolve(10) == 5
    _, augs = build_augmentations(["random_yaw"], frame="body")
    assert augs == []
    with pytest.raises(ValueError, match="unknown augmentation"):
        build_augmentations(["mixup"])
