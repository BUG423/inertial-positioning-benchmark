from dataclasses import replace

import numpy as np
import pytest
from synthetic import Motion, make_sequence

from inertial_benchmark.data.augment import (
    Bias,
    BiasShift,
    GravityPerturb,
    Noise,
    RandomYaw,
    TimeShift,
    build_augmentations,
)
from inertial_benchmark.data.views import (
    SequenceView,
    ViewConfig,
    window_valid_mask,
)
from inertial_benchmark.utils.geometry import (
    GRAVITY,
    quat_conjugate,
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
    with pytest.raises(ValueError, match="output_steps"):
        ViewConfig(target="multi_displacement")
    with pytest.raises(ValueError, match="history_stride"):
        ViewConfig(history=3)
    with pytest.raises(ValueError, match="extra_inputs"):
        ViewConfig(extra_inputs=("magnetometer",))
    with pytest.raises(ValueError, match="overlap"):
        ViewConfig(overlap="first")
    cfg = ViewConfig.from_cfg({"window": "100", "dims": 3, "frame": "body", "extra": 1})
    assert cfg.window == 100 and cfg.dims == 3
    assert ViewConfig(window=201).target_offset == pytest.approx(0.5)
    assert ViewConfig(window=201, target="velocity_at_end").target_offset == pytest.approx(1.0)


def test_output_layout_offsets_and_scales():
    """InputSpec/ViewConfig 必须声明输出布局与各输出的时间偏移（DESIGN 第 3 节）。"""
    window = ViewConfig(window=201)
    assert window.output_layout == "window" and window.num_outputs == 1
    assert window.output_shape == (2,)
    np.testing.assert_allclose(window.output_offsets, [0.5])
    np.testing.assert_allclose(window.output_scales, [1.0])
    disp = ViewConfig(window=201, target="displacement")
    np.testing.assert_allclose(disp.output_scales, [1.0])  # 位移 → 速度：除以 1 s
    frames = ViewConfig(window=4, rate=2.0, target="frame_velocity")
    assert frames.output_layout == "frame" and frames.num_outputs == 4
    assert frames.output_shape == (4, 2)
    np.testing.assert_allclose(frames.output_offsets, [0.0, 0.5, 1.0, 1.5])
    steps = ViewConfig(window=201, target="multi_displacement", output_steps=4, dims=3)
    assert steps.output_layout == "steps" and steps.num_outputs == 4
    np.testing.assert_array_equal(steps.step_bounds, [[0, 50], [50, 100], [100, 150], [150, 200]])
    np.testing.assert_allclose(steps.output_offsets, [0.125, 0.375, 0.625, 0.875])
    np.testing.assert_allclose(steps.output_scales, 4.0)  # 每段 0.25 s
    # 时间偏移一律是半个采样间隔的整数倍（重叠合并要精确分组）
    for cfg in (window, disp, frames, steps):
        np.testing.assert_allclose(np.round(cfg.output_offsets / (0.5 * cfg.dt)),
                                   cfg.output_offsets / (0.5 * cfg.dt), atol=1e-9)


def test_frame_velocity_targets_match_reference_velocity(seq):
    view = SequenceView(seq, ViewConfig(window=100, dims=3, target="frame_velocity"))
    starts = np.array([0, 300, 900])
    targets = view.targets(starts)
    assert targets.shape == (3, 100, 3)
    for row, start in enumerate(starts):
        expected = seq.velocity[start:start + 100]
        np.testing.assert_allclose(targets[row], expected, atol=1e-5)
    times = view.output_times(starts)
    np.testing.assert_allclose(times[0], seq.timestamp[:100], atol=1e-12)
    np.testing.assert_array_equal(view.target_mask(starts).shape, (3, 100))
    np.testing.assert_allclose(view.to_world_velocity(targets, starts), targets, atol=1e-5)


def test_multi_step_displacement_targets(seq):
    cfg = ViewConfig(window=101, dims=3, target="multi_displacement", output_steps=10)
    view = SequenceView(seq, cfg)
    starts = np.array([0, 500])
    targets = view.targets(starts)
    assert targets.shape == (2, 10, 3)
    for row, start in enumerate(starts):
        for step, (lo, hi) in enumerate(cfg.step_bounds):
            expected = seq.position[start + hi] - seq.position[start + lo]
            np.testing.assert_allclose(targets[row, step], expected, atol=1e-6)
    # 位移换算为速度：每段除以自己的时间跨度
    speeds = view.to_world_velocity(targets, starts)
    np.testing.assert_allclose(speeds, targets * cfg.output_scales[None, :, None], atol=1e-6)
    # 10 段位移之和 = 整窗位移
    np.testing.assert_allclose(targets.sum(axis=1),
                               seq.position[starts + 100] - seq.position[starts], atol=1e-6)


def test_history_sub_windows_are_contiguous_in_time(seq):
    cfg = ViewConfig(window=100, dims=3, history=4, history_stride=50)
    view = SequenceView(seq, cfg)
    assert cfg.input_span == 250 and cfg.history_offset == 150
    starts = view.starts(50)
    assert starts[0] == 150  # 起点保证历史子窗口不越过序列开头
    x = view.imu_windows(starts[:2])
    assert x.shape == (2, 4, 6, 100)
    plain = SequenceView(seq, ViewConfig(window=100, dims=3))
    for sub in range(4):
        offset = (sub - 3) * 50
        expected = plain.imu_windows(starts[:2] + offset)
        np.testing.assert_allclose(x[:, sub], expected, atol=1e-6)  # 子窗口在时间上连续
    # 有效性按整个输入跨度判断：跨越无效区的窗口被排除
    gapped = make_sequence(duration=12.0, seed=2)
    gapped.valid_imu[100:120] = False
    view2 = SequenceView(gapped, cfg)
    assert not view2.window_valid_input(np.array([200]))[0]  # 历史子窗口落在缺口上
    assert view2.window_valid_input(np.array([400]))[0]


def test_extra_inputs_and_privileged_marking(seq):
    cfg = ViewConfig(window=100, dims=3,
                     extra_inputs=("orientation", "gravity", "init_velocity"))
    view = SequenceView(seq, cfg)
    starts = np.array([0, 400])
    extra = view.extra_inputs(starts)
    assert set(extra) == {"orientation", "gravity", "init_velocity"}
    assert extra["orientation"].shape == (2, 100, 4)
    np.testing.assert_allclose(extra["orientation"][1], seq.orientation[400:500], atol=1e-6)
    np.testing.assert_allclose(extra["gravity"][0], np.tile([0, 0, GRAVITY], (100, 1)), atol=1e-6)
    np.testing.assert_allclose(extra["init_velocity"], seq.velocity[starts], atol=1e-5)
    assert cfg.privileged_inputs == ("init_velocity",)
    assert ViewConfig(extra_inputs=("orientation",)).privileged_inputs == ()
    # body 系下重力随姿态变化，姿态输入按视图坐标系给出
    body = SequenceView(seq, ViewConfig(window=100, dims=3, frame="body",
                                        extra_inputs=("gravity",)))
    gravity = body.extra_inputs(starts)["gravity"]
    expected = quat_rotate(quat_conjugate(seq.orientation[:100].astype(np.float64)),
                           np.array([0.0, 0.0, GRAVITY]))
    np.testing.assert_allclose(gravity[0], expected, atol=1e-5)


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


def test_device_orientation_gap_only_affects_device_windows(seq):
    import copy

    gap = copy.deepcopy(seq)
    gap.valid_device_orientation = np.ones(len(gap), bool)
    gap.valid_device_orientation[600:800] = False  # 1 s 设备姿态缺口
    ref = SequenceView(gap, ViewConfig(dims=2))
    dev = SequenceView(gap, ViewConfig(dims=2, orientation="device"))
    assert len(ref.starts(10)) == len(SequenceView(seq, ViewConfig(dims=2)).starts(10))
    assert len(dev.starts(10)) < len(ref.starts(10))
    assert not dev.window_valid(np.array([500, 700, 790])).any()
    assert dev.window_valid(np.array([0, 800])).all()
    # 偏航对齐在首个 IMU/位姿/设备姿态都有效的样本处估计
    gap.valid_device_orientation[:10] = False
    assert SequenceView(gap, ViewConfig(dims=2, orientation="device")).yaw_offset == \
        pytest.approx(dev.yaw_offset)


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


def test_bias_shift_and_gravity_perturb(seq):
    view = SequenceView(seq, ViewConfig(dims=3, frame="gravity_yaw_local"))
    base = view.imu_windows(np.array([200]))[0]
    target = view.targets(np.array([200]))[0]
    sample = {"imu": base.copy(), "target": target.copy()}
    BiasShift(gyro=0.05, acc=0.2)(sample, np.random.default_rng(3))
    diff = sample["imu"] - base
    # 整窗常值、逐轴独立、在范围内
    np.testing.assert_allclose(diff, np.repeat(diff[:, :1], base.shape[1], axis=1), atol=1e-6)
    assert np.all(np.abs(diff[:3, 0]) <= 0.05) and np.all(np.abs(diff[3:, 0]) <= 0.2)
    assert np.unique(np.round(diff[:, 0], 6)).size == 6

    sample = {"imu": base.copy(), "target": target.copy()}
    GravityPerturb(max_deg=5.0)(sample, np.random.default_rng(4))
    np.testing.assert_array_equal(sample["target"], target)  # 目标不旋转
    for sl in (slice(0, 3), slice(3, 6)):
        np.testing.assert_allclose(np.linalg.norm(sample["imu"][sl], axis=0),
                                   np.linalg.norm(base[sl], axis=0), rtol=1e-5)
    # 绕水平轴的旋转：重力方向（加计均值）与 z 轴的夹角变化不超过 5°
    g0 = base[3:].mean(axis=1)
    g1 = sample["imu"][3:].mean(axis=1)
    cos = g0 @ g1 / (np.linalg.norm(g0) * np.linalg.norm(g1))
    assert 0 < np.degrees(np.arccos(min(cos, 1.0))) <= 5.0 + 1e-3
    _, augs = build_augmentations([{"gravity_perturb": {"max_deg": 5}}, "random_yaw",
                                   {"bias_shift": {"acc": 0.1}}])
    assert [a.name for a in augs] == ["bias_shift", "gravity_perturb", "random_yaw"]
    _, augs = build_augmentations(["gravity_perturb"], frame="body")
    assert augs == []

    # 历史子窗口布局 (H, 6, T)：两者都作用于整个输入跨度，所有子窗口共用同一次抽样
    hist = SequenceView(seq, ViewConfig(dims=3, frame="gravity_yaw_local", history=3,
                                        history_stride=50))
    base_h = hist.imu_windows(np.array([300]))[0]
    assert base_h.shape == (3, 6, 200)
    sample = {"imu": base_h.copy(), "target": target.copy()}
    BiasShift(gyro=0.05, acc=0.2)(sample, np.random.default_rng(3))
    diff_h = sample["imu"] - base_h
    np.testing.assert_allclose(diff_h[0], diff_h[2], atol=1e-6)
    np.testing.assert_allclose(diff_h[0][:, 0], diff[:, 0], atol=1e-6)  # 同一随机流
    sample = {"imu": base_h.copy(), "target": target.copy()}
    GravityPerturb(max_deg=5.0)(sample, np.random.default_rng(4))
    np.testing.assert_array_equal(sample["target"], target)
    for h in range(3):
        for sl in (slice(0, 3), slice(3, 6)):
            np.testing.assert_allclose(np.linalg.norm(sample["imu"][h, sl], axis=0),
                                       np.linalg.norm(base_h[h, sl], axis=0), rtol=1e-5)
    # 与无历史时同一旋转：重叠部分逐样本一致
    single = {"imu": base_h[2].copy(), "target": target.copy()}
    GravityPerturb(max_deg=5.0)(single, np.random.default_rng(4))
    np.testing.assert_allclose(sample["imu"][2], single["imu"], atol=1e-6)


def test_time_shift_range():
    assert TimeShift().resolve_range(10) == (-5, 5)
    assert TimeShift(max_shift=9, min_shift=0).resolve_range(10) == (0, 9)
    assert TimeShift(max_shift=-3).resolve_range(10) == (0, 0)
    with pytest.raises(ValueError, match="min_shift"):
        TimeShift(max_shift=1, min_shift=2).resolve_range(10)
