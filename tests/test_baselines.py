"""经典基线（``pdr`` / ``mean_speed_heading``）的确定性测试。

合成信号：设备平放、姿态恒定为 ``Rz(yaw)``，世界系比力 ``f = [0, 0, g₀ + A·sin(2π f_step t)]``，
参考轨迹为沿固定航向的匀速直线。检验步数、步长/距离误差、航向约定、窗口汇总公式与标定流程。
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from inertial_benchmark.cfg import get_cfg  # noqa: E402
from inertial_benchmark.data.format import Sequence  # noqa: E402
from inertial_benchmark.data.views import SequenceView, ViewConfig  # noqa: E402
from inertial_benchmark.engine import Predictor  # noqa: E402
from inertial_benchmark.models.pdr.heading import resolve_axes, segment_heading  # noqa: E402
from inertial_benchmark.models.pdr.model import G0, PDR, valid_segments  # noqa: E402
from inertial_benchmark.nn import build_model  # noqa: E402
from inertial_benchmark.utils.geometry import (  # noqa: E402
    quat_conjugate,
    quat_from_yaw,
    quat_rotate,
)

RATE = 200.0
BODY = "synthetic_device"
AXES = {BODY: [[0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]}  # 与 android_device 同约定
WINDOW = 200
YAW30 = np.deg2rad(30.0)  # 设备平放、机顶朝 30° 方位 → 航向 120°
COURSE120 = np.deg2rad(120.0)


def walk_sequence(duration=20.0, amplitude=2.0, step_hz=2.0, yaw=YAW30,
                  speed=1.2, heading=None, noise=0.0, seed=0, upright=False):
    """平放手持、姿态恒定的合成走路信号 + 匀速直线参考轨迹。"""
    n = int(round(duration * RATE)) + 1
    t = np.arange(n) / RATE
    world_acc = np.zeros((n, 3))
    world_acc[:, 2] = G0 + amplitude * np.sin(2 * np.pi * step_hz * t)
    if noise:
        world_acc += np.random.default_rng(seed).normal(0.0, noise, world_acc.shape)
    if upright:  # 机顶朝上（前向轴竖直）→ 必须换用备用轴 −z
        q_tilt = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])  # 绕 x 转 90°
        q = np.tile(quat_from_yaw(yaw), (n, 1))
        from inertial_benchmark.utils.geometry import quat_multiply

        q = quat_multiply(q, np.tile(q_tilt, (n, 1)))
    else:
        q = np.tile(quat_from_yaw(yaw), (n, 1))
    acc_body = quat_rotate(quat_conjugate(q), world_acc)
    course = yaw + np.pi / 2 if heading is None else heading
    position = np.zeros((n, 3))
    position[:, 0] = speed * t * np.cos(course)
    position[:, 1] = speed * t * np.sin(course)
    velocity = np.zeros((n, 3))
    velocity[:, 0], velocity[:, 1] = speed * np.cos(course), speed * np.sin(course)
    attrs = {"sequence_id": "walk", "dataset": "synthetic", "sample_rate_hz": RATE,
             "body_frame": BODY, "group_id": "g0"}
    return Sequence(timestamp=t, gyroscope=np.zeros((n, 3)), accelerometer=acc_body,
                    orientation=q, position=position, valid_imu=np.ones(n, bool),
                    valid_pose=np.ones(n, bool), velocity=velocity, attrs=attrs)


def make_pdr(**args):
    cfg = get_cfg({"model": "pdr", "window": WINDOW, "device": "cpu",
                   "orientation": "reference", "model_args": {"body_axes": AXES, **args}})
    return build_model(cfg), cfg


def view_of(seq, cfg=None):
    return SequenceView(seq, ViewConfig(window=WINDOW, frame="gravity_world",
                                        orientation="reference"))


# --------------------------------------------------------------------------- 步检测
@pytest.mark.parametrize("step_hz,expected", [(2.0, 40), (1.0, 20), (4.0, 0)])
def test_step_count_on_synthetic_walking_signal(step_hz, expected):
    """2 Hz → 40 步，1 Hz → 20 步，4 Hz（高于 3 Hz 截止）→ 检测不到步。"""
    model, _ = make_pdr()
    seq = walk_sequence(duration=20.0, step_hz=step_hz)
    steps = model.detect_steps(view_of(seq))
    assert len(steps["time"]) == expected
    if expected:
        intervals = np.diff(steps["time"])
        assert intervals.min() >= 0.3 - 1e-9  # 最小步间隔


def test_step_amplitude_matches_the_filter_response():
    """2 Hz 正弦经 4 阶 3 Hz 巴特沃斯后的峰峰值 = 2A/(1+(f/f_c)^{2·order})。"""
    model, _ = make_pdr()
    steps = model.detect_steps(view_of(walk_sequence(duration=20.0, step_hz=2.0)))
    expected = 4.0 / (1.0 + (2.0 / 3.0) ** 8)
    np.testing.assert_allclose(steps["amplitude"][1:], expected, rtol=0.01)
    steps1 = model.detect_steps(view_of(walk_sequence(duration=20.0, step_hz=1.0)))
    np.testing.assert_allclose(steps1["amplitude"][1:], 4.0 / (1.0 + (1.0 / 3.0) ** 8), rtol=0.01)


def test_static_signal_yields_no_step_and_zero_velocity():
    model, cfg = make_pdr()
    seq = walk_sequence(duration=20.0, amplitude=0.0, noise=0.05, speed=0.0, seed=0)
    view = view_of(seq)
    assert len(model.detect_steps(view)["time"]) == 0
    starts = view.starts(10, require_valid=False)
    _, velocity, _ = model.predict_sequence(seq, view, starts)
    np.testing.assert_allclose(velocity, 0.0)


def test_valid_segments_helper():
    mask = np.array([1, 1, 0, 1, 1, 1, 0, 1], bool)
    assert valid_segments(mask, 1) == [(0, 2), (3, 6), (7, 8)]
    assert valid_segments(mask, 3) == [(3, 6)]
    assert valid_segments(np.zeros(5, bool), 1) == []


# --------------------------------------------------------------------------- 航向
def test_heading_convention_and_yaw_equivariance():
    """``body_frame`` 平放、``R = Rz(30°)``、δ=0 → 航向 = 120°；姿态左乘 Rz(α) 使航向加 α。"""
    model, _ = make_pdr()
    seq = walk_sequence(yaw=YAW30)
    steps = model.detect_steps(view_of(seq))
    np.testing.assert_allclose(np.degrees(steps["heading"]), 120.0, atol=1e-5)
    from inertial_benchmark.utils.geometry import quat_multiply, wrap_angle

    alpha = 0.7
    rotated = walk_sequence(yaw=YAW30 + alpha)
    rotated_steps = model.detect_steps(view_of(rotated))
    np.testing.assert_allclose(wrap_angle(rotated_steps["heading"] - steps["heading"] - alpha),
                               0.0, atol=1e-6)  # 序列以 float32 存储，容差到 1e-6
    assert quat_multiply(quat_from_yaw(alpha), seq.orientation).shape == seq.orientation.shape
    # 前向轴竖直时改用备用轴 −z，航向仍为 120°
    upright = walk_sequence(yaw=YAW30, upright=True)
    up_steps = model.detect_steps(view_of(upright))
    assert up_steps["alt_axis"].all()
    np.testing.assert_allclose(np.degrees(up_steps["heading"]), 120.0, atol=1e-6)


def test_resolve_axes_requires_a_known_body_frame():
    fwd, alt = resolve_axes("android_device")
    np.testing.assert_allclose(fwd, [0, 1, 0])
    np.testing.assert_allclose(alt, [0, 0, -1])
    with pytest.raises(ValueError, match="body_frame"):
        resolve_axes("head_mounted_rig")
    assert resolve_axes(BODY, AXES)[0][1] == 1.0
    angle, used = segment_heading(np.tile(quat_from_yaw(0.0), (10, 1)), fwd, alt)
    assert angle == pytest.approx(np.pi / 2) and not used


# --------------------------------------------------------------------------- 窗口汇总
def test_window_aggregation_matches_the_definition():
    """``v_k = Σ_{t_i ∈ (t_s, t_e]} L_i·u_i / ((T−1)·dt)``；t_s 处的步不计入、t_e 处的计入。"""
    model, _ = make_pdr()
    seq = walk_sequence(duration=6.0)
    view = view_of(seq)
    with torch.no_grad():
        model.step_gain.fill_(0.5)
    steps = {"time": np.array([0.0, 0.5, 0.995, 2.0]),
             "amplitude": np.full(4, 16.0), "heading": np.zeros(4),
             "alt_axis": np.zeros(4, bool)}
    model.detect_steps = lambda view: steps  # noqa: ARG005 - 用人工步序列检验公式
    starts = np.array([0, 200])
    _, velocity, extras = model.predict_sequence(seq, view, starts)
    length = 0.5 * 16.0 ** 0.25  # L = K·amp^{1/4}
    span = (WINDOW - 1) / RATE
    # 窗口 0 覆盖 (0.0, 0.995]：0.0 不计入，0.5 与 0.995 计入
    np.testing.assert_allclose(velocity[0], [2 * length / span, 0.0], rtol=1e-9)
    # 窗口 1 覆盖 (1.0, 1.995]：没有步
    np.testing.assert_allclose(velocity[1], 0.0)
    np.testing.assert_array_equal(extras["num_steps"], [2, 0])


def test_pdr_reconstructs_a_straight_constant_speed_walk():
    """40 步、每步 L、航向恒定 → 第 5 节重建的终点位移 ≈ 40·L（误差 < 5%）。"""
    seq = walk_sequence(duration=20.0, step_hz=2.0, speed=1.2)
    cfg = get_cfg({"model": "pdr", "window": WINDOW, "device": "cpu", "eval_stride": 10,
                   "orientation": "reference", "model_args": {"body_axes": AXES}})
    model = build_model(cfg)
    view = SequenceView(seq, ViewConfig(window=WINDOW, frame="gravity_world",
                                        orientation="reference"))
    stats = model.calibrate([view])
    assert stats["segments"] == 2 and stats["length_rmse"] < 0.1  # 12 m 段上 < 1%
    assert stats["step_gain"] > 0.0
    res = Predictor(cfg, model=model).predict_sequence(seq)
    metrics = res.compute_metrics()
    steps = model.detect_steps(view)
    expected = float(np.linalg.norm(model.step_displacements(steps).sum(axis=0)))
    travelled = float(np.linalg.norm(res.pos_pred[-1] - res.pos_pred[0]))
    assert travelled == pytest.approx(expected, rel=0.05), (travelled, expected)
    # 标定后的 PDR 在这条合成序列上距离误差很小
    assert metrics["pde"] < 5.0 and abs(metrics["plr"] - 1.0) < 0.05


def test_pdr_calibration_recovers_a_known_step_gain():
    """用已知 K 生成参考路径长 → 标定结果误差 < 1e-6；δ 已知偏置 0.3 rad → 误差 < 1e-3。"""
    model, _ = make_pdr(calibrate_delta=False)
    seq = walk_sequence(duration=40.0, step_hz=2.0, speed=1.2)
    view = view_of(seq)
    steps = model.detect_steps(view)
    amp_quarter = np.power(steps["amplitude"], 0.25)
    gain = 0.45
    # 参考轨迹的速度按“每 10 s 段内 K·Σamp^{1/4}”重新生成，使标定必须恢复出 K
    length = int(round(10.0 * RATE))
    position = np.zeros_like(seq.position)
    course = COURSE120
    for lo in range(0, len(seq) - length + 1, length):
        mask = ((steps["time"] >= seq.timestamp[lo])
                & (steps["time"] <= seq.timestamp[lo + length - 1]))
        distance = gain * float(amp_quarter[mask].sum())
        local = np.arange(length + 1)[:, None] / length * distance
        position[lo:lo + length + 1, 0] = position[lo, 0] + local[:, 0] * np.cos(course)
        position[lo:lo + length + 1, 1] = position[lo, 1] + local[:, 0] * np.sin(course)
    seq.position = position
    stats = model.calibrate([view_of(seq)])
    assert stats["step_gain"] == pytest.approx(gain, rel=2e-3)

    # δ：真值航向相对设备航向偏 0.3 rad
    biased = walk_sequence(duration=20.0, yaw=YAW30,
                           heading=COURSE120 + 0.3)
    model2, _ = make_pdr()
    stats2 = model2.calibrate([view_of(biased)])
    assert np.radians(stats2["heading_bias_deg"]) == pytest.approx(0.3, abs=1e-3)
    assert stats2["heading_resultant"] == pytest.approx(1.0, abs=1e-6)


def test_calibration_refuses_non_train_splits():
    model, _ = make_pdr()
    view = view_of(walk_sequence(duration=6.0))
    with pytest.raises(ValueError, match="train split"):
        model.calibrate([view], split="test")
    msh = build_model(get_cfg({"model": "mean_speed_heading", "window": WINDOW,
                               "orientation": "reference",
                               "model_args": {"body_axes": AXES}}))
    with pytest.raises(ValueError, match="train split"):
        msh.calibrate([view], split="val")


def test_pdr_requires_a_gravity_aligned_world_frame():
    with pytest.raises(ValueError, match="gravity_world"):
        build_model(get_cfg({"model": "pdr", "frame": "body", "dims": 3}))
    with pytest.raises(ValueError, match="one average velocity"):
        build_model(get_cfg({"model": "pdr", "target": "frame_velocity"}))


# --------------------------------------------------------------------------- 常速基线
def make_msh(variant="constant", **args):
    cfg = get_cfg({"model": "mean_speed_heading", "window": WINDOW, "device": "cpu",
                   "eval_stride": 10, "orientation": "reference",
                   "model_args": {"variant": variant, "body_axes": AXES, **args}})
    return build_model(cfg), cfg


def test_constant_variant_uses_the_calibrated_speed_and_heading():
    model, cfg = make_msh()
    seq = walk_sequence(duration=20.0, speed=1.2)
    view = view_of(seq)
    stats = model.calibrate([view])
    assert stats["mean_speed"] == pytest.approx(1.2, rel=1e-6)
    assert abs(stats["heading_bias_deg"]) < 1e-6
    starts = view.starts(10, require_valid=False)
    _, velocity, _ = model.predict_sequence(seq, view, starts)
    np.testing.assert_allclose(np.linalg.norm(velocity, axis=1), 1.2, atol=1e-9)
    np.testing.assert_allclose(np.degrees(np.arctan2(velocity[:, 1], velocity[:, 0])), 120.0,
                               atol=1e-6)
    # 与 IMU 数值无关：把加计换成噪声，输出不变
    noisy = walk_sequence(duration=20.0, speed=1.2)
    rng = np.random.default_rng(1)
    noisy.accelerometer = rng.normal(0.0, 3.0, noisy.accelerometer.shape).astype(np.float32)
    _, again, _ = model.predict_sequence(noisy, view_of(noisy), starts)
    np.testing.assert_allclose(again, velocity, atol=1e-9)


def test_constant_variant_is_yaw_equivariant():
    model, _ = make_msh()
    seq = walk_sequence(duration=15.0, speed=1.0, yaw=0.2)
    model.calibrate([view_of(seq)])
    starts = view_of(seq).starts(10, require_valid=False)
    _, base, _ = model.predict_sequence(seq, view_of(seq), starts)
    alpha = 0.9
    rotated = walk_sequence(duration=15.0, speed=1.0, yaw=0.2 + alpha,
                            heading=np.arctan2(base[0, 1], base[0, 0]) + alpha)
    _, turned, _ = model.predict_sequence(rotated, view_of(rotated), starts)
    from inertial_benchmark.utils.geometry import rotate_z

    np.testing.assert_allclose(turned, rotate_z(base, alpha), atol=1e-6)


def test_gated_and_sanity_check_variants():
    walking = walk_sequence(duration=20.0, amplitude=2.0, speed=1.2)
    still = walk_sequence(duration=20.0, amplitude=0.0, noise=0.05, speed=0.0, seed=3)
    gated, _ = make_msh("gated")
    stats = gated.calibrate([view_of(walking), view_of(still)])
    assert stats["mean_speed_moving"] > stats["mean_speed"]  # 静止窗口被门控排除
    starts = view_of(still).starts(10, require_valid=False)
    _, zero, _ = gated.predict_sequence(still, view_of(still), starts)
    np.testing.assert_allclose(zero, 0.0)
    _, moving, _ = gated.predict_sequence(walking, view_of(walking),
                                          view_of(walking).starts(10, require_valid=False))
    np.testing.assert_allclose(np.linalg.norm(moving, axis=1),
                               float(gated.mean_speed_moving), atol=1e-9)
    # zero_velocity：轨迹恒等于锚点
    zero_model, cfg = make_msh("zero_velocity")
    res = Predictor(cfg, model=zero_model).predict_sequence(walking)
    np.testing.assert_allclose(res.pos_pred, np.tile(walking.position[0, :2], (len(walking), 1)),
                               atol=1e-9)
    # train_mean_velocity：与姿态无关，等于向量平均
    mean_model, _ = make_msh("train_mean_velocity")
    mean_stats = mean_model.calibrate([view_of(walking)])
    _, vectors, _ = mean_model.predict_sequence(walking, view_of(walking), starts[:5])
    np.testing.assert_allclose(vectors, np.tile(mean_stats["mean_velocity"], (len(vectors), 1)),
                               atol=1e-6)


def test_mean_speed_heading_reconstructs_a_constant_heading_line():
    """恒定航向直线：重建终点位移 == s̄·(N−1)·dt（首尾常值外推后速度处处相等）。"""
    model, cfg = make_msh()
    seq = walk_sequence(duration=20.0, speed=1.1)
    model.calibrate([view_of(seq)])
    res = Predictor(cfg, model=model).predict_sequence(seq)
    expected = float(model.mean_speed) * (len(seq) - 1) / RATE
    assert float(np.linalg.norm(res.pos_pred[-1] - res.pos_pred[0])) == pytest.approx(expected,
                                                                                      rel=1e-6)
    metrics = res.compute_metrics()
    assert metrics["ate"] < 0.5 and abs(metrics["plr"] - 1.0) < 1e-3


@pytest.mark.parametrize("name", ["pdr", "mean_speed_heading"])
def test_baseline_train_val_writes_the_calibration(name, tmp_path):
    """序列级基线走同一套 train（只标定）→ val 流程，标定量写入 metrics.json 并存进 checkpoint。"""
    import json

    from test_engine import make_dataset

    from inertial_benchmark import NIO

    data = make_dataset(tmp_path / "ds")
    axes = {"synthetic_device": [[0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]}
    model = NIO(name, device="cpu", workers=0, efficiency=False, plots=False, window=100,
                eval_stride=20, orientation="reference",
                model_args={"body_axes": axes, "delta_stride": 20})
    model.train(data=str(data), project=str(tmp_path), name=name, epochs=4)
    meta = json.loads((tmp_path / "train" / name / "metrics.json").read_text())
    assert meta["calibration"]["heading_resultant"] >= 0.0
    # 序列级模型没有 train()/eval() 语义差异：失配诊断跳过而不是报错
    assert "sequence model" in meta["train_eval_gap"]["skipped"]
    rows = (tmp_path / "train" / name / "results.csv").read_text().strip().splitlines()
    assert len(rows) == 2  # 只标定一次，不跑 4 轮梯度训练
    # checkpoint 里带着标定量：独立 val 直接复用
    loaded = NIO(str(tmp_path / "train" / name / "weights" / "best.pt"), device="cpu",
                 workers=0, efficiency=False, plots=False)
    loaded.val(data=str(data), split="test", project=str(tmp_path), name=name + "_val")
    val_meta = json.loads((tmp_path / "val" / (name + "_val") / "metrics.json").read_text())
    assert val_meta["calibration"] == meta["calibration"]
    assert val_meta["metrics"]["ate"] > 0


def test_baselines_have_no_learnable_parameters():
    from inertial_benchmark.metrics.efficiency import efficiency_metrics

    for name in ("pdr", "mean_speed_heading"):
        model = build_model(get_cfg({"model": name, "orientation": "reference"}))
        assert model.num_params == 0, name
        assert isinstance(model, torch.nn.Module)
        # 序列级模型没有逐窗口前向：效率指标只记录参数量，不因此崩掉
        eff = efficiency_metrics(model, model.input_spec.window,
                                 input_shape=model.input_spec.input_shape)
        assert eff["params"] == 0 and eff["flops"] is None
    assert PDR.__mro__[1].__name__ == "SequenceModel"
