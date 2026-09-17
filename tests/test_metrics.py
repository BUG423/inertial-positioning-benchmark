"""指标的解析解测试：匀速直线、圆周运动、刚体变换、速度旋转/缩放。"""

import math

import numpy as np
import pytest

from inertial_benchmark.metrics import (
    align_trajectory,
    ate,
    ate_aligned,
    d_rte,
    drift,
    length_ratios,
    path_length,
    rte,
    sequence_metrics,
    trajectory_metrics,
    window_metrics,
)
from inertial_benchmark.utils.geometry import rotate_z

RATE = 200.0


def line(duration, speed=1.5, heading=0.3):
    t = np.arange(int(duration * RATE) + 1) / RATE
    v = speed * np.array([math.cos(heading), math.sin(heading), 0.0])
    return t, t[:, None] * v


def circle(duration, radius=5.0, omega=0.5):
    t = np.arange(int(duration * RATE) + 1) / RATE
    pos = np.column_stack([radius * np.cos(omega * t), radius * np.sin(omega * t), 0 * t])
    return t, pos


def test_constant_velocity_with_scale_error():
    s, speed, duration = 0.1, 1.5, 120.0
    t, gt = line(duration, speed)
    pred = gt * (1 + s)
    valid = np.ones(len(t), bool)
    assert ate(pred, gt) == pytest.approx(s * speed * math.sqrt(np.mean(t**2)), rel=1e-9)
    assert rte(pred, gt, valid, RATE, 60.0) == pytest.approx(s * speed * 60.0, rel=1e-9)
    assert rte(pred, gt, valid, RATE, 1.0) == pytest.approx(s * speed, rel=1e-9)
    assert rte(pred, gt, valid, RATE, 10.0) == pytest.approx(s * speed * 10.0, rel=1e-9)
    assert drift(pred, gt, valid, RATE) == pytest.approx(100 * s, rel=1e-9)
    assert d_rte(pred, gt, valid, RATE, 10.0) == pytest.approx(10 * s, rel=1e-3)
    ratios = length_ratios(pred, gt, valid, RATE)
    assert ratios["plr"] == pytest.approx(1 + s, rel=1e-9)
    assert ratios["plr_dense"] == pytest.approx(1 + s, rel=1e-9)
    assert path_length(gt, valid, RATE) == pytest.approx(speed * duration, rel=1e-9)
    # 缩放误差不是刚体误差：对齐后 ATE 仍非零，但小于未对齐
    assert 0 < ate_aligned(pred, gt) < ate(pred, gt)


def test_rigid_transform_is_removed_by_alignment():
    t, gt = circle(60.0)
    pred = rotate_z(gt, 0.4) + np.array([3.0, -2.0, 0.0])
    assert ate(pred, gt) > 3
    assert ate_aligned(pred, gt) == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_allclose(align_trajectory(pred, gt), gt[:, :2], atol=1e-9)
    # 3D 对齐：z 只平移
    pred3 = pred + np.array([0, 0, 1.5])
    assert ate_aligned(pred3, gt, dims=3) == pytest.approx(0.0, abs=1e-9)


def test_circle_length_ratios_and_identity():
    radius, omega, duration = 5.0, 0.8, 100.0
    t, gt = circle(duration, radius, omega)
    valid = np.ones(len(t), bool)
    dense = path_length(gt, valid, RATE, None)
    assert dense == pytest.approx(radius * omega * duration, rel=1e-5)
    coarse = path_length(gt, valid, RATE, 1.0)
    chord = 2 * radius * math.sin(omega / 2)
    assert coarse == pytest.approx(chord * duration, rel=1e-9)
    # oracle = 真值的平滑版本（半径略小的同相圆），预测 = 1.05 × oracle
    oracle = gt * 0.98
    pred = oracle * 1.05
    r = length_ratios(pred, gt, valid, RATE, oracle)
    assert r["plr"] == pytest.approx(0.98 * 1.05, rel=1e-9)
    assert r["plr_oracle"] == pytest.approx(1.05, rel=1e-9)
    assert r["oracle_ratio"] == pytest.approx(0.98 * chord / (radius * omega), rel=1e-5)
    assert r["plr_dense"] == pytest.approx(r["oracle_ratio"] * r["plr_oracle"], rel=1e-12)


def test_circle_rte_with_heading_error():
    # 预测轨迹相对真值有常值角速度偏差：p̂ 在半径相同但角速度为 ω+δ 的圆上
    radius, omega, delta = 5.0, 2 * math.pi / 31.4, 0.01  # 周期恰为 6280 个样本
    t, gt = circle(200.0, radius, omega)
    pred = np.column_stack([radius * np.cos((omega + delta) * t),
                            radius * np.sin((omega + delta) * t), 0 * t])
    valid = np.ones(len(t), bool)
    d = int(60 * RATE)
    i = np.arange(len(t) - d)

    def unit(w, tt):
        return np.column_stack([np.cos(w * tt), np.sin(w * tt)])

    # 逐对的解析相对位移误差
    e = radius * ((unit(omega + delta, t[i + d]) - unit(omega + delta, t[i]))
                  - (unit(omega, t[i + d]) - unit(omega, t[i])))
    expected = math.sqrt(np.mean(np.sum(e**2, axis=1)))
    assert rte(pred, gt, valid, RATE, 60.0) == pytest.approx(expected, rel=1e-9)
    # 一个圆周（2π/ω）后真值相对位移为零，预测的相对位移长度为 2R|sin(δΔ/2)|
    assert rte(pred, gt, valid, RATE, 31.4) == pytest.approx(
        2 * radius * abs(math.sin(delta * 31.4 / 2)), rel=1e-9)


def test_rte_short_sequence_is_scaled():
    b = np.array([0.02, -0.01, 0.0])  # 常值速度偏差 → 终点误差 |b|·T
    t, gt = line(30.0)
    pred = gt + t[:, None] * b
    valid = np.ones(len(t), bool)
    expected = np.linalg.norm(b[:2]) * 30.0 * (60.0 / 30.0)
    assert rte(pred, gt, valid, RATE, 60.0) == pytest.approx(expected, rel=1e-9)
    # 首末样本无效时使用首末有效样本及其实际跨度
    valid[:200] = False
    valid[-400:] = False
    span = (len(t) - 401 - 200) / RATE
    expected = np.linalg.norm(b[:2]) * span * 60.0 / span
    assert rte(pred, gt, valid, RATE, 60.0) == pytest.approx(expected, rel=1e-9)


def test_valid_masks_and_edge_cases():
    t, gt = line(90.0)
    pred = gt.copy()
    pred[1000:1200] += 50.0  # 仅在无效区出错
    valid = np.ones(len(t), bool)
    valid[1000:1200] = False
    assert ate(pred, gt, valid) == 0.0
    assert rte(pred, gt, valid, RATE, 10.0) == 0.0
    # 终点漂移取末个有效样本
    pred2 = gt.copy()
    pred2[-10:] += 100.0
    v2 = np.ones(len(t), bool)
    v2[-10:] = False
    assert drift(pred2, gt, v2, RATE) == 0.0
    # 无有效样本 / 长度过短 → NaN
    none = np.zeros(len(t), bool)
    assert math.isnan(ate(pred, gt, none))
    assert math.isnan(rte(pred, gt, none, RATE))
    assert math.isnan(ate_aligned(pred, gt, none))
    still = np.zeros_like(gt)
    assert math.isnan(drift(still, still, None, RATE))
    assert math.isnan(d_rte(still, still, None, RATE, 10.0))
    assert math.isnan(length_ratios(still, still, None, RATE)["plr"])
    with pytest.raises(ValueError):
        ate(pred[:10], gt)


def test_trajectory_metrics_keys():
    t, gt = line(70.0)
    out = trajectory_metrics(gt * 1.02, gt, None, RATE, oracle=gt, t_rte=(1, 10),
                             d_rte_distance=(10, 20))
    assert {"ate", "ate_aligned", "rte", "t_rte_1s", "t_rte_10s", "d_rte_10m", "d_rte_20m",
            "pde", "plr", "plr_dense", "plr_oracle", "oracle_ratio", "ate_oracle"} <= set(out)
    assert out["ate_oracle"] == 0.0 and out["oracle_ratio"] == pytest.approx(1.0)


def test_window_metrics_rotation_and_scale():
    rng = np.random.default_rng(0)
    heading = rng.uniform(-np.pi, np.pi, 500)
    speed = rng.uniform(0.5, 2.0, 500)
    target = np.column_stack([speed * np.cos(heading), speed * np.sin(heading)])
    theta = np.radians(10.0)
    pred = rotate_z(target, theta)
    m = window_metrics(pred, target)
    assert m["dir_err_mean"] == pytest.approx(10.0)
    assert m["dir_err_median"] == pytest.approx(10.0)
    assert m["speed_bias"] == pytest.approx(0.0, abs=1e-12)
    assert m["speed_ratio"] == pytest.approx(1.0)
    assert m["along_bias"] == pytest.approx(np.mean(speed) * (math.cos(theta) - 1))
    assert m["cross_rmse"] == pytest.approx(math.sin(theta) * math.sqrt(np.mean(speed**2)))
    assert m["vel_rmse"] == pytest.approx(2 * math.sin(theta / 2) * math.sqrt(np.mean(speed**2)))

    scaled = window_metrics(1.1 * target, target)
    assert scaled["speed_ratio"] == pytest.approx(1.1)
    assert scaled["speed_bias"] == pytest.approx(0.1 * np.mean(speed))
    assert scaled["dir_err_mean"] == pytest.approx(0.0, abs=1e-6)
    assert scaled["cross_rmse"] == pytest.approx(0.0, abs=1e-12)
    assert scaled["along_rmse"] == pytest.approx(0.1 * math.sqrt(np.mean(speed**2)))


def test_window_metrics_thresholds_and_masks():
    target = np.array([[1.0, 0.0], [0.1, 0.0], [0.0, 1.0], [1.0, 1.0]])
    pred = np.array([[0.0, 1.0], [-0.1, 0.0], [0.0, 0.0], [5.0, 5.0]])
    valid = np.array([True, True, True, False])
    m = window_metrics(pred, target, valid, min_speed=0.2)
    assert m["num_windows"] == 3 and m["num_moving_windows"] == 2
    # 慢窗口（180° 误差）被排除；零预测记 90°
    assert m["dir_err_mean"] == pytest.approx(90.0)
    empty = window_metrics(pred, target, np.zeros(4, bool))
    assert empty["num_windows"] == 0 and math.isnan(empty["vel_rmse"])
    still = window_metrics(np.zeros((3, 2)), np.zeros((3, 2)))
    assert math.isnan(still["dir_err_mean"]) and math.isnan(still["speed_ratio"])
    three = window_metrics(np.array([[0, 1.0, 0]]), np.array([[1.0, 0, 5.0]]), dims=3)
    assert three["dir_err_mean"] == pytest.approx(90.0)


def test_sequence_metrics_combines_both():
    t, gt = line(65.0)
    vel = np.tile([1.5 * math.cos(0.3), 1.5 * math.sin(0.3)], (100, 1))
    out = sequence_metrics(gt, gt, np.ones(len(t), bool), vel, vel, np.ones(100, bool),
                           pos_oracle=gt)
    assert out["ate"] == 0 and out["vel_rmse"] == 0 and out["num_windows"] == 100


def test_efficiency_metrics():
    torch = pytest.importorskip("torch")
    from inertial_benchmark.metrics.efficiency import (
        count_flops,
        count_params,
        efficiency_metrics,
        hook_flops,
    )

    model = torch.nn.Sequential(torch.nn.Conv1d(6, 4, 3, padding=1), torch.nn.Flatten(),
                                torch.nn.Linear(4 * 10, 2))
    assert count_params(model) == 6 * 4 * 3 + 4 + 40 * 2 + 2
    expected = 10 * 4 * (6 * 3) + 10 * 4 + (40 * 2 + 2)
    assert hook_flops(model, torch.zeros(1, 6, 10)) == expected
    assert hook_flops(model, torch.zeros(3, 6, 10)) == 3 * expected
    assert count_flops(model, (1, 6, 10), "hooks") == {"flops": expected,
                                                       "flops_backend": "hooks"}
    lstm = torch.nn.LSTM(6, 8, batch_first=True)
    per_step = 4 * (6 * 8 + 8 * 8 + 16) + 3 * 8

    class Wrap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = lstm

        def forward(self, x):
            return self.lstm(x.transpose(1, 2))[0]

    assert count_flops(Wrap(), (1, 6, 5), "hooks")["flops"] == 5 * per_step
    eff = efficiency_metrics(model, window=10, runs=3)
    assert eff["params"] == count_params(model) and eff["latency_ms_cpu"] > 0
    assert eff["flops_backend"] in ("thop", "fvcore", "hooks")
    with pytest.raises(ValueError):
        count_flops(model, (1, 6, 10), "magic")
