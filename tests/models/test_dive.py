"""DIVE 移植测试：夹具参数量与变体、局部重力对齐输入的构造、偏航不变性与阶段切换。"""

import math

import pytest

from .model_testing import load_fixture, param_shapes, run_end_to_end, shape_multiset

torch = pytest.importorskip("torch")

from inertial_benchmark.cfg import get_cfg  # noqa: E402
from inertial_benchmark.models.dive.model import (  # noqa: E402
    SCIPY_GRAVITY,
    body_measurements,
    gravity_aligned_input,
    heading_from_quaternion,
    local_attitude_frames,
    rotation_z,
    so3_log_principal,
)
from inertial_benchmark.nn import build_model  # noqa: E402
from inertial_benchmark.nn.losses import build_loss  # noqa: E402
from inertial_benchmark.nn.modules.lie_events import quat_to_matrix, so3_exp  # noqa: E402

FIXTURE = load_fixture("dive")
BATCH = 4
DT = 1.0 / 200.0


def build(**overrides):
    return build_model(get_cfg({"model": "dive", **overrides}))


def quat_from_yaw(yaw: float, dtype=torch.float64) -> torch.Tensor:
    return torch.tensor([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)], dtype=dtype)


def quat_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz,
                        aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw], dim=-1)


def test_parameters_match_the_ipb_fixture():
    model = build()
    assert model.num_params == FIXTURE["total_params"] == 7_516_420
    shapes = param_shapes(model)
    assert shape_multiset(shapes) == shape_multiset(FIXTURE["param_shapes"])
    assert shapes == FIXTURE["param_shapes"] and len(shapes) == 102
    buffers = [list(b.shape) for _, b in model.named_buffers()]
    assert buffers == FIXTURE["buffer_shapes"] and len(buffers) == 90
    component = FIXTURE["component_params"]
    backbone = model.backbone
    assert sum(p.numel() for p in backbone.backbone.stem.parameters()) \
        == component["input_block"]
    assert sum(p.numel() for p in backbone.backbone.stages.parameters()) \
        == component["residual_groups"]
    assert sum(p.numel() for p in backbone.mean_head.parameters()) == component["output_block1"]
    assert sum(p.numel() for p in backbone.logstd_head.parameters()) == component["output_block2"]
    assert model.feature_length == 7


def test_official_and_other_variants():
    """夹具的 ``official_config`` 与 ``variant_total_params``（卡 §4 的参照配置）。"""
    variants = FIXTURE["variant_total_params"]
    assert build(dims=3, window=1400).num_params == FIXTURE["official_config"]["total_params"]
    assert build(dims=3).num_params \
        == variants["200Hz_1.0s_3d_depth3 (IPB window=200, dims=3)"]
    legacy = build(dims=3, window=1400, model_args={"group_sizes": [2, 2, 2, 2]})
    assert legacy.num_params == variants[
        "legacy network/net.py group_sizes [2,2,2,2], 1400 samples, 3d"]


def test_output_shapes_and_independent_heads():
    model = build().eval()
    spec = model.input_spec
    with torch.no_grad():
        out = model(torch.randn(BATCH, 6, 200), spec.dummy_extra(BATCH))
    assert out["vel"].shape == (BATCH, 2) and out["logstd"].shape == (BATCH, 2)
    mean = dict(model.backbone.mean_head.named_parameters())
    logstd = dict(model.backbone.logstd_head.named_parameters())
    assert set(mean) == set(logstd)
    assert all(mean[k] is not logstd[k] for k in mean)


def test_preprocessing_of_a_level_stationary_window():
    """静止、水平、偏航 γ 的窗口：``φ ≡ 0``、``a ≡ 0``（规格卡 §8）。"""
    model = build().double().eval()
    yaw = 0.8
    quat = quat_from_yaw(yaw).expand(1, 200, 4).contiguous()
    imu = torch.zeros(1, 6, 200, dtype=torch.float64)
    # 视图坐标系下的比力：R_z(γ)·[0, 0, g] = [0, 0, g]
    imu[:, 5] = SCIPY_GRAVITY
    with torch.no_grad():
        channels = model.preprocess(imu, {"orientation": quat})
    assert float(channels.abs().max()) < 1e-5


def test_preprocessing_is_yaw_invariant():
    """把 IMU 与姿态一起绕 z 轴旋转 θ → ``[φ, a]`` 完全不变（规格卡 §8）。"""
    torch.manual_seed(0)
    model = build().double().eval()
    quat = torch.randn(2, 200, 4, dtype=torch.float64)
    quat = quat / quat.norm(dim=-1, keepdim=True)
    imu = torch.randn(2, 6, 200, dtype=torch.float64)
    theta = 1.3
    spin = rotation_z(torch.tensor(theta, dtype=torch.float64))
    rotated_imu = torch.cat([torch.einsum("ij,bjt->bit", spin, imu[:, 0:3]),
                             torch.einsum("ij,bjt->bit", spin, imu[:, 3:6])], dim=1)
    rotated_quat = quat_multiply(quat_from_yaw(theta).expand_as(quat), quat)
    with torch.no_grad():
        base = model.preprocess(imu, {"orientation": quat})
        turned = model.preprocess(rotated_imu, {"orientation": rotated_quat})
    assert float((turned - base).abs().max()) < 1e-6


def test_backward_integration_matches_the_analytic_frames():
    """恒定 ω 时向后积分得到的 ``D_k`` 应等于 ``R_z(−γ)·C_k``（规格卡 §8）。"""
    omega = torch.tensor([0.1, -0.2, 0.35], dtype=torch.float64)
    steps = 200
    start = so3_exp(torch.tensor([0.05, -0.03, 0.5], dtype=torch.float64))
    rot = [start]
    for _ in range(steps - 1):
        rot.append(rot[-1] @ so3_exp(omega * DT))
    rot = torch.stack(rot).unsqueeze(0)
    body_gyro = omega.expand(1, steps, 3)
    yaw = torch.zeros(1, dtype=torch.float64)
    frames = local_attitude_frames(body_gyro, rot[:, -1], yaw, DT)
    torch.testing.assert_close(frames, rot, atol=1e-9, rtol=0)
    # 加上非零 γ 后每一帧都左乘 R_z(−γ)
    yaw = torch.tensor([0.7], dtype=torch.float64)
    shifted = local_attitude_frames(body_gyro, rot[:, -1], yaw, DT)
    expected = rotation_z(-yaw).unsqueeze(1) @ rot
    torch.testing.assert_close(shifted, expected, atol=1e-9, rtol=0)


def test_channels_are_attitude_log_and_gravity_free_acceleration():
    """通道语义：前三个是 ``Log(D_k)``，后三个是 ``D_k f^b_k − [0,0,g]``（规格卡 §2）。"""
    torch.manual_seed(1)
    quat = torch.randn(2, 200, 4, dtype=torch.float64)
    quat = quat / quat.norm(dim=-1, keepdim=True)
    imu = torch.randn(2, 6, 200, dtype=torch.float64)
    rot = quat_to_matrix(quat)
    body_gyro, body_acc = body_measurements(imu, rot)
    frames = local_attitude_frames(body_gyro, rot[:, -1], heading_from_quaternion(quat[:, -1]), DT)
    channels = gravity_aligned_input(imu, quat, DT)
    torch.testing.assert_close(channels[:, 0:3], so3_log_principal(frames).transpose(1, 2),
                               atol=1e-12, rtol=0)
    expected = torch.einsum("btij,btj->bti", frames, body_acc)
    expected[..., 2] -= SCIPY_GRAVITY
    torch.testing.assert_close(channels[:, 3:6], expected.transpose(1, 2), atol=1e-12, rtol=0)
    # 陀螺不直接进入通道：只改陀螺的**最后一个**样本不影响任何输出（向后积分只用 k <= T−2）
    tweaked = imu.clone()
    tweaked[:, 0:3, -1] += 1.0
    torch.testing.assert_close(gravity_aligned_input(tweaked, quat, DT), channels,
                               atol=1e-12, rtol=0)


def test_so3_log_principal_roundtrip():
    torch.manual_seed(2)
    phi = torch.randn(64, 3, dtype=torch.float64)
    phi = phi / phi.norm(dim=-1, keepdim=True) * torch.rand(64, 1, dtype=torch.float64) * math.pi
    torch.testing.assert_close(so3_log_principal(so3_exp(phi)), phi, atol=1e-8, rtol=0)
    # 与 lie_events 的官方变体不同：θ 接近 π 时不返回零向量
    axis = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64) * (math.pi - 5e-4)
    assert float(so3_log_principal(so3_exp(axis)).abs().max()) > 3.0


def test_loss_stage_switch_and_offset():
    model = build()
    spec = model.input_spec
    x = torch.randn(BATCH, 6, 200)
    extra = spec.dummy_extra(BATCH)
    target = torch.randn(BATCH, 2)
    loss_fn = build_loss("dive_mse_then_nll")
    assert loss_fn.switch_epoch == 10 and loss_fn.offset == 1e-7
    assert loss_fn.stage(9) == "mse" and loss_fn.stage(10) == "nll"
    for epoch, expect_grad in ((9, False), (10, True)):
        model.zero_grad(set_to_none=True)
        loss, items = model.loss(model(x, extra), {"target": target, "imu": x}, epoch)
        loss.backward()
        grad = model.backbone.logstd_head.fc3.weight.grad
        assert (grad is not None and float(grad.abs().max()) > 0) is expect_grad
        assert ("nll" in items) is expect_grad
    # s = 0 时 NLL == mean(r²/2) + 1e-7（官方在 s 上加的 1e-7 偏移）
    residual = torch.tensor([[0.3, -0.4]], dtype=torch.float64)
    out = {"vel": residual, "logstd": torch.zeros(1, 2, dtype=torch.float64)}
    value, _ = loss_fn(out, torch.zeros(1, 2, dtype=torch.float64), epoch=10)
    expected = float((residual**2 / (2 * math.exp(2e-7))).mean()) + 1e-7
    assert float(value) == pytest.approx(expected, abs=1e-12)


def test_loss_masks_invalid_outputs():
    model = build()
    spec = model.input_spec
    x = torch.randn(BATCH, 6, 200)
    extra = spec.dummy_extra(BATCH)
    out = model(x, extra)
    mask = torch.zeros(BATCH, 1)
    for epoch in (0, 99):
        loss, _ = model.loss(out, {"target": torch.randn(BATCH, 2), "imu": x, "mask": mask},
                             epoch)
        assert float(loss.detach()) == 0.0


def test_forward_backward_and_initialisation():
    model = build()
    spec = model.input_spec
    x = torch.randn(BATCH, 6, 200)
    extra = spec.dummy_extra(BATCH)
    model.eval()
    with torch.no_grad():
        first, second = model(x, extra)["vel"], model(x, extra)["vel"]
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    model.train()
    model.zero_grad(set_to_none=True)
    loss, _ = model.loss(model(x, extra), {"target": torch.randn(BATCH, 2), "imu": x}, 99)
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"parameters without gradient: {missing[:5]}"
    # 官方初始化：Linear 为 N(0, 0.01²)，bias 为 0
    linear = model.backbone.mean_head.fc1
    assert float(linear.weight.detach().std()) == pytest.approx(0.01, rel=0.1)
    assert float(linear.bias.detach().abs().max()) == 0.0


def test_configuration_constraints():
    with pytest.raises(ValueError, match="extra_inputs"):
        build(extra_inputs=[])
    with pytest.raises(ValueError, match="gravity-aligned"):
        build(frame="body", dims=3)
    cfg = get_cfg({"model": "dive", "recipe": "official"})
    assert cfg.epochs == 100 and cfg.batch == 32 and cfg.lr == 1e-4
    assert cfg.optimizer == "adamw" and cfg.weight_decay == 0.01 and cfg.scheduler == "none"
    assert cfg.frame == "gravity_yaw_local" and cfg.target == "velocity_at_end"
    assert cfg.dims == 2 and cfg.extra_inputs == ["orientation"]
    assert cfg.augment == get_cfg({"model": "dive"}).augment
    # 非特权：DIVE 只用逐样本姿态，不用参考真值速度
    assert build().input_spec.privileged_inputs == ()


@pytest.mark.slow
def test_end_to_end_train_and_val(synthetic_dataset, tmp_path):
    run_end_to_end("dive", synthetic_dataset, tmp_path)
