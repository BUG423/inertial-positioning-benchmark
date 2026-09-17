"""NIO from Lie Events 移植测试：SE(3) 工具、事件生成 golden、分桶、偏航等变与特权输入标记。"""

import math

import pytest

from .model_testing import load_fixture, param_shapes, run_end_to_end, shape_multiset

torch = pytest.importorskip("torch")

from inertial_benchmark.cfg import get_cfg  # noqa: E402
from inertial_benchmark.nn import build_model  # noqa: E402
from inertial_benchmark.nn.modules.lie_events import (  # noqa: E402
    generate_event_stack,
    left_jacobian,
    left_jacobian_inverse,
    lie_events,
    quat_to_matrix,
    se3_exp,
    se3_log,
    so3_exp,
    so3_log,
)

RONIN = load_fixture("nio_lie_events_ronin")
TLIO = load_fixture("nio_lie_events_tlio")
DT = 1.0 / 200.0


def build(name: str, **overrides):
    return build_model(get_cfg({"model": name, **overrides}))


def rotation_z(angle: float, dtype=torch.float64) -> torch.Tensor:
    cos, sin = math.cos(angle), math.sin(angle)
    return torch.tensor([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]], dtype=dtype)


def synthetic_window() -> tuple:
    """夹具 ``synthetic_trajectory`` 定义的合成轨迹：世界系 IMU、逐样本姿态与 ``v0``。"""
    t = torch.arange(200, dtype=torch.float64) * DT
    body_gyro = torch.stack([0.2 * torch.sin(2 * math.pi * t),
                             0.1 * torch.cos(2 * math.pi * t),
                             torch.full_like(t, 0.5)], dim=-1)
    world_acc = torch.stack([torch.sin(4 * math.pi * t),
                             0.5 * torch.cos(4 * math.pi * t),
                             torch.full_like(t, 9.81)], dim=-1)
    angles = torch.tensor([0.05, -0.03, math.radians(30.0)], dtype=torch.float64)
    rot = [so3_exp(torch.tensor([0.0, 0.0, float(angles[2])], dtype=torch.float64))
           @ so3_exp(torch.tensor([0.0, float(angles[1]), 0.0], dtype=torch.float64))
           @ so3_exp(torch.tensor([float(angles[0]), 0.0, 0.0], dtype=torch.float64))]
    for k in range(199):
        rot.append(rot[-1] @ so3_exp(body_gyro[k] * DT))
    rot = torch.stack(rot)
    world_gyro = torch.einsum("tij,tj->ti", rot, body_gyro)
    v0 = torch.tensor([[1.2, 0.3, 0.0]], dtype=torch.float64)
    return world_gyro.unsqueeze(0), world_acc.unsqueeze(0), rot.unsqueeze(0), v0


# --------------------------------------------------------------------- SE(3) 工具


def test_se3_exp_log_roundtrip():
    torch.manual_seed(0)
    xi = torch.randn(32, 6, dtype=torch.float64) * 0.4
    rot, trans = se3_exp(xi)
    torch.testing.assert_close(se3_log(rot, trans), xi, atol=1e-10, rtol=1e-10)
    eye = torch.eye(3, dtype=torch.float64).expand(32, 3, 3)
    torch.testing.assert_close(rot @ rot.transpose(1, 2), eye, atol=1e-12, rtol=0)


def test_so3_log_returns_zero_near_pi():
    """官方行为：``|θ − π| < 1e-3`` 时 ``log(R)`` 直接返回零向量（不连续，必须覆盖）。"""
    axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    near = so3_exp(axis * (math.pi - 5e-4))
    assert float(so3_log(near).abs().max()) == 0.0
    away = so3_exp(axis * (math.pi - 5e-2))
    torch.testing.assert_close(so3_log(away), axis * (math.pi - 5e-2), atol=1e-9, rtol=0)


def test_left_jacobian_inverse():
    torch.manual_seed(1)
    phi = torch.randn(16, 3, dtype=torch.float64) * 0.5
    eye = torch.eye(3, dtype=torch.float64).expand(16, 3, 3)
    torch.testing.assert_close(left_jacobian(phi) @ left_jacobian_inverse(phi), eye,
                               atol=1e-10, rtol=0)
    tiny = torch.zeros(2, 3, dtype=torch.float64)
    torch.testing.assert_close(left_jacobian(tiny) @ left_jacobian_inverse(tiny),
                               torch.eye(3, dtype=torch.float64).expand(2, 3, 3),
                               atol=1e-12, rtol=0)


def test_quaternion_to_matrix_matches_numpy_helper():
    from inertial_benchmark.utils.geometry import quat_to_matrix as numpy_quat_to_matrix

    torch.manual_seed(2)
    quat = torch.randn(8, 4, dtype=torch.float64)
    quat = quat / quat.norm(dim=-1, keepdim=True)
    expected = torch.as_tensor(numpy_quat_to_matrix(quat.numpy()))
    torch.testing.assert_close(quat_to_matrix(quat), expected, atol=1e-12, rtol=0)


# --------------------------------------------------------------------- 事件生成 golden


@pytest.mark.parametrize("key,threshold", [("theta_0.1", 0.1), ("theta_0.01", 0.01)])
def test_event_generation_matches_the_fixture(key, threshold):
    """夹具 ``event_generation_evidence``：事件数、空桶数、逐桶取值与末端预积分位姿。"""
    want = RONIN["event_generation_evidence"][key]
    gyro, acc, rot, v0 = synthetic_window()
    got = lie_events(gyro, acc, rot, v0, DT, threshold, bins=200)
    stack = got["stack"][0]
    assert int(got["counts"][0]) == want["n_events_excluding_first_last"]
    assert list(got["stack"].shape) == want["stack_shape"]
    assert int((stack.abs().sum(0) == 0).sum()) == want["empty_bins"]
    for index, values in ((0, "bin0"), (199, "bin199")):
        torch.testing.assert_close(stack[:, index],
                                   torch.tensor(want[values], dtype=torch.float64),
                                   atol=1e-6, rtol=0)
    for row in want["nonempty_bins_sparse"]:
        torch.testing.assert_close(stack[:, int(row[0])],
                                   torch.tensor(row[1:], dtype=torch.float64),
                                   atol=1e-6, rtol=0)
    pose = torch.tensor(want["final_preintegrated_pose_T"], dtype=torch.float64)
    torch.testing.assert_close(got["pose_rotation"][0], pose[:3, :3], atol=1e-6, rtol=0)
    torch.testing.assert_close(got["pose_translation"][0], pose[:3, 3], atol=1e-6, rtol=0)


def test_polarity_norms_and_bin_indices():
    """非空桶的极性范数落在 ``(0, 1)`` 内且接近 1；桶号为 ``int(linspace(0, 199, M))``。"""
    want = RONIN["event_generation_evidence"]["theta_0.1"]
    gyro, acc, rot, v0 = synthetic_window()
    stack = lie_events(gyro, acc, rot, v0, DT, 0.1, bins=200)["stack"][0]
    occupied = torch.nonzero(stack.abs().sum(0) > 0).reshape(-1)
    count = want["n_events_excluding_first_last"] + 2       # 加首尾两条伪事件
    expected = torch.linspace(0.0, 199.0, count, dtype=torch.float64).to(torch.int64).unique()
    torch.testing.assert_close(occupied, expected, atol=0, rtol=0)
    norms = torch.linalg.vector_norm(stack[6:, occupied], dim=0)
    low, high = want["polarity_bin_norm_min_max_over_nonempty"]
    assert torch.all(norms > 0) and torch.all(norms < 1.0)
    assert float(norms.min()) == pytest.approx(low, abs=1e-6)
    assert float(norms.max()) == pytest.approx(high, abs=1e-6)
    empty = torch.nonzero(stack.abs().sum(0) == 0).reshape(-1)
    assert torch.all(stack[:, empty] == 0)


def test_generate_event_stack_index_rule():
    torch.manual_seed(3)
    count = 7
    times = torch.arange(count, dtype=torch.float64)
    measurements = torch.randn(count, 6, dtype=torch.float64)
    polarities = torch.randn(count, 6, dtype=torch.float64)
    stack = generate_event_stack(times, measurements, polarities, 200)
    index = torch.linspace(0.0, 199.0, count, dtype=torch.float64).to(torch.int64)
    assert torch.all(stack[:6, index[0]] == measurements[0])
    # 分桶只看序号：把时间戳整体平移不改变结果
    shifted = generate_event_stack(times + 100.0, measurements, polarities, 200)
    torch.testing.assert_close(shifted, stack, atol=0, rtol=0)
    # 没有事件时输出全零
    assert torch.all(generate_event_stack(times[:0], measurements[:0], polarities[:0], 8) == 0)


def test_event_stack_is_yaw_equivariant():
    """同时旋转 IMU、姿态与 ``v0`` 后 ``X' = diag(R_z × 4)·X``（规格卡 §2.3，实测 5.4e-8）。"""
    gyro, acc, rot, v0 = synthetic_window()
    base = lie_events(gyro, acc, rot, v0, DT, 0.01, bins=200)["stack"][0]
    yaw = rotation_z(0.9)
    rotated = lie_events(torch.einsum("ij,btj->bti", yaw, gyro),
                         torch.einsum("ij,btj->bti", yaw, acc),
                         torch.einsum("ij,btjk->btik", yaw, rot),
                         (yaw @ v0.T).T, DT, 0.01, bins=200)["stack"][0]
    expected = torch.cat([yaw @ base[k:k + 3] for k in (0, 3, 6, 9)], dim=0)
    assert float((rotated - expected).abs().max()) < 1e-6


def test_stationary_window_stays_finite():
    """``‖w‖ = 0`` 时官方产生 NaN；IPB 取 ``u = 0``、不产生事件（规格卡 §8、§10-5）。"""
    rot = torch.eye(3, dtype=torch.float64).expand(1, 200, 3, 3).clone()
    gyro = torch.zeros(1, 200, 3, dtype=torch.float64)
    acc = torch.zeros(1, 200, 3, dtype=torch.float64)
    acc[..., 2] = 9.81
    got = lie_events(gyro, acc, rot, torch.zeros(1, 3, dtype=torch.float64), DT, 0.1, bins=200)
    assert int(got["counts"][0]) == 0
    assert torch.isfinite(got["stack"]).all()
    # 完全静止：只剩首尾两条伪事件，其余桶为 0
    assert int((got["stack"][0].abs().sum(0) != 0).sum()) == 2


# --------------------------------------------------------------------- 模型


@pytest.mark.parametrize("name,fixture,params,count", [
    ("nio_lie_events_ronin", RONIN, 4_637_570, 69),
    ("nio_lie_events_tlio", TLIO, 5_427_334, 78),
])
def test_parameters_match_official_fixture(name, fixture, params, count):
    model = build(name)
    assert model.num_params == fixture["total_params"] == params
    shapes = param_shapes(model)
    assert shape_multiset(shapes) == shape_multiset(fixture["param_shapes"])
    assert shapes == fixture["param_shapes"] and len(shapes) == count
    buffers = [list(b.shape) for _, b in model.named_buffers()]
    assert buffers == fixture["buffer_shapes"]
    assert shapes[0] == [64, 12, 7]      # 第一层卷积输入通道由 6 改为 12
    assert model.feature_length == 7


@pytest.mark.parametrize("name,shape", [("nio_lie_events_ronin", (3, 2)),
                                        ("nio_lie_events_tlio", (3, 3))])
def test_forward_shapes_and_event_channels(name, shape):
    model = build(name).eval()
    spec = model.input_spec
    extra = spec.dummy_extra(3)
    x = torch.randn(3, 6, 200)
    with torch.no_grad():
        stack = model.event_stack(x, extra)
        out = model(x, extra)
    assert stack.shape == (3, 12, 200)
    assert out["vel"].shape == shape
    if name.endswith("tlio"):
        assert out["logstd"].shape == shape
    assert len(model.event_spec.channels) == 12


def test_privileged_input_is_declared_and_required():
    for name in ("nio_lie_events_ronin", "nio_lie_events_tlio"):
        spec = build(name).input_spec
        assert spec.extra_inputs == ("orientation", "init_velocity")
        assert spec.privileged_inputs == ("init_velocity",)
    with pytest.raises(ValueError, match="privileged"):
        build("nio_lie_events_ronin", extra_inputs=["orientation"])


def test_ronin_zeroes_the_vertical_initial_velocity():
    """官方 RoNIN 的 ``v0_z`` 恒为 0（训练与测试都是），TLIO 变体保留 3 维 ``v0``。"""
    ronin = build("nio_lie_events_ronin").eval()
    extra = {"init_velocity": torch.tensor([[1.0, 2.0]])}
    torch.testing.assert_close(ronin.initial_velocity(extra, torch.float64),
                               torch.tensor([[1.0, 2.0, 0.0]], dtype=torch.float64),
                               atol=0, rtol=0)
    tlio = build("nio_lie_events_tlio").eval()
    extra = {"init_velocity": torch.tensor([[1.0, 2.0, 3.0]])}
    torch.testing.assert_close(tlio.initial_velocity(extra, torch.float64),
                               torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64),
                               atol=0, rtol=0)


def test_method_noise_only_applies_in_training_mode():
    """``v0`` 噪声与极性噪声是方法自带的正则，验证/推理必须确定（规格卡 §6）。"""
    model = build("nio_lie_events_ronin")
    extra = {"init_velocity": torch.tensor([[1.0, 2.0]], dtype=torch.float64)}
    model.eval()
    torch.testing.assert_close(model.initial_velocity(extra, torch.float64),
                               model.initial_velocity(extra, torch.float64), atol=0, rtol=0)
    model.train()
    torch.manual_seed(0)
    first = model.initial_velocity(extra, torch.float64)
    second = model.initial_velocity(extra, torch.float64)
    assert float((first - second).abs().max()) > 0
    assert float(first[0, 2]) == 0.0 and float(second[0, 2]) == 0.0   # v0_z 保持 0
    # 极性噪声：训练时改变非零项、eval 时逐位不变；桶内极性仍是单位范数量级
    stack = torch.zeros(1, 12, 200)
    stack[0, 6:, 5] = torch.tensor([0.9, 0.2, 0.0, 0.0, 0.1, 0.35])
    model.eval()
    torch.testing.assert_close(model.perturb_polarity(stack), stack, atol=0, rtol=0)
    model.train()
    noisy = model.perturb_polarity(stack)
    assert float((noisy[0, 6:, 5] - stack[0, 6:, 5]).abs().max()) > 0
    assert torch.all(noisy[0, 6:, 6] == 0)          # 空桶保持为 0
    assert float(noisy[0, 6:, 5].norm()) < 1.0


def test_event_generation_ignores_gradients_but_backbone_trains():
    model = build("nio_lie_events_ronin")
    spec = model.input_spec
    x = torch.randn(2, 6, 200, requires_grad=True)
    extra = spec.dummy_extra(2)
    model.train()
    out = model(x, extra)
    loss, _ = model.loss(out, {"target": torch.randn(2, 2), "imu": x})
    loss.backward()
    assert x.grad is None                            # 事件生成是输入表示，不对 IMU 求导
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"parameters without gradient: {missing[:5]}"
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())


def test_tlio_variant_loss_stage_switch():
    model = build("nio_lie_events_tlio")
    assert model.loss_name == "nll_detach_then_nll"
    spec = model.input_spec
    x = torch.randn(2, 6, 200)
    extra = spec.dummy_extra(2)
    target = torch.randn(2, 3)
    for epoch, expect_grad in ((8, False), (9, True)):
        model.zero_grad(set_to_none=True)
        loss, items = model.loss(model(x, extra), {"target": target, "imu": x}, epoch)
        loss.backward()
        grad = model.backbone.logstd_head.fc3.weight.grad
        assert (grad is not None and float(grad.abs().max()) > 0) is expect_grad
        assert "mse" in items
    mask = torch.zeros(2, 1)
    loss, _ = model.loss(model(x, extra), {"target": target, "imu": x, "mask": mask}, 99)
    assert float(loss.detach()) == 0.0


def test_configuration():
    ronin = get_cfg({"model": "nio_lie_events_ronin", "recipe": "official"})
    assert ronin.epochs == 120 and ronin.batch == 128 and ronin.scheduler == "plateau"
    assert ronin.frame == "gravity_world" and ronin.dims == 2
    assert ronin.target == "avg_velocity" and ronin.augment == []
    tlio = get_cfg({"model": "nio_lie_events_tlio", "recipe": "official"})
    assert tlio.epochs == 50 and tlio.batch == 1024 and tlio.grad_clip == 0.1
    assert tlio.frame == "gravity_yaw_local" and tlio.dims == 3
    assert tlio.target == "displacement"
    assert tlio.augment == get_cfg({"model": "nio_lie_events_tlio"}).augment
    assert build("nio_lie_events_ronin").threshold == 0.1
    assert build("nio_lie_events_tlio").threshold == 0.01
    with pytest.raises(ValueError, match="dims=3"):
        build("nio_lie_events_tlio", dims=2, frame="gravity_world")


@pytest.mark.slow
@pytest.mark.parametrize("name", ["nio_lie_events_ronin", "nio_lie_events_tlio"])
def test_end_to_end_train_and_val(name, synthetic_dataset, tmp_path):
    metrics = run_end_to_end(name, synthetic_dataset, tmp_path / name)
    assert metrics["ate"] >= 0.0


@pytest.mark.slow
def test_privileged_inputs_are_recorded_in_metrics(synthetic_dataset, tmp_path):
    """DESIGN §3.3：``metrics.json`` 必须记录 ``privileged_inputs``，报表才能单列。"""
    import json

    run_end_to_end("nio_lie_events_ronin", synthetic_dataset, tmp_path)
    meta = json.loads((tmp_path / "val" / "val" / "metrics.json").read_text(encoding="utf-8"))
    assert meta["privileged_inputs"] == ["init_velocity"]
