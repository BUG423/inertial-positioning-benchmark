"""RIO 移植测试：GroupNorm 骨干、旋转算子与负余弦、联合损失门控、A-TTT 状态机与端到端。"""

import math

import pytest

from .model_testing import (
    assert_backprop,
    assert_deterministic_eval,
    load_fixture,
    param_shapes,
    run_end_to_end,
    shape_multiset,
)

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from inertial_benchmark.cfg import get_cfg  # noqa: E402
from inertial_benchmark.models.rio import (  # noqa: E402
    DEFAULT_ANGLES_DEG,
    AdaptiveTTT,
    RIOJointLoss,
    negative_cosine,
    rotate_imu_z,
    rotate_vector_z,
)
from inertial_benchmark.nn import build_model  # noqa: E402

# RIO 没有官方代码与夹具：骨干与 ronin_resnet18 同构（BN → GN，参数量不变），以该夹具锁定
FIXTURE = load_fixture("ronin_resnet18")
# A-TTT 测试用的小模型：关掉 dropout，使梯度步与预测完全可复现
TINY = {"model_args": {"group_sizes": [1, 1], "base_plane": 8, "fc_dim": 16, "trans_planes": 4,
                       "dropout": 0.0},
        "window": 100}


def build(**overrides):
    return build_model(get_cfg({"model": "rio", **overrides}))


def test_parameters_match_ronin_resnet18():
    model = build()
    assert model.num_params == FIXTURE["total_params"] == 4_634_882
    shapes = param_shapes(model)
    assert shape_multiset(shapes) == shape_multiset(FIXTURE["param_shapes"])
    assert shapes == FIXTURE["param_shapes"]      # GN 与 BN 的仿射参数形状相同
    assert build(dims=3).num_params == 4_635_395


def test_groupnorm_replaces_every_batchnorm():
    model = build()
    assert model.num_groupnorm == 21
    assert sum(isinstance(m, nn.GroupNorm) for m in model.modules()) == 21
    assert sum(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules()) == 0
    assert list(model.buffers()) == []            # GN 没有 running statistics
    assert all(m.num_groups == 32 for m in model.modules() if isinstance(m, nn.GroupNorm))
    # batch=1 的训练模式前向（BN 版会报错），A-TTT 需要这一性质
    model.train()
    assert model(torch.randn(1, 6, 200))["vel"].shape == (1, 2)


def test_forward_modes_and_gradients():
    torch.manual_seed(0)
    model = build()
    x = torch.randn(4, 6, 200)
    model.train()
    out = model(x)
    assert set(out) == {"vel", "vel_conj", "phi"}
    assert out["vel"].shape == (4, 2) and out["vel_conj"].shape == (4, 2)
    assert out["phi"].shape == (4,)
    assert torch.all((out["phi"] > 0) & (out["phi"] <= 2 * math.pi))
    model.eval()
    assert set(model(x)) == {"vel"}
    assert_deterministic_eval(model, x)
    assert_backprop(build(), x, torch.randn(4, 2))
    assert model.loss_name == "rio_joint"
    with pytest.raises(ValueError, match="gravity-aligned"):
        build(frame="body", dims=3)


def test_rotation_operator():
    x = torch.randn(3, 6, 50)
    a = torch.tensor([0.3, -1.2, 2.0])
    b = torch.tensor([1.1, 0.4, -0.7])
    torch.testing.assert_close(rotate_imu_z(rotate_imu_z(x, a), b), rotate_imu_z(x, a + b),
                               atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(rotate_imu_z(x, torch.full((3,), 2 * math.pi)), x,
                               atol=1e-5, rtol=1e-5)
    rotated = rotate_imu_z(x, a)
    torch.testing.assert_close(rotated[:, 2], x[:, 2])        # 陀螺 z 不变
    torch.testing.assert_close(rotated[:, 5], x[:, 5])        # 加计 z 不变
    for sl in (slice(0, 2), slice(3, 5)):
        torch.testing.assert_close(rotated[:, sl].norm(dim=1), x[:, sl].norm(dim=1),
                                   atol=1e-5, rtol=1e-5)
    v = torch.randn(3, 2)
    torch.testing.assert_close(rotate_vector_z(v, torch.zeros(3)), v)
    v3 = torch.randn(3, 3)
    torch.testing.assert_close(rotate_vector_z(v3, a)[:, 2], v3[:, 2])


def test_negative_cosine():
    v = torch.tensor([[1.0, 2.0], [-3.0, 0.5]])
    assert torch.allclose(negative_cosine(v, v), torch.full((2,), -1.0), atol=1e-6)
    assert torch.allclose(negative_cosine(v, -v), torch.ones(2), atol=1e-6)
    perp = rotate_vector_z(v, torch.full((2,), math.pi / 2))
    assert torch.allclose(negative_cosine(v, perp), torch.zeros(2), atol=1e-6)
    zero = torch.zeros(1, 2)
    assert torch.isfinite(negative_cosine(zero, v[:1])).all()


def test_joint_loss_gate_and_weights():
    loss_fn = RIOJointLoss()
    target = torch.tensor([[1.0, 0.0], [0.1, 0.0]])
    vel = torch.tensor([[1.0, 0.0], [0.1, 0.0]], requires_grad=True)
    phi = torch.tensor([0.7, 0.7])
    out = {"vel": vel, "phi": phi, "vel_conj": rotate_vector_z(vel, phi).detach()}
    loss, items = loss_fn(out, target)
    # 完全等变 → 未被门控的样本辅助项为 −1；第二个样本 ‖v̂‖ = 0.1 ≤ 0.5 被门控为 0
    assert items["ssl"].item() == pytest.approx(-0.5)
    assert items["vel_loss"].item() == pytest.approx(0.0)
    assert loss.item() == pytest.approx(-0.5)
    # 权重 1:1 与 ssl_weight 生效
    weighted, _ = RIOJointLoss(ssl_weight=2.0)(out, target)
    assert weighted.item() == pytest.approx(-1.0)
    # 门控样本没有梯度
    slow = torch.tensor([[0.1, 0.0]], requires_grad=True)
    out_slow = {"vel": slow, "phi": torch.tensor([0.4]),
                "vel_conj": torch.tensor([[0.3, 0.2]])}
    ssl = RIOJointLoss().ssl(out_slow, torch.zeros(1, 2))
    assert ssl.item() == 0.0
    ssl.backward()
    assert torch.all(slow.grad == 0)
    # 缺少共轭分支（验证/推理）时只有速度项
    only_vel, items = loss_fn({"vel": torch.ones(2, 2)}, torch.zeros(2, 2))
    assert only_vel.item() == pytest.approx(2.0) and "ssl" not in items
    with pytest.raises(ValueError, match="gate_on"):
        RIOJointLoss(gate_on="both")


def test_ssl_loss_is_minimal_for_equivariant_outputs():
    """精确等变的输出使辅助项取最小值 −1；对旋转不变（常值）的输出期望值约为 0。"""
    torch.manual_seed(0)
    loss_fn = RIOJointLoss(speed_gate=-1.0)   # 关掉门控，只看辅助项本身
    vel = torch.randn(256, 2) * 2.0
    target = vel.clone()
    phi = torch.rand(256) * 2 * math.pi
    equivariant = {"vel": vel, "phi": phi, "vel_conj": rotate_vector_z(vel, phi)}
    assert loss_fn.ssl(equivariant, target).item() == pytest.approx(-1.0, abs=1e-5)
    invariant = {"vel": vel, "phi": phi, "vel_conj": vel}
    assert abs(loss_fn.ssl(invariant, target).item()) < 0.1
    # φ = 0 时共轭输入与原输入相同（eval 模式下模型输出也相同）→ 辅助项为 −1
    model = build(**TINY).eval()
    x = torch.randn(8, 6, 100)
    out = model(x, angle=torch.zeros(8))
    torch.testing.assert_close(out["vel"], out["vel_conj"])
    # 随机初始化的小模型输出接近 0，负余弦分母里的 eps 会占主导；先归一化到单位范数
    scale = 1.0 / out["vel"].norm(dim=-1, keepdim=True).clamp_min(1e-12)
    unit = {"vel": out["vel"] * scale, "vel_conj": out["vel_conj"] * scale, "phi": out["phi"]}
    assert loss_fn.ssl(unit, unit["vel"]).item() == pytest.approx(-1.0, abs=1e-5)


def test_conjugate_pass_equals_two_separate_forwards():
    """GroupNorm 使 2B 拼批与分别前向等价（eval 模式下逐元素一致）。"""
    model = build(**TINY).eval()
    x = torch.randn(5, 6, 100)
    phi = torch.rand(5) * 2 * math.pi
    with torch.no_grad():
        vel, vel_conj = model.conjugate_pass(x, phi)
        torch.testing.assert_close(vel, model.predict(x), atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(vel_conj, model.predict(rotate_imu_z(x, phi)),
                                   atol=1e-6, rtol=1e-5)


class _MockTTT(AdaptiveTTT):
    """用给定方差序列替换集成方差，并统计优化器步数。"""

    def __init__(self, model, variances, **kwargs):
        self.variances = list(variances)
        self.calls = 0
        self.step_count = 0
        super().__init__(model, [], **kwargs)

    def ensemble_variance(self, imu):
        value = self.variances[min(self.calls, len(self.variances) - 1)]
        self.calls += 1
        return torch.as_tensor(value, dtype=torch.float32)

    def update(self, imu):
        before = self.step_count
        super().update(imu)
        assert self.step_count - before == self.steps

    def reset(self):
        super().reset()
        original = self.optimizer.step

        def counting_step(*args, **kwargs):
            self.step_count += 1
            return original(*args, **kwargs)

        self.optimizer.step = counting_step


def test_adaptive_ttt_state_machine():
    torch.manual_seed(0)
    model = build(**TINY)
    variances = [
        torch.tensor([1e-6, 1.0]),      # min < 1e-4 → 恢复
        torch.tensor([0.01, 0.02]),     # mean < 0.04 → 保持
        torch.tensor([1.0, 2.0]),       # → 更新 5 次
    ]
    model.set_loss("rio_joint", speed_gate=-1.0)   # 关掉门控，保证更新步真的改变参数
    ttt = _MockTTT(model, variances, lr=1e-3, steps=5, batch=2)
    assert [round(math.degrees(a)) for a in ttt.angles] == [72, 144, 216, 288]
    assert tuple(DEFAULT_ANGLES_DEG) == (72.0, 144.0, 216.0, 288.0)
    windows = torch.randn(6, 6, 100)
    initial = {k: v.clone() for k, v in model.state_dict().items()}

    out = ttt.step(windows[:2])           # 恢复
    assert out.shape == (2, 2) and ttt.actions == ["reset"]
    assert all(torch.equal(v, initial[k]) for k, v in model.state_dict().items())

    ttt.step(windows[2:4])                # 保持
    assert ttt.actions[-1] == "hold"
    assert all(torch.equal(v, initial[k]) for k, v in model.state_dict().items())

    ttt.step(windows[4:6])                # 更新：恰好 5 次 Adam step
    assert ttt.actions[-1] == "update" and ttt.step_count == 5
    assert any(not torch.equal(v, initial[k]) for k, v in model.state_dict().items())

    # 恢复分支把参数完全还原并重置优化器
    ttt.variances = [torch.tensor([1e-9, 1e-9])]
    ttt.calls = 0
    ttt.step(windows[:2])
    assert ttt.actions[-1] == "reset"
    assert all(torch.equal(v, initial[k]) for k, v in model.state_dict().items())


def test_identical_ensemble_always_resets():
    torch.manual_seed(0)
    model = build(**TINY)
    member = build(**TINY)
    member.load_state_dict(model.state_dict())
    ttt = AdaptiveTTT(model, [member, member], lr=1e-3, steps=2, batch=4)
    windows = torch.randn(8, 6, 100)
    variance = ttt.ensemble_variance(windows)
    assert torch.all(variance == 0)
    out = ttt.run(windows, batch=4)
    assert out.shape == (8, 2) and ttt.actions == ["reset", "reset"]


def test_naive_ttt_always_updates_and_is_causal():
    torch.manual_seed(0)
    model = build(**TINY)
    model.set_loss("rio_joint", speed_gate=-1.0)
    ttt = AdaptiveTTT(model, [], lr=1e-3, steps=1, batch=4, mode="naive")
    windows = torch.randn(12, 6, 100)
    first = ttt.run(windows, batch=4)
    assert ttt.actions == ["update"] * 3
    # 因果性：打乱后续批不改变前面批的输出
    shuffled = windows.clone()
    shuffled[8:] = shuffled[8:].flip(0)
    replica = build(**TINY)
    replica.set_loss("rio_joint", speed_gate=-1.0)
    second = AdaptiveTTT(replica, [], lr=1e-3, steps=1, batch=4, mode="naive")
    second.model.load_state_dict(model.state_dict())
    second.initial_state = {k: v.clone() for k, v in ttt.initial_state.items()}
    out = second.run(shuffled, batch=4)
    torch.testing.assert_close(out[:8], first[:8], atol=1e-6, rtol=1e-5)
    with pytest.raises(ValueError, match="mode must be"):
        AdaptiveTTT(model, [], mode="fancy")


@pytest.mark.slow
def test_end_to_end_train_and_val(synthetic_dataset, tmp_path):
    run_end_to_end("rio", synthetic_dataset, tmp_path)
