"""GNIO 移植测试：参数量（规格卡推导值）、门控头性质、Motion Bank、静态加权损失。"""

import math

import pytest

from .model_testing import (
    assert_backprop,
    assert_deterministic_eval,
    run_end_to_end,
)

torch = pytest.importorskip("torch")

from inertial_benchmark.cfg import get_cfg  # noqa: E402
from inertial_benchmark.nn import build_model  # noqa: E402
from inertial_benchmark.nn.heads import GatedHead, build_head  # noqa: E402
from inertial_benchmark.nn.losses import build_loss  # noqa: E402

BATCH = 4
# docs/algorithms/gnio.md §4.3 的推导值（论文 Table I 只报告 4.90 M）
DERIVED_PARAMS = 4_934_153
BACKBONE_PARAMS = 3_846_144
ATTENTION_PARAMS = 1_050_624
BANK_PARAMS = 64 * 512


def build(**overrides):
    return build_model(get_cfg({"model": "gnio", **overrides}))


def windows(batch: int = BATCH, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    gyro = 0.8 * torch.randn(batch, 3, 200, generator=generator)
    acc = 2.0 * torch.randn(batch, 3, 200, generator=generator)
    acc[:, 2] += 9.81
    return torch.cat([gyro, acc], dim=1)


def test_parameter_count_matches_the_spec_card():
    model = build()
    assert model.num_params == DERIVED_PARAMS
    assert sum(p.numel() for p in model.backbone.parameters()) == BACKBONE_PARAMS
    assert model.motion_bank.bank.numel() == BANK_PARAMS
    assert sum(p.numel() for p in model.motion_bank.attention.parameters()) == ATTENTION_PARAMS
    # 三个输出支路：门控头 2×(512·3+3)，不确定度支路 512·3+3
    assert sum(p.numel() for p in model.head.parameters()) == 2 * 1539
    assert sum(p.numel() for p in model.logstd_head.parameters()) == 1539
    assert model.feature_length == 7


@pytest.mark.parametrize("prototypes,expected", [(16, 4_909_577), (32, 4_917_769),
                                                 (64, 4_934_153), (128, 4_966_921)])
def test_motion_bank_size_changes_the_parameter_count(prototypes, expected):
    assert build(model_args={"prototypes": prototypes}).num_params == expected


def test_two_dimensional_output_variant():
    assert build(dims=2).num_params == 4_932_614


def test_output_shapes_and_backbone_pooling():
    model = build().eval()
    x = windows()
    with torch.no_grad():
        out = model(x)
        features = model.backbone(x)
        hidden, weights = model.features(x)
    assert features.shape == (BATCH, 512, 7)
    assert hidden.shape == (BATCH, 512) and weights.shape == (BATCH, 64)
    assert out["vel"].shape == (BATCH, 3) and out["logstd"].shape == (BATCH, 3)
    assert out["scale"].shape == (BATCH, 3) and out["gate"].shape == (BATCH, 3)
    # 注意力权重是 64 个原型上的分布
    torch.testing.assert_close(weights.sum(-1), torch.ones(BATCH), atol=1e-5, rtol=0)


def test_gated_head_algebra():
    """``s̃ > 0``、``g ∈ [−1,1]``、``|d̂| ≤ s̃``，且 ``d̂ = s̃ ⊙ g``（规格卡 §8）。"""
    model = build().eval()
    with torch.no_grad():
        out = model(windows(seed=1))
    scale, gate, vel = out["scale"], out["gate"], out["vel"]
    assert torch.all(scale > 0)
    assert torch.all(gate >= -1.0) and torch.all(gate <= 1.0)
    torch.testing.assert_close(vel, scale * gate, atol=0, rtol=0)
    assert torch.all(vel.abs() <= scale)


def test_gate_zero_is_a_hard_zupt_and_unit_gate_is_plain_softplus():
    model = build().eval()
    x = windows(seed=2)
    with torch.no_grad():
        torch.nn.init.zeros_(model.head.gate_linear.weight)
        torch.nn.init.zeros_(model.head.gate_linear.bias)
        out = model(x)
        assert torch.all(out["vel"] == 0)
        # b_g 很大、W_g = 0 → g → 1，退化为 Softplus 正值回归
        torch.nn.init.constant_(model.head.gate_linear.bias, 20.0)
        out = model(x)
        torch.testing.assert_close(out["vel"], out["scale"], atol=1e-6, rtol=0)


def test_gated_head_gradients():
    """``∂d̂_i/∂s̃_i = g_i``、``∂d̂_i/∂g_i = s̃_i``（规格卡 §8）。"""
    head = GatedHead(8, 3)
    x = torch.randn(2, 8, dtype=torch.float64)
    head.double()
    scale = head.scale_linear(x).detach().requires_grad_(True)
    gate = head.gate_linear(x).detach().requires_grad_(True)
    s = torch.nn.functional.softplus(scale)
    g = torch.tanh(gate)
    (s * g).sum().backward()
    torch.testing.assert_close(scale.grad, g * torch.sigmoid(scale), atol=1e-12, rtol=0)
    torch.testing.assert_close(gate.grad, s * (1 - g**2), atol=1e-12, rtol=0)


@pytest.mark.parametrize("scale_fn", ["softplus", "pos_elu", "abs", "exp", "linear"])
@pytest.mark.parametrize("gate_fn", ["tanh", "sigmoid"])
def test_reusable_gated_head_variants(scale_fn, gate_fn):
    head = build_head("gated", 512, 3, scale_fn=scale_fn, gate_fn=gate_fn)
    assert sum(p.numel() for p in head.parameters()) == 3078   # 普通线性头为 1539
    out = head(torch.randn(BATCH, 512))
    assert out["vel"].shape == (BATCH, 3)
    if scale_fn != "linear":
        assert torch.all(out["scale"] > 0) and torch.all(out["vel"].abs() <= out["scale"] + 1e-6)
        assert head.bounded
    else:
        assert not head.bounded
    if gate_fn == "sigmoid":
        assert torch.all(out["gate"] > 0)     # Sigmoid 无法翻转方向（论文 Table IV）
    with pytest.raises(ValueError, match="scale_fn"):
        build_head("gated", 8, 2, scale_fn="nope")


def test_motion_bank_is_trained_and_can_be_switched_off():
    model = build()
    x = windows(seed=3)
    model(x)["vel"].sum().backward()
    assert model.motion_bank.bank.grad is not None
    assert torch.isfinite(model.motion_bank.bank.grad).all()
    plain = build(model_args={"motion_bank": False})
    assert plain.num_params == DERIVED_PARAMS - BANK_PARAMS - ATTENTION_PARAMS
    assert "aux" not in plain.eval()(x)


def test_motion_bank_residual_and_one_hot_attention():
    """``h = f + c``；注意力权重接近 one-hot 时 ``c`` 等于该原型的 V 投影。"""
    model = build().double().eval()
    bank = model.motion_bank
    with torch.no_grad():
        # Q/K/V 投影取恒等、无偏置，注意力权重就只由 ``f·M_n`` 决定
        bank.attention.in_proj_weight.copy_(torch.eye(512, dtype=torch.float64).repeat(3, 1))
        bank.attention.in_proj_bias.zero_()
        bank.bank.zero_()
        bank.bank[7] = 1e3          # 让第 7 个原型与查询的内积远大于其余
        features = torch.ones(2, 512, dtype=torch.float64)
        hidden, weights = bank(features)
        value = bank.attention.out_proj(bank.bank[7])
    assert torch.all(weights[:, 7] > 0.999)
    torch.testing.assert_close(hidden[0] - features[0], value, atol=1e-6, rtol=1e-6)


def test_static_weighted_loss():
    """式 13 是静态加权和：权重比 1e6，且不随 epoch 切换（规格卡 §10-1）。"""
    loss_fn = build_loss("gnio_static_weighted")
    assert loss_fn.mse_weight == 1.0e2 and loss_fn.nll_weight == 1.0e-4
    assert loss_fn.mse_weight / loss_fn.nll_weight == pytest.approx(1.0e6)
    target = torch.randn(BATCH, 3)
    perfect = {"vel": target.clone(), "logstd": torch.zeros(BATCH, 3)}
    value, items = loss_fn(perfect, target, epoch=0)
    assert float(value) == pytest.approx(0.0, abs=1e-9)     # Σ = I 时 log det Σ = 0
    assert set(items) == {"sq", "nll", "mse"}
    # 不同 epoch 的损失完全相同（没有阶段切换）
    out = {"vel": torch.randn(BATCH, 3), "logstd": torch.randn(BATCH, 3)}
    early, _ = loss_fn(out, target, epoch=0)
    late, _ = loss_fn(out, target, epoch=199)
    torch.testing.assert_close(early, late, atol=0, rtol=0)
    # λ_NLL = 0 时梯度只来自 MSE 项（logstd 不再收梯度）
    only_mse = build_loss("gnio_static_weighted", nll_weight=0.0)
    logstd = torch.zeros(BATCH, 3, requires_grad=True)
    value, _ = only_mse({"vel": torch.zeros(BATCH, 3, requires_grad=True), "logstd": logstd},
                        target, epoch=0)
    value.backward()
    assert logstd.grad is None or float(logstd.grad.abs().max()) == 0.0


def test_loss_masks_invalid_outputs():
    loss_fn = build_loss("gnio_static_weighted")
    target = torch.randn(BATCH, 3)
    out = {"vel": torch.randn(BATCH, 3), "logstd": torch.zeros(BATCH, 3)}
    mask = torch.zeros(BATCH, 1)
    value, _ = loss_fn(out, target, 0, mask)
    assert float(value) == 0.0 and value.requires_grad is False
    mask[0] = 1.0
    part, _ = loss_fn(out, target, 0, mask)
    single, _ = loss_fn({k: v[:1] for k, v in out.items()}, target[:1], 0)
    torch.testing.assert_close(part, single, atol=1e-6, rtol=1e-6)


def test_displacement_to_velocity_conversion():
    """``vel · 0.995 s == d̂``，``logstd_vel == u − log(0.995)``（规格卡 §8）。"""
    cfg = get_cfg({"model": "gnio"})
    spec = build().input_spec
    assert cfg.target == "displacement" and cfg.window == 200 and cfg.dims == 3
    span = (spec.window - 1) * spec.dt
    assert span == pytest.approx(0.995)
    assert float(spec.output_scales[0]) == pytest.approx(1.0 / 0.995)
    assert math.log(1.0 / span) == pytest.approx(-math.log(0.995))


def test_forward_backward_and_determinism():
    model = build()
    x = windows(seed=4)
    assert model.loss_name == "gnio_static_weighted"
    assert_deterministic_eval(model, x)
    assert_backprop(build(), x, torch.randn(BATCH, 3))


def test_channel_order_option():
    model = build(model_args={"channel_order": "acc_gyro"}).eval()
    x = windows(seed=5)
    swapped = torch.cat([x[:, 3:6], x[:, 0:3]], dim=1)
    torch.testing.assert_close(model.order_channels(x), swapped, atol=0, rtol=0)
    assert model.num_params == DERIVED_PARAMS
    with pytest.raises(ValueError, match="channel_order"):
        build(model_args={"channel_order": "nope"})


def test_configuration():
    cfg = get_cfg({"model": "gnio", "recipe": "official"})
    assert cfg.epochs == 200 and cfg.batch == 1024 and cfg.lr == 1e-4
    assert cfg.warmup_epochs == 5 and cfg.scheduler == "cosine" and cfg.stride == 20
    assert cfg.optimizer == "adam" and cfg.grad_clip == 0.1
    official = get_cfg({"model": "gnio", "recipe": "official"}).augment
    assert get_cfg({"model": "gnio"}).augment == official   # 增强两个配方一致
    with pytest.raises(ValueError, match="one displacement per window"):
        build(target="frame_velocity")


@pytest.mark.slow
def test_end_to_end_train_and_val(synthetic_dataset, tmp_path):
    run_end_to_end("gnio", synthetic_dataset, tmp_path)
