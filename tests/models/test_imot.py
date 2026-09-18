"""iMoT 移植测试：参考规格的参数量分解、PSD 性质、APE、DSM 凸组合与共享 MLP。"""

import math

import pytest

from .model_testing import assert_backprop, assert_deterministic_eval, run_end_to_end

torch = pytest.importorskip("torch")

from inertial_benchmark.cfg import get_cfg  # noqa: E402
from inertial_benchmark.models.imot.model import (  # noqa: E402
    AdaptivePositionalEncoding,
    ProgressiveSeriesDecoupler,
    moving_average,
)
from inertial_benchmark.nn import build_model  # noqa: E402
from inertial_benchmark.nn.heads import sinusoidal_encoding  # noqa: E402
from inertial_benchmark.nn.losses import build_loss  # noqa: E402

BATCH = 3
# docs/algorithms/imot.md §4 的推导值（论文 Table 3 报告 14.49 M，见卡 §10-1）
DERIVED_PARAMS = 7_137_482


def build(**overrides):
    return build_model(get_cfg({"model": "imot", **overrides}))


def windows(batch: int = BATCH, window: int = 200, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    gyro = 0.8 * torch.randn(batch, 3, window, generator=generator)
    acc = 2.0 * torch.randn(batch, 3, window, generator=generator)
    acc[:, 2] += 9.81
    return torch.cat([gyro, acc], dim=1)


def test_parameter_decomposition_matches_the_spec_card():
    """卡 §4 的分解表逐项核对（总计 7,137,482）。"""
    model = build()
    assert model.num_params == DERIVED_PARAMS
    encoder = model.encoder[0]
    assert sum(p.numel() for p in encoder.positional.parameters()) == 160_800   # APE 两个 MLP
    assert sum(p.numel() for p in encoder.attention.parameters()) == 160_800    # MHA
    assert sum(p.numel() for p in encoder.asc1.parameters()) == 681_000         # ASC
    assert sum(p.numel() for p in encoder.asc2.parameters()) == 681_000
    assert sum(p.numel() for p in encoder.feedforward.parameters()) == 321_000  # FFN 4T
    assert sum(p.numel() for p in encoder.norm1.parameters()) \
        + sum(p.numel() for p in encoder.norm2.parameters()) == 800
    assert sum(p.numel() for p in encoder.parameters()) == 2_005_400
    assert sum(p.numel() for p in model.encoder.parameters()) == 4_010_800
    assert model.particles.numel() == 256
    assert sum(p.numel() for p in model.position_mlp.parameters()) == 80_400    # MLP_pos
    assert sum(p.numel() for p in model.refine_mlp.parameters()) == 40_602      # MLP_Delta
    decoder = model.decoder[0]
    assert sum(p.numel() for p in decoder.attention.parameters()) == 160_800
    assert sum(p.numel() for p in decoder.scale_mlp.parameters()) == 80_400     # MLP_s
    assert sum(p.numel() for p in decoder.cross_acc.parameters()) \
        + sum(p.numel() for p in decoder.cross_gyro.parameters()) == 802_400
    assert sum(p.numel() for p in decoder.fuse.parameters()) == 120_400         # MLP_c
    assert sum(p.numel() for p in decoder.feedforward.parameters()) == 321_000
    assert sum(p.numel() for p in decoder.parameters()) == 1_486_200
    assert sum(p.numel() for p in model.decoder.parameters()) == 2_972_400
    assert sum(p.numel() for p in model.scoring.parameters()) == 33_024         # DSM


def test_reference_variants_from_the_card():
    """T=100 必须改用 4 头（1,815,382）；FFN 宽度 2048 时为 9,139,274（卡 §4）。"""
    assert build(window=100, model_args={"heads": 4}).num_params == 1_815_382
    assert build(model_args={"feedforward": 2048}).num_params == 9_139_274
    with pytest.raises(ValueError, match="must divide the token length"):
        build(window=100)


def test_shapes_and_token_layout():
    model = build().eval()
    x = windows()
    with torch.no_grad():
        tokens = model.tokenize(x)
        out = model(x)
    assert tokens.shape == (BATCH, 18, 200)
    assert out["vel"].shape == (BATCH, 2)
    assert out["aux"]["particles"].shape == (BATCH, 128, 2)
    assert out["aux"]["scores"].shape == (BATCH, 2, 128)
    assert out["aux"]["attention"].shape == (BATCH, 8, 128, 9)   # 交叉注意力图


def test_channel_permutation_into_the_paper_order():
    """raw 槽 0:3 是 IPB 输入的加计（3:6），raw 槽 9:12 是陀螺（0:3）（规格卡 §8）。"""
    model = build().eval()
    x = windows(seed=1)
    with torch.no_grad():
        tokens = model.tokenize(x)
    torch.testing.assert_close(tokens[:, 0:3], x[:, 3:6], atol=0, rtol=0)
    torch.testing.assert_close(tokens[:, 9:12], x[:, 0:3], atol=0, rtol=0)


def test_progressive_series_decoupler_properties():
    """PSD 无参数：常值序列的趋势等于该常值、seasonal ≡ 0，且 ``A_t + A_s == A``（规格卡 §8）。"""
    psd = ProgressiveSeriesDecoupler()
    assert sum(p.numel() for p in psd.parameters()) == 0
    constant = torch.full((2, 6, 200), 1.7, dtype=torch.float64)
    trend, seasonal = psd.decompose(constant)
    torch.testing.assert_close(trend, constant, atol=1e-12, rtol=0)
    torch.testing.assert_close(seasonal, torch.zeros_like(constant), atol=1e-12, rtol=0)
    torch.manual_seed(0)
    raw = torch.randn(2, 6, 200, dtype=torch.float64)
    trend, seasonal = psd.decompose(raw)
    assert trend.shape == raw.shape                     # 输出长度 = T
    torch.testing.assert_close(trend + seasonal, raw, atol=1e-12, rtol=0)
    # 线性斜坡在远离边界处 seasonal ≈ 0（居中对称滑动平均）
    ramp = (torch.arange(200, dtype=torch.float64) * 0.01).expand(1, 6, 200).contiguous()
    _, ramp_seasonal = psd.decompose(ramp)
    assert float(ramp_seasonal[..., 20:-20].abs().max()) < 1e-12
    # 重组的 18 个 token：raw / trend / seasonal 各 3 轴，acc 在前
    tokens = psd(raw)
    assert tokens.shape == (2, 18, 200)
    torch.testing.assert_close(psd.raw_slots(tokens), raw, atol=0, rtol=0)
    torch.testing.assert_close(tokens[:, 3:6], trend[:, 0:3], atol=0, rtol=0)
    torch.testing.assert_close(tokens[:, 6:9], seasonal[:, 0:3], atol=0, rtol=0)


def test_moving_average_keeps_the_length_and_is_centred():
    x = torch.randn(1, 2, 51, dtype=torch.float64)
    for kernel in (3, 9):
        assert moving_average(x, kernel).shape == x.shape
    # 居中：对称序列的滑动平均仍然对称
    symmetric = torch.cat([x, x.flip(-1)], dim=-1)
    averaged = moving_average(symmetric, 9)
    torch.testing.assert_close(averaged, averaged.flip(-1), atol=1e-12, rtol=0)


def test_adaptive_positional_encoding_reduces_to_the_tiled_sinusoid():
    """把 ``MLP_a`` 的输出置为全 1 时 ``Ẽ_a`` 等于平铺后的正弦编码（规格卡 §8）。"""
    ape = AdaptivePositionalEncoding(200).double().eval()
    with torch.no_grad():
        for mlp in (ape.acc_mlp, ape.gyro_mlp):
            torch.nn.init.zeros_(mlp[0].weight)
            torch.nn.init.zeros_(mlp[0].bias)
            torch.nn.init.zeros_(mlp[2].weight)
            torch.nn.init.ones_(mlp[2].bias)
        encoding = ape(torch.randn(2, 18, 200, dtype=torch.float64))
    torch.testing.assert_close(encoding[:, :9], ape.base.expand(2, 9, 200), atol=1e-12, rtol=0)
    torch.testing.assert_close(encoding[:, 9:], ape.base.expand(2, 9, 200), atol=1e-12, rtol=0)
    # 基础编码就是按轴序号（0–2）的正弦编码沿 token 维平铺 3 份（buffer 以模型 dtype 存储）
    reference = sinusoidal_encoding(torch.arange(3, dtype=torch.float64), 200).repeat(3, 1)
    torch.testing.assert_close(ape.base, reference, atol=1e-6, rtol=0)


def test_dynamic_scoring_is_a_convex_combination():
    """``S`` 沿粒子维求和为 1；``v_m`` 的每个分量落在该轴粒子取值的 ``[min, max]`` 内（卡 §8）。"""
    model = build().eval()
    x = windows(seed=2)
    with torch.no_grad():
        out = model(x)
    scores, particles = out["aux"]["scores"], out["aux"]["particles"]
    torch.testing.assert_close(scores.sum(-1), torch.ones(BATCH, 2), atol=1e-5, rtol=0)
    assert torch.all(scores >= 0)
    lower, upper = particles.amin(dim=1), particles.amax(dim=1)
    assert torch.all(out["vel"] >= lower - 1e-5) and torch.all(out["vel"] <= upper + 1e-5)
    torch.testing.assert_close(out["vel"], (scores * particles.transpose(1, 2)).sum(-1),
                               atol=1e-6, rtol=0)


def test_position_and_refinement_mlps_are_shared_across_layers():
    """``MLP_pos`` 与 ``MLP_Δ`` 在各解码层之间是同一对象（参数只计一次，规格卡 §8）。"""
    model = build(model_args={"decoder_layers": 3})
    assert sum(p.numel() for p in model.position_mlp.parameters()) == 80_400
    assert sum(p.numel() for p in model.refine_mlp.parameters()) == 40_602
    # 每层独立的 MLP_s 随层数线性增长，共享的两个不增长
    expected = DERIVED_PARAMS + 1_486_200
    assert model.num_params == expected
    assert model.particles.requires_grad


def test_particles_are_learnable_and_initialised_in_the_unit_range():
    model = build()
    assert float(model.particles.detach().abs().max()) <= 1.0
    x = windows(seed=3)
    model(x)["vel"].sum().backward()
    assert model.particles.grad is not None
    assert torch.isfinite(model.particles.grad).all()


def test_velocity_loss_only():
    """最终模型只用 ``J_vel``（IPB 的 ``mse_sum``）：``v_m == v_GT`` 时为 0（规格卡 §8）。"""
    model = build()
    assert model.loss_name == "mse_sum"
    loss_fn = build_loss("mse_sum")
    target = torch.randn(BATCH, 2)
    value, items = loss_fn({"vel": target.clone()}, target, 0)
    assert float(value) == pytest.approx(0.0, abs=1e-12) and "mse" in items
    mask = torch.zeros(BATCH, 1)
    value, _ = loss_fn({"vel": torch.randn(BATCH, 2)}, target, 0, mask)
    assert float(value) == 0.0


def test_forward_backward_and_determinism():
    model = build()
    x = windows(seed=4)
    assert_deterministic_eval(model, x)
    assert_backprop(build(), x, torch.randn(BATCH, 2))


def test_configuration():
    cfg = get_cfg({"model": "imot", "recipe": "official"})
    assert cfg.epochs == 100 and cfg.batch == 128 and cfg.lr == 1e-4
    assert cfg.optimizer == "adam" and cfg.scheduler == "none"
    assert cfg.frame == "gravity_world" and cfg.target == "avg_velocity" and cfg.dims == 2
    assert cfg.augment == [] == get_cfg({"model": "imot"}).augment
    model = build()
    assert model.particle_scale == pytest.approx(2.0 * math.pi)
    with pytest.raises(ValueError, match="one velocity segment per window"):
        build(target="frame_velocity")


@pytest.mark.slow
def test_end_to_end_train_and_val(synthetic_dataset, tmp_path):
    run_end_to_end("imot", synthetic_dataset, tmp_path)
