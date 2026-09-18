"""IONext 移植测试：解析参数量、ADE 块的融合权重与门控、退化检查与通道置换等价。"""

import pytest

from .model_testing import assert_backprop, assert_deterministic_eval, run_end_to_end

torch = pytest.importorskip("torch")

from inertial_benchmark.cfg import get_cfg  # noqa: E402
from inertial_benchmark.models.ionext.model import (  # noqa: E402
    AdaptiveDynamicModule,
    AdaptiveGatingUnit,
    ADEBlock,
)
from inertial_benchmark.nn import build_model  # noqa: E402

BATCH = 4
# docs/algorithms/ionext.md §4.1 的解析计数（论文 v1 表 I 报告 1.1e7）
DERIVED_PARAMS = 10_641_506


def build(**overrides):
    return build_model(get_cfg({"model": "ionext", **overrides}))


def windows(batch: int = BATCH, window: int = 200, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    gyro = 0.8 * torch.randn(batch, 3, window, generator=generator)
    acc = 2.0 * torch.randn(batch, 3, window, generator=generator)
    acc[:, 2] += 9.81
    return torch.cat([gyro, acc], dim=1)


def test_parameter_count_matches_the_spec_card():
    model = build()
    assert model.num_params == DERIVED_PARAMS
    assert build(dims=3).num_params == 10_642_275
    assert sum(p.numel() for p in model.stem.parameters()) == 2_592
    stage_totals = [89_664, 382_464, 4_210_176, 5_953_536]
    for index, expected in enumerate(stage_totals):
        got = sum(p.numel() for p in model.stages[index].parameters())
        if index:
            got += sum(p.numel() for p in model.downsamples[index - 1].parameters())
        assert got == expected
    assert sum(p.numel() for p in model.norm.parameters()) \
        + sum(p.numel() for p in model.head.parameters()) == 3_074


@pytest.mark.parametrize("channels", [96, 192, 384, 768])
def test_ade_block_parameter_formula(channels):
    """单块参数量为 ``4.5·C² + 35·C``（ADM ``2.5C² + 26C``、AGU ``2C² + 5C``、两个 BN ``4C``）。"""
    block = ADEBlock(channels)
    assert sum(p.numel() for p in block.parameters()) == 4.5 * channels**2 + 35 * channels
    assert sum(p.numel() for p in block.adm.parameters()) == 2.5 * channels**2 + 26 * channels
    assert sum(p.numel() for p in block.agu.parameters()) == 2 * channels**2 + 5 * channels


@pytest.mark.parametrize("window,lengths", [(200, [50, 25, 12, 6]), (100, [25, 12, 6, 3])])
def test_stage_lengths_and_shapes(window, lengths):
    model = build(window=window).eval()
    assert model.stage_lengths == lengths and model.feature_length == lengths[-1]
    x = windows(2, window)
    with torch.no_grad():
        stem = model.stem(x)
        out = model(x)
    assert stem.shape == (2, 96, lengths[0])
    assert out["vel"].shape == (2, 2)
    with torch.no_grad():
        assert model.features(x).shape == (2, 768, lengths[-1])


def test_fusion_weights_are_a_branch_distribution():
    """``ω`` 形状 ``(B, 3, h, 1)``、非负、沿分支轴求和为 1（规格卡 §8）。"""
    torch.manual_seed(0)
    adm = AdaptiveDynamicModule(96)
    x = torch.randn(3, 48, 50)
    for wing in (0, 1):
        weights = adm.fusion_weights(wing, x)
        assert weights.shape == (3, 3, 48, 1)
        assert torch.all(weights >= 0)
        torch.testing.assert_close(weights.sum(dim=1), torch.ones(3, 48, 1), atol=1e-6, rtol=0)
    # channel 轴的 softmax 变体：沿 3h 个通道求和为 1
    other = AdaptiveDynamicModule(96, softmax_axis="channel")
    weights = other.fusion_weights(0, x)
    torch.testing.assert_close(weights.reshape(3, -1, 1).sum(dim=1), torch.ones(3, 1),
                               atol=1e-6, rtol=0)
    with pytest.raises(ValueError, match="softmax_axis"):
        AdaptiveDynamicModule(96, softmax_axis="nope")


def test_gate_is_a_channel_wise_sigmoid():
    """``ξ`` 形状 ``(B, C, 1)``、取值在 ``(0, 1)``（v2 的通道轴门控，规格卡 §8）。"""
    torch.manual_seed(1)
    agu = AdaptiveGatingUnit(64)
    x = torch.randn(5, 64, 25)
    gate = agu.gating(x)
    assert gate.shape == (5, 64, 1)
    assert torch.all(gate > 0) and torch.all(gate < 1)


def test_depthwise_convolutions_use_the_declared_kernels():
    adm = AdaptiveDynamicModule(96)
    assert adm.kernels == ((1, 3, 11), (1, 5, 17))
    for wing, kernels in enumerate(adm.kernels):
        for conv, kernel in zip(adm.wings[wing], kernels):
            assert conv.groups == conv.in_channels == 48
            assert tuple(conv.weight.shape) == (48, 1, kernel)
    agu = AdaptiveGatingUnit(96)
    assert agu.value.groups == agu.value.in_channels == 96
    assert tuple(agu.value.weight.shape) == (96, 1, 3)


def test_degenerate_fusion_weights_average_the_branches():
    """``W1`` 置零 → ``ω ≡ 1/3``，ADM 等于 ``W2(concat(mean_i Y₀, mean_i Y₁))``（规格卡 §8）。"""
    torch.manual_seed(2)
    adm = AdaptiveDynamicModule(64).double().eval()
    for weights in adm.weights:
        torch.nn.init.zeros_(weights.weight)
        torch.nn.init.zeros_(weights.bias)
    x = torch.randn(2, 64, 30, dtype=torch.float64)
    with torch.no_grad():
        got = adm(x)
        assert torch.allclose(adm.fusion_weights(0, x[:, :32]),
                              torch.full((2, 3, 32, 1), 1 / 3, dtype=torch.float64))
        halves = (x[:, :32], x[:, 32:])
        means = [torch.stack([conv(part) for conv in adm.wings[wing]], dim=1).mean(dim=1)
                 for wing, part in enumerate(halves)]
        expected = adm.project(torch.cat(means, dim=1))
    torch.testing.assert_close(got, expected, atol=1e-10, rtol=0)


def test_degenerate_gate_reduces_to_a_depthwise_convolution():
    """``W3`` 权重置零、偏置取很大的正数 → ``ξ → 1``，AGU 退化为纯深度卷积（规格卡 §8）。"""
    torch.manual_seed(3)
    agu = AdaptiveGatingUnit(32).double().eval()
    torch.nn.init.zeros_(agu.gate.weight)
    torch.nn.init.constant_(agu.gate.bias, 50.0)
    x = torch.randn(2, 32, 20, dtype=torch.float64)
    with torch.no_grad():
        torch.testing.assert_close(agu(x), agu.value(x), atol=1e-12, rtol=0)


def test_residual_identity_when_both_submodules_vanish():
    """两个子模块输出为 0 时 ADE 是恒等映射（规格卡 §8）。"""
    torch.manual_seed(4)
    block = ADEBlock(32).double().eval()
    torch.nn.init.zeros_(block.adm.project.weight)
    torch.nn.init.zeros_(block.adm.project.bias)
    torch.nn.init.zeros_(block.agu.value.weight)
    torch.nn.init.zeros_(block.agu.value.bias)
    x = torch.randn(2, 32, 20, dtype=torch.float64)
    with torch.no_grad():
        torch.testing.assert_close(block(x), x, atol=1e-12, rtol=0)


def test_input_channel_permutation_is_absorbed_by_the_stem():
    """stem 是全通道卷积：置换输入通道并同步置换 stem 权重后输出逐元素一致（规格卡 §8）。"""
    torch.manual_seed(5)
    model = build().double().eval()
    x = windows(2).double()
    order = [3, 4, 5, 0, 1, 2]                    # [gyro, acc] → [acc, gyro]
    with torch.no_grad():
        base = model(x)["vel"]
        model.stem[0].weight.copy_(model.stem[0].weight[:, order])
        permuted = model(x[:, order])["vel"]
    torch.testing.assert_close(permuted, base, atol=1e-10, rtol=0)


def test_eval_output_is_independent_of_batch_composition():
    """``eval()`` 下 BN 使用滑动统计，单样本与整批前向必须一致（规格卡 §8）。"""
    model = build().eval()
    x = windows(4)
    with torch.no_grad():
        batched = model(x)["vel"]
        single = torch.cat([model(x[i:i + 1])["vel"] for i in range(4)])
    torch.testing.assert_close(single, batched, atol=1e-5, rtol=1e-5)


def test_forward_backward_and_determinism():
    model = build()
    x = windows(seed=6)
    assert model.loss_name == "mse"
    assert_deterministic_eval(model, x)
    assert_backprop(build(), x, torch.randn(BATCH, 2))


def test_configuration():
    cfg = get_cfg({"model": "ionext", "recipe": "official"})
    assert cfg.epochs == 100 and cfg.batch == 512 and cfg.lr == 1e-4
    assert cfg.optimizer == "adam" and cfg.scheduler == "plateau"
    assert cfg.frame == "gravity_world" and cfg.target == "avg_velocity" and cfg.dims == 2
    assert cfg.augment == [] == get_cfg({"model": "ionext"}).augment
    with pytest.raises(ValueError, match="one average velocity per window"):
        build(target="frame_velocity")
    with pytest.raises(ValueError, match="must match"):
        build(model_args={"depths": [2, 2]})


@pytest.mark.slow
def test_end_to_end_train_and_val(synthetic_dataset, tmp_path):
    run_end_to_end("ionext", synthetic_dataset, tmp_path)
