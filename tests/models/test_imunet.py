"""IMUNet 及其 4 个移动端基线的移植测试：参数量锁定、噪声层语义、结构性质与端到端。"""

import pytest
from model_testing import (
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
from inertial_benchmark.models.imunet.mobile import InvertedResidual1d  # noqa: E402
from inertial_benchmark.nn import build_model  # noqa: E402

NAMES = ("imunet", "imunet_mobilenet", "imunet_mobilenetv2", "imunet_mnasnet",
         "imunet_efficientnetb0")
# 规格卡 §8 的残差块下标（features 中 use_res 为真的倒残差块）
RESIDUALS = {
    "imunet_mobilenetv2": [3, 5, 6, 8, 9, 10, 12, 13, 15, 16],
    "imunet_mnasnet": [3, 4, 6, 7, 9, 10, 12, 14, 15, 16],
    "imunet_efficientnetb0": [3, 5, 7, 8, 10, 11, 13, 14, 15],
}


def build(name, **overrides):
    return build_model(get_cfg({"model": name, **overrides}))


@pytest.mark.parametrize("name", NAMES)
def test_parameters_match_official_fixture(name):
    fixture = load_fixture(name)
    model = build(name)
    assert model.num_params == fixture["total_params"]
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) \
        == fixture["trainable_params"]
    shapes = param_shapes(model)
    assert shape_multiset(shapes) == shape_multiset(fixture["param_shapes"])
    assert shapes == fixture["param_shapes"]  # 注册顺序也一致
    assert [list(b.shape) for _, b in model.named_buffers()] == fixture["buffer_shapes"]


@pytest.mark.parametrize("name", NAMES)
def test_forward_backward_and_determinism(name):
    model = build(name)
    x = torch.randn(4, 6, 200)
    out = model(x)
    assert set(out) == {"vel"} and out["vel"].shape == (4, 2)
    assert_deterministic_eval(model, x)
    assert_backprop(build(name), x, torch.randn(4, 2))
    assert model.loss_name == "mse"


def test_imunet_layer_shapes_and_window_constraint():
    model = build("imunet").eval()
    x = torch.randn(1, 6, 200)
    assert model.input_block(x).shape == (1, 64, 50)
    lengths = []
    feat = model.input_block(x)
    for block in (model.conv_1_1, model.conv_1_2, model.conv_3_1, model.conv_3_2, model.conv_4_1,
                  model.conv_4_2, model.conv_5_1, model.conv_5_2, model.conv_6_1, model.conv_6_2,
                  model.conv_7_1, model.conv_7_2):
        feat = block(feat)
        lengths.append(feat.shape[-1])
    assert lengths == [50, 50, 50, 50, 25, 25, 13, 13, 7, 7, 4, 4]
    assert model.features(x).shape == (1, 400, 3) and model.feature_length == 3
    # 噪声层把结构锁死在 6×200：其他窗口必须在构造时报错（官方是运行时形状错误）
    for window in (100, 400):
        with pytest.raises(ValueError, match="noise layer"):
            build("imunet", window=window)
    # 激活：stem 为 ReLU，其余块为 ELU；全网无 dropout
    assert isinstance(model.input_block[2], nn.ReLU)
    assert isinstance(model.conv_1_1.act, nn.ELU) and isinstance(model.conv_7_1.act, nn.ELU)
    assert not any(isinstance(m, nn.Dropout) for m in model.modules())


def test_imunet_noise_layer_pairs_by_flat_index():
    """规格卡 §8：f=0、b=0、W=1、fc 取 one-hot 时输出 ``ELU(−x[:, c, t])``，``i = c·200 + t``。"""
    model = build("imunet").eval()
    with torch.no_grad():
        model.output_block[1].weight.zero_()
        model.output_block[1].bias.zero_()
        model.noise.W.fill_(1.0)
        model.noise.b.zero_()
        model.head.linear.weight.zero_()
        model.head.linear.bias.zero_()
        for row, (c, t) in enumerate(((0, 0), (4, 137))):
            model.head.linear.weight[row, c * 200 + t] = 1.0
    x = torch.randn(3, 6, 200)
    with torch.no_grad():
        out = model(x)["vel"]
    expected = torch.stack([nn.functional.elu(-x[:, 0, 0]), nn.functional.elu(-x[:, 4, 137])], -1)
    torch.testing.assert_close(out, expected)


def test_imunet_noise_initialisation():
    values = torch.cat([build("imunet").noise.W.detach().flatten() for _ in range(3)])
    assert abs(float(values.mean())) < 0.1 and abs(float(values.std()) - 1.0) < 0.1
    assert torch.all(build("imunet").noise.b == 0)


@pytest.mark.parametrize("name", sorted(RESIDUALS))
def test_residual_positions_and_linear_init(name):
    model = build(name)
    found = [i for i, block in enumerate(model.features)
             if isinstance(block, InvertedResidual1d) and block.use_res]
    assert found == RESIDUALS[name]
    # 官方 _initialize_weights 在一维模型上只影响 Linear
    linear = model.classifier[-1] if isinstance(model.classifier, nn.Sequential) \
        else model.classifier
    weight, bias = linear.weight.detach(), linear.bias.detach()
    assert torch.all(bias == 0) and abs(float(weight.std()) - 0.01) < 0.004
    convs = [m for m in model.modules() if isinstance(m, nn.Conv1d)]
    assert all(c.bias is None for c in convs)
    # 每个倒残差块的最后一层是 BN（线性瓶颈，无激活）
    for block in model.features:
        if isinstance(block, InvertedResidual1d):
            assert isinstance(block.conv[-1], nn.BatchNorm1d)


def test_dropout_presence_matches_cards():
    counts = {name: sum(isinstance(m, nn.Dropout) for m in build(name).modules())
              for name in NAMES}
    assert counts == {"imunet": 0, "imunet_mobilenet": 0, "imunet_mobilenetv2": 0,
                      "imunet_mnasnet": 1, "imunet_efficientnetb0": 1}
    for name in ("imunet_mnasnet", "imunet_efficientnetb0"):
        model = build(name)
        assert model.classifier[0].p == 0.5
        x = torch.randn(4, 6, 200)
        model.train()
        assert not torch.allclose(model(x)["vel"], model(x)["vel"])


@pytest.mark.parametrize("name", NAMES[1:])
@pytest.mark.parametrize("window", [100, 400])
def test_pooled_backbones_accept_other_windows(name, window):
    """全局池化头：窗口改变时仍可前向且参数量不变（规格卡 §8）。"""
    model = build(name, window=window).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 6, window))
    assert out["vel"].shape == (2, 2)
    assert model.num_params == load_fixture(name)["total_params"]


def test_efficientnet_variant_details():
    model = build("imunet_efficientnetb0")
    # t=1 的块仍保留 1×1 扩张卷积（3 个卷积）
    first = model.features[1]
    assert isinstance(first, InvertedResidual1d)
    assert sum(isinstance(m, nn.Conv1d) for m in first.conv) == 3
    # Swish == x·sigmoid(x)，且没有 SE 门控
    act = [m for m in model.modules() if isinstance(m, nn.SiLU)]
    assert act and not any(isinstance(m, nn.AdaptiveAvgPool1d) for m in model.features.modules())
    x = torch.randn(7)
    torch.testing.assert_close(act[0](x.clone()), x * torch.sigmoid(x))


@pytest.mark.slow
@pytest.mark.parametrize("name", NAMES)
def test_end_to_end_train_and_val(name, synthetic_dataset, tmp_path):
    run_end_to_end(name, synthetic_dataset, tmp_path)
