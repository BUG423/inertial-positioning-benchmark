"""TinyOdom 移植测试：参数量锁定（Keras 形状换算）、因果性与感受野、池化轴、通道置换与端到端。"""

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

from inertial_benchmark.cfg import get_cfg  # noqa: E402
from inertial_benchmark.models.tinyodom import CHANNEL_PERMUTATION, TinyOdom  # noqa: E402
from inertial_benchmark.nn import build_model  # noqa: E402
from inertial_benchmark.nn.base import InputSpec  # noqa: E402

FIXTURE = load_fixture("tinyodom")


def build(**overrides):
    return build_model(get_cfg({"model": "tinyodom", **overrides}))


def to_keras_shape(shape):
    """PyTorch → Keras：卷积 ``(out, in, k) → (k, in, out)``、Linear ``(out, in) → (in, out)``。"""
    if len(shape) == 3:
        return [shape[2], shape[1], shape[0]]
    if len(shape) == 2:
        return [shape[1], shape[0]]
    return list(shape)


def test_parameters_match_official_fixture():
    model = build()
    assert model.num_params == FIXTURE["total_params"] == 100_481
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) \
        == FIXTURE["trainable_params"]
    shapes = [to_keras_shape(s) for s in param_shapes(model)]
    assert shape_multiset(shapes) == shape_multiset(FIXTURE["param_shapes"])
    assert shapes == FIXTURE["param_shapes"]      # Keras model.weights 顺序也一致
    assert not list(model.buffers())
    # dims=2 的消融（保留官方 2 个头）
    assert build(dims=2, frame="gravity_world").num_params \
        == FIXTURE["benchmark_config"]["variant_2d_total_params"] == 100_448
    # 官方 10 通道 + 2 头配置
    channels = tuple(f"c{i}" for i in range(10))
    official = TinyOdom(InputSpec(window=400, dims=2, frame="gravity_world", channels=channels))
    assert official.num_params == FIXTURE["official_config"]["total_params"] == 102_008


def test_forward_shapes_and_window_independence():
    model = build().eval()
    x = torch.randn(3, 6, 400)
    out = model(x)
    assert set(out) == {"vel"} and out["vel"].shape == (3, 3)
    assert model.loss_name == "mse_sum"
    for window in (50, 1000):
        assert model(torch.randn(2, 6, window))["vel"].shape == (2, 3)
    assert build(window=1000).num_params == FIXTURE["total_params"]   # 与 T 无关
    assert_deterministic_eval(model, x)
    assert_backprop(build(), x, torch.randn(3, 3))


def test_causality_and_receptive_field():
    """输出只取末时间步，感受野 ``1 + 2·(k−1)·Σd = 1739`` 个样本（规格卡 §8）。"""
    model = build().eval()
    assert model.receptive_field == 1739
    x = torch.randn(1, 6, 2000)
    with torch.no_grad():
        base = model(x)["vel"]
        outside = x.clone()
        outside[0, :, 100] += 10.0            # 2000 − 1739 = 261 之前 → 感受野外
        torch.testing.assert_close(model(outside)["vel"], base, atol=1e-6, rtol=1e-5)
        inside = x.clone()
        inside[0, :, 1999] += 10.0            # 最后一个样本 → 一定在感受野内
        assert not torch.allclose(model(inside)["vel"], base, atol=1e-4)


def test_maxpool_runs_along_the_filter_axis():
    """末时间步的 30 维特征两两取最大 → 15 维（规格卡 §8）。"""
    model = build().eval()
    x = torch.randn(2, 6, 400)
    with torch.no_grad():
        feature = x[:, list(CHANNEL_PERMUTATION)]
        for block in model.blocks:
            feature = block(feature)
        last = feature[:, :, -1]
        pooled = model.pool(last.unsqueeze(1)).flatten(1)
    assert last.shape == (2, 30) and pooled.shape == (2, 15)
    expected = torch.maximum(last[:, 0::2], last[:, 1::2])
    torch.testing.assert_close(pooled, expected)
    # pre 层是线性激活（无非线性）：pre + 输出头合起来是线性映射
    with torch.no_grad():
        a, b = torch.randn(1, 15), torch.randn(1, 15)
        lin = model.heads[0](model.pre(a + b)) - model.heads[0](model.pre(a)) \
            - model.heads[0](model.pre(b)) + model.heads[0](model.pre(torch.zeros(1, 15)))
    assert float(lin.abs().max()) < 1e-5


def test_channel_permutation():
    """IPB 的 ``[gyro, acc]`` 按下标 ``[3,4,5,0,1,2]`` 重排为官方的 ``[acc, gyro]``。"""
    assert CHANNEL_PERMUTATION == (3, 4, 5, 0, 1, 2)
    model = build().eval()
    x = torch.randn(2, 6, 400)
    swapped = torch.cat([x[:, 3:6], x[:, 0:3]], dim=1)
    plain = build(model_args={"permute_channels": False}).eval()
    plain.load_state_dict(model.state_dict())
    with torch.no_grad():
        torch.testing.assert_close(model(x)["vel"], plain(swapped)["vel"])


def test_residual_shortcut_only_in_first_block():
    model = build()
    matching = [block.matching_conv1D is not None for block in model.blocks]
    assert matching == [True, False, False, False, False]
    assert model.blocks[0].matching_conv1D.weight.shape == (30, 6, 1)
    assert [block.pad for block in model.blocks] == [11, 22, 44, 88, 704]
    # Keras 初始化：卷积偏置与 Dense 偏置为 0，卷积权重为截断 he_normal
    for block in model.blocks:
        assert torch.all(block.conv1D_0.bias.detach() == 0)
    weight = model.blocks[1].conv1D_0.weight.detach()
    std = (2.0 / (30 * 12)) ** 0.5 / 0.8796256610342398
    assert float(weight.abs().max()) <= 2 * std + 1e-6
    assert abs(float(weight.std()) - std * 0.88) < 0.3 * std
    assert torch.all(model.pre.bias.detach() == 0)


def test_loss_is_sum_of_per_head_mse():
    model = build()
    pred = torch.tensor([[1.0, 2.0, -1.0]])
    target = torch.zeros(1, 3)
    imu = torch.zeros(1, 6, model.input_spec.window)
    loss, items = model.loss({"vel": pred}, {"target": target, "imu": imu})
    assert loss.item() == pytest.approx(1.0 + 4.0 + 1.0)      # Σ_d mean_b (·)²
    assert items["mse"].item() == pytest.approx(6.0 / 3)


def test_body_frame_view_and_recipe():
    cfg = get_cfg({"model": "tinyodom", "recipe": "official"})
    assert cfg.window == 400 and cfg.frame == "body" and cfg.dims == 3
    assert cfg.target == "avg_velocity" and cfg.augment == []
    assert cfg.epochs == 900 and cfg.batch == 256 and cfg.lr == 1e-3
    assert cfg.scheduler == "none" and cfg.stride == 20 and cfg.eval_stride == 10
    assert cfg.fitness == "ate"


@pytest.mark.slow
def test_end_to_end_train_and_val(synthetic_dataset, tmp_path):
    run_end_to_end("tinyodom", synthetic_dataset, tmp_path)
