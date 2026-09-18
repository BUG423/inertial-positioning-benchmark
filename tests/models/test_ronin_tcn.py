"""RoNIN-TCN 移植测试：参数量/形状锁定、感受野与严格因果、weight norm 初始化、part 模式损失。"""

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

from inertial_benchmark.cfg import get_cfg  # noqa: E402
from inertial_benchmark.data.augment import build_augmentations  # noqa: E402
from inertial_benchmark.models.ronin.temporal import WeightNormConv1d  # noqa: E402
from inertial_benchmark.nn import build_model  # noqa: E402
from inertial_benchmark.nn.losses import build_loss  # noqa: E402

FIXTURE = load_fixture("ronin_tcn")


def build(**overrides):
    return build_model(get_cfg({"model": "ronin_tcn", **overrides}))


def test_parameters_match_official_fixture():
    model = build()
    assert model.num_params == FIXTURE["total_params"] == 540_488
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) \
        == FIXTURE["trainable_params"]
    shapes = param_shapes(model.net)
    assert shape_multiset(shapes) == shape_multiset(FIXTURE["param_shapes"])
    assert shapes == FIXTURE["param_shapes"]  # 注册顺序 bias, weight_g, weight_v（68 个张量）
    assert len(shapes) == 68
    assert [list(b.shape) for _, b in model.named_buffers()] == FIXTURE["buffer_shapes"] == []


def test_paper_channel_variant_is_smaller():
    """论文正文写 [16,32,64,128,72,36]，官方 config 为 [32,...]；以代码为准（规格卡 §4）。"""
    variant = build(model_args={"layer_channels": [16, 32, 64, 128, 72, 36]})
    assert variant.num_params == FIXTURE["paper_channels_variant"]["total_params"] == 177_176


def test_shapes_and_backprop():
    model = build().eval()
    spec = model.input_spec
    assert spec.output_layout == "frame" and spec.output_shape == (400, 2)
    x = torch.randn(4, 6, 400)
    assert model(x)["vel"].shape == (4, 400, 2)
    assert model.net(x.transpose(1, 2)).shape == tuple(FIXTURE["output_shape"])
    assert_deterministic_eval(model, x)
    assert_backprop(build(), x, torch.randn(4, 400, 2))


@pytest.mark.parametrize("length", [1, 7, 400, 1000])
def test_output_length_matches_input_length(length):
    model = build().eval()
    with torch.no_grad():
        assert model.net(torch.randn(2, length, 6)).shape == (2, length, 2)


def test_receptive_field_is_253_and_strictly_causal():
    model = build().eval().double()   # 末端的影响量级 ~1e-10，float32 会被舍入掉
    assert model.receptive_field == FIXTURE["receptive_field"] == 253
    x = torch.zeros(1, 6, 600, dtype=torch.float64)
    with torch.no_grad():
        base = model.net(x.transpose(1, 2))
        impulse = x.clone()
        impulse[0, :, 0] = 1.0
        changed = (model.net(impulse.transpose(1, 2)) - base).abs().sum(dim=-1)[0] > 0
        last = int(torch.nonzero(changed)[-1])
        assert last == 252 and not bool(changed[253:].any())  # 脉冲只影响第 0..252 帧
        # 严格因果：改动 t >= 300 的输入不改变 t < 300 的输出
        tail = torch.randn(1, 6, 600, dtype=torch.float64)
        tail[:, :, :300] = x[:, :, :300]
        assert torch.allclose(model.net(tail.transpose(1, 2))[:, :300], base[:, :300], atol=0)


def test_weight_norm_initialisation_is_the_effective_one():
    """官方的 ``N(0, 0.01)`` 对 WN 卷积不生效：``g = ‖v‖``、有效权重是 Conv1d 默认初始化。"""
    block = build().net.tcn.network[3]
    for conv in (block.conv1, block.conv2):
        assert isinstance(conv, WeightNormConv1d)
        norm = torch.linalg.vector_norm(conv.weight_v, dim=(1, 2), keepdim=True)
        torch.testing.assert_close(conv.weight_g, norm, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(conv.weight, conv.weight_v, rtol=1e-6, atol=1e-7)
        assert conv.weight.std().item() > 0.01     # 约 0.1（kaiming_uniform），不是 0.01
    # 1×1 捷径与输出层的 N(0, 0.01) / N(0, 0.001) 才真正生效
    assert block.downsample.weight.std().item() < 0.02
    assert build().net.output_layer.bias.std().item() < 0.01


def test_global_pos_loss_part_mode():
    model = build()
    assert model.loss_name == "ronin_global_pos"
    assert model.loss_kwargs == {"mode": "part", "history": 253}
    loss_fn = build_loss("ronin_global_pos", mode="part", history=253)
    target = torch.randn(2, 400, 2)
    terms, valid = loss_fn.terms(target + 0.5, target, None)
    assert terms.shape == (2, 146, 2) and valid is None      # T=400 时 146 个 253 帧位移项
    value, _ = loss_fn({"vel": target + 0.5}, target, 0, None)
    assert value.item() == pytest.approx(253 ** 2 * 0.5 ** 2, rel=1e-5)
    zero, _ = loss_fn({"vel": target.clone()}, target, 0, None)
    assert zero.item() == pytest.approx(0.0, abs=1e-8)


def test_part_mode_mask_behaviour():
    loss_fn = build_loss("ronin_global_pos", mode="part", history=2)
    target = torch.zeros(1, 6, 2)
    pred = torch.zeros(1, 6, 2, requires_grad=True)
    mask = torch.ones(1, 6, dtype=torch.bool)
    mask[0, 3] = False
    terms, valid = loss_fn.terms(pred, target, mask)
    # 第 j 项覆盖帧 j+2 … j+3；帧 3 无效 ⇒ 第 0、1 项无效，第 2 项（帧 4-5）有效
    assert terms.shape == (1, 3, 2)
    assert valid.tolist() == [[False, False, True]]
    empty, _ = loss_fn({"vel": pred}, target, 0, torch.zeros(1, 6, dtype=torch.bool))
    assert empty.item() == 0.0 and empty.requires_grad


def test_stream_hook_matches_window_mode_after_the_receptive_field():
    model = build().eval()
    long = torch.randn(1, 6, 900)
    stream = model.forward_sequence(long)
    with torch.no_grad():
        window = model(long[:, :, 200:600])["vel"]
    # 窗口内 t >= 252 的帧拥有完整历史，与整序列推理一致
    torch.testing.assert_close(stream[:, 452:600], window[:, 252:], rtol=1e-4, atol=1e-5)


def test_recipes_official_and_unified():
    official = get_cfg({"model": "ronin_tcn", "recipe": "official"})
    unified = get_cfg({"model": "ronin_tcn", "recipe": "unified"})
    assert official.batch == 72 and official.lr == pytest.approx(3e-4)
    assert official.scheduler == "plateau" and official.gamma == 0.75
    assert official.stride == 100 and official.amp is False and official.fitness == "loss"
    assert official.augment == unified.augment
    shift, augs = build_augmentations(official.augment, official.frame)
    assert [a.name for a in augs] == ["random_yaw"]
    assert shift.resolve_range(official.stride) == (-50, 50)
    assert build(recipe="official").net.tcn.network[0].dropout1.p == 0.2


@pytest.mark.slow
@pytest.mark.parametrize("overlap", ["mean", "center"])
def test_end_to_end_train_and_val(synthetic_dataset, tmp_path, overlap):
    metrics = run_end_to_end("ronin_tcn", synthetic_dataset, tmp_path / overlap, overlap=overlap)
    assert math.isfinite(metrics["rte"]) and math.isfinite(metrics["ate"])
