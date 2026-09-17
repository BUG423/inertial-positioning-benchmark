"""LLIO 移植测试：参数量锁定、patch 展平顺序、ResMLP 子层性质、抽取层与端到端。"""

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
from inertial_benchmark.models.llio import Affine, PatchFlatten  # noqa: E402
from inertial_benchmark.nn import build_model  # noqa: E402

FIXTURE = load_fixture("llio")


def build(**overrides):
    return build_model(get_cfg({"model": "llio", **overrides}))


def test_parameters_match_official_fixture():
    model = build()
    assert model.num_params == FIXTURE["total_params"] == 7_191_654
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) \
        == FIXTURE["trainable_params"]
    shapes = param_shapes(model)
    assert shape_multiset(shapes) == shape_multiset(FIXTURE["param_shapes"])
    assert shapes == FIXTURE["param_shapes"]            # 注册顺序也一致
    assert [name for name, _ in model.named_parameters()] == FIXTURE["param_names"]
    assert not list(model.buffers())                    # 没有 buffer
    # out_dim=2 的变体（仅对照，不注册）
    assert build(dims=2).num_params \
        == FIXTURE["benchmark_config"]["variant_out_dim2_total_params"]


def test_output_fields_and_window_constraint():
    model = build().eval()
    x = torch.randn(2, 6, 200)
    out = model(x)
    assert set(out) == {"vel", "logstd"}
    assert out["vel"].shape == (2, 3) and out["logstd"].shape == (2, 3)
    assert tuple(FIXTURE["output_shape"]["disp"]) == (1, 3)
    with pytest.raises(ValueError, match="expected input"):
        model(torch.randn(2, 6, 100))
    with pytest.raises(ValueError, match="input_len × decimation"):
        build(window=100)
    assert_deterministic_eval(model, x)
    assert_backprop(build(), x, torch.randn(2, 3))
    assert model.loss_name == "nll_detach_then_nll" and model.loss_kwargs["switch_epoch"] == 9


def test_patch_flatten_order():
    """patch 内按时间优先展平：token ``l`` 的第 ``τ·6 + c`` 维取 ``x[:, c, 25l + τ]``。"""
    x = (torch.arange(6).reshape(1, 6, 1) * 1000 + torch.arange(100).reshape(1, 1, 100)).float()
    tokens = PatchFlatten(25)(x)
    assert tokens.shape == (1, 4, 150)
    expected = [0.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 1.0, 1001.0]
    assert tokens[0, 0, :8].tolist() == expected
    assert tokens[0, 1, 0].item() == 25.0
    with pytest.raises(ValueError, match="multiple of patch_len"):
        PatchFlatten(25)(torch.zeros(1, 6, 110))


def test_decimation_is_a_two_tap_average():
    model = build().eval()
    x = torch.zeros(1, 6, 200)
    x[:, :, 0::2] = 1.0        # 偶数样本 1、奇数样本 0 → 抽取后恒为 0.5
    decimated = nn.functional.avg_pool1d(x, model.decimation, model.decimation)
    assert decimated.shape == (1, 6, 100) and torch.all(decimated == 0.5)
    const = torch.full((1, 6, 200), 2.5)
    assert torch.all(nn.functional.avg_pool1d(const, 2, 2) == 2.5)


def test_resmlp_sublayer_properties():
    model = build()
    layer = model.extractor.net[2]
    token_mixing, channel_mixing = layer[0], layer[1]
    # 初始化：仿射 g=1、b=0，LayerScale 为 0.1
    for block in (token_mixing, channel_mixing):
        assert torch.all(block.affine.g.detach() == 1) and torch.all(block.affine.b.detach() == 0)
        assert torch.all(block.affine_out.g.detach() == 1)
        assert torch.all(block.affine_out.b.detach() == 0)
        assert torch.all(block.scale.detach() == 0.1)
    # 仿射作用在残差相加之后（post-affine）：把 scale 置零后子层退化为纯仿射
    x = torch.randn(3, 4, 512)
    with torch.no_grad():
        token_mixing.scale.zero_()
        token_mixing.affine_out.g.fill_(2.0)
        token_mixing.affine_out.b.fill_(1.0)
        torch.testing.assert_close(token_mixing(x), x * 2.0 + 1.0)
        # 单位阵 + scale=0 的 token-mixing 为恒等映射
        token_mixing.affine_out.g.fill_(1.0)
        token_mixing.affine_out.b.zero_()
        torch.testing.assert_close(token_mixing(x), x)
    # token-mixing 把 patch 轴当通道：Conv1d(4, 4, 1)
    assert isinstance(token_mixing.fn, nn.Conv1d)
    assert token_mixing.fn.weight.shape == (4, 4, 1) and token_mixing.fn.bias is None
    # channel-mixing：512 → 1024 → 512，隐层 dropout 0.2
    assert channel_mixing.fn[0].out_features == 1024 and channel_mixing.fn[0].bias is None
    assert isinstance(channel_mixing.fn[2], nn.Dropout) and channel_mixing.fn[2].p == 0.2
    assert isinstance(model.extractor.net[8], Affine)
    assert model.reg.net[1][2].p == 0.5      # 回归 MLP 的 dropout 固定为 0.5


def test_patch_permutation_equivalence():
    """置换 patch 与同样置换 token-mixing 权重的行/列后，池化输出不变（规格卡 §8）。"""
    model = build().eval()
    x = torch.randn(2, 6, 200)
    with torch.no_grad():
        base = model(x)["vel"]
        perm = torch.tensor([2, 0, 3, 1])
        for layer in model.extractor.net[2:8]:
            weight = layer[0].fn.weight.data
            layer[0].fn.weight.data = weight[perm][:, perm]
        tokens = PatchFlatten(25)(nn.functional.avg_pool1d(x, 2, 2))
        tokens = tokens[:, perm]
        feature = model.extractor.net[1:](tokens)
        disp, _ = model.reg(feature)
    torch.testing.assert_close(disp, base, atol=1e-5, rtol=1e-4)


def test_dropout_only_in_training_mode():
    model = build()
    x = torch.randn(4, 6, 200)
    model.train()
    assert not torch.allclose(model(x)["vel"], model(x)["vel"])
    model.eval()
    torch.testing.assert_close(model(x)["vel"], model(x)["vel"], atol=0, rtol=0)


def test_recipes():
    official = get_cfg({"model": "llio", "recipe": "official"})
    assert official.lr == 1e-4 and official.batch == 1024 and official.grad_clip == 0.1
    assert official.scheduler == "none" and official.optimizer == "adam"
    unified = get_cfg({"model": "llio", "recipe": "unified"})
    assert unified.grad_clip == 0.1 and unified.scheduler == "cosine"
    assert unified.target == "displacement" and unified.dims == 3
    assert unified.frame == "gravity_yaw_local"


@pytest.mark.slow
def test_end_to_end_train_and_val(synthetic_dataset, tmp_path):
    run_end_to_end("llio", synthetic_dataset, tmp_path)
