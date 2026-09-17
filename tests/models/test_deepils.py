"""DeepILS 移植测试：参数量锁定、注意力数学、窗口约束、MSE+L1 损失与端到端。"""

import numpy as np
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
from inertial_benchmark.data.augment import RandomYaw  # noqa: E402
from inertial_benchmark.nn import build_model  # noqa: E402

FIXTURE = load_fixture("deepils")


def build(**overrides):
    return build_model(get_cfg({"model": "deepils", **overrides}))


def test_parameters_match_official_fixture():
    model = build()
    assert model.num_params == FIXTURE["total_params"] == 2_290_226
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) \
        == FIXTURE["trainable_params"]
    shapes = param_shapes(model)
    assert shape_multiset(shapes) == shape_multiset(FIXTURE["param_shapes"])
    assert shapes == FIXTURE["param_shapes"]  # 注册顺序也一致
    assert [list(b.shape) for _, b in model.named_buffers()] == FIXTURE["buffer_shapes"]


def test_forward_shapes_and_gradients():
    model = build().eval()
    x = torch.randn(2, 6, 200)
    out = model(x)
    assert set(out) == {"vel"} and out["vel"].shape == (2, 2)
    # 长度 200 → 101 → 51 → 51 → 26 → 13 → 7（k=5 配 p=3 的非对称 same）
    assert model.input_block[0](x).shape[-1] == 101
    h = model.input_block(x)
    lengths = [h.shape[-1]]
    for group in model.residual_groups:
        h = group(h)
        lengths.append(h.shape[-1])
    assert lengths == [51, 51, 26, 13, 7]
    assert model.feature_length == 7 and model.transition(h).shape[1] == 128
    assert_deterministic_eval(model, x)
    assert_backprop(build(), x, torch.randn(2, 2))


def test_window_constraint():
    for window in (100, 400):
        with pytest.raises(ValueError, match="temporal length"):
            build(window=window)
    # 显式关闭锁定时按窗口推导（改变结构）
    model = build(window=400, model_args={"feature_length": None})
    assert model.feature_length == 13 and model.num_params != FIXTURE["total_params"]
    assert model.eval()(torch.randn(1, 6, 400))["vel"].shape == (1, 2)
    # T ∈ [191, 222] 仍得到长度 7，参数量不变
    assert build(window=222).num_params == FIXTURE["total_params"]


def test_attention_modules():
    model = build()
    block = model.residual_groups[3][0]
    y = torch.randn(3, 512, 7)
    ca, sa = block.ca(y), block.sa(y)
    assert ca.shape == (3, 512, 1) and sa.shape == (3, 1, 7)
    assert torch.all((ca > 0) & (ca < 1)) and torch.all((sa > 0) & (sa < 1))
    # 缩减比恒为 16（ratio 形参无效）：64 → 4，512 → 32
    assert model.residual_groups[0][0].ca.fc[0].out_channels == 4
    assert block.ca.fc[0].out_channels == 32
    # 权重置零后门控恒为 0.5
    with torch.no_grad():
        for p in list(block.ca.parameters()) + list(block.sa.parameters()):
            p.zero_()
    torch.testing.assert_close(block.ca(y), torch.full((3, 512, 1), 0.5))
    torch.testing.assert_close(block.sa(y), torch.full((3, 1, 7), 0.5))
    # 每个块都带 CA/SA，第二个 conv 之后没有 ReLU（相加后才有）
    blocks = [b for group in model.residual_groups for b in group]
    assert len(blocks) == 8 and all(b.ca is not None and b.sa is not None for b in blocks)
    assert sum(b.downsample is not None for b in blocks) == 3


def test_loss_is_mse_plus_l1():
    model = build()
    assert model.loss_name == "mse_l1"
    pred = torch.tensor([[1.0, -2.0], [0.5, 0.0]])
    target = torch.zeros(2, 2)
    loss, items = model.loss({"vel": pred}, {"target": target})
    mse = float(((pred - target) ** 2).mean())
    l1 = float((pred - target).abs().mean())
    assert loss.item() == pytest.approx(mse + l1)
    assert items["mse"].item() == pytest.approx(mse) and items["l1"].item() == pytest.approx(l1)


def test_initialisation():
    model = build()
    linear = [m for m in model.modules() if isinstance(m, nn.Linear)]
    assert len(linear) == 3
    for layer in linear:
        assert abs(float(layer.weight.detach().std()) - 0.01) < 0.004
        assert torch.all(layer.bias.detach() == 0)
    for bn in (m for m in model.modules() if isinstance(m, nn.BatchNorm1d)):
        assert torch.all(bn.weight.detach() == 1) and torch.all(bn.bias.detach() == 0)
    assert all(c.bias is None for c in model.modules() if isinstance(c, nn.Conv1d))


def test_random_yaw_preserves_norms():
    rng = np.random.default_rng(0)
    sample = {"imu": rng.normal(size=(6, 200)).astype(np.float32),
              "target": np.array([1.0, 2.0], np.float32)}
    before = sample["imu"].copy()
    RandomYaw()(sample, rng)
    for sl in (slice(0, 2), slice(3, 5)):
        np.testing.assert_allclose(np.linalg.norm(sample["imu"][sl], axis=0),
                                   np.linalg.norm(before[sl], axis=0), rtol=1e-5)
    np.testing.assert_allclose(sample["imu"][2], before[2])   # z 分量不变
    np.testing.assert_allclose(sample["imu"][5], before[5])
    assert np.linalg.norm(sample["target"]) == pytest.approx(np.sqrt(5.0), rel=1e-5)


def test_official_recipe():
    cfg = get_cfg({"model": "deepils", "recipe": "official"})
    assert cfg.epochs == 200 and cfg.batch == 128 and cfg.lr == 1e-4
    assert cfg.optimizer == "adam" and cfg.scheduler == "plateau" and cfg.gamma == 0.1
    assert cfg.weight_decay == 0.0 and cfg.grad_clip == 0.0 and cfg.amp is False
    assert cfg.augment == ["random_yaw", "time_shift"] and cfg.stride == 10


@pytest.mark.slow
def test_end_to_end_train_and_val(synthetic_dataset, tmp_path):
    run_end_to_end("deepils", synthetic_dataset, tmp_path)
