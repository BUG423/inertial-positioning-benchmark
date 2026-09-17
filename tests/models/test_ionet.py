"""IONet 移植测试：推导参数量锁定、``h_n`` 读出、极坐标头与角度损失、抽取与端到端。"""

import math

import pytest
from model_testing import (
    assert_backprop,
    assert_deterministic_eval,
    run_end_to_end,
    shape_multiset,
)

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from inertial_benchmark.cfg import get_cfg  # noqa: E402
from inertial_benchmark.models.ionet import (  # noqa: E402
    CHANNEL_PERMUTATION,
    IONetPolarLoss,
    wrap_angle,
)
from inertial_benchmark.nn import build_model  # noqa: E402

# paper-only：没有官方夹具，锁定规格卡 §4/§8 推导的 PyTorch 计数与参数形状
DERIVED_PARAMS = 302_978
DERIVED_PARAMS_H128 = 535_042
DERIVED_SHAPES = ([[384, 6], [384, 96], [384], [384]] * 2
                  + [[384, 192], [384, 96], [384], [384]] * 2
                  + [[2, 192], [2]])


def build(**overrides):
    return build_model(get_cfg({"model": "ionet", **overrides}))


def test_parameter_count_matches_spec_card():
    model = build()
    assert model.num_params == DERIVED_PARAMS
    shapes = [list(p.shape) for _, p in model.named_parameters()]
    assert len(shapes) == 18
    assert shapes == DERIVED_SHAPES
    assert shape_multiset(shapes) == shape_multiset(DERIVED_SHAPES)
    assert not list(model.buffers())
    # OxIOD 的 DeepIO 变体（hidden=128）
    assert build(model_args={"hidden": 128}).num_params == DERIVED_PARAMS_H128


def test_forward_shapes_and_gradients():
    model = build().eval()
    x = torch.randn(4, 6, 400)
    out = model(x)
    assert set(out) == {"vel", "speed", "heading"}
    assert out["vel"].shape == (4, 2) and out["speed"].shape == (4,)
    with pytest.raises(ValueError, match="expected input"):
        model(torch.randn(2, 6, 200))
    with pytest.raises(ValueError, match="multiple of"):
        build(window=401)
    with pytest.raises(ValueError, match="dims=2"):
        build(dims=3)
    assert_deterministic_eval(model, x)
    assert_backprop(build(), x, torch.randn(4, 2))
    assert model.loss_name == "ionet_polar"


def test_channel_permutation_and_decimation():
    model = build().eval()
    assert CHANNEL_PERMUTATION == (3, 4, 5, 0, 1, 2)
    x = torch.randn(2, 6, 400)
    swapped = torch.cat([x[:, 3:6], x[:, 0:3]], dim=1)
    plain = build(model_args={"permute_channels": False}).eval()
    plain.load_state_dict(model.state_dict())
    with torch.no_grad():
        torch.testing.assert_close(model(x)["vel"], plain(swapped)["vel"])
    # 抽取：常值输入保持常值，长度 400 → 200
    const = torch.full((1, 6, 400), 1.5)
    decimated = nn.functional.avg_pool1d(const, model.decimation, model.decimation)
    assert decimated.shape == (1, 6, 200) and torch.all(decimated == 1.5)


def test_hidden_state_readout():
    """读出为 ``h_n`` 的前向/反向拼接（不是 ``out[:, -1, :]``，规格卡 §8、§10.4）。"""
    model = build().eval()
    x = torch.randn(3, 6, 400)
    with torch.no_grad():
        feature = model.features(x)
        permuted = x[:, list(CHANNEL_PERMUTATION)]
        decimated = nn.functional.avg_pool1d(permuted, 2, 2).transpose(1, 2)
        sequence, _ = model.lstm1(decimated)
        outputs, _ = model.lstm2(sequence)
    assert feature.shape == (3, 192) and outputs.shape == (3, 200, 192)
    torch.testing.assert_close(feature[:, :96], outputs[:, -1, :96])    # 前向末状态
    torch.testing.assert_close(feature[:, 96:], outputs[:, 0, 96:])     # 反向末状态
    assert not torch.allclose(feature[:, 96:], outputs[:, -1, 96:])


def test_polar_head():
    model = build().eval()
    x = torch.randn(8, 6, 400)
    with torch.no_grad():
        out = model(x)
    torch.testing.assert_close(out["vel"].norm(dim=-1), out["speed"].abs(), atol=1e-6, rtol=1e-5)
    positive = out["speed"] > 0
    angle = torch.atan2(out["vel"][:, 1], out["vel"][:, 0])
    torch.testing.assert_close(angle[positive], wrap_angle(out["heading"])[positive],
                               atol=1e-5, rtol=1e-5)


def test_polar_loss():
    loss_fn = IONetPolarLoss(kappa=1.0, min_speed=0.1)
    target = torch.tensor([[1.0, 0.0], [0.05, 0.0]])
    # 角度差 2π − ε：wrap 之后约为 ε，不是 (2π)²
    eps = 1e-3
    out = {"speed": torch.tensor([1.0, 0.05]),
           "heading": torch.tensor([2 * math.pi - eps, 3.0]),
           "vel": torch.zeros(2, 2)}
    loss, items = loss_fn(out, target)
    assert items["speed"].item() == pytest.approx(0.0)
    # 只有第一个样本参与角度项（第二个 ‖v̄‖ = 0.05 ≤ 0.1 被掩掉）
    assert items["heading"].item() == pytest.approx(eps ** 2 / 2, rel=1e-3)
    assert loss.item() == pytest.approx(eps ** 2 / 2, rel=1e-3)
    # κ 生效；速度项为普通 MSE
    scaled, _ = IONetPolarLoss(kappa=10.0)(out, target)
    assert scaled.item() == pytest.approx(10 * eps ** 2 / 2, rel=1e-3)
    fast = {"speed": torch.tensor([2.0]), "heading": torch.zeros(1), "vel": torch.zeros(1, 2)}
    value, items = loss_fn(fast, torch.tensor([[1.0, 0.0]]))
    assert items["speed"].item() == pytest.approx(1.0) and value.item() == pytest.approx(1.0)
    # 包裹到 ±π 区间（边界处的符号由浮点误差决定）
    assert abs(wrap_angle(torch.tensor(3 * math.pi)).item()) == pytest.approx(math.pi, abs=1e-6)
    assert wrap_angle(torch.tensor(0.5 - 2 * math.pi)).item() == pytest.approx(0.5, abs=1e-6)


def test_official_recipe_and_semantics():
    cfg = get_cfg({"model": "ionet", "recipe": "official"})
    assert cfg.window == 400 and cfg.stride == 20 and cfg.eval_stride == 10
    assert cfg.lr == 1.5e-3 and cfg.epochs == 100 and cfg.batch == 128
    assert cfg.scheduler == "none" and cfg.augment == []
    # IPB 默认适配：重力对齐世界系 + 2D 平均速度（语义改变，已在 docstring/YAML 注明）
    assert cfg.frame == "gravity_world" and cfg.target == "avg_velocity" and cfg.dims == 2


@pytest.mark.slow
def test_end_to_end_train_and_val(synthetic_dataset, tmp_path):
    run_end_to_end("ionet", synthetic_dataset, tmp_path)
