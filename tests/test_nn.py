import math

import pytest

torch = pytest.importorskip("torch")

from inertial_benchmark.cfg import get_cfg  # noqa: E402
from inertial_benchmark.nn import (  # noqa: E402
    MODELS,
    BaseModel,
    GaussianHead,
    InputSpec,
    PolarHead,
    VelocityHead,
    build_loss,
    build_model,
    gaussian_nll,
    list_models,
    register_model,
)
from inertial_benchmark.nn.losses import MIN_LOGSTD, MSEThenNLL  # noqa: E402
from inertial_benchmark.nn.modules import ResNet1DBackbone  # noqa: E402

# 官方 RoNIN 仓库（Sachini/ronin@805b7f0）get_model() 在 window=200 下的参数量，
# 由 third_party 中实例化官方 ResNet1D 得到；本实现与官方逐参数对齐后输出完全一致。
OFFICIAL_PARAMS_2D = {
    "ronin_resnet18": 4_634_882,
    "ronin_resnet50": 9_256_450,
    "ronin_resnet101": 15_958_530,
}


@pytest.mark.parametrize("name", sorted(OFFICIAL_PARAMS_2D))
def test_ronin_parameter_count_matches_official(name):
    model = build_model(get_cfg({"model": name}))
    assert model.num_params == OFFICIAL_PARAMS_2D[name]
    assert model.feature_length == 7


def test_ronin_3d_and_other_windows():
    model = build_model(get_cfg({"model": "ronin_resnet18", "dims": 3}))
    assert model.num_params == 4_635_395
    for window in (64, 100, 400):
        m = build_model(get_cfg({"model": "ronin_resnet18", "window": window})).eval()
        out = m(torch.randn(3, 6, window))
        assert out["vel"].shape == (3, 2)
        backbone = ResNet1DBackbone(6)
        assert backbone(torch.zeros(1, 6, window)).shape[-1] == m.feature_length
    with pytest.raises(ValueError, match="expected input"):
        model.check_input(torch.zeros(2, 6, 100))


def test_model_args_override_and_loss():
    cfg = get_cfg({"model": "ronin_resnet18", "model_args": {"fc_dim": 64},
                   "loss": "mse_then_nll", "loss_switch_epoch": 3})
    model = build_model(cfg)
    assert model.model_cfg["args"]["fc_dim"] == 64
    assert model.loss_name == "mse_then_nll" and model.loss_kwargs == {"switch_epoch": 3}
    out = model(torch.randn(4, 6, 200))
    loss, items = model.loss(out, {"target": torch.zeros(4, 2)}, epoch=0)
    assert loss.ndim == 0 and "mse" in items
    with pytest.raises(KeyError, match="logstd"):
        model.loss(out, {"target": torch.zeros(4, 2)}, epoch=5)


def test_registry():
    assert "ronin_resnet" in list_models()

    with pytest.raises(KeyError, match="already registered"):
        @register_model("ronin_resnet")
        class Other(BaseModel):
            pass

    @register_model("tiny_test_model")
    class Tiny(BaseModel):
        default_loss = "gaussian_nll"

        def __init__(self, input_spec, hidden=8):
            super().__init__(input_spec)
            self.body = torch.nn.Linear(6 * input_spec.window, hidden)
            self.head = GaussianHead(hidden, input_spec.dims)

        def forward(self, imu):
            return self.head(self.body(imu.flatten(1)))

    try:
        model = build_model(get_cfg({"model": {"name": "tiny", "arch": "tiny_test_model",
                                               "args": {"hidden": 4}}, "window": 10}))
        out = model(torch.randn(2, 6, 10))
        loss, items = model.loss(out, {"target": torch.zeros(2, 2)})
        assert set(items) == {"nll", "mse"} and torch.isfinite(loss)
    finally:
        MODELS.pop("tiny_test_model")


def test_input_spec_roundtrip():
    spec = InputSpec(window=100, dims=3, frame="gravity_yaw_local")
    again = InputSpec.from_dict(spec.to_dict())
    assert again == spec and again.num_channels == 6
    assert spec.to_dict()["channels"][0] == "gyro_x"
    assert InputSpec.from_cfg(get_cfg({"window": 50})).window == 50


def test_polar_head():
    torch.manual_seed(0)
    head = PolarHead(16, dims=2)
    x = torch.randn(32, 16, requires_grad=True)
    out = head(x)
    assert out["vel"].shape == (32, 2) and out["speed"].shape == (32,)
    assert torch.all(out["speed"] >= 0)
    torch.testing.assert_close(out["direction"].norm(dim=-1), torch.ones(32))
    torch.testing.assert_close(out["vel"], out["speed"][:, None] * out["direction"])
    torch.testing.assert_close(out["vel"].norm(dim=-1), out["speed"])
    out["vel"].sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    # 方向向量接近零时仍然数值稳定
    with torch.no_grad():
        head.direction.weight.zero_()
        head.direction.bias.zero_()
    zero = head(torch.randn(4, 16))
    assert torch.isfinite(zero["vel"]).all()
    # 3D 版本
    out3 = PolarHead(8, dims=3)(torch.randn(5, 8))
    assert out3["direction"].shape == (5, 3)


def test_velocity_and_gaussian_heads():
    assert VelocityHead(8, 3)(torch.randn(2, 8))["vel"].shape == (2, 3)
    out = GaussianHead(8, 2)(torch.randn(2, 8))
    assert set(out) == {"vel", "logstd"}


def test_gaussian_nll_value_and_clamp():
    pred = torch.tensor([[0.0, 1.0]])
    target = torch.tensor([[1.0, 1.0]])
    logstd = torch.tensor([[math.log(2.0), 0.0]])
    expected = ((1.0 / (2 * 4.0) + math.log(2.0)) + 0.0) / 2
    assert gaussian_nll(pred, logstd, target).item() == pytest.approx(expected)
    tiny = torch.full((1, 2), -50.0, requires_grad=True)
    value = gaussian_nll(pred, tiny, target)
    clamped = ((1.0 / (2 * math.exp(2 * MIN_LOGSTD)) + MIN_LOGSTD) + MIN_LOGSTD) / 2
    assert value.item() == pytest.approx(clamped)
    value.backward()
    assert torch.all(tiny.grad == 0)  # 裁剪区间外无梯度（与 TLIO 相同）


def test_mse_then_nll_schedule():
    loss_fn = MSEThenNLL(switch_epoch=2)
    logstd = torch.zeros(3, 2, requires_grad=True)
    out = {"vel": torch.ones(3, 2), "logstd": logstd}
    target = torch.zeros(3, 2)
    early, items = loss_fn(out, target, epoch=1)
    assert early.item() == pytest.approx(1.0) and set(items) == {"mse"}
    early.backward()
    assert torch.all(logstd.grad == 0)
    late, items = loss_fn(out, target, epoch=2)
    assert late.item() == pytest.approx(0.5) and "nll" in items
    assert loss_fn.stage(1) == "mse" and loss_fn.stage(2) == "nll"
    with pytest.raises(ValueError, match="unknown loss"):
        build_loss("huber")
