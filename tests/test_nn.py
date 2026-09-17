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
    mse,
    register_model,
)
from inertial_benchmark.nn.losses import (  # noqa: E402
    LOSSES,
    MIN_LOGSTD,
    DetachedNLLThenNLL,
    MSEThenNLL,
    expand_mask,
    masked_output_mean,
    register_loss,
)
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
    imu = torch.randn(4, 6, 200)
    out = model(imu)
    batch = {"target": torch.zeros(4, 2), "imu": imu}
    loss, items = model.loss(out, batch, epoch=0)
    assert loss.ndim == 0 and "mse" in items
    with pytest.raises(KeyError, match="logstd"):
        model.loss(out, batch, epoch=5)
    # 训练与验证必须传同样的键（至少 target 与 imu）
    with pytest.raises(KeyError, match="loss batch is missing"):
        model.loss(out, {"target": torch.zeros(4, 2)}, epoch=0)
    # 训练时的损失写入模型配置，从 checkpoint 重建时保持一致
    assert model.model_cfg["loss"] == "mse_then_nll"
    rebuilt = build_model(get_cfg({"model": "ronin_resnet18"}), model_cfg=model.model_cfg)
    assert rebuilt.loss_name == "mse_then_nll" and rebuilt.loss_kwargs == {"switch_epoch": 3}
    assert build_model(get_cfg({})).model_cfg["loss"] == "mse"


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
        imu = torch.randn(2, 6, 10)
        out = model(imu)
        loss, items = model.loss(out, {"target": torch.zeros(2, 2), "imu": imu})
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


def test_losses_support_per_frame_and_multi_step_targets_with_masks():
    """逐帧/多步目标：损失按掩码跳过无效的帧/步（DESIGN 第 3 节）。"""
    pred = torch.tensor([[[1.0, 0.0], [0.0, 0.0], [5.0, 0.0]]])  # (B=1, R=3, D=2)
    target = torch.zeros(1, 3, 2)
    assert mse(pred, target).item() == pytest.approx((1.0 + 25.0) / 6)
    mask = torch.tensor([[True, True, False]])
    assert mse(pred, target, mask).item() == pytest.approx(1.0 / 4)  # 第 3 帧不计入
    # 全部无效：损失为 0 但保留计算图（优化器不会拿不到梯度）
    grad_pred = pred.clone().requires_grad_(True)
    zero = mse(grad_pred, target, torch.zeros(1, 3, dtype=torch.bool))
    assert zero.item() == 0.0
    zero.backward()
    assert torch.all(grad_pred.grad == 0)
    # 高斯 NLL 与调度损失同样接受掩码
    logstd = torch.zeros(1, 3, 2)
    nll = build_loss("gaussian_nll")({"vel": pred, "logstd": logstd}, target, 0, mask)[0]
    assert nll.item() == pytest.approx(0.5 * (1.0 / 4))
    scheduled, items = build_loss("mse_then_nll", switch_epoch=2)(
        {"vel": pred, "logstd": logstd}, target, 0, mask)
    assert scheduled.item() == pytest.approx(0.25) and set(items) == {"mse"}
    assert expand_mask(mask, pred).shape == pred.shape
    assert expand_mask(None, pred) is None


def test_base_model_passes_the_mask_through_the_loss_batch():
    @register_model("test_mask_model")
    class Masked(BaseModel):
        def __init__(self, input_spec):
            super().__init__(input_spec)
            self.bias = torch.nn.Parameter(torch.zeros(input_spec.output_shape))

        def forward(self, imu):
            return {"vel": self.bias.expand(imu.shape[0], *self.bias.shape)}

    try:
        cfg = get_cfg({"model": {"name": "masked", "arch": "test_mask_model"},
                       "window": 4, "target": "frame_velocity", "rate": 4.0})
        model = build_model(cfg)
        assert model.input_spec.output_shape == (4, 2)
        imu = torch.zeros(2, 6, 4)
        out = model(imu)
        target = torch.ones(2, 4, 2)
        mask = torch.tensor([[True, True, False, False], [True, False, False, False]])
        loss, _ = model.loss(out, {"target": target, "imu": imu, "mask": mask})
        assert loss.item() == pytest.approx(1.0)
        with pytest.raises(ValueError, match="expected input"):
            model.check_input(torch.zeros(2, 6, 5))
    finally:
        MODELS.pop("test_mask_model")


def test_input_spec_declares_layouts_and_extra_inputs():
    spec = InputSpec(window=100, history=3, history_stride=50, dims=3,
                     target="multi_displacement", output_steps=5,
                     extra_inputs=("orientation", "init_velocity"))
    assert spec.input_shape == (3, 6, 100) and spec.output_shape == (5, 3)
    assert spec.input_span == 200 and spec.privileged_inputs == ("init_velocity",)
    assert InputSpec.from_dict(spec.to_dict()) == spec
    assert InputSpec(window=50).input_shape == (6, 50)


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


def test_sum_and_l1_losses():
    out = {"vel": torch.tensor([[1.0, 2.0], [0.0, -1.0]])}
    target = torch.zeros(2, 2)
    loss, items = build_loss("mse_sum")(out, target)
    assert loss.item() == pytest.approx((1 + 4 + 0 + 1) / 2)  # mean_b Σ_d
    assert items["mse"].item() == pytest.approx(6 / 4)
    loss, items = build_loss("mse_l1")(out, target)
    assert loss.item() == pytest.approx(6 / 4 + 4 / 4)
    assert set(items) == {"mse", "l1"}
    loss, _ = build_loss("mse_l1", mse_weight=0.0, l1_weight=2.0)(out, target)
    assert loss.item() == pytest.approx(2.0)


def test_detached_nll_schedule():
    loss_fn = DetachedNLLThenNLL(switch_epoch=9)
    target = torch.zeros(4, 3)
    for epoch, logstd_grad in ((8, False), (9, True)):
        vel = torch.ones(4, 3, requires_grad=True)
        logstd = torch.full((4, 3), 0.3, requires_grad=True)
        loss, items = loss_fn({"vel": vel, "logstd": logstd}, target, epoch)
        # 两个阶段的数值相同（detach 不改变前向）
        expected = gaussian_nll(vel, logstd, target, MIN_LOGSTD, None).item()
        assert loss.item() == pytest.approx(expected) and set(items) == {"nll", "mse"}
        loss.backward()
        assert vel.grad.abs().sum() > 0
        assert bool(logstd.grad.abs().sum() > 0) is logstd_grad
    assert loss_fn.stage(8) == "nll_detached" and loss_fn.stage(9) == "nll"
    # 缺省只有下限裁剪：大 logstd 不被截断
    big = torch.full((1, 3), 10.0)
    value, _ = loss_fn({"vel": torch.zeros(1, 3), "logstd": big}, torch.zeros(1, 3), 20)
    assert value.item() == pytest.approx(10.0)
    with pytest.raises(KeyError, match="logstd"):
        loss_fn({"vel": torch.zeros(1, 3)}, torch.zeros(1, 3), 0)
    cfg = get_cfg({"model": "ronin_resnet18", "loss": "nll_detach_then_nll",
                   "loss_switch_epoch": 4})
    assert build_model(cfg).loss_kwargs == {"switch_epoch": 4}


def test_register_loss():
    @register_loss("test_tmp_loss")
    class Tmp:
        def __call__(self, out, target, epoch=0, mask=None):
            return masked_output_mean(out["vel"].sum(dim=-1), mask), {}

    try:
        fn = build_loss("test_tmp_loss")
        assert fn.name == "test_tmp_loss"
        # 注册的专用损失按四参数签名调用，并按掩码跳过无效输出
        value, _ = fn({"vel": torch.ones(2, 2)}, torch.zeros(2, 2), 0,
                      torch.tensor([[True], [False]]))
        assert value.item() == pytest.approx(2.0)
        assert build_loss("test_tmp_loss").name == "test_tmp_loss"
        with pytest.raises(KeyError, match="already registered"):
            register_loss("test_tmp_loss")(type("Other", (), {}))
    finally:
        LOSSES.pop("test_tmp_loss")
