"""AirIO 移植测试：夹具参数量（含未参与前向的注册参数）、标签取样、通道置换与协方差损失。"""

import math

import pytest

from .model_testing import load_fixture, param_shapes, run_end_to_end, shape_multiset

torch = pytest.importorskip("torch")

from inertial_benchmark.cfg import get_cfg  # noqa: E402
from inertial_benchmark.models.airio.model import COV_LOG_OFFSET, CNNEncoder  # noqa: E402
from inertial_benchmark.nn import build_model  # noqa: E402
from inertial_benchmark.nn.losses import build_loss  # noqa: E402

FIXTURE = load_fixture("airio")
BATCH = 4
WINDOW = 1000


def build(**overrides):
    return build_model(get_cfg({"model": "airio", **overrides}))


def inputs(model, batch: int = BATCH, window: int = WINDOW) -> tuple:
    torch.manual_seed(0)
    return torch.randn(batch, 6, window), model.input_spec.dummy_extra(batch)


def test_parameters_match_official_fixture():
    model = build()
    assert model.num_params == FIXTURE["total_params"] == 387_014
    names = [n for n, _ in model.named_parameters()]
    assert names == FIXTURE["param_names"]      # 注册顺序与官方逐项一致
    shapes = param_shapes(model)
    assert shape_multiset(shapes) == shape_multiset(FIXTURE["param_shapes"])
    assert shapes == FIXTURE["param_shapes"] and len(shapes) == 56
    buffers = [list(b.shape) for _, b in model.named_buffers()]
    assert buffers == FIXTURE["buffer_shapes"] and len(buffers) == 24
    for component, count in FIXTURE["component_params"].items():
        assert sum(p.numel() for p in getattr(model, component).parameters()) == count


def test_unused_registered_parameters_get_no_gradient():
    """``cnn``/``fcn1``/``batchnorm1`` 计入参数量但不参与前向（夹具逐项锁定，卡 §4）。"""
    model = build()
    x, extra = inputs(model)
    model.train()
    model.zero_grad(set_to_none=True)
    out = model(x, extra)
    loss, _ = model.loss(out, {"target": torch.randn(BATCH, 3), "imu": x})
    loss.backward()
    unused = {n for n, p in model.named_parameters() if p.grad is None}
    assert unused == set(FIXTURE["params_without_grad_in_forward"])
    assert sum(p.numel() for n, p in model.named_parameters() if n in unused) \
        == sum(FIXTURE["unused_params"].values()) == 32_736
    effective = model.num_params - sum(FIXTURE["unused_params"].values())
    assert effective == FIXTURE["effective_params"] == 354_278
    assert all(torch.isfinite(p.grad).all()
               for n, p in model.named_parameters() if n not in unused)


def test_body_only_variant_parameter_count():
    """论文 “Body” 消融（``CodeNetMotion``，无姿态支路）为 330,598（夹具 variants）。"""
    model = build()
    variants = FIXTURE["variant_total_params"]
    body_only = sum(sum(p.numel() for p in getattr(model, k).parameters())
                    for k in variants["codenetmotion components"])
    assert body_only == variants[
        "codenetmotion (Body, no attitude encoding; configs/*/motion_body.conf)"] == 330_598


@pytest.mark.parametrize("window,length", [(200, 23), (1000, 112), (1001, 112)])
def test_sequence_output_length(window, length):
    assert CNNEncoder.output_length(window) == length
    assert FIXTURE["output_length_by_T"][str(window)] == [1, length, 3]
    model = build(window=window).eval()
    x, extra = inputs(model, 2, window)
    with torch.no_grad():
        velocity, variance = model.sequence(x, extra)
    assert velocity.shape == (2, length, 3) and variance.shape == (2, length, 3)
    assert model.feature_length == length
    assert model.num_params == FIXTURE["total_params"]      # 参数量与窗口无关


def test_label_indices_follow_the_official_rule():
    """``get_label``：下标 ``14 + 9j``，不足 ``T_out`` 时用末端补齐（规格卡 §3、§8）。"""
    model = build()
    short = model.label_indices(201)
    assert short[:3] == [14, 23, 32] and short[-4:] == [185, 194, 200, 200] and len(short) == 23
    long = model.label_indices(1001)
    assert long[:3] == [14, 23, 32] and long[-4:] == [986, 995, 1000, 1000] and len(long) == 112
    # IPB 的窗口只有 T 个位姿样本，补齐改用 T−1（末端相差一个样本）
    ipb = model.label_indices(1000)
    assert ipb[-1] == 999 and len(ipb) == 112


def test_covariance_is_a_variance_and_window_output_is_the_last_token():
    """``cov = exp(raw − 5)`` 是逐轴**方差**（不是 logstd）；窗口级输出取最后一个 token。"""
    model = build().eval()
    x, extra = inputs(model)
    with torch.no_grad():
        out = model(x, extra)
        velocity, variance = model.sequence(x, extra)
    assert torch.all(out["variance"] > 0) and torch.all(variance > 0)
    torch.testing.assert_close(out["vel"], velocity[:, -1], atol=0, rtol=0)
    torch.testing.assert_close(out["variance"], variance[:, -1], atol=0, rtol=0)
    torch.testing.assert_close(out["logstd"], 0.5 * torch.log(out["variance"]),
                               atol=1e-6, rtol=0)
    # 随机初始化时 cov 的量级约 e^{-5}（解码器输出接近 0）
    assert abs(math.log(float(variance.mean())) + COV_LOG_OFFSET) < 1.0


def test_channel_permutation_to_the_official_order():
    """IPB 的 ``[gyro, acc]`` 在模型内部置换为官方的 ``[acc, gyro]``：交换后输出必须改变。"""
    model = build().eval()
    x, extra = inputs(model)
    swapped = torch.cat([x[:, 3:6], x[:, 0:3]], dim=1)
    with torch.no_grad():
        base, other = model(x, extra)["vel"], model(swapped, extra)["vel"]
        encoded = model.feature_encoder(swapped)
    assert float((base - other).abs().max()) > 1e-6
    # feature_encoder 收到的确实是 [acc, gyro]
    with torch.no_grad():
        features = model.feature_encoder(torch.cat([x[:, 3:6], x[:, 0:3]], dim=1))
    torch.testing.assert_close(features, encoded, atol=0, rtol=0)


def test_attitude_channels_are_so3_logarithms():
    model = build().eval()
    quat = torch.zeros(2, 8, 4, dtype=torch.float64)
    quat[..., 0] = 1.0                                    # 单位四元数 → 零旋转向量
    torch.testing.assert_close(model.attitude_channels({"orientation": quat}, torch.float64),
                               torch.zeros(2, 3, 8, dtype=torch.float64), atol=1e-12, rtol=0)
    angle = 0.9
    quat[..., 0] = math.cos(angle / 2)
    quat[..., 3] = math.sin(angle / 2)
    channels = model.attitude_channels({"orientation": quat}, torch.float64)
    assert channels.shape == (2, 3, 8)
    torch.testing.assert_close(channels[:, 2], torch.full((2, 8), angle, dtype=torch.float64),
                               atol=1e-9, rtol=0)


def test_loss_weights_and_covariance_gradient():
    loss_fn = build_loss("airio_huber_cov")
    assert loss_fn.weight == 1.0e2 and loss_fn.cov_weight == 1.0e-4
    assert loss_fn.delta == 0.005 and loss_fn.covaug is True
    target = torch.randn(BATCH, 3)
    # covaug=True：协方差项对 vel 有梯度
    pred = torch.randn(BATCH, 3, requires_grad=True)
    logstd = torch.zeros(BATCH, 3, requires_grad=True)
    only_cov = build_loss("airio_huber_cov", weight=1.0, cov_weight=1.0)
    value, _ = only_cov({"vel": pred, "logstd": logstd}, target, 0)
    value.backward()
    assert pred.grad is not None and float(pred.grad.abs().max()) > 0
    assert logstd.grad is not None and float(logstd.grad.abs().max()) > 0
    # covaug=False：协方差项的残差被 detach，速度头只从 Huber 项收梯度
    detached = build_loss("airio_huber_cov", weight=1.0, cov_weight=1.0, covaug=False)
    pred2 = torch.randn(BATCH, 3, requires_grad=True)
    logstd2 = torch.zeros(BATCH, 3, requires_grad=True)
    value, items = detached({"vel": pred2, "logstd": logstd2}, target, 0)
    assert set(items) == {"huber", "cov", "mse"}
    value.backward()
    assert float(logstd2.grad.abs().max()) > 0
    huber_only = build_loss("airio_huber_cov", weight=1.0, cov_weight=0.0)
    pred3 = pred2.detach().clone().requires_grad_(True)
    reference, _ = huber_only({"vel": pred3, "logstd": torch.zeros(BATCH, 3)}, target, 0)
    reference.backward()
    torch.testing.assert_close(pred2.grad, pred3.grad, atol=1e-6, rtol=1e-6)
    # 没有阶段切换：任意 epoch 的损失相同
    out = {"vel": torch.randn(BATCH, 3), "logstd": torch.randn(BATCH, 3)}
    early, _ = loss_fn(out, target, epoch=0)
    late, _ = loss_fn(out, target, epoch=99)
    torch.testing.assert_close(early, late, atol=0, rtol=0)


def test_loss_masks_invalid_outputs():
    model = build()
    x, extra = inputs(model)
    out = model(x, extra)
    mask = torch.zeros(BATCH, 1)
    loss, _ = model.loss(out, {"target": torch.randn(BATCH, 3), "imu": x, "mask": mask})
    assert float(loss.detach()) == 0.0


def test_determinism_in_eval_mode():
    model = build().eval()
    x, extra = inputs(model)
    with torch.no_grad():
        first, second = model(x, extra)["vel"], model(x, extra)["vel"]
    torch.testing.assert_close(first, second, atol=0, rtol=0)


def test_configuration_constraints():
    with pytest.raises(ValueError, match="extra_inputs"):
        build(extra_inputs=[])
    with pytest.raises(ValueError, match="body frame"):
        build(frame="gravity_world", dims=3)
    cfg = get_cfg({"model": "airio", "recipe": "official"})
    assert cfg.epochs == 100 and cfg.batch == 128 and cfg.lr == 1e-3
    assert cfg.optimizer == "adam" and cfg.scheduler == "plateau" and cfg.gamma == 0.2
    assert cfg.frame == "body" and cfg.dims == 3 and cfg.window == 1000
    assert cfg.target == "velocity_at_end" and cfg.extra_inputs == ["orientation"]
    # 增强两个配方一致（DESIGN §4）；姿态编码含绝对偏航，body 系下不能套用 random_yaw
    assert cfg.augment == [] == get_cfg({"model": "airio"}).augment
    assert build().input_spec.privileged_inputs == ()


@pytest.mark.slow
def test_end_to_end_train_and_val(synthetic_dataset, tmp_path):
    run_end_to_end("airio", synthetic_dataset, tmp_path)
