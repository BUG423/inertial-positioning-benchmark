"""TLIO 移植测试：参数量/形状锁定、两个输出头、NLL 阶段切换、位移↔速度换算与端到端。"""

import math

import numpy as np
import pytest
from synthetic import make_sequence

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
from inertial_benchmark.data.views import SequenceView, ViewConfig  # noqa: E402
from inertial_benchmark.nn import build_model  # noqa: E402
from inertial_benchmark.nn.losses import MIN_LOGSTD, gaussian_nll  # noqa: E402

FIXTURE = load_fixture("tlio")


def build(**overrides):
    return build_model(get_cfg({"model": "tlio", **overrides}))


def test_parameters_match_official_fixture():
    model = build()
    assert model.num_params == FIXTURE["total_params"] == 5_424_646
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) \
        == FIXTURE["trainable_params"]
    shapes = param_shapes(model)
    assert shape_multiset(shapes) == shape_multiset(FIXTURE["param_shapes"])
    assert shapes == FIXTURE["param_shapes"]  # 注册顺序也一致
    assert [list(b.shape) for _, b in model.named_buffers()] == FIXTURE["buffer_shapes"]


def test_output_fields_and_layer_shapes():
    model = build().eval()
    x = torch.randn(2, 6, 200)
    out = model(x)
    assert set(out) == {"vel", "logstd"}
    assert out["vel"].shape == (2, 3) and out["logstd"].shape == (2, 3)
    assert tuple(FIXTURE["output_shape"]["mean"]) == (2, 3)
    # 时间维 100 → 50 → 50 → 25 → 13 → 7（规格卡 §8）
    lengths = [model.backbone.stem(x).shape[-1]]
    feat = model.backbone.stem(x)
    for stage in model.backbone.stages:
        feat = stage(feat)
        lengths.append(feat.shape[-1])
    assert lengths == [50, 50, 25, 13, 7]
    assert model.feature_length == 7 and feat.shape[1] == 512
    # 两个头不共享参数
    assert model.mean_head.fc3.weight is not model.logstd_head.fc3.weight
    assert_deterministic_eval(model, x)
    assert_backprop(build(), x, torch.randn(2, 3))


@pytest.mark.parametrize("window,expected", [(100, 5_031_430), (400, 6_211_078)])
def test_parameter_count_follows_window(window, expected):
    assert build(window=window).num_params == expected


def test_dims2_variant_matches_fixture():
    assert build(dims=2).num_params == FIXTURE["variants"]["dims2_output"]["total_params"]


def test_loss_stage_switch():
    """官方：epoch（1 起）< 10 时 NLL 的 logstd 被 detach → IPB 的 switch_epoch=9。"""
    model = build()
    assert model.loss_name == "nll_detach_then_nll" and model.loss_kwargs["switch_epoch"] == 9
    target = torch.randn(4, 3)
    grads = {}
    for epoch in (8, 9):
        model.zero_grad(set_to_none=True)
        out = model(torch.randn(4, 6, 200))
        loss, items = model.loss(out, {"target": target}, epoch)
        loss.backward()
        grads[epoch] = (float(model.mean_head.fc3.weight.grad.abs().sum()),
                        float(model.logstd_head.fc3.weight.grad.abs().sum()))
        assert set(items) == {"nll", "mse"}
    assert grads[8][0] > 0 and grads[8][1] == 0     # 第一阶段协方差头无梯度
    assert grads[9][0] > 0 and grads[9][1] > 0      # 第二阶段两头都有梯度


def test_logstd_lower_bound_and_no_upper_bound():
    model = build()
    loss_fn = model.loss
    out = {"vel": torch.zeros(1, 3), "logstd": torch.full((1, 3), -10.0)}
    value, _ = loss_fn(out, {"target": torch.zeros(1, 3)}, 100)
    assert value.item() == pytest.approx(MIN_LOGSTD)  # u' = log(1e-3)
    expected = gaussian_nll(out["vel"], torch.full((1, 3), MIN_LOGSTD),
                            torch.zeros(1, 3), None, None).item()
    assert value.item() == pytest.approx(expected)
    # 官方只截下限：logstd 很大时不被截断
    high, _ = loss_fn({"vel": torch.zeros(1, 3), "logstd": torch.full((1, 3), 8.0)},
                      {"target": torch.zeros(1, 3)}, 100)
    assert high.item() == pytest.approx(8.0)


def test_displacement_target_and_velocity_conversion():
    seq = make_sequence(duration=8.0, seed=2)
    disp_cfg = ViewConfig(window=200, frame="gravity_yaw_local", target="displacement", dims=3)
    vel_cfg = ViewConfig(window=200, frame="gravity_yaw_local", target="avg_velocity", dims=3)
    starts = np.arange(0, 1000, 100)
    disp = SequenceView(seq, disp_cfg).targets(starts)
    vel = SequenceView(seq, vel_cfg).targets(starts)
    span = 199 / 200.0
    assert disp_cfg.velocity_scale == pytest.approx(1.0 / span)
    np.testing.assert_allclose(disp, vel * span, rtol=1e-5)
    # 模型输出（位移）经视图换算即为窗口平均速度
    view = SequenceView(seq, disp_cfg)
    world = view.to_world_velocity(disp, starts)
    np.testing.assert_allclose(world, SequenceView(seq, vel_cfg).targets_world(starts)[:, :3],
                               rtol=1e-4, atol=1e-6)
    assert disp_cfg.target_offset == pytest.approx(0.5 * span)  # 时间戳在窗口中心


def test_official_augmentation_pipeline():
    cfg = get_cfg({"model": "tlio", "recipe": "official"})
    shift, augs = build_augmentations(cfg.augment, cfg.frame)
    assert [a.name for a in augs] == ["bias_shift", "gravity_perturb", "random_yaw"]
    assert shift.resolve_range(cfg.stride) == (0, 9)
    assert augs[0].gyro == 0.05 and augs[0].acc == 0.2 and augs[1].max_deg == 5.0
    assert cfg.grad_clip == 0.1 and cfg.scheduler == "none" and cfg.lr == 1e-4
    assert cfg.batch == 1024 and cfg.optimizer == "adam" and cfg.fitness == "loss"
    # 重力扰动不改变目标
    target = np.ones(3, dtype=np.float32)
    sample = {"imu": np.zeros((6, 200), np.float32), "target": target.copy()}
    augs[1](sample, np.random.default_rng(0))
    np.testing.assert_array_equal(sample["target"], target)


@pytest.mark.slow
def test_end_to_end_train_and_val(synthetic_dataset, tmp_path):
    metrics = run_end_to_end("tlio", synthetic_dataset, tmp_path)
    assert math.isfinite(metrics["rte"])
