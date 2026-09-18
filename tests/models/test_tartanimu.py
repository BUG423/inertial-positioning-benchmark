"""TartanIMU 移植测试：参数量/形状锁定、子窗抽取与时间对齐、多头合成、official/unified 阶段差异。"""

import json
import math

import numpy as np
import pytest

from .model_testing import (
    assert_deterministic_eval,
    load_fixture,
    param_shapes,
    run_end_to_end,
    shape_multiset,
)

torch = pytest.importorskip("torch")

from inertial_benchmark.cfg import get_cfg  # noqa: E402
from inertial_benchmark.data.augment import build_augmentations  # noqa: E402
from inertial_benchmark.data.convert import convert_dataset  # noqa: E402
from inertial_benchmark.models.tartanimu.model import HEAD_ORDER  # noqa: E402
from inertial_benchmark.nn import build_model  # noqa: E402
from inertial_benchmark.nn.losses import build_loss  # noqa: E402

FIXTURE = load_fixture("tartanimu")
BENCH = FIXTURE["benchmark_config"]


def build(**overrides):
    return build_model(get_cfg({"model": "tartanimu", **overrides}))


def bn_buffers(model):
    """夹具的 ``buffer_shapes`` 只含 BatchNorm 缓冲区；本实现另有一个非持久化的段时长缓冲。"""
    return [list(b.shape) for name, b in model.named_buffers() if "step_durations" not in name]


@pytest.fixture(scope="module")
def long_dataset(tmp_path_factory):
    """40 s 的合成数据集：TartanIMU 的输入跨度是 10 s。"""
    from pathlib import Path

    root = tmp_path_factory.mktemp("tartanimu_data")
    entries = [{"id": f"tr{k}", "seed": k, "group": f"g{k}", "split": "train",
                "duration": 40.0, "imu_rate": 200.0} for k in range(4)]
    entries.append({"id": "te0", "seed": 50, "group": "gx", "split": "test",
                    "duration": 40.0, "imu_rate": 200.0})
    (root / "raw").mkdir(parents=True)
    (root / "raw" / "spec.json").write_text(json.dumps({"sequences": entries}))
    convert_dataset("fake", root / "raw", root / "fake",
                    converter=Path(__file__).resolve().parents[1] / "fake_converter.py",
                    val_fraction=0.25)
    return root / "fake"


def test_parameters_match_official_fixture():
    model = build()
    assert model.num_params == FIXTURE["total_params"] == 5_698_346
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) \
        == FIXTURE["trainable_params"]
    shapes = param_shapes(model)
    assert shape_multiset(shapes) == shape_multiset(FIXTURE["param_shapes"])
    assert shapes == FIXTURE["param_shapes"] and len(shapes) == 159
    assert bn_buffers(model) == FIXTURE["buffer_shapes"]
    assert list(model.heads) == list(HEAD_ORDER) == ["dog", "human", "car", "drone"]


def test_parameter_breakdown():
    model = build()
    trunk = sum(p.numel() for p in model.model.parameters())
    unused = sum(p.numel() for p in model.model.output_block1.parameters())
    unused += sum(p.numel() for p in model.model.output_block2.parameters())
    human = sum(p.numel() for p in model.heads["human"].parameters())
    assert unused == BENCH["unused_trunk_fcblock_params"] == 199_174
    assert trunk + human == BENCH["trunk_plus_human_head_params"] == 4_804_367
    assert trunk + human - unused == BENCH["human_forward_params"] == 4_605_193
    assert list(model.model.lstm.weight_ih_l0.shape) == [512, 384]   # resnet_code = 128×3
    assert model.model.lstm.num_layers == 2 and model.model.lstm.bidirectional is False


def test_step_layout_shapes_and_determinism():
    model = build(recipe="official").eval()
    spec = model.input_spec
    assert spec.output_layout == "steps" and spec.output_shape == (10, 3)
    assert spec.frame == "body" and spec.dims == 3 and spec.remove_gravity is False
    x = torch.randn(2, 6, 2000)
    out = model(x)
    assert set(out) == {"vel"} and out["vel"].shape == (2, 10, 3)
    assert tuple(FIXTURE["output_shape"]["human"]) == (1, 10, 3)
    assert_deterministic_eval(model, x)
    cov = build(recipe="unified").eval()(x)
    assert cov["logstd"].shape == (2, 10, 3)
    assert tuple(FIXTURE["cov_output_shape"]["human"]) == (1, 10, 3)


def test_decimation_matches_manual_slicing():
    model = build()
    x = torch.randn(3, 6, 2000)
    split = model.split_sub_windows(x)
    assert split.shape == (3, 10, 6, 40)
    for k in range(10):
        torch.testing.assert_close(split[:, k], x[:, :, 200 * k:200 * k + 200:5], rtol=0, atol=0)


def test_sub_window_length_outside_the_supported_range_is_rejected():
    """抽取后长度 ∉ [33, 48] 时 ResNet 展平宽度改变，必须报错（规格卡 §4）。"""
    model = build()
    bad = torch.randn(1, 10, 6, 64)
    with pytest.raises(ValueError, match="flattened ResNet code"):
        model.model.encode(bad)


def test_sub_window_outputs_are_causal_across_the_lstm():
    """单向 LSTM：改动第 k 个子窗之后的输入不影响前 k 个子窗的输出（规格卡 §8）。"""
    model = build(recipe="official").eval()
    x = torch.randn(1, 6, 2000)
    with torch.no_grad():
        base = model(x)["vel"]
        tail = x.clone()
        tail[:, :, 1000:] += 3.0
        changed = model(tail)["vel"]
    torch.testing.assert_close(changed[:, :5], base[:, :5], rtol=1e-5, atol=1e-6)
    assert not torch.allclose(changed[:, 5:], base[:, 5:])


def test_head_composition_split_z_and_velocity_scale():
    model = build(recipe="unified").eval()
    head = model.heads["human"]
    x = torch.randn(2, 6, 2000)
    with torch.no_grad():
        base = model(x)
        # z 分支置零 ⇒ z 分量恒为 0（split_z 的合成方式）
        head.output_block1_z.fcs[6].weight.zero_()
        head.output_block1_z.fcs[6].bias.zero_()
        zeroed = model(x)
        assert torch.count_nonzero(zeroed["vel"][..., 2]) == 0
        # velocity_scale 只缩放均值、不影响 log σ
        head.velocity_scale.copy_(torch.tensor([[2.0, 1.0, 1.0]]))
        scaled = model(x)
    torch.testing.assert_close(scaled["vel"][..., 0], 2.0 * zeroed["vel"][..., 0],
                               rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(scaled["logstd"], base["logstd"], rtol=1e-6, atol=1e-7)


def test_step_durations_convert_velocity_to_displacement():
    """官方 ``pred *= window_time``、``log σ += log(window_time)``，这里按每段自己的时长换算。"""
    model = build(recipe="unified").eval()
    durations = model.step_durations.flatten()
    assert durations.shape == (10,)
    assert durations[0].item() == pytest.approx(1.0) and durations[5].item() == pytest.approx(
        0.995)
    bounds = model.input_spec.step_bounds
    np.testing.assert_allclose(durations.numpy(),
                               (bounds[:, 1] - bounds[:, 0]) * model.input_spec.dt, rtol=1e-6)
    x = torch.randn(2, 6, 2000)
    with torch.no_grad():
        feat = model.model.encode(model.split_sub_windows(x))
        mean, logstd = model.heads["human"](feat, True)
        out = model(x)
    torch.testing.assert_close(out["vel"], mean * model.step_durations, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(out["logstd"], logstd + torch.log(model.step_durations),
                               rtol=1e-6, atol=1e-7)


def test_official_loss_is_epoch_independent_and_leaves_the_cov_head_without_gradient():
    """规格卡 §10.1：body 分支下损失恒为 20·L1，协方差头拿不到梯度。"""
    model = build(recipe="official").eval()   # 关掉 dropout，两次前向才可逐位比较
    assert model.predict_cov is False and model.loss_name == "tartanimu_velocity"
    x = torch.randn(2, 6, 2000)
    target = torch.randn(2, 10, 3)
    mask = torch.ones(2, 10, dtype=torch.bool)
    values = []
    for epoch in (1, 300):
        model.zero_grad(set_to_none=True)
        out = model(x)
        assert "logstd" not in out
        loss, items = model.loss(out, {"target": target, "imu": x, "mask": mask}, epoch)
        loss.backward()
        values.append(loss.item())
        assert all(p.grad is None for p in model.heads["human"].output_block2.parameters())
        assert "nll" not in items
    assert values[0] == pytest.approx(values[1])
    # 其余三个平台头在行人训练中同样没有梯度
    assert all(p.grad is None for name in ("dog", "car", "drone")
               for p in model.heads[name].parameters())


def test_unified_loss_ramps_in_the_nll_and_trains_the_cov_head():
    model = build(recipe="unified")
    assert model.predict_cov is True
    x = torch.randn(2, 6, 2000)
    target = torch.randn(2, 10, 3)
    mask = torch.ones(2, 10, dtype=torch.bool)
    out = model(x)
    early, items_early = model.loss(out, {"target": target, "imu": x, "mask": mask}, 0)
    late, items_late = model.loss(out, {"target": target, "imu": x, "mask": mask}, 300)
    assert float(items_early["cov_weight"]) == pytest.approx(0.2)   # 5 轮线性引入
    assert float(items_late["cov_weight"]) == pytest.approx(1.0)
    assert early.item() != pytest.approx(late.item())
    model.zero_grad(set_to_none=True)
    late.backward()
    assert model.heads["human"].output_block2.fcs[6].weight.grad.abs().sum() > 0


def test_l1_weight_and_mask_behaviour():
    loss_fn = build_loss("tartanimu_velocity", l1_weight=20.0)
    target = torch.zeros(1, 4, 3)
    pred = torch.ones(1, 4, 3, requires_grad=True)
    value, items = loss_fn({"vel": pred}, target, 0, None)
    assert value.item() == pytest.approx(20.0) and float(items["l1"]) == pytest.approx(1.0)
    mask = torch.tensor([[True, True, False, False]])
    with torch.no_grad():
        pred[0, 2:] = 9.0
    partial, _ = loss_fn({"vel": pred}, target, 0, mask)
    assert partial.item() == pytest.approx(20.0)   # 无效步不计入
    empty, _ = loss_fn({"vel": pred}, target, 0, torch.zeros(1, 4, dtype=torch.bool))
    assert empty.item() == 0.0 and empty.requires_grad
    empty.backward()
    assert pred.grad is not None and torch.count_nonzero(pred.grad) == 0


def test_window_level_layout_is_the_degraded_fallback():
    """规格卡 §6 方案 A/B：窗口级目标时取最后一个子窗；``window=200`` 即 ``@seq1``。"""
    model = build(target="avg_velocity", output_steps=0, recipe="unified").eval()
    assert model.num_params == FIXTURE["total_params"]
    x = torch.randn(2, 6, 2000)
    out = model(x)
    assert out["vel"].shape == (2, 3) and out["seq_vel"].shape == (2, 10, 3)
    torch.testing.assert_close(out["vel"], out["seq_vel"][:, -1])
    seq1 = build(window=200, target="avg_velocity", output_steps=0).eval()
    assert seq1.num_params == FIXTURE["total_params"] and seq1.sub_windows == 1
    assert seq1(torch.randn(3, 6, 200))["vel"].shape == (3, 3)


def test_lstm_initialisation():
    lstm = build().model.lstm
    hidden = lstm.hidden_size
    for layer in (0, 1):
        bias_hh = getattr(lstm, f"bias_hh_l{layer}")
        assert torch.all(bias_hh[hidden:2 * hidden] == 1.0)
        assert torch.all(bias_hh[:hidden] == 0.0) and torch.all(bias_hh[2 * hidden:] == 0.0)
        assert torch.all(getattr(lstm, f"bias_ih_l{layer}") == 0.0)
        weight = getattr(lstm, f"weight_hh_l{layer}")
        # 正交初始化：列向量近似标准正交
        gram = weight.T @ weight
        torch.testing.assert_close(gram, torch.eye(hidden), rtol=1e-4, atol=1e-4)


def test_body_frame_augmentations_preserve_norms():
    official = get_cfg({"model": "tartanimu", "recipe": "official"})
    unified = get_cfg({"model": "tartanimu", "recipe": "unified"})
    assert official.augment == unified.augment      # 增强在两个配方中一致（DESIGN §4）
    _, augs = build_augmentations(official.augment, official.frame)
    assert [a.name for a in augs] == ["bias_shift", "noise", "body_yaw", "body_tilt"]
    assert augs[0].gyro == 0.002 and augs[0].acc == 0.1
    assert augs[1].gyro == 1e-5 and augs[1].acc == 1e-4 and augs[3].max_deg == 5.0
    rng = np.random.default_rng(0)
    imu = rng.normal(size=(6, 2000)).astype(np.float32)
    target = rng.normal(size=(10, 3)).astype(np.float32)
    for aug in (augs[2], augs[3]):
        sample = {"imu": imu.copy(), "target": target.copy()}
        aug(sample, np.random.default_rng(1))
        for channels in (slice(0, 3), slice(3, 6)):
            np.testing.assert_allclose(np.linalg.norm(sample["imu"][channels], axis=0),
                                       np.linalg.norm(imu[channels], axis=0), rtol=1e-4)
        np.testing.assert_allclose(np.linalg.norm(sample["target"], axis=-1),
                                   np.linalg.norm(target, axis=-1), rtol=1e-4)
        assert not np.allclose(sample["target"], target)


def test_recipes_official_and_unified():
    official = get_cfg({"model": "tartanimu", "recipe": "official"})
    unified = get_cfg({"model": "tartanimu", "recipe": "unified"})
    assert official.batch == 512 and official.lr == pytest.approx(1e-4)
    assert official.weight_decay == pytest.approx(0.01) and official.optimizer == "adam"
    assert official.scheduler == "plateau" and official.gamma == 0.1
    assert official.plateau_patience == 8 and official.epochs == 250
    assert official.amp is True and official.stride == 5 and official.fitness == "loss"
    assert unified.model_args == {"predict_cov": True}


@pytest.mark.slow
@pytest.mark.parametrize("overlap", ["mean", "center"])
def test_end_to_end_train_and_val(long_dataset, tmp_path, overlap):
    metrics = run_end_to_end("tartanimu", long_dataset, tmp_path / overlap, overlap=overlap,
                             stride=200, eval_stride=200, batch=4)
    assert math.isfinite(metrics["ate"]) and math.isfinite(metrics["vel_rmse"])
