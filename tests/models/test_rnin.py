"""RNIN 移植测试：参数量/形状锁定、10 步位移布局、子窗口时间对齐、official/unified 阶段差异。"""

import json
import math

import numpy as np
import pytest
from synthetic import make_sequence

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
from inertial_benchmark.data.views import SequenceView, ViewConfig  # noqa: E402
from inertial_benchmark.nn import build_model  # noqa: E402
from inertial_benchmark.nn.losses import build_loss  # noqa: E402

FIXTURE = load_fixture("rnin")
FAKE = "tests/fake_converter.py"


def build(**overrides):
    return build_model(get_cfg({"model": "rnin", **overrides}))


@pytest.fixture(scope="module")
def long_dataset(tmp_path_factory):
    """40 s 的合成数据集：RNIN 的输入跨度是 10 s，10 s 的默认夹具只够一个窗口。"""
    from pathlib import Path

    root = tmp_path_factory.mktemp("rnin_data")
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
    assert model.num_params == FIXTURE["total_params"] == 5_358_342
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) \
        == FIXTURE["trainable_params"]
    shapes = param_shapes(model.trunk)
    assert shape_multiset(shapes) == shape_multiset(FIXTURE["param_shapes"])
    assert shapes == FIXTURE["param_shapes"] and len(shapes) == 79
    assert [list(b.shape) for _, b in model.named_buffers()] == FIXTURE["buffer_shapes"]
    assert list(model.trunk.lstm.weight_ih_l0.shape) == [1024, 896]   # resnet_code = 128×7
    assert model.trunk.lstm.bidirectional is False and model.trunk.lstm.num_layers == 1
    cov_head = sum(p.numel() for p in model.trunk.output_block2.parameters())
    assert cov_head == FIXTURE["official_config"]["cov_head_params"] == 132_355


def test_native_200hz_subwindow_variant_is_detected():
    """误把子窗口当成 200 Hz 的 200 样本会变成 6,144,774 参数（规格卡 §8）。"""
    wrong = build(model_args={"decimate": 1})
    assert wrong.num_params == FIXTURE["variants"]["native_200hz_subwindow"]["total_params"]
    assert wrong.num_params == 6_144_774


def test_step_layout_shapes_and_determinism():
    model = build(recipe="official").eval()
    spec = model.input_spec
    assert spec.output_layout == "steps" and spec.output_shape == (10, 3)
    assert spec.window == 2000 and spec.remove_gravity is True and spec.dims == 3
    x = torch.randn(2, 6, 2000)
    out = model(x)
    assert set(out) == {"vel"} and out["vel"].shape == (2, 10, 3)   # official：不输出 logstd
    assert tuple(FIXTURE["output_shape"]["mean"]) == (2, 10, 3)
    assert_deterministic_eval(model, x)
    cov_out = build(recipe="unified").eval()(x)   # unified：均值与 log σ 同形
    assert cov_out["vel"].shape == (2, 10, 3) and cov_out["logstd"].shape == (2, 10, 3)
    assert tuple(FIXTURE["output_shape"]["logstd"]) == (2, 10, 3)


def test_trunk_accepts_any_sequence_length():
    trunk = build().trunk.eval()
    for steps in (1, 5, 10):
        assert trunk(torch.randn(2, steps, 6, 100))["mean"].shape == (2, steps, 3)


def test_sub_window_split_is_time_contiguous():
    """第 k 个子窗口恰为 ``x[:, :, 200k : 200k+200 : 2]``（规格卡 §8 的抽取/切分检查）。"""
    model = build()
    x = torch.randn(3, 6, 2000)
    split = model.split_sub_windows(x)
    assert split.shape == (3, 10, 6, 100)
    for k in range(10):
        torch.testing.assert_close(split[:, k], x[:, :, 200 * k:200 * k + 200:2], rtol=0, atol=0)
    # 子窗口在时间上连续、覆盖整个输入跨度且互不重叠
    starts = [200 * k for k in range(10)]
    assert starts == sorted(starts) and starts[-1] + 200 == x.shape[-1]


def test_history_alignment_and_invalid_spans(long_dataset):
    """输入跨度 = 窗口，有效性按整个 10 s 判断：跨无效区的窗口不参与训练与窗口级指标。"""
    from inertial_benchmark.data.format import load_sequence

    cfg = ViewConfig(window=2000, target="multi_displacement", output_steps=10, dims=3,
                     remove_gravity=True)
    assert cfg.input_span == 2000 and cfg.history_offset == 0
    seq = load_sequence(sorted((long_dataset / "sequences").glob("*.h5"))[0])
    view = SequenceView(seq, cfg)
    view.valid_input = view.valid_input.copy()
    view.valid_input[3000:3010] = False          # 在序列中间挖一个输入缺口
    view.valid = view.valid_input & view.valid_target
    starts = view.starts(200)
    assert len(starts) > 0
    # 任何包含缺口的窗口都被排除
    assert all(not (start <= 3000 < start + 2000) for start in starts)
    # 10 个输出的时间戳为各段中心，严格递增且落在窗口内
    times = view.output_times(np.array([starts[0]]))[0]
    assert times.shape == (10,) and np.all(np.diff(times) > 0)
    assert times[0] > seq.timestamp[starts[0]] and times[-1] < seq.timestamp[starts[0] + 2000]


def test_step_targets_sum_to_the_window_displacement():
    """等分段的位移之和严格等于窗口首末位移——绝对损失 AL 依赖这一“可伸缩求和”性质。"""
    seq = make_sequence(duration=30.0, seed=5)
    cfg = ViewConfig(window=2000, target="multi_displacement", output_steps=10, dims=3,
                     remove_gravity=True)
    view = SequenceView(seq, cfg)
    starts = np.array([0, 500, 1000])
    steps = view._targets_world_rows(starts)
    expected = seq.position[starts + 1999] - seq.position[starts]
    np.testing.assert_allclose(steps.sum(axis=1), expected, rtol=1e-6, atol=1e-9)
    bounds = cfg.step_bounds
    assert bounds[0].tolist() == [0, 200] and bounds[-1].tolist() == [1799, 1999]


def test_default_loss_value_matches_the_card():
    """``pred=0``、``targ=1``、``(1,10,3)`` 时 loss = (30 + 3·8·Σ_{m=2}^{10} m²)/57。"""
    loss_fn = build_loss("rnin_displacement", absolute_weight=8.0, start_cov_epoch=0)
    value, items = loss_fn({"vel": torch.zeros(1, 10, 3)}, torch.ones(1, 10, 3), 0, None)
    assert value.item() == pytest.approx(9246 / 57, rel=1e-6)
    assert float(items["stage"]) == 0.0


def test_official_recipe_never_trains_the_covariance_head():
    """规格卡 §10.1：官方默认配置下 ``output_block2`` 全程拿不到梯度、模型不输出 σ。"""
    model = build(recipe="official")
    assert model.predict_cov is False
    x = torch.randn(2, 6, 2000)
    for epoch in (0, 1, 2500):
        model.zero_grad(set_to_none=True)
        out = model(x)
        assert "logstd" not in out
        loss, items = model.loss(out, {"target": torch.randn(2, 10, 3), "imu": x,
                                       "mask": torch.ones(2, 10, dtype=torch.bool)}, epoch)
        loss.backward()
        assert all(p.grad is None for p in model.trunk.output_block2.parameters())
        assert model.trunk.output_block1.fcs[6].weight.grad.abs().sum() > 0
        assert float(items["stage"]) == 0.0


def test_unified_recipe_trains_the_covariance_head_with_cumulative_variance():
    model = build(recipe="unified")
    assert model.predict_cov is True and model.loss_kwargs["start_cov_epoch"] == 0
    x = torch.randn(2, 6, 2000)
    out = model(x)
    loss, items = model.loss(out, {"target": torch.randn(2, 10, 3), "imu": x,
                                   "mask": torch.ones(2, 10, dtype=torch.bool)}, 1)
    loss.backward()
    assert float(items["stage"]) == 1.0
    assert model.trunk.output_block2.fcs[6].weight.grad.abs().sum() > 0
    # AL 项按独立假设使用累积方差 cumsum(exp(2u))
    loss_fn = build_loss("rnin_displacement", absolute_weight=8.0, start_cov_epoch=0)
    pred, logstd = torch.zeros(1, 3, 2), torch.zeros(1, 3, 2)
    target = torch.ones(1, 3, 2)
    value, _ = loss_fn({"vel": pred, "logstd": logstd}, target, 0, None)
    var = torch.cumsum(torch.ones(3), 0)[1:]
    rel = (0.5 + 0.0) * 6
    absolute = 8.0 * ((torch.tensor([2.0, 3.0]) ** 2 / (2 * var) + 0.5 * torch.log(var)).sum() * 2)
    assert value.item() == pytest.approx(float((rel + absolute) / 10), rel=1e-5)


def test_mask_behaviour_on_steps():
    loss_fn = build_loss("rnin_displacement", absolute_weight=8.0, start_cov_epoch=0)
    target = torch.zeros(1, 4, 3)
    pred = torch.zeros(1, 4, 3, requires_grad=True)
    mask = torch.ones(1, 4, dtype=torch.bool)
    mask[0, 1] = False          # 第 1 步无效 ⇒ 其 RL 项与第 1..3 步的 AL 项都被跳过
    with torch.no_grad():
        pred[0, 1] = 5.0
    value, _ = loss_fn({"vel": pred}, target, 0, mask)
    assert value.item() == pytest.approx(0.0, abs=1e-10)
    empty, _ = loss_fn({"vel": pred}, target, 0, torch.zeros(1, 4, dtype=torch.bool))
    assert empty.item() == 0.0 and empty.requires_grad
    empty.backward()
    assert pred.grad is not None and torch.count_nonzero(pred.grad) == 0


def test_window_level_layout_is_the_degraded_fallback():
    """规格卡 §6 的降级映射：窗口级目标时只用最后一个子窗，AL 项消失。"""
    model = build(target="displacement", output_steps=0).eval()
    assert model.num_params == FIXTURE["total_params"]
    x = torch.randn(2, 6, 2000)
    out = model(x)
    assert out["vel"].shape == (2, 3) and out["seq_disp"].shape == (2, 10, 3)
    torch.testing.assert_close(out["vel"], out["seq_disp"][:, -1])
    loss, items = model.loss(out, {"target": torch.randn(2, 3), "imu": x,
                                   "mask": torch.ones(2, 1, dtype=torch.bool)}, 0)
    assert float(items["degraded"]) == 1.0 and math.isfinite(loss.item())


def test_recipes_official_and_unified():
    official = get_cfg({"model": "rnin", "recipe": "official"})
    unified = get_cfg({"model": "rnin", "recipe": "unified"})
    assert official.batch == 32 and official.lr == pytest.approx(1e-4)
    assert official.optimizer == "adam" and official.scheduler == "plateau"
    assert official.gamma == 0.1 and official.plateau_patience == 10 and official.seed == 42
    assert official.epochs == 200 and official.weight_decay == 0.0 and official.amp is False
    assert official.augment == unified.augment        # 增强在两个配方中一致（DESIGN §4）
    _, augs = build_augmentations(official.augment, official.frame)
    assert [a.name for a in augs] == ["bias_shift", "noise", "random_yaw", "gravity_perturb"]
    assert augs[0].gyro == 0.002 and augs[0].acc == 0.1
    assert augs[1].gyro == 1e-5 and augs[1].acc == 1e-4 and augs[3].max_deg == 5.0
    assert official.model_args == {"predict_cov": False} or official.model_args == {}
    assert unified.model_args == {"predict_cov": True}


@pytest.mark.slow
@pytest.mark.parametrize("overlap", ["mean", "center"])
def test_end_to_end_train_and_val(long_dataset, tmp_path, overlap):
    metrics = run_end_to_end("rnin", long_dataset, tmp_path / overlap, overlap=overlap,
                             stride=200, eval_stride=200, batch=4)
    assert math.isfinite(metrics["ate"]) and math.isfinite(metrics["rte"])
