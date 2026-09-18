"""RoNIN-LSTM 移植测试：参数量/形状锁定、逐帧布局、GlobalPosLoss、因果性与端到端。"""

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
from inertial_benchmark.nn.losses import build_loss  # noqa: E402

FIXTURE = load_fixture("ronin_lstm")


def build(**overrides):
    return build_model(get_cfg({"model": "ronin_lstm", **overrides}))


def test_parameters_match_official_fixture():
    model = build()
    assert model.num_params == FIXTURE["total_params"] == 216_620
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) \
        == FIXTURE["trainable_params"]
    shapes = param_shapes(model.net)
    assert shape_multiset(shapes) == shape_multiset(FIXTURE["param_shapes"])
    assert shapes == FIXTURE["param_shapes"]  # 注册顺序也一致（18 个张量）
    assert len(shapes) == 18
    assert [list(b.shape) for _, b in model.named_buffers()] == FIXTURE["buffer_shapes"] == []


def test_lstm_is_unidirectional_not_bidirectional():
    """规格卡 §1：官方 ``--type lstm_bi`` 的 ``bi`` 指 bilinear，LSTM 本身单向。"""
    lstm = build().net.lstm
    assert lstm.bidirectional is False and lstm.num_layers == 3 and lstm.hidden_size == 100
    for layer in range(3):
        assert list(getattr(lstm, f"weight_hh_l{layer}").shape) == [400, 100]
    assert list(lstm.weight_ih_l0.shape) == [400, 30]   # bilinear 混合后 6 + 24 = 30 通道
    assert build().net.bilinear.weight.shape == (24, 6, 6)


def test_frame_layout_shapes_and_backprop():
    model = build().eval()
    spec = model.input_spec
    assert spec.output_layout == "frame" and spec.output_shape == (400, 2)
    x = torch.randn(4, 6, 400)
    out = model(x)
    assert set(out) == {"vel"} and out["vel"].shape == (4, 400, 2)
    # 官方模块口径：batch-first (B, T, 6) → (B, T, 2)
    assert model.net(x.transpose(1, 2))[0].shape == tuple(FIXTURE["output_shape"])
    assert_deterministic_eval(model, x)
    assert_backprop(build(), x, torch.randn(4, 400, 2))


def test_window_layout_pools_frames():
    """配成窗口级目标时按规格卡 §6 取前 T−1 帧的均值（不改变参数）。"""
    model = build(target="avg_velocity").eval()
    assert model.num_params == FIXTURE["total_params"]
    x = torch.randn(3, 6, 400)
    out = model(x)
    assert out["vel"].shape == (3, 2) and out["frame_vel"].shape == (3, 400, 2)
    torch.testing.assert_close(out["vel"], out["frame_vel"][:, :-1].mean(dim=1))
    # 逐帧输出恒为常数 c 时窗口速度就是 c
    model.module_forward = lambda x: torch.full((x.shape[0], 400, 2), 2.5)
    torch.testing.assert_close(model(x)["vel"], torch.full((3, 2), 2.5))


def test_frame_pooling_equals_avg_velocity_on_a_synthetic_trajectory():
    """``mean_{k=0..T−2} (p[k+1]−p[k])/dt`` 恰为 IPB ``avg_velocity``（规格卡 §6 的推导）。"""
    seq = make_sequence(duration=6.0, seed=3)
    view = SequenceView(seq, ViewConfig(window=400, target="avg_velocity", dims=2))
    for start in (0, 150, 400):
        frames = seq.position[start + 1:start + 401] - seq.position[start:start + 400]
        pooled = (frames / (1.0 / 200.0))[:-1].mean(axis=0)[:2]
        np.testing.assert_allclose(pooled, view.targets(np.array([start]))[0], rtol=1e-5,
                                   atol=1e-6)


def test_causality_and_batch_independence():
    model = build().eval()
    x = torch.randn(2, 6, 400)
    with torch.no_grad():
        base = model(x)["vel"]
        perturbed = x.clone()
        perturbed[:, :, 300:] += 5.0
        assert torch.allclose(model(perturbed)["vel"][:, :300], base[:, :300], atol=0)
        assert not torch.allclose(model(perturbed)["vel"][:, 300:], base[:, 300:])
        # 隐藏状态不再绑定构造时的 batch（规格卡 §6/§10.2）
        big = torch.randn(7, 6, 400)
        big[3] = x[0]
        torch.testing.assert_close(model(big)["vel"][3], base[0], rtol=1e-5, atol=1e-6)


def test_forward_sequence_stream_hook():
    """``forward_sequence`` 的前 400 帧与窗口模式的第一个窗口一致（规格卡 §8）。"""
    model = build().eval()
    long = torch.randn(1, 6, 800)
    stream = model.forward_sequence(long)
    assert stream.shape == (1, 800, 2)
    with torch.no_grad():
        window = model(long[:, :, :400])["vel"]
    torch.testing.assert_close(stream[:, :400], window, rtol=1e-5, atol=1e-6)


def test_global_pos_loss_full_mode():
    model = build()
    assert model.loss_name == "ronin_global_pos" and model.loss_kwargs == {"mode": "full"}
    loss_fn = build_loss("ronin_global_pos", mode="full")
    target = torch.randn(2, 400, 2)
    zero, _ = loss_fn({"vel": target.clone()}, target, 0, None)
    assert zero.item() == pytest.approx(0.0, abs=1e-10)
    # pred = targ + δ ⇒ 前缀误差为 δ·j，损失为 δ²·mean_{j=1..T−1} j²
    delta = 0.25
    value, items = loss_fn({"vel": target + delta}, target, 0, None)
    j = torch.arange(1, 400, dtype=torch.float64)
    assert value.item() == pytest.approx(delta ** 2 * float((j ** 2).mean()), rel=1e-5)
    assert set(items) == {"global_pos", "mse"}


def test_global_pos_loss_mask_behaviour():
    """部分无效时只算有效项；全部无效时为 0 且保留计算图（DESIGN §3.1）。"""
    loss_fn = build_loss("ronin_global_pos", mode="full")
    target = torch.zeros(1, 6, 2)
    pred = torch.zeros(1, 6, 2, requires_grad=True)
    with torch.no_grad():
        pred[0, 4] = 10.0      # 只有第 4 帧有误差
    mask = torch.ones(1, 6, dtype=torch.bool)
    mask[0, 4] = False         # 该帧无效 → 误差被跳过，且含它的前缀项也无效
    value, _ = loss_fn({"vel": pred}, target, 0, mask)
    assert value.item() == pytest.approx(0.0, abs=1e-10)
    full, _ = loss_fn({"vel": pred}, target, 0, None)
    assert full.item() > 0.0
    empty, _ = loss_fn({"vel": pred}, target, 0, torch.zeros(1, 6, dtype=torch.bool))
    assert empty.item() == 0.0 and empty.requires_grad
    empty.backward()
    assert pred.grad is not None and torch.count_nonzero(pred.grad) == 0


def test_window_level_targets_degrade_to_mse():
    loss_fn = build_loss("ronin_global_pos", mode="full")
    target = torch.randn(3, 2)
    value, items = loss_fn({"vel": target + 1.0}, target, 0, None)
    assert value.item() == pytest.approx(1.0) and float(items["degraded"]) == 1.0


def test_recipes_official_and_unified():
    official = get_cfg({"model": "ronin_lstm", "recipe": "official"})
    unified = get_cfg({"model": "ronin_lstm", "recipe": "unified"})
    assert official.batch == 72 and official.lr == pytest.approx(3e-4)
    assert official.optimizer == "adam" and official.scheduler == "plateau"
    assert official.gamma == 0.75 and official.plateau_patience == 10
    assert official.stride == 100 and official.weight_decay == 0.0 and official.amp is False
    assert official.fitness == "loss" and official.patience == 0
    # 增强在两个配方中一致（DESIGN §4 的公平性口径）
    assert official.augment == unified.augment
    shift, augs = build_augmentations(official.augment, official.frame)
    assert [a.name for a in augs] == ["random_yaw"]
    assert shift.resolve_range(official.stride) == (-50, 50)   # 官方 randrange(-50, 50)
    assert unified.epochs == get_cfg({}).epochs                 # unified 用统一预算


@pytest.mark.slow
@pytest.mark.parametrize("overlap", ["mean", "center"])
def test_end_to_end_train_and_val(synthetic_dataset, tmp_path, overlap):
    metrics = run_end_to_end("ronin_lstm", synthetic_dataset, tmp_path / overlap,
                             overlap=overlap)
    assert math.isfinite(metrics["rte"]) and math.isfinite(metrics["vel_rmse"])
