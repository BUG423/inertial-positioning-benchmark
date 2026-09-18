"""CTIN（paper-only）移植测试：推导参数量锁定、逐帧布局、非因果性、IVL/CNL 多任务损失。

夹具：**无**（官方仓库没有源码）。参数量锁的是 ``docs/algorithms/ctin.md`` §4 的推导值
477,572（+2 个多任务权重 = 477,574），只有论文 Table 3 报告的 0.5571e6 的 85.7%；
论文没有给出任何宽度，无法唯一复原该数字，因此**参数量不能作为忠实性证据**。
"""

import math

import numpy as np
import pytest

from .model_testing import (
    assert_backprop,
    assert_deterministic_eval,
    run_end_to_end,
)

torch = pytest.importorskip("torch")

from inertial_benchmark.cfg import get_cfg  # noqa: E402
from inertial_benchmark.data.augment import build_augmentations  # noqa: E402
from inertial_benchmark.nn import build_model  # noqa: E402
from inertial_benchmark.nn.losses import build_loss  # noqa: E402

# 规格卡 §4 的参数分解（参考规格 d=64, heads=8, ff=512, groups=4, reduction=2）
CARD_BREAKDOWN = {"spatial_embedding": 5_504, "encoder": 41_088, "temporal_embedding": 23_040,
                  "decoder": 399_104, "velocity_head": 4_418, "covariance_head": 4_418,
                  "log_sigma": 2}
NETWORK_PARAMS = 477_572
PAPER_PARAMS = 557_100      # 论文 Table 3 的 0.5571e6，本实现为其 85.7%


def build(**overrides):
    return build_model(get_cfg({"model": "ctin", **overrides}))


def module_params(model):
    out = {}
    for name, param in model.named_parameters():
        out[name.split(".")[0]] = out.get(name.split(".")[0], 0) + param.numel()
    return out


def test_derived_parameter_count_is_locked():
    model = build()
    assert model.num_params == NETWORK_PARAMS + 2 == 477_574
    assert module_params(model) == CARD_BREAKDOWN
    assert sum(CARD_BREAKDOWN.values()) - 2 == NETWORK_PARAMS
    # 与论文报告值的差距（规格卡 §4/§7）：85.7%，差异来自论文未给出的宽度
    assert NETWORK_PARAMS / PAPER_PARAMS == pytest.approx(0.857, abs=0.001)
    assert model.num_params < PAPER_PARAMS


def test_reference_widths_are_configurable():
    """卡片标为假设的宽度都是构造参数；例如 ff=640 得 543,620（仍不等于论文值）。"""
    assert build(model_args={"ff_dim": 640}).num_params - 2 == 543_620


def test_frame_layout_shapes_and_backprop():
    model = build().eval()
    spec = model.input_spec
    assert spec.output_layout == "frame" and spec.output_shape == (200, 2)
    x = torch.randn(2, 6, 200)
    out = model(x)
    assert out["vel"].shape == (2, 200, 2) and out["logstd"].shape == (2, 200, 2)
    assert out["log_sigma"].shape == (2, 2)   # 每个样本一行，验证路径的 index_select 才能用
    assert_deterministic_eval(model, x)
    assert_backprop(build(), x, torch.randn(2, 200, 2))


def test_decoder_self_attention_mask_is_strictly_upper_triangular():
    mask = build().causal_mask(5, torch.device("cpu"))
    assert torch.equal(torch.isinf(mask), torch.triu(torch.ones(5, 5, dtype=torch.bool), 1))
    assert torch.all(mask[torch.tril(torch.ones(5, 5, dtype=torch.bool))] == 0.0)


def test_model_is_not_causal():
    """规格卡 §8/§10.5：解码器有因果掩码，但 tgt 来自双向 LSTM、memory 非因果 ⇒ 整体非因果。"""
    model = build().eval()
    x = torch.randn(1, 6, 200)
    with torch.no_grad():
        base = model(x)["vel"]
        perturbed = x.clone()
        perturbed[0, :, -1] += 10.0         # 只扰动窗口最后一个样本
        changed = model(perturbed)["vel"]
    assert (changed[0, 0] - base[0, 0]).abs().max().item() > 1e-6


def test_window_level_layout_aggregates_frames():
    """规格卡 §3 的窗口级降级：速度取均值，σ 取“误差完全相关”的保守近似。"""
    model = build(target="avg_velocity").eval()
    assert model.num_params == NETWORK_PARAMS + 2
    x = torch.randn(2, 6, 200)
    out = model(x)
    assert out["vel"].shape == (2, 2) and out["logstd"].shape == (2, 2)
    torch.testing.assert_close(out["vel"], out["vel_seq"].mean(dim=1))
    # logstd_seq 恒为常数 c 时窗口级 logstd 就是 c（不是 c − ½log m）
    with torch.no_grad():
        model.covariance_head.out.weight.zero_()
        model.covariance_head.out.bias.fill_(-0.75)
        aggregated = model(x)["logstd"]
    torch.testing.assert_close(aggregated, torch.full((2, 2), -0.75), rtol=1e-5, atol=1e-6)


def test_multitask_loss_at_initial_weights():
    """``u_v = u_c = 0`` 时 ``L = ½(Lv + Lc)``（规格卡 §8）。"""
    loss_fn = build_loss("ctin_multitask", rate=200.0)
    target = torch.randn(2, 50, 2)
    out = {"vel": target + 0.1, "logstd": torch.zeros(2, 50, 2),
           "log_sigma": torch.zeros(2, 2)}
    value, items = loss_fn(out, target, 0, None)
    assert value.item() == pytest.approx(0.5 * (float(items["lv"]) + float(items["lc"])),
                                         rel=1e-6)
    # v̂ ≡ v ⇒ Lv = 0（Lpv 与 Lev 都为 0）
    exact, items_exact = loss_fn({"vel": target.clone(), "logstd": torch.zeros(2, 50, 2),
                                  "log_sigma": torch.zeros(2, 2)}, target, 0, None)
    assert float(items_exact["lv"]) == pytest.approx(0.0, abs=1e-12)
    assert float(items_exact["lpv"]) == pytest.approx(0.0, abs=1e-12)
    assert exact.item() == pytest.approx(0.0, abs=1e-12)


def test_covariance_loss_is_minimised_at_the_empirical_log_sigma():
    """``logσ`` 取常数 a 时，``Lc`` 对 a 的导数在 ``a = ½·ln mean(err²)`` 处为 0（规格卡 §8）。"""
    loss_fn = build_loss("ctin_multitask", rate=200.0)
    target = torch.zeros(1, 40, 2)
    error = torch.randn(1, 40, 2)
    best = 0.5 * math.log(float((error ** 2).mean()))
    a = torch.tensor(best, requires_grad=True)
    out = {"vel": error, "logstd": a.expand(1, 40, 2), "log_sigma": torch.zeros(1, 2)}
    lc = loss_fn._covariance(out["vel"], out["logstd"], target, None)
    lc.backward()
    assert abs(float(a.grad)) < 1e-4


def test_multitask_weights_receive_gradients():
    model = build()
    x = torch.randn(2, 6, 200)
    out = model(x)
    loss, items = model.loss(out, {"target": torch.randn(2, 200, 2), "imu": x,
                                   "mask": torch.ones(2, 200, dtype=torch.bool)}, 0)
    loss.backward()
    assert model.log_sigma.grad is not None and model.log_sigma.grad.abs().sum() > 0
    assert float(items["u_v"]) == 0.0 and float(items["u_c"]) == 0.0


def test_mask_behaviour_on_frames():
    loss_fn = build_loss("ctin_multitask", rate=200.0)
    target = torch.zeros(1, 6, 2)
    pred = torch.zeros(1, 6, 2, requires_grad=True)
    logstd = torch.zeros(1, 6, 2)
    mask = torch.ones(1, 6, dtype=torch.bool)
    mask[0, 2] = False
    with torch.no_grad():
        pred[0, 2] = 4.0          # 误差全部落在无效帧上
    out = {"vel": pred, "logstd": logstd, "log_sigma": torch.zeros(1, 2)}
    value, items = loss_fn(out, target, 0, mask)
    assert float(items["lev"]) == pytest.approx(0.0, abs=1e-10)
    assert float(items["lpv"]) == pytest.approx(0.0, abs=1e-10)   # 前缀项要求 0..t 全部有效
    empty, _ = loss_fn(out, target, 0, torch.zeros(1, 6, dtype=torch.bool))
    assert empty.item() == 0.0 and empty.requires_grad
    empty.backward()
    assert pred.grad is not None and torch.count_nonzero(pred.grad) == 0
    assert value.item() >= 0.0


def test_integrated_position_term_uses_dt():
    """``Lpv`` 用 ``dt·cumsum``；把 rate 提高一倍，该项应缩小 4 倍。"""
    target = torch.zeros(1, 20, 2)
    out = {"vel": torch.ones(1, 20, 2), "logstd": torch.zeros(1, 20, 2),
           "log_sigma": torch.zeros(1, 2)}
    slow = build_loss("ctin_multitask", rate=200.0)(out, target, 0, None)[1]["lpv"]
    fast = build_loss("ctin_multitask", rate=400.0)(out, target, 0, None)[1]["lpv"]
    assert float(slow) / float(fast) == pytest.approx(4.0, rel=1e-6)
    j = torch.arange(1, 21, dtype=torch.float64)
    assert float(slow) == pytest.approx(float((2 * (j / 200.0) ** 2).mean()), rel=1e-5)


def test_recipes_official_and_unified():
    official = get_cfg({"model": "ctin", "recipe": "official"})
    unified = get_cfg({"model": "ctin", "recipe": "unified"})
    assert official.lr == pytest.approx(5e-4) and official.weight_decay == pytest.approx(1e-6)
    assert official.optimizer == "adam" and official.scheduler == "none"
    assert official.batch == 128 and official.epochs == 300 and official.patience == 30
    assert official.amp is False and official.fitness == "loss"
    assert official.augment == unified.augment     # 增强在两个配方中一致（DESIGN §4）
    shift, augs = build_augmentations(official.augment, official.frame)
    assert [a.name for a in augs] == ["bias_shift", "random_yaw"]
    assert augs[0].gyro == 0.05 and augs[0].acc == 0.2
    assert shift.resolve_range(official.stride) == (-5, 5)
    # 论文没有阶段切换：两个配方都不做 MSE→NLL 热身（规格卡 §10.7）
    model = build(recipe="official")
    assert model.loss_name == "ctin_multitask"
    assert "switch_epoch" not in model.loss_kwargs
    # 空间编码器 dropout 0.5、时间解码器 dropout 0.05（A.3）
    assert model.encoder[0].dropout.p == 0.5 and model.decoder[0].dropout1.p == 0.05
    assert len(model.encoder) == 1 and len(model.decoder) == 4


def test_local_attention_weights_sum_over_time():
    """局部注意力的 γ 沿时间维 softmax，融合权重沿分支维 softmax（规格卡 §4 E2c/E2e）。"""
    block = build().eval().encoder[0]
    x = torch.randn(2, 64, 200)
    with torch.no_grad():
        gamma = torch.softmax(block.local.attend(
            torch.cat([x, block.local.key(x)], dim=1)), dim=-1)
        fused = torch.softmax(block.local.fuse(x.mean(dim=-1)).reshape(-1, 2, 64), dim=1)
    np.testing.assert_allclose(gamma.sum(dim=-1).numpy(), np.ones((2, 64)), rtol=1e-5)
    np.testing.assert_allclose(fused.sum(dim=1).numpy(), np.ones((2, 64)), rtol=1e-5)


@pytest.mark.slow
@pytest.mark.parametrize("overlap", ["mean", "center"])
def test_end_to_end_train_and_val(synthetic_dataset, tmp_path, overlap):
    metrics = run_end_to_end("ctin", synthetic_dataset, tmp_path / overlap, overlap=overlap)
    assert math.isfinite(metrics["ate"]) and math.isfinite(metrics["rte"])
