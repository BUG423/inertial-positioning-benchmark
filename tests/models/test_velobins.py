"""VeloBins 移植测试：参数量（与箱数无关）、分箱头解码、误差条件高斯标签与 KL+Huber 损失。"""

import pytest

from .model_testing import assert_backprop, assert_deterministic_eval, run_end_to_end

torch = pytest.importorskip("torch")

from inertial_benchmark.cfg import get_cfg  # noqa: E402
from inertial_benchmark.models.velobins.model import (  # noqa: E402
    ModalityEncoder,
    tilted_gaussian_bins,
)
from inertial_benchmark.nn import build_model  # noqa: E402
from inertial_benchmark.nn.heads import BinsHead, bin_edges, build_head  # noqa: E402
from inertial_benchmark.nn.losses import build_loss  # noqa: E402

BATCH = 4
# docs/algorithms/velobins.md §4 的推导值（论文报告约 76.6 k）
DERIVED_PARAMS = 69_968
OFFICIAL_SHAPE_PARAMS = 68_128
DECODER_PARAMS = 6_848


def build(**overrides):
    return build_model(get_cfg({"model": "velobins", **overrides}))


def windows(batch: int = BATCH, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    gyro = 0.8 * torch.randn(batch, 3, 200, generator=generator)
    acc = 2.0 * torch.randn(batch, 3, 200, generator=generator)
    acc[:, 2] += 9.81
    return torch.cat([gyro, acc], dim=1)


def test_parameter_count_matches_the_spec_card():
    model = build()
    assert model.num_params == DERIVED_PARAMS
    conv = sum(p.numel() for p in model.acc_encoder.parameters()) \
        + sum(p.numel() for p in model.gyro_encoder.parameters())
    assert conv == 6_576
    assert sum(p.numel() for p in model.transformer.parameters()) == 56_544
    assert sum(p.numel() for p in model.head.parameters()) == DECODER_PARAMS


def test_official_shape_with_the_rotor_branch():
    """官方形状（3 条支路、W=16，含 rotor）为 68,128：卡 §8 的第二个核对值。"""
    model = build()
    branches = [ModalityEncoder(3, 16), ModalityEncoder(3, 16), ModalityEncoder(4, 16)]
    conv = sum(sum(p.numel() for p in b.parameters()) for b in branches)
    assert conv == 4_736
    total = conv + sum(p.numel() for p in model.transformer.parameters()) \
        + sum(p.numel() for p in model.head.parameters())
    assert total == OFFICIAL_SHAPE_PARAMS


@pytest.mark.parametrize("bins", [16, 512, 2048])
def test_decoder_parameters_are_independent_of_the_bin_count(bins):
    """“箱坐标分类器”的核心好处：参数量与 N 无关（对照 Linear(48, 3N) 在 N=512 时 75,264）。"""
    model = build(model_args={"bins": bins})
    assert model.num_params == DERIVED_PARAMS
    assert sum(p.numel() for p in model.head.parameters()) == DECODER_PARAMS
    plain = torch.nn.Linear(48, 3 * 512)
    assert sum(p.numel() for p in plain.parameters()) == 75_264


def test_output_shapes_and_normalised_distribution():
    model = build().eval()
    with torch.no_grad():
        out = model(windows())
    assert out["vel"].shape == (BATCH, 3) and out["logstd"].shape == (BATCH, 3)
    assert out["probs"].shape == (BATCH, 3, 512) and out["logits"].shape == (BATCH, 3, 512)
    torch.testing.assert_close(out["probs"].sum(-1), torch.ones(BATCH, 3), atol=1e-6, rtol=0)


def test_expectation_decoding_of_known_distributions():
    """one-hot → ``v̂ = b_n``、``σ̂² = 0``；相邻两箱各 0.5 → 中点，``σ̂² = (Δ/2)²``（卡 §8）。"""
    head = BinsHead(8, 3, bins=512, value_range=5.0, clamp_std=False)
    centers, width = head.centers, head.bin_width
    probs = torch.zeros(2, 3, 512)
    probs[0, :, 100] = 1.0
    probs[1, :, 100] = 0.5
    probs[1, :, 101] = 0.5
    vel, std = head.decode_probs(probs)
    torch.testing.assert_close(vel[0], centers[100].expand(3), atol=1e-6, rtol=0)
    torch.testing.assert_close(std[0], torch.zeros(3), atol=1e-6, rtol=0)
    torch.testing.assert_close(vel[1], (0.5 * (centers[100] + centers[101])).expand(3),
                               atol=1e-6, rtol=0)
    torch.testing.assert_close(std[1], torch.full((3,), width / 2), atol=1e-6, rtol=0)


def test_decode_modes_can_be_switched():
    """``topk_expectation(k=N)`` 与 ``expectation`` 逐位相等；``argmax`` 落在箱中心上。"""
    torch.manual_seed(0)
    head = BinsHead(8, 3, bins=64, value_range=2.0, clamp_std=False)
    probs = torch.softmax(torch.randn(5, 3, 64), dim=-1)
    head.set_decode("expectation")
    expectation, expectation_std = head.decode_probs(probs)
    head.set_decode("topk_expectation", topk=64)
    topk, topk_std = head.decode_probs(probs)
    torch.testing.assert_close(topk, expectation, atol=0, rtol=0)
    torch.testing.assert_close(topk_std, expectation_std, atol=0, rtol=0)
    head.set_decode("argmax")
    argmax, argmax_std = head.decode_probs(probs)
    assert torch.isin(argmax, head.centers).all()
    torch.testing.assert_close(argmax, head.centers[probs.argmax(-1)], atol=0, rtol=0)
    # argmax 的方差解码仍来自完整分布（围绕解码均值），因此不为零
    assert torch.all(argmax_std > 0)
    assert float((argmax - expectation).abs().max()) <= 2.0    # 平凡上界 R
    # 单峰对称分布上两者只差量化误差 Δ/2
    unimodal = torch.zeros(1, 1, 64)
    unimodal[0, 0, 30:33] = torch.tensor([0.25, 0.5, 0.25])
    head.set_decode("expectation")
    peak, _ = head.decode_probs(unimodal)
    head.set_decode("argmax")
    hard, _ = head.decode_probs(unimodal)
    assert float((hard - peak).abs().max()) <= head.bin_width / 2 + 1e-6
    with pytest.raises(ValueError, match="unknown decode"):
        head.set_decode("nope")


def test_model_level_decode_switch_keeps_the_distribution():
    model = build(model_args={"decode": "argmax"}).eval()
    assert model.num_params == DERIVED_PARAMS
    with torch.no_grad():
        out = model(windows(seed=2))
        expectation = (out["probs"] * model.head.centers).sum(-1)
    torch.testing.assert_close(out["vel"], model.head.centers[out["probs"].argmax(-1)],
                               atol=0, rtol=0)
    assert float((out["vel"] - expectation).abs().max()) < 2 * model.head.value_range


def test_reusable_bins_head_on_another_feature_width():
    """``bins`` 头可以插到任意骨干上（``D_h`` 换成骨干宽度），参数量按卡 §6 的公式。"""
    head = build_head("bins", 512, 2, bins=256, value_range=3.0)
    expected = 64 + 32 * (64 + 1) + 2 * 32 * (512 + 1)
    assert sum(p.numel() for p in head.parameters()) == expected
    out = head(torch.randn(3, 512))
    assert out["vel"].shape == (3, 2) and out["probs"].shape == (3, 2, 256)
    # 固定位置编码变体没有可学习频率 γ
    fixed = build_head("bins", 512, 2, bins=256, value_range=3.0, encoding="pe")
    assert sum(p.numel() for p in fixed.parameters()) == expected - 64
    # direct 变体（消融）退化为 Linear(D_h, dims·N)，参数量随 N 增长
    direct = build_head("bins", 512, 2, bins=256, value_range=3.0, encoding="direct")
    assert sum(p.numel() for p in direct.parameters()) == 2 * 256 * (512 + 1)


def test_error_conditioned_gaussian_labels():
    """式 6–7：归一、σ 越小越尖、倾斜后均值精确等于真值（卡 §8）。"""
    head = BinsHead(8, 3, bins=512, value_range=5.0)
    centers = head.centers.double()
    lower, upper = bin_edges(centers, head.value_range)
    target = torch.tensor([[0.3, -1.2, 2.0]], dtype=torch.float64)
    for sigma_value in (0.02, 0.2, 1.0):
        sigma = torch.full_like(target, sigma_value)
        q = tilted_gaussian_bins(target, sigma, lower, upper, centers)
        torch.testing.assert_close(q.sum(-1), torch.ones_like(target), atol=1e-9, rtol=0)
        assert torch.all(q >= 0)
        torch.testing.assert_close((q * centers).sum(-1), target, atol=1e-6, rtol=0)
    sharp = tilted_gaussian_bins(target, torch.full_like(target, 0.02), lower, upper, centers)
    broad = tilted_gaussian_bins(target, torch.full_like(target, 1.0), lower, upper, centers)

    def entropy(q):
        return -(q * torch.log(q.clamp_min(1e-30))).sum(-1)

    assert torch.all(entropy(sharp) < entropy(broad))
    # σ 被 clamp 到 Δ/10 时至少有一个箱带非零质量
    floor = tilted_gaussian_bins(target, torch.full_like(target, head.bin_width / 10),
                                 lower, upper, centers)
    assert torch.all(floor.max(-1).values > 0.5)


def test_loss_is_zero_when_the_distribution_matches_the_label():
    model = build().eval()
    loss_fn = build_loss("velobins_bins")
    assert loss_fn.delta == 0.1 and loss_fn.huber_weight == loss_fn.kl_weight == 1.0
    target = torch.tensor([[0.4, -0.7, 0.1]])
    head = model.head
    centers = head.centers
    lower, upper = bin_edges(centers, head.value_range)
    sigma = torch.full_like(target, head.bin_width / 10)
    labels = tilted_gaussian_bins(target, sigma, lower, upper, centers)
    vel, std = head.decode_probs(labels)
    out = {"vel": vel, "logstd": torch.log(std), "probs": labels, "head": head}
    value, items = loss_fn(out, target, 0)
    assert set(items) == {"huber", "kl", "mse"}
    assert float(items["kl"]) == pytest.approx(0.0, abs=1e-6)
    assert float(items["huber"]) == pytest.approx(0.0, abs=1e-6)
    assert float(value) == pytest.approx(0.0, abs=1e-6)


def test_labels_are_detached_and_no_nll_term():
    model = build()
    x = windows(seed=3)
    out = model(x)
    target = 0.5 * torch.randn(BATCH, 3)
    labels = build_loss("velobins_bins").labels({**out, "head": model.head}, target)
    assert labels.requires_grad is False
    # 只有 Huber 与 KL 两项：把两个权重都设为 0 时损失恒为 0
    zero = build_loss("velobins_bins", huber_weight=0.0, kl_weight=0.0)
    value, _ = zero({**out, "head": model.head}, target, 0)
    assert float(value.detach()) == pytest.approx(0.0, abs=1e-9)


def test_huber_is_continuous_at_the_knee():
    loss_fn = build_loss("velobins_bins")
    delta = loss_fn.delta
    eps = 1e-6
    pred = torch.tensor([[delta - eps, delta + eps, 0.0]], dtype=torch.float64)
    target = torch.zeros(1, 3, dtype=torch.float64)
    values = torch.nn.functional.huber_loss(pred, target, reduction="none", delta=delta)
    assert float(values[0, 0] - values[0, 1]).__abs__() < 1e-6


def test_loss_masks_invalid_outputs():
    model = build().eval()
    x = windows(seed=4)
    with torch.no_grad():
        out = model(x)
    target = 0.5 * torch.randn(BATCH, 3)
    mask = torch.zeros(BATCH, 1)
    value, _ = model.loss(out, {"target": target, "imu": x, "mask": mask})
    assert float(value) == 0.0


def test_body_frame_view_and_decimation():
    cfg = get_cfg({"model": "velobins"})
    assert cfg.frame == "body" and cfg.dims == 3 and cfg.window == 200
    assert cfg.target == "avg_velocity"
    model = build(model_args={"decimate": 2}).eval()
    assert model.num_params == DERIVED_PARAMS
    with torch.no_grad():
        assert model.encode(windows(seed=5)).shape == (BATCH, 48)


def test_channel_permutation_to_the_official_modality_order():
    """IPB 的 ``[gyro, acc]`` 在模型内部置换为官方的 ``[acc, gyro]`` 两条支路。"""
    model = build().eval()
    x = windows(seed=6)
    with torch.no_grad():
        tokens = torch.cat([model.acc_encoder(x[:, 3:6]), model.gyro_encoder(x[:, 0:3])], dim=1)
        expected = model.transformer(tokens.transpose(1, 2))[:, -1]
        torch.testing.assert_close(model.encode(x), expected, atol=0, rtol=0)


def test_forward_backward_and_determinism():
    model = build()
    assert model.loss_name == "velobins_bins"
    x = windows(seed=7)
    assert_deterministic_eval(model, x)
    assert_backprop(build(), x, 0.5 * torch.randn(BATCH, 3))


def test_configuration_constraints():
    with pytest.raises(ValueError, match="one velocity per window"):
        build(target="frame_velocity")
    with pytest.raises(ValueError, match="d_model"):
        build(model_args={"width": 20})
    cfg = get_cfg({"model": "velobins", "recipe": "official"})
    assert cfg.epochs == 100 and cfg.batch == 128 and cfg.lr == 3e-4
    assert cfg.scheduler == "none" and cfg.optimizer == "adam"
    assert cfg.augment == [] == get_cfg({"model": "velobins"}).augment


@pytest.mark.slow
def test_end_to_end_train_and_val(synthetic_dataset, tmp_path):
    run_end_to_end("velobins", synthetic_dataset, tmp_path)
