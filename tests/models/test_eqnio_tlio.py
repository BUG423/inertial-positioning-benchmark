"""EqNIO（TLIO 骨干）移植测试：夹具参数量、O(2) 等变性、规范帧对角 NLL 与阶段切换。"""

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
from inertial_benchmark.models.eqnio.tlio import (  # noqa: E402
    canonical_diagonal_to_world_cov,
    frame_to_world_vector,
)
from inertial_benchmark.nn import build_model  # noqa: E402
from inertial_benchmark.nn.losses import build_loss  # noqa: E402
from inertial_benchmark.nn.modules.equivariant import (  # noqa: E402
    act_o2,
    reflection_matrix_2d,
    rotation_matrix_2d,
)

FIXTURE = load_fixture("eqnio_tlio")
BATCH = 8


def build(**overrides):
    return build_model(get_cfg({"model": "eqnio_tlio", **overrides}))


def windows(dtype=torch.float64, seed: int = 0) -> torch.Tensor:
    """与 `eqnio_ronin` 相同的等变性测试输入（规格卡 §8）。"""
    generator = torch.Generator().manual_seed(seed)
    gyro = 0.8 * torch.randn(BATCH, 3, 200, generator=generator, dtype=dtype)
    acc = 2.0 * torch.randn(BATCH, 3, 200, generator=generator, dtype=dtype)
    acc[:, 2] += 9.81
    return torch.cat([gyro, acc], dim=1)


def block_diagonal(mats: torch.Tensor) -> torch.Tensor:
    """``M̃ = diag(M, 1)``：水平面按 ``M`` 变换，z 轴不动。"""
    out = torch.zeros(mats.shape[0], 3, 3, dtype=mats.dtype)
    out[:, :2, :2] = mats
    out[:, 2, 2] = 1.0
    return out


@pytest.fixture(scope="module")
def model64():
    torch.manual_seed(0)
    return build().double().eval()


def test_parameters_match_official_fixture():
    model = build()
    assert model.num_params == FIXTURE["total_params"] == 6_020_230
    assert sum(p.numel() for p in model.frame_net.parameters()) \
        == FIXTURE["frame_network_params"] == 595_584
    assert sum(p.numel() for p in model.tlio.parameters()) \
        == FIXTURE["backbone_params"] == 5_424_646
    shapes = param_shapes(model)
    assert shape_multiset(shapes) == shape_multiset(FIXTURE["param_shapes"])
    assert shapes == FIXTURE["param_shapes"]      # 注册顺序也一致（规范帧网络 23 + 骨干 78）
    assert len(shapes) == 101
    buffers = [list(b.shape) for _, b in model.named_buffers()]
    assert buffers == FIXTURE["buffer_shapes"] and len(buffers) == 75
    # 规范帧网络的参数形状与 eqnio_ronin 夹具中非 ronin. 前缀的部分逐项相同
    ronin = load_fixture("eqnio_ronin")
    frame_only = [s for s, n in zip(ronin["param_shapes"], ronin["param_names"])
                  if not n.startswith("ronin.")]
    assert shapes[:len(frame_only)] == frame_only
    assert model.feature_length == 7


def test_output_fields_and_frame_properties(model64):
    x = windows()
    with torch.no_grad():
        out = model64(x)
    assert out["vel"].shape == (BATCH, 3) and out["logstd"].shape == (BATCH, 3)
    assert out["disp_canonical"].shape == (BATCH, 3)
    assert out["logstd_canonical"].shape == (BATCH, 3)
    assert out["cov"].shape == (BATCH, 3, 3)
    assert out["aux"]["frame"].shape == (BATCH, 2, 2)
    frame = out["aux"]["frame"]
    eye = torch.eye(2, dtype=torch.float64).expand(BATCH, 2, 2)
    torch.testing.assert_close(frame @ frame.transpose(1, 2), eye, atol=1e-12, rtol=0)
    assert set(float(v) for v in torch.linalg.det(frame).sign().unique()) <= {-1.0, 1.0}
    # 规范帧内对角、旋回世界系后 xy 块成为满阵、z 与 xy 不耦合（卡 §10-9）
    cov = out["cov"]
    torch.testing.assert_close(cov, cov.transpose(1, 2), atol=1e-12, rtol=0)
    assert torch.all(cov[:, :2, 2] == 0) and torch.all(cov[:, 2, :2] == 0)
    torch.testing.assert_close(cov[:, 2, 2], torch.exp(2 * out["logstd_canonical"][:, 2]),
                               atol=1e-12, rtol=1e-12)


def test_world_mapping_of_the_canonical_outputs(model64):
    """``d_w = [F d_c,xy, d_c,z]``、``Σ_w = blkdiag(F diag(σ²) Fᵀ, σ_z²)``（规格卡 §3）。"""
    x = windows(seed=2)
    with torch.no_grad():
        out = model64(x)
        frame, _ = model64.frame(x)
    torch.testing.assert_close(out["vel"], frame_to_world_vector(frame, out["disp_canonical"]),
                               atol=1e-12, rtol=0)
    torch.testing.assert_close(
        out["cov"], canonical_diagonal_to_world_cov(frame, out["logstd_canonical"]),
        atol=1e-12, rtol=0)
    # 报告用的 logstd 是 Σ_w 的对角（边缘标准差）
    torch.testing.assert_close(
        out["logstd"], 0.5 * torch.log(torch.diagonal(out["cov"], dim1=1, dim2=2)),
        atol=1e-12, rtol=0)


@pytest.mark.parametrize("kind", ["rotation", "reflection"])
def test_o2_equivariance(model64, kind):
    """旋转/反射（陀螺按赝矢量）时 ``d_w``、``Σ_w`` 协变，规范帧内的量与骨干输入不变。"""
    angles = [0.7, -2.3] * (BATCH // 2) if kind == "rotation" else [0.4, -1.1] * (BATCH // 2)
    mats = (rotation_matrix_2d if kind == "rotation" else reflection_matrix_2d)(angles)
    x = windows()
    big = block_diagonal(mats)
    with torch.no_grad():
        out, out_t = model64(x), model64(act_o2(x, mats))
        _, canonical = model64.frame(x)
        _, canonical_t = model64.frame(act_o2(x, mats))
    expected_det = 1.0 if kind == "rotation" else -1.0
    torch.testing.assert_close(torch.linalg.det(mats),
                               torch.full((BATCH,), expected_det, dtype=torch.float64),
                               atol=1e-12, rtol=0)
    torch.testing.assert_close(out_t["vel"], torch.einsum("bij,bj->bi", big, out["vel"]),
                               atol=1e-8, rtol=0)
    torch.testing.assert_close(out_t["cov"], big @ out["cov"] @ big.transpose(1, 2),
                               atol=1e-8, rtol=0)
    # 规范帧内的位移与 logstd 是不变量，frame（= Fᵀ）按 frame·Qᵀ 变换
    torch.testing.assert_close(out_t["disp_canonical"], out["disp_canonical"], atol=1e-8, rtol=0)
    torch.testing.assert_close(out_t["logstd_canonical"], out["logstd_canonical"],
                               atol=1e-8, rtol=0)
    torch.testing.assert_close(out_t["aux"]["frame"], out["aux"]["frame"] @ mats.transpose(1, 2),
                               atol=1e-8, rtol=0)
    torch.testing.assert_close(canonical_t, canonical, atol=1e-7, rtol=0)
    assert float(out["vel"].abs().max()) > 1e-4   # 输出不是恒零，等变性才有意义


def test_gyroscope_must_be_a_pseudovector(model64):
    """负向测试：反射时把陀螺当普通矢量，等变性必须明显不成立（规格卡 §8）。"""
    mats = reflection_matrix_2d([0.4, -1.1] * (BATCH // 2))
    x = windows()
    wrong = x.clone()
    wrong[:, :2] = torch.einsum("bij,bjt->bit", mats, x[:, :2])     # 陀螺 xy 按普通矢量
    wrong[:, 3:5] = torch.einsum("bij,bjt->bit", mats, x[:, 3:5])   # 加计 xy 正常
    big = block_diagonal(mats)
    with torch.no_grad():
        out, out_wrong = model64(x), model64(wrong)
        _, canonical = model64.frame(x)
        _, canonical_wrong = model64.frame(wrong)
    err = (out_wrong["vel"] - torch.einsum("bij,bj->bi", big, out["vel"])).abs().max()
    assert float(err) > 1e-2
    assert float((canonical_wrong - canonical).abs().max()) > 1e-2


def test_backbone_channel_order(model64):
    """骨干输入通道为 ``[Fᵀa_xy, a_z, s·Fᵀω_xy, s·ω_z]``，``s = det F``（规格卡 §4.2）。"""
    x = windows()
    with torch.no_grad():
        frame, canonical = model64.frame(x)
    scale = torch.linalg.det(frame).reshape(-1, 1, 1)
    gyro, acc = x[:, 0:3], x[:, 3:6]
    acc_c = torch.einsum("bji,bjt->bit", frame, acc[:, :2])
    gyro_c = torch.einsum("bji,bjt->bit", frame, gyro[:, :2])
    expected = torch.cat([acc_c, acc[:, 2:3], scale * gyro_c, scale * gyro[:, 2:3]], dim=1)
    torch.testing.assert_close(canonical, expected, atol=1e-12, rtol=0)


def test_float32_equivariance_is_approximate():
    """float32 下等变性只是近似成立，只锁中位数（`eqnio_ronin` 卡 §10.6）。"""
    torch.manual_seed(0)
    model = build().eval()
    mats = rotation_matrix_2d([0.7, -2.3] * (BATCH // 2), dtype=torch.float32)
    big = block_diagonal(mats)
    x = windows(torch.float32, seed=1)
    with torch.no_grad():
        out = model(x)["vel"]
        out_t = model(act_o2(x, mats))["vel"]
    err = (out_t - torch.einsum("bij,bj->bi", big, out)).abs().amax(dim=1)
    assert float(err.median()) < 1e-4


def test_loss_stage_switch_and_detached_logstd():
    """第 9 个 epoch 是 MSE（协方差头无梯度），第 10 个起为规范帧对角 NLL（规格卡 §8）。"""
    model = build()
    x = windows(torch.float32, seed=3)
    target = torch.randn(BATCH, 3)
    loss_fn = build_loss("eqnio_canonical_nll", switch_epoch=9)
    assert loss_fn.stage(8) == "mse" and loss_fn.stage(9) == "nll"
    for epoch, expect_grad in ((8, False), (9, True)):
        model.zero_grad(set_to_none=True)
        out = model(x)
        loss, items = model.loss(out, {"target": target, "imu": x}, epoch)
        loss.backward()
        grads = [model.tlio.logstd_head.fc3.weight.grad,
                 model.tlio.logstd_head.fc3.bias.grad]
        has_grad = any(g is not None and float(g.abs().max()) > 0 for g in grads)
        assert has_grad is expect_grad
        assert ("mse" in items) and (("nll" in items) is expect_grad)
        assert all(p.grad is not None for p in model.tlio.mean_head.parameters())


def test_canonical_diagonal_nll_equals_world_full_covariance_nll():
    """规范帧对角 NLL 的三轴之和 == 世界系满协方差 NLL（规格卡 §5、§8）。"""
    model = build().double().eval()
    x = windows(seed=4)
    target = torch.randn(BATCH, 3, dtype=torch.float64)
    with torch.no_grad():
        out = model(x)
        loss_fn = build_loss("eqnio_canonical_nll", switch_epoch=0)
        canonical, _ = loss_fn(out, target, epoch=0)
        error = (out["vel"] - target).unsqueeze(-1)
        cov = out["cov"]
        quadratic = (error.transpose(1, 2) @ torch.linalg.inv(cov) @ error).reshape(-1)
        world = (0.5 * quadratic + 0.5 * torch.logdet(cov)).mean() / 3.0
    torch.testing.assert_close(canonical, world, atol=1e-10, rtol=1e-10)


def test_loss_masks_invalid_outputs():
    model = build()
    x = windows(torch.float32, seed=5)
    target = torch.randn(BATCH, 3)
    out = model(x)
    mask = torch.zeros(BATCH, 1)
    for epoch in (0, 99):
        loss, _ = model.loss(out, {"target": target, "imu": x, "mask": mask}, epoch)
        assert float(loss.detach()) == 0.0


def test_forward_backward_and_determinism():
    model = build()
    x = windows(torch.float32, seed=6)
    assert model.loss_name == "eqnio_canonical_nll"
    assert_deterministic_eval(model, x)
    model.train()
    model.zero_grad(set_to_none=True)
    out = model(x)
    loss, _ = model.loss(out, {"target": torch.randn(BATCH, 3), "imu": x}, epoch=99)
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"parameters without gradient: {missing[:5]}"
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())


def test_degenerate_windows_stay_finite(model64):
    x = torch.zeros(3, 6, 200, dtype=torch.float64)
    x[1, 2] = 0.5           # 只有 omega_z（ω_xy == 0，走退化分支）
    x[2, 5] = 9.81          # 静止：只有重力
    with torch.no_grad():
        out = model64(x)
    assert torch.isfinite(out["vel"]).all() and torch.isfinite(out["cov"]).all()


def test_configuration_constraints():
    with pytest.raises(ValueError, match="dims=3"):
        build(dims=2)
    with pytest.raises(ValueError, match="gravity-aligned"):
        build(frame="body")
    cfg = get_cfg({"model": "eqnio_tlio", "recipe": "official"})
    assert cfg.epochs == 50 and cfg.batch == 1024 and cfg.lr == 1e-4
    assert cfg.scheduler == "none" and cfg.grad_clip == 0.1 and cfg.optimizer == "adam"
    assert cfg.target == "displacement" and cfg.dims == 3 and cfg.frame == "gravity_world"
    # 增强属于算法本身，两个配方一致；等变架构不做随机偏航
    assert cfg.augment == get_cfg({"model": "eqnio_tlio"}).augment
    assert "random_yaw" not in cfg.augment


@pytest.mark.slow
def test_end_to_end_train_and_val(synthetic_dataset, tmp_path):
    run_end_to_end("eqnio_tlio", synthetic_dataset, tmp_path)
