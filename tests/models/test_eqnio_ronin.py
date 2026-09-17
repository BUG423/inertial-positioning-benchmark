"""EqNIO（RoNIN 骨干）移植测试：参数量锁定与 O(2) 等变性（含陀螺赝矢量的负向测试）。"""

import pytest
from model_testing import (
    assert_backprop,
    assert_deterministic_eval,
    load_fixture,
    param_shapes,
    run_end_to_end,
    shape_multiset,
)

torch = pytest.importorskip("torch")

from inertial_benchmark.cfg import get_cfg  # noqa: E402
from inertial_benchmark.nn import build_model  # noqa: E402
from inertial_benchmark.nn.modules.equivariant import (  # noqa: E402
    act_o2,
    o2_frame_features,
    reflection_matrix_2d,
    rotation_matrix_2d,
)

FIXTURE = load_fixture("eqnio_ronin")
BATCH = 8


def build(**overrides):
    return build_model(get_cfg({"model": "eqnio_ronin", **overrides}))


def windows(dtype=torch.float64, seed: int = 0) -> torch.Tensor:
    """规格卡 §8 的等变性测试输入：陀螺 σ=0.8、加计 σ=2 并在 z 上加 9.81。"""
    generator = torch.Generator().manual_seed(seed)
    gyro = 0.8 * torch.randn(BATCH, 3, 200, generator=generator, dtype=dtype)
    acc = 2.0 * torch.randn(BATCH, 3, 200, generator=generator, dtype=dtype)
    acc[:, 2] += 9.81
    return torch.cat([gyro, acc], dim=1)


@pytest.fixture(scope="module")
def model64():
    torch.manual_seed(0)
    return build().double().eval()


def test_parameters_match_official_fixture():
    model = build()
    assert model.num_params == FIXTURE["total_params"] == 5_230_466
    assert sum(p.numel() for p in model.frame_net.parameters()) \
        == FIXTURE["frame_network_params"] == 595_584
    assert sum(p.numel() for p in model.ronin.parameters()) == FIXTURE["backbone_params"]
    shapes = param_shapes(model)
    assert shape_multiset(shapes) == shape_multiset(FIXTURE["param_shapes"])
    assert shapes == FIXTURE["param_shapes"]  # 注册顺序也一致
    buffers = [list(b.shape) for _, b in model.named_buffers()]
    assert buffers == FIXTURE["buffer_shapes"] and len(buffers) == 72
    # 9 个 LayerNorm 的 beta 是 buffer（不参与训练）
    betas = [name for name, _ in model.named_buffers() if name.endswith("beta")]
    assert len(betas) == 9
    assert all(torch.all(b == 0) for name, b in model.named_buffers() if name.endswith("beta"))


def test_output_fields_and_frame_properties(model64):
    x = windows()
    with torch.no_grad():
        out = model64(x)
    assert set(out) == {"vel", "aux"} and set(out["aux"]) == {"frame"}
    assert out["vel"].shape == (BATCH, 2) and out["aux"]["frame"].shape == (BATCH, 2, 2)
    frame = out["aux"]["frame"]
    eye = torch.eye(2, dtype=torch.float64).expand(BATCH, 2, 2)
    torch.testing.assert_close(frame @ frame.transpose(1, 2), eye, atol=1e-12, rtol=0)
    det = torch.linalg.det(frame)
    assert torch.all(det.abs() - 1.0 < 1e-12)
    assert set(float(v) for v in det.sign().unique()) <= {-1.0, 1.0}


def test_forward_backward_and_determinism():
    model = build()
    x = windows(torch.float32, seed=3)
    assert_deterministic_eval(model, x)
    assert_backprop(build(), x, torch.randn(BATCH, 2))
    assert model.loss_name == "mse"


@pytest.mark.parametrize("kind", ["rotation", "reflection"])
def test_o2_equivariance(model64, kind):
    """输入绕 z 旋转/反射（陀螺按赝矢量）时输出按 O(2) 协变，骨干输入不变。"""
    angles = [0.7, -2.3] * (BATCH // 2) if kind == "rotation" else [0.4, -1.1] * (BATCH // 2)
    mats = (rotation_matrix_2d if kind == "rotation" else reflection_matrix_2d)(angles)
    x = windows()
    with torch.no_grad():
        frame, canonical = model64.frame(x)
        out = model64(x)
        frame_t, canonical_t = model64.frame(act_o2(x, mats))
        out_t = model64(act_o2(x, mats))
    assert torch.allclose(torch.linalg.det(mats),
                          torch.full((BATCH,), 1.0 if kind == "rotation" else -1.0,
                                     dtype=torch.float64))
    # vel' = Q·vel，frame'（= Fᵀ）= frame·Qᵀ，骨干输入不变
    torch.testing.assert_close(out_t["vel"], torch.einsum("bij,bj->bi", mats, out["vel"]),
                               atol=1e-8, rtol=0)
    torch.testing.assert_close(out_t["aux"]["frame"], out["aux"]["frame"] @ mats.transpose(1, 2),
                               atol=1e-8, rtol=0)
    torch.testing.assert_close(canonical_t, canonical, atol=1e-7, rtol=0)
    assert float(out["vel"].abs().max()) > 1e-3   # 输出不是恒零，等变性才有意义


def test_gyroscope_must_be_a_pseudovector(model64):
    """负向测试：反射时把陀螺当普通矢量，等变性必须明显不成立（规格卡 §8）。"""
    mats = reflection_matrix_2d([0.4, -1.1] * (BATCH // 2))
    x = windows()
    wrong = x.clone()
    wrong[:, :2] = torch.einsum("bij,bjt->bit", mats, x[:, :2])       # 陀螺 xy 按普通矢量
    wrong[:, 3:5] = torch.einsum("bij,bjt->bit", mats, x[:, 3:5])     # 加计 xy 正常
    with torch.no_grad():
        out = model64(x)
        _, canonical = model64.frame(x)
        out_wrong = model64(wrong)
        _, canonical_wrong = model64.frame(wrong)
    vel_err = (out_wrong["vel"] - torch.einsum("bij,bj->bi", mats, out["vel"])).abs().max()
    assert float(vel_err) > 1e-2
    assert float((canonical_wrong - canonical).abs().max()) > 1e-2


def test_backbone_channel_order(model64):
    """骨干输入通道为 ``[Fᵀa_xy, a_z, s·Fᵀω_xy, s·ω_z]``，``s = det F``（规格卡 §4.3）。"""
    x = windows()
    with torch.no_grad():
        frame, canonical = model64.frame(x)
    scale = torch.linalg.det(frame).reshape(-1, 1, 1)
    gyro, acc = x[:, 0:3], x[:, 3:6]
    acc_c = torch.einsum("bji,bjt->bit", frame, acc[:, :2])
    gyro_c = torch.einsum("bji,bjt->bit", frame, gyro[:, :2])
    expected = torch.cat([acc_c, acc[:, 2:3], scale * gyro_c, scale * gyro[:, 2:3]], dim=1)
    torch.testing.assert_close(canonical, expected, atol=1e-12, rtol=0)


def test_preprocessing_vectors(model64):
    """``v1 × v2 = ‖ω‖·ω``，且 ``v1``、``v2`` 与 ``ω`` 的范数关系符合规格卡 §2.1。"""
    x = windows()
    vec, sca, original = o2_frame_features(x)
    assert vec.shape == (BATCH, 200, 2, 3) and sca.shape == (BATCH, 200, 9)
    assert original.shape == (BATCH, 200, 3)
    gyro = x[:, 0:3].transpose(1, 2)
    v1 = torch.cat([vec[..., 1], original[..., 1:2]], dim=-1)
    v2 = torch.cat([vec[..., 2], original[..., 2:3]], dim=-1)
    norm = torch.linalg.vector_norm(gyro, dim=-1, keepdim=True)
    torch.testing.assert_close(torch.linalg.cross(v1, v2, dim=-1), norm * gyro,
                               atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(torch.linalg.vector_norm(v1, dim=-1), norm[..., 0],
                               atol=1e-9, rtol=1e-9)
    # 标量特征为 O(2) 不变量
    mats = reflection_matrix_2d([0.4, -1.1] * (BATCH // 2))
    _, sca_t, original_t = o2_frame_features(act_o2(x, mats))
    torch.testing.assert_close(sca_t, sca, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(original_t, original, atol=1e-9, rtol=1e-9)


def test_degenerate_windows_stay_finite(model64):
    x = torch.zeros(3, 6, 200, dtype=torch.float64)
    x[1, 2] = 0.5           # 只有 omega_z（ω_xy == 0，走退化分支）
    x[2, 5] = 9.81          # 静止：只有重力
    with torch.no_grad():
        out = model64(x)
    assert torch.isfinite(out["vel"]).all() and torch.isfinite(out["aux"]["frame"]).all()


def test_float32_equivariance_is_approximate():
    """float32 下等变性只是近似成立（LayerNorm 链会放大少数窗口的误差，规格卡 §10.6）。"""
    torch.manual_seed(0)
    model = build().eval()
    mats = rotation_matrix_2d([0.7, -2.3] * (BATCH // 2), dtype=torch.float32)
    x = windows(torch.float32, seed=1)
    with torch.no_grad():
        out = model(x)["vel"]
        out_t = model(act_o2(x, mats))["vel"]
    err = (out_t - torch.einsum("bij,bj->bi", mats, out)).abs().amax(dim=1)
    assert float(err.median()) < 1e-4


def test_configuration_constraints():
    with pytest.raises(ValueError, match="dims=2"):
        build(dims=3)
    with pytest.raises(ValueError, match="gravity-aligned"):
        build(frame="body", dims=3)
    cfg = get_cfg({"model": "eqnio_ronin", "recipe": "official"})
    assert cfg.epochs == 120 and cfg.lr == 1e-4 and cfg.batch == 128
    assert cfg.scheduler == "plateau" and cfg.optimizer == "adam"
    assert cfg.augment == ["time_shift"]   # 等变架构不做随机偏航
    assert get_cfg({"model": "eqnio_ronin"}).augment == ["time_shift"]


@pytest.mark.slow
def test_end_to_end_train_and_val(synthetic_dataset, tmp_path):
    run_end_to_end("eqnio_ronin", synthetic_dataset, tmp_path)
