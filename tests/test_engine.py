import json
import math
from pathlib import Path

import numpy as np
import pytest
from synthetic import make_sequence

torch = pytest.importorskip("torch")

from inertial_benchmark import NIO  # noqa: E402
from inertial_benchmark.cfg import ConfigError, get_cfg  # noqa: E402
from inertial_benchmark.data.convert import convert_dataset  # noqa: E402
from inertial_benchmark.data.views import SequenceView, ViewConfig  # noqa: E402
from inertial_benchmark.engine import Predictor, RunResult, SequenceResult, Trainer  # noqa: E402
from inertial_benchmark.engine import trainer as trainer_module  # noqa: E402
from inertial_benchmark.engine.predictor import (  # noqa: E402
    fill_invalid_windows,
    integrate,
    merge_outputs,
    reconstruct,
)
from inertial_benchmark.nn import (  # noqa: E402
    MODELS,
    BaseModel,
    SequenceModel,
    register_model,
)
from inertial_benchmark.utils import LOGGER, yaml_load  # noqa: E402
from inertial_benchmark.utils.torch_utils import (  # noqa: E402
    EarlyStopping,
    build_optimizer,
    build_scheduler,
    epoch_seed,
    load_checkpoint,
)

FAKE = Path(__file__).with_name("fake_converter.py")
TINY = {
    "model": "ronin_resnet18",
    "model_args": {"group_sizes": [1, 1], "base_plane": 8, "fc_dim": 16, "trans_planes": 4},
    "window": 100, "stride": 25, "eval_stride": 25, "batch": 16, "device": "cpu",
    "workers": 0, "efficiency": False, "plots": False, "lr": 0.003,
}


def make_dataset(root: Path, n_train: int = 4, duration: float = 10.0) -> Path:
    entries = [{"id": f"tr{k}", "seed": k, "group": f"g{k}", "split": "train",
                "duration": duration, "imu_rate": 200.0} for k in range(n_train)]
    entries.append({"id": "te0", "seed": 50, "group": "gx", "split": "test",
                    "duration": duration, "imu_rate": 200.0})
    (root / "raw").mkdir(parents=True)
    (root / "raw" / "spec.json").write_text(json.dumps({"sequences": entries}))
    convert_dataset("fake", root / "raw", root / "fake", converter=FAKE, val_fraction=0.25)
    return root / "fake"


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    return make_dataset(tmp_path_factory.mktemp("engine"))


@pytest.fixture(scope="module")
def zero_model():
    @register_model("test_zero_model")
    class Zero(BaseModel):
        def __init__(self, input_spec):
            super().__init__(input_spec)
            self.bias = torch.nn.Parameter(torch.zeros(input_spec.dims))

        def forward(self, imu):
            return {"vel": self.bias.expand(imu.shape[0], -1)}

    yield {"name": "zero", "arch": "test_zero_model"}
    MODELS.pop("test_zero_model")


def test_integration_helpers():
    t = np.arange(0, 5, 0.005)
    vel = np.tile([1.0, -2.0], (len(t), 1))
    np.testing.assert_allclose(integrate(t, vel)[-1], [t[-1], -2 * t[-1]])
    # 线性变化速度的梯形积分对分段线性速度是精确的
    tw = np.array([1.0, 3.0])
    pos = reconstruct(t, tw, np.array([[0.0, 0.0], [2.0, 0.0]]), 10, np.array([5.0, 5.0, 1.0]))
    np.testing.assert_allclose(pos[10], [5.0, 5.0])
    v = np.interp(t, tw, [0.0, 2.0])
    expected = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * 0.005)])
    np.testing.assert_allclose(pos[:, 0] - pos[10, 0], expected - expected[10], atol=1e-12)
    values = np.array([[1.0], [9.0], [3.0], [9.0]])
    filled = fill_invalid_windows(np.arange(4.0), values, np.array([True, False, True, False]))
    np.testing.assert_allclose(filled[:, 0], [1.0, 2.0, 3.0, 3.0])


def test_window_center_timestamps_make_oracle_exact(zero_model):
    seq = make_sequence(duration=30.0, seed=4)
    cfg = get_cfg({"model": zero_model, "device": "cpu", "eval_stride": 10})
    predictor = Predictor(cfg)
    res = predictor.predict_sequence(seq)
    metrics = res.compute_metrics()
    # 零速度模型：预测轨迹停在起点
    np.testing.assert_allclose(res.pos_pred, np.tile(seq.position[0, :2], (len(seq), 1)))
    assert metrics["ate"] > 1.0
    # oracle 轨迹（窗口目标 + 中心时间戳）几乎复现参考轨迹
    assert metrics["ate_oracle"] < 0.02
    # 若把时间戳错误地放在窗口起点（旧实现的错误），oracle 误差明显变大
    view = SequenceView(seq, predictor.view_cfg)
    starts = view.starts(10, require_valid=False)
    wrong = reconstruct(seq.timestamp, seq.timestamp[starts], view.targets_world(starts)[:, :2],
                        0, seq.position[0])
    wrong_ate = float(np.sqrt(np.mean(np.sum((wrong - seq.position[:, :2]) ** 2, axis=1))))
    assert wrong_ate > 5 * metrics["ate_oracle"]
    # 首尾半窗常值外推，覆盖完整时间轴
    assert res.pos_pred.shape == (len(seq), 2) and np.isfinite(res.pos_oracle).all()


def test_predictor_handles_invalid_and_short(zero_model):
    cfg = get_cfg({"model": zero_model, "device": "cpu"})
    predictor = Predictor(cfg)
    short = predictor.predict_sequence(make_sequence(duration=0.5))
    assert short.skipped and "shorter" in short.skipped
    seq = make_sequence(duration=12.0, seed=1)
    seq.valid_pose[:300] = False
    seq.valid_imu[1000:1100] = False
    res = predictor.predict_sequence(seq, collect_loss=True)
    assert not res.window_valid.all()
    np.testing.assert_allclose(res.pos_pred[300], seq.position[300, :2])  # 锚定首个有效位姿
    assert res.loss is not None and res.loss > 0
    m = res.compute_metrics()
    assert m["num_windows"] == int(res.window_valid.sum())


def test_predictor_saves_extra_window_outputs(tmp_path):
    @register_model("test_extra_outputs_model")
    class Extra(BaseModel):
        def __init__(self, input_spec):
            super().__init__(input_spec)
            self.scale = torch.nn.Parameter(torch.ones(1))

        def forward(self, imu):
            b = imu.shape[0]
            mean = imu[:, 3:5].mean(dim=-1) * self.scale
            return {"vel": mean, "speed": mean.norm(dim=-1),
                    "cov": torch.eye(2).expand(b, 2, 2) * self.scale,
                    "scalar": self.scale.sum(), "aux": {"ignored": mean}}

    try:
        cfg = get_cfg({"model": {"name": "extra", "arch": "test_extra_outputs_model"},
                       "device": "cpu", "val_batch": 7})
        predictor = Predictor(cfg)
        res = predictor.predict_sequence(make_sequence(duration=6.0, seed=2), collect_loss=True)
        k = len(res.starts)
        assert set(res.outputs) == {"speed", "cov"}  # 非逐窗口张量与非张量不保存
        assert res.outputs["speed"].shape == (k,) and res.outputs["cov"].shape == (k, 2, 2)
        np.testing.assert_allclose(res.outputs["speed"],
                                   np.linalg.norm(res.vel_pred, axis=1), rtol=1e-5)
        res.compute_metrics()
        back = SequenceResult.load(res.save(tmp_path / "x.npz"))
        assert set(back.outputs) == {"speed", "cov"}
        np.testing.assert_array_equal(back.outputs["cov"], res.outputs["cov"])
        # saved_outputs 限定保存范围
        predictor.model.saved_outputs = ("cov",)
        res = predictor.predict_sequence(make_sequence(duration=6.0, seed=2))
        assert set(res.outputs) == {"cov"}
    finally:
        MODELS.pop("test_extra_outputs_model")


def test_sequence_and_run_result_io(tmp_path, zero_model):
    predictor = Predictor(get_cfg({"model": zero_model, "device": "cpu"}))
    results = [predictor.predict_sequence(make_sequence(duration=8.0, seed=k,
                                                        sequence_id=f"s{k}")) for k in range(2)]
    results.append(SequenceResult(sequence_id="skip", skipped="too short"))
    for r in results:
        r.compute_metrics()
    path = results[0].save(tmp_path / "a.npz")
    back = SequenceResult.load(path)
    np.testing.assert_array_equal(back.pos_pred, results[0].pos_pred)
    assert back.metrics["ate"] == pytest.approx(results[0].metrics["ate"])
    run = RunResult(results, dataset="syn", split="test", model="zero",
                    cfg={"seed": 3, "window": 200})
    agg = run.aggregate()
    assert agg["count"]["ate"] == 2
    assert run.metrics["ate"] == pytest.approx(np.mean([r.metrics["ate"] for r in results[:2]]))
    out = run.save(tmp_path / "run", plots=True, max_plots=1)
    meta = json.loads((out / "metrics.json").read_text())
    assert meta["num_sequences"] == 2 and meta["num_skipped"] == 1 and meta["seed"] == 3
    assert (out / "sequences.csv").read_text().count("\n") == 4
    assert sorted(p.name for p in (out / "predictions").iterdir()) == ["s0.npz", "s1.npz"]
    assert (out / "plots" / "trajectories.png").exists()
    assert "ate=" in run.summary()


def test_trainer_outputs_callbacks_and_checkpoint(dataset, tmp_path):
    events = []
    trainer = Trainer(overrides={**TINY, "data": str(dataset), "epochs": 2,
                                 "project": str(tmp_path), "name": "t", "plots": True})
    for event in ("on_train_start", "on_train_epoch_end", "on_val_end", "on_fit_epoch_end",
                  "on_model_save", "on_train_end"):
        trainer.add_callback(event, lambda obj, e=event: events.append(e))
    metrics = trainer.train()
    d = tmp_path / "train" / "t"
    assert trainer.save_dir == d
    for name in ("args.yaml", "env.json", "results.csv", "metrics.json", "sequences.csv",
                 "log.txt", "weights/best.pt", "weights/last.pt", "plots/results.png"):
        assert (d / name).exists(), name
    assert events[0] == "on_train_start" and events[-1] == "on_train_end"
    assert events.count("on_fit_epoch_end") == 2
    rows = (d / "results.csv").read_text().strip().splitlines()
    assert len(rows) == 3 and "val/ate" in rows[0] and "train/loss" in rows[0]
    meta = json.loads((d / "metrics.json").read_text())
    assert meta["mode"] == "train" and meta["split"] == "val" and "best_epoch" in meta
    assert meta["metrics"]["ate"] == pytest.approx(metrics["ate"])
    env = json.loads((d / "env.json").read_text())
    assert env["dataset"]["fingerprint"] and "commit" in env["git"]
    ckpt = load_checkpoint(d / "weights" / "last.pt")
    for key in ("model", "model_cfg", "input_spec", "cfg", "epoch", "optimizer", "git",
                "dataset_fingerprint", "model_name"):
        assert key in ckpt, key
    assert ckpt["model_cfg"]["args"]["base_plane"] == 8
    assert ckpt["input_spec"]["window"] == 100
    best = load_checkpoint(d / "weights" / "best.pt")
    assert best["optimizer"] is None
    assert yaml_load(d / "args.yaml")["epochs"] == 2


def test_resume_after_interruption(dataset, tmp_path):
    over = {**TINY, "data": str(dataset), "epochs": 3, "project": str(tmp_path), "name": "r"}
    trainer = Trainer(overrides=over)

    def stop(obj):
        if obj.epoch == 0:
            raise KeyboardInterrupt

    trainer.add_callback("on_fit_epoch_end", stop)
    with pytest.raises(KeyboardInterrupt):
        trainer.train()
    last = tmp_path / "train" / "r" / "weights" / "last.pt"
    assert load_checkpoint(last)["epoch"] == 0
    resumed = Trainer(overrides={"resume": True, "project": str(tmp_path), "name": "r"})
    resumed.train()
    assert resumed.start_epoch == 1 and resumed.save_dir == tmp_path / "train" / "r"
    rows = (tmp_path / "train" / "r" / "results.csv").read_text().strip().splitlines()[1:]
    assert [int(r.split(",")[0]) for r in rows] == [0, 1, 2]
    assert load_checkpoint(last)["epoch"] == 2


def test_training_is_deterministic(dataset, tmp_path):
    weights = []
    for name in ("a", "b"):
        over = {**TINY, "data": str(dataset), "epochs": 1, "project": str(tmp_path),
                "name": name, "save_predictions": False}
        Trainer(overrides=over).train()
        weights.append(load_checkpoint(tmp_path / "train" / name / "weights" / "last.pt"))
    for k, v in weights[0]["model"].items():
        assert torch.equal(v, weights[1]["model"][k]), k


def test_nio_facade(dataset, tmp_path):
    model = NIO("ronin_resnet18", **{k: v for k, v in TINY.items() if k != "model"})
    assert model.model.num_params < 50_000
    model.train(data=str(dataset), epochs=1, project=str(tmp_path), name="n")
    ckpt = tmp_path / "train" / "n" / "weights" / "best.pt"
    assert model.ckpt_path == str(ckpt)

    loaded = NIO(str(ckpt))
    result = loaded.val(data=str(dataset), split="test", project=str(tmp_path), name="v",
                        device="cpu", efficiency=True, plots=False)
    meta = json.loads((tmp_path / "val" / "v" / "metrics.json").read_text())
    assert meta["split"] == "test" and meta["weights"] == str(ckpt)
    assert meta["protocol"]["window"] == 100 and meta["seed"] == 0
    assert meta["efficiency"]["params"] == loaded.model.num_params
    assert result.metrics["ate"] == pytest.approx(meta["metrics"]["ate"])
    traj = loaded.predict(dataset / "sequences" / "te0.h5", device="cpu", save=True,
                          project=str(tmp_path), name="p")
    assert traj.pos.shape[1] == 2 and (tmp_path / "predict" / "p" / "predictions").is_dir()
    trajs = loaded.predict(dataset, device="cpu")
    assert len(trajs) == 5
    info = loaded.info(verbose=False)
    assert info["parameters"] == loaded.model.num_params and info["flops"] > 0
    with pytest.raises(ConfigError, match="conflicts"):
        get_cfg({"model": str(ckpt), "window": 200})
    # 用 checkpoint 继续训练：作为 pretrained 加载
    loaded.train(data=str(dataset), epochs=1, project=str(tmp_path), name="ft", device="cpu")
    ft = load_checkpoint(tmp_path / "train" / "ft" / "weights" / "last.pt")
    assert ft["cfg"]["pretrained"] == str(ckpt)


def test_nio_custom_trainer(dataset, tmp_path):
    class Counting(Trainer):
        calls = 0

        def fitness(self, metrics):
            type(self).calls += 1
            return super().fitness(metrics)

    model = NIO("ronin_resnet18", **{k: v for k, v in TINY.items() if k != "model"})
    model.train(data=str(dataset), epochs=1, project=str(tmp_path), name="c", trainer=Counting,
                save_predictions=False)
    assert isinstance(model.trainer, Counting) and Counting.calls == 1
    with pytest.raises(TypeError, match="Trainer subclass"):
        model.train(data=str(dataset), trainer=object)


def test_training_requires_data():
    with pytest.raises(ConfigError, match="data is required"):
        Trainer(overrides={"model": "ronin_resnet18"})


def test_early_stopping_and_schedulers():
    stopper = EarlyStopping(patience=2)
    assert stopper.update(0, 1.0) and not stopper.update(1, 1.5)
    assert not stopper.update(2, math.nan)
    assert not stopper.should_stop(1) and stopper.should_stop(2)
    assert not EarlyStopping(0).should_stop(100)

    params = [torch.nn.Parameter(torch.zeros(1))]
    cases = (("cosine", [1.0, 0.7525, 0.2575, 0.01]), ("step", [1.0, 0.1, 0.01, 1e-3]),
             ("none", [1.0] * 4))
    for name, expected in cases:
        opt = torch.optim.SGD(params, lr=1.0)
        cfg = get_cfg({"scheduler": name, "epochs": 3, "lr_final": 0.01, "step_size": 1,
                       "gamma": 0.1})
        sched = build_scheduler(opt, cfg)
        lrs = []
        for _ in range(4):
            lrs.append(opt.param_groups[0]["lr"])
            opt.step()
            sched.step()
        np.testing.assert_allclose(lrs, expected, rtol=1e-6)
    opt = torch.optim.SGD(params, lr=1.0)
    warm = build_scheduler(opt, get_cfg({"scheduler": "none", "warmup_epochs": 3}))
    lrs = []
    for _ in range(4):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        warm.step()
    np.testing.assert_allclose(lrs, [0.25, 0.5, 0.75, 1.0])
    plateau = build_scheduler(torch.optim.SGD(params, lr=1.0), get_cfg({"scheduler": "plateau"}))
    assert isinstance(plateau, torch.optim.lr_scheduler.ReduceLROnPlateau)


def test_merge_outputs_strategies():
    times = np.array([0.0, 0.5, 0.5, 1.0])
    values = np.array([[1.0], [2.0], [6.0], [3.0]])
    distance = np.array([0.0, 0.4, 0.1, 0.0])
    valid = np.ones(4, bool)
    t, mean, ok = merge_outputs(times, values, valid, distance, "mean", resolution=0.5)
    np.testing.assert_allclose(t, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(mean[:, 0], [1.0, 4.0, 3.0])  # 同一时刻取平均
    assert ok.all()
    _, center, _ = merge_outputs(times, values, valid, distance, "center", resolution=0.5)
    np.testing.assert_allclose(center[:, 0], [1.0, 6.0, 3.0])  # 取最靠窗口中心的那个
    # 无效输出不参与平均；某一时刻全部无效时保留数值但标记为无效
    valid = np.array([True, False, True, False])
    _, mean2, ok2 = merge_outputs(times, values, valid, distance, "mean", resolution=0.5)
    np.testing.assert_allclose(mean2[:, 0], [1.0, 6.0, 3.0])
    np.testing.assert_array_equal(ok2, [True, True, False])
    with pytest.raises(ValueError, match="overlap"):
        merge_outputs(times, values, valid, distance, "first")


@pytest.fixture()
def layout_model():
    @register_model("test_layout_model")
    class Layout(BaseModel):
        """输出恒为零、但形状随 ``output_shape`` 变化的模型（用于检验协议映射）。"""

        def __init__(self, input_spec):
            super().__init__(input_spec)
            self.bias = torch.nn.Parameter(torch.zeros(input_spec.output_shape))

        def forward(self, imu):
            self.check_input(imu)
            return {"vel": self.bias.expand(imu.shape[0], *self.bias.shape)}

    yield {"name": "layout", "arch": "test_layout_model"}
    MODELS.pop("test_layout_model")


@pytest.mark.parametrize("overrides,tol", [
    ({"target": "avg_velocity"}, 0.05),
    ({"target": "frame_velocity"}, 0.05),
    # 窗口末端瞬时速度在 eval_stride 上采样较粗（合成信号含 1.8 Hz 起伏），容许更大误差
    ({"target": "velocity_at_end"}, 0.3),
    ({"target": "multi_displacement", "output_steps": 10, "window": 101}, 0.05),
    ({"target": "frame_velocity", "overlap": "center"}, 0.05),
    ({"target": "multi_displacement", "output_steps": 5, "window": 101, "overlap": "center"},
     0.05),
])
def test_multi_output_protocol_reconstructs_the_reference(layout_model, overrides, tol):
    """每个输出映射到正确的时间戳：用窗口目标积分（oracle）必须几乎复现参考轨迹。"""
    seq = make_sequence(duration=25.0, seed=6)
    cfg = get_cfg({"model": layout_model, "device": "cpu", "window": 100, "dims": 3,
                   "eval_stride": 10, "metric_dims": 3, **overrides})
    res = Predictor(cfg).predict_sequence(seq)
    metrics = res.compute_metrics(dims=3)
    assert metrics["ate_oracle"] < tol, metrics["ate_oracle"]
    assert len(res.t_window) == len(res.vel_pred) == len(res.vel_target)
    assert np.all(np.diff(res.t_window) > 0)  # 合并后时间轴严格递增
    np.testing.assert_allclose(res.vel_pred, 0.0)  # 零输出模型
    np.testing.assert_allclose(res.pos_pred, np.tile(seq.position[0], (len(seq), 1)), atol=1e-9)


def test_frame_velocity_overlap_strategies_agree(layout_model):
    """逐帧速度的目标与窗口无关，因此 mean 与 center 合并结果必须一致。"""
    seq = make_sequence(duration=15.0, seed=7)
    base = {"model": layout_model, "device": "cpu", "window": 100, "dims": 3,
            "target": "frame_velocity", "eval_stride": 10, "metric_dims": 3}
    a = Predictor(get_cfg({**base, "overlap": "mean"})).predict_sequence(seq)
    b = Predictor(get_cfg({**base, "overlap": "center"})).predict_sequence(seq)
    np.testing.assert_allclose(a.t_window, b.t_window)
    np.testing.assert_allclose(a.vel_target, b.vel_target, atol=1e-6)
    # 逐帧布局的时间轴就是样本时间轴（步长 1 个样本）
    np.testing.assert_allclose(a.t_window, seq.timestamp[: len(a.t_window)], atol=1e-9)


def test_history_and_extra_inputs_reach_the_model():
    @register_model("test_history_extra_model")
    class HistoryExtra(BaseModel):
        def __init__(self, input_spec):
            super().__init__(input_spec)
            self.gain = torch.nn.Parameter(torch.ones(1))
            self.seen: dict = {}

        def forward(self, imu, extra):
            self.check_input(imu)
            self.seen = {k: tuple(v.shape) for k, v in extra.items()}
            # 特权输入：直接把参考初速度当作预测
            return {"vel": extra["init_velocity"] * self.gain}

    try:
        seq = make_sequence(duration=15.0, seed=8)
        cfg = get_cfg({"model": {"name": "hx", "arch": "test_history_extra_model"},
                       "device": "cpu", "window": 100, "dims": 3, "metric_dims": 3,
                       "history": 3, "history_stride": 50, "eval_stride": 10,
                       "extra_inputs": ["init_velocity", "orientation"]})
        predictor = Predictor(cfg)
        res = predictor.predict_sequence(seq)
        assert predictor.model.seen["init_velocity"][1:] == (3,)
        assert predictor.model.seen["orientation"][1:] == (3, 100, 4)
        assert res.starts[0] == 100  # 历史子窗口不越过序列开头
        metrics = res.compute_metrics(dims=3)
        assert metrics["ate"] < 2.0  # 用真值初速度预测，轨迹接近参考
    finally:
        MODELS.pop("test_history_extra_model")


def test_privileged_inputs_are_marked_in_results(dataset, tmp_path):
    @register_model("test_privileged_model")
    class Privileged(BaseModel):
        def __init__(self, input_spec):
            super().__init__(input_spec)
            self.gain = torch.nn.Parameter(torch.ones(1))

        def forward(self, imu, extra):
            return {"vel": extra["init_velocity"] * self.gain}

    try:
        model = NIO({"name": "priv", "arch": "test_privileged_model",
                     "input": {"window": 100, "extra_inputs": ["init_velocity"]}},
                    device="cpu", workers=0, efficiency=False, plots=False)
        result = model.val(data=str(dataset), split="test", project=str(tmp_path), name="priv")
        meta = json.loads((tmp_path / "val" / "priv" / "metrics.json").read_text())
        assert meta["privileged_inputs"] == ["init_velocity"]
        assert result.privileged_inputs == ["init_velocity"]
    finally:
        MODELS.pop("test_privileged_model")


def test_sequence_model_uses_the_same_trajectory_pipeline():
    @register_model("test_oracle_sequence_model")
    class OracleSequence(SequenceModel):
        def __init__(self, input_spec):
            super().__init__(input_spec)
            self.gain = torch.nn.Parameter(torch.ones(1))
            self.calibrated = 0

        def calibrate(self, views, split="train"):
            self.calibrated = len(views)
            return {"scale": 1.0, "sequences": len(views)}

        def predict_sequence(self, seq, view, starts):
            return view.target_times(starts), view.targets(starts), {"note": np.zeros(len(starts))}

    try:
        seq = make_sequence(duration=20.0, seed=9)
        cfg = get_cfg({"model": {"name": "oracle_seq", "arch": "test_oracle_sequence_model"},
                       "device": "cpu", "window": 100, "dims": 3, "metric_dims": 3,
                       "eval_stride": 10})
        res = Predictor(cfg).predict_sequence(seq)
        metrics = res.compute_metrics(dims=3)
        assert metrics["ate"] < 0.05 and metrics["ate"] == pytest.approx(metrics["ate_oracle"])
        assert res.outputs["note"].shape == res.starts.shape
        # 时间戳必须落在窗口网格上
        model = Predictor(cfg).model
        model.predict_sequence = lambda seq, view, starts: (np.array([0.0]), np.zeros((1, 3)))
        with pytest.raises(ValueError, match="window grid"):
            Predictor(cfg, model=model).predict_sequence(seq)
    finally:
        MODELS.pop("test_oracle_sequence_model")


def test_predictions_are_only_filled_where_the_input_is_invalid():
    """位姿缺口不应替换模型预测：模型穿越缺口的漂移必须被评测到。"""
    @register_model("test_mean_acc_model")
    class MeanAcc(BaseModel):
        def __init__(self, input_spec):
            super().__init__(input_spec)
            self.gain = torch.nn.Parameter(torch.ones(1))

        def forward(self, imu):
            return {"vel": imu[:, 3:5].mean(dim=-1) * self.gain}

    seq = make_sequence(duration=20.0, seed=7, device_yaw_offset=0.4)
    seq.valid_pose[1000:2000] = False  # 只缺位姿，IMU 与设备姿态正常
    try:
        cfg = get_cfg({"model": {"name": "mean", "arch": "test_mean_acc_model"},
                       "device": "cpu", "orientation": "device", "eval_stride": 20})
        predictor = Predictor(cfg)
        res = predictor.predict_sequence(seq)
        gap = ~res.window_valid_target
        assert gap.any() and res.window_valid_input.all()
        assert not res.window_valid[gap].any()  # 窗口级指标仍然排除这些窗口
        view = SequenceView(seq, predictor.view_cfg)
        raw = predictor.infer(view)
        world = view.to_world_velocity(raw["vel"], raw["starts"])
        np.testing.assert_allclose(res.vel_pred[gap], world[gap], rtol=1e-6)  # 预测未被插值
        # 目标在位姿缺口处被插值替换（否则位置差分会横跨缺口）
        assert np.isfinite(res.vel_target).all()
        m = res.compute_metrics()
        assert m["num_windows"] == int(res.window_valid.sum()) < len(res.starts)
    finally:
        MODELS.pop("test_mean_acc_model")


def test_body_frame_windows_ignore_pose_gaps_for_the_input(zero_model):
    """frame=body 且不去重力时输入完全不用姿态，输入掩码只看 valid/imu。"""
    seq = make_sequence(duration=8.0, seed=8)
    seq.valid_pose[400:800] = False
    seq.valid_imu[1000:1100] = False
    view = SequenceView(seq, ViewConfig(frame="body", dims=3, window=100))
    assert view.valid_input[500] and not view.valid_input[1050]
    np.testing.assert_array_equal(view.valid_input, seq.valid_imu)
    np.testing.assert_array_equal(view.valid_target, seq.valid_pose)
    # orientation=reference 时输入旋转来自参考姿态，因此位姿缺口也让输入无效
    world = SequenceView(seq, ViewConfig(frame="gravity_world", window=100))
    assert not world.valid_input[500]


def _interrupt_after(trainer, epoch: int) -> None:
    def stop(obj):
        if obj.epoch == epoch:
            raise KeyboardInterrupt

    trainer.add_callback("on_fit_epoch_end", stop)


def test_resume_reproduces_uninterrupted_weights(dataset, tmp_path):
    """连续训练 3 轮与“中断后续训到 3 轮”必须得到逐位相同的权重。"""
    base = {**TINY, "data": str(dataset), "epochs": 3, "project": str(tmp_path),
            "save_predictions": False, "plots": False}
    Trainer(overrides={**base, "name": "full"}).train()
    part = Trainer(overrides={**base, "name": "part"})
    _interrupt_after(part, 0)
    with pytest.raises(KeyboardInterrupt):
        part.train()
    Trainer(overrides={"resume": True, "project": str(tmp_path), "name": "part"}).train()
    full = load_checkpoint(tmp_path / "train" / "full" / "weights" / "last.pt")
    resumed = load_checkpoint(tmp_path / "train" / "part" / "weights" / "last.pt")
    assert full["epoch"] == resumed["epoch"] == 2
    for k, v in full["model"].items():
        assert torch.equal(v, resumed["model"][k]), k


def test_epoch_seeding_shuffles_differently_each_epoch(dataset, tmp_path):
    """每轮的 shuffle 顺序由 ``(seed, epoch)`` 决定：同一轮可复现，不同轮不相同。"""
    orders: dict = {}

    def record(obj):
        orders.setdefault((obj.args.name, obj.epoch), []).append(
            int(obj.batch["start"][0]) * 10 + int(obj.batch["seq"][0]))

    for name in ("s1", "s2"):
        trainer = Trainer(overrides={**TINY, "data": str(dataset), "epochs": 2,
                                     "project": str(tmp_path), "name": name,
                                     "save_predictions": False, "plots": False})
        trainer.add_callback("on_train_batch_start", record)
        trainer.train()
    assert orders[("s1", 0)] == orders[("s2", 0)]  # 同一轮可复现
    assert orders[("s1", 1)] == orders[("s2", 1)]
    assert orders[("s1", 0)] != orders[("s1", 1)]  # 不同轮顺序不同
    assert epoch_seed(0, 0) != epoch_seed(0, 1) and epoch_seed(0, 3) == epoch_seed(0, 3)


def test_setup_failure_releases_the_log_handler(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "spec.json").write_text(json.dumps({"sequences": [
        {"id": "tr0", "seed": 0, "group": "g0", "split": "train", "duration": 0.4,
         "imu_rate": 200.0},
        {"id": "te0", "seed": 1, "group": "g1", "split": "test", "duration": 0.4,
         "imu_rate": 200.0}]}))
    convert_dataset("fake", tmp_path / "raw", tmp_path / "tiny", converter=FAKE,
                    val_fraction=0.5, min_duration=0.1)
    before = list(LOGGER.handlers)
    trainer = Trainer(overrides={**TINY, "data": str(tmp_path / "tiny"), "epochs": 1,
                                 "project": str(tmp_path), "name": "bad"})
    with pytest.raises(RuntimeError, match="no valid windows"):
        trainer.train()
    assert LOGGER.handlers == before


def test_best_checkpoint_is_written_before_last(dataset, tmp_path, monkeypatch):
    """``last.pt`` 记录的早停状态必须与磁盘上的 ``best.pt`` 对应（best 先落盘）。"""
    order = []
    real = trainer_module.save_checkpoint

    def spy(path, ckpt):
        order.append(Path(path).name)
        return real(path, ckpt)

    monkeypatch.setattr(trainer_module, "save_checkpoint", spy)
    Trainer(overrides={**TINY, "data": str(dataset), "epochs": 1, "project": str(tmp_path),
                       "name": "order", "save_predictions": False, "plots": False}).train()
    assert order[:2] == ["best.pt", "last.pt"]
    last = load_checkpoint(tmp_path / "train" / "order" / "weights" / "last.pt")
    best = load_checkpoint(tmp_path / "train" / "order" / "weights" / "best.pt")
    assert last["stopper"]["best_epoch"] == best["epoch"]


def test_load_checkpoint_returns_independent_copies(dataset, tmp_path):
    """缓存不能与调用方共享存储：优化器状态是原地更新的。"""
    Trainer(overrides={**TINY, "data": str(dataset), "epochs": 1, "project": str(tmp_path),
                       "name": "cache", "save_predictions": False, "plots": False}).train()
    last = tmp_path / "train" / "cache" / "weights" / "last.pt"
    first, second = load_checkpoint(last), load_checkpoint(last)
    key = next(iter(first["model"]))
    assert first is not second and first["model"][key] is not second["model"][key]
    first["model"][key].add_(1.0)
    assert not torch.equal(first["model"][key], second["model"][key])
    assert torch.equal(load_checkpoint(last)["model"][key], second["model"][key])
    # 优化器状态经 load_state_dict 后原地更新，不得污染缓存
    state = load_checkpoint(last)["optimizer"]
    tensors = [v for s in state["state"].values() for v in s.values() if torch.is_tensor(v)]
    if tensors:
        before = [t.clone() for t in tensors]
        for t in tensors:
            t.mul_(2.0)
        after = load_checkpoint(last)["optimizer"]
        again = [v for s in after["state"].values() for v in s.values() if torch.is_tensor(v)]
        for expected, actual in zip(before, again):
            assert torch.equal(expected, actual)


def test_sgd_without_momentum_disables_nesterov():
    model = torch.nn.Linear(3, 2)
    assert not build_optimizer(model, "sgd", 0.1, momentum=0.0).param_groups[0]["nesterov"]
    assert build_optimizer(model, "sgd", 0.1, momentum=0.9).param_groups[0]["nesterov"]


def test_view_config_from_model_spec(zero_model):
    cfg = get_cfg({"model": zero_model, "frame": "gravity_yaw_local", "dims": 3})
    predictor = Predictor(cfg, device=torch.device("cpu"))
    assert predictor.view_cfg == ViewConfig(frame="gravity_yaw_local", dims=3)
    res = predictor.predict_sequence(make_sequence(duration=5.0))
    assert res.pos_pred.shape[1] == 3
