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
from inertial_benchmark.engine.benchmark import run_benchmark  # noqa: E402
from inertial_benchmark.engine.budget import (  # noqa: E402
    budget_epochs,
    count_train_windows,
    is_sequence_model,
    resolve_budget,
)
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
    check_loss_batch,
    register_model,
)
from inertial_benchmark.nn.losses import masked_output_mean  # noqa: E402
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


def test_median_fitness_selects_on_the_per_sequence_median(dataset, tmp_path):
    """``fitness_stat=median`` 必须按逐序列 ATE 的中位数选模，而不是均值。

    理由：自动生成的 val 划分可能只有一名受试者、且含 train 中不存在的携带方式（RIDI），
    均值会被少数序列主导，选模信号很噪。
    """
    data = make_dataset(tmp_path / "ds", n_train=12, duration=8.0)  # val 有 3 条，中位数≠均值
    over = {**TINY, "data": str(data), "epochs": 1, "project": str(tmp_path),
            "save_predictions": False, "fitness": "ate", "fitness_stat": "median",
            "name": "med"}
    trainer = Trainer(overrides=over)
    trainer.train()
    agg = trainer.val_result.aggregate()
    assert len(trainer.val_result.evaluated) == 3
    assert agg["median"]["ate"] != pytest.approx(agg["mean"]["ate"], rel=1e-6)
    assert trainer.stopper.best == pytest.approx(agg["median"]["ate"], rel=1e-9)
    assert trainer.val_result.value("ate", "median") == pytest.approx(agg["median"]["ate"],
                                                                      rel=1e-9)
    assert trainer.val_result.value("ate") == pytest.approx(agg["mean"]["ate"], rel=1e-9)
    assert trainer.val_result.value("nope") is None
    with pytest.raises(ValueError, match="stat must be one of"):
        trainer.val_result.value("ate", "mode")
    # results.csv 的列集合与 fitness/fitness_stat 无关：只有固定的 fitness 一列
    header = (tmp_path / "train" / "med" / "results.csv").read_text().splitlines()[0].split(",")
    assert "fitness" in header and "val/ate_median" not in header
    # 缺省统计量仍是均值（默认行为不变），且两种设置给出同样的列
    mean_trainer = Trainer(overrides={**over, "fitness_stat": "mean", "name": "mean"})
    mean_trainer.train()
    assert mean_trainer.stopper.best == pytest.approx(
        mean_trainer.val_result.aggregate()["mean"]["ate"], rel=1e-9)
    mean_header = (tmp_path / "train" / "mean" / "results.csv").read_text().splitlines()[0]
    assert mean_header.split(",") == header


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
                    device="cpu", workers=0, efficiency=True, plots=False)
        result = model.val(data=str(dataset), split="test", project=str(tmp_path), name="priv")
        meta = json.loads((tmp_path / "val" / "priv" / "metrics.json").read_text())
        assert meta["privileged_inputs"] == ["init_velocity"]
        assert result.privileged_inputs == ["init_velocity"]
        # 声明了额外输入的模型也能算效率指标（用 InputSpec.dummy_extra 造占位输入）
        assert meta["efficiency"]["latency_ms_cpu"] > 0  # 前向真的跑通了
        assert model.info(verbose=False)["flops_backend"] != "not applicable"
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


def test_train_eval_gap_is_recorded_and_warns(dataset, tmp_path, caplog):
    """train/eval 失配诊断：正常模型比值≈1，故意在 eval 模式缩小输出的模型必须触发告警。

    真实案例是 ronin_resnet18 在统一配方下 eval 模式速度只有训练模式的三分之一。
    """
    over = {**TINY, "data": str(dataset), "epochs": 1, "project": str(tmp_path),
            "save_predictions": False}
    trainer = Trainer(overrides={**over, "name": "gap_ok"})
    trainer.train()
    gap = json.loads((tmp_path / "train" / "gap_ok" / "metrics.json").read_text())
    assert 0.2 < gap["train_eval_gap"]["ratio"] < 5.0
    assert gap["train_eval_gap"]["num_windows"] > 0

    @register_model("test_shrinking_model")
    class Shrinking(BaseModel):
        """eval 模式输出被平移（模拟 dropout/BN 在两种模式下行为不同的头）。"""

        def __init__(self, input_spec):
            super().__init__(input_spec)
            self.scale = torch.nn.Parameter(torch.ones(input_spec.dims))

        def forward(self, imu):
            vel = self.scale.expand(imu.shape[0], -1)
            return {"vel": vel if self.training else vel + 3.0}

    try:
        bad = Trainer(overrides={**over, "name": "gap_bad", "model_args": {},
                                 "model": {"name": "shrink", "arch": "test_shrinking_model"}})
        bad.train()
        with caplog.at_level("WARNING"):
            g = bad.train_eval_gap()
        assert g["ratio"] > 2.0
        assert any("train/eval mismatch" in r.getMessage() for r in caplog.records)
    finally:
        MODELS.pop("test_shrinking_model")


def test_train_eval_gap_handles_per_frame_targets(dataset, tmp_path):
    """逐帧目标（``(B,T,D)`` 输出）下诊断也必须工作：batch 走 ``SequenceView.windows``，带 mask。"""

    @register_model("test_frame_gap_model")
    class Frames(BaseModel):
        def __init__(self, input_spec):
            super().__init__(input_spec)
            self.bias = torch.nn.Parameter(torch.zeros(input_spec.output_shape))

        def forward(self, imu):
            vel = self.bias.expand(imu.shape[0], *self.bias.shape)
            return {"vel": vel if self.training else vel + 0.05}

    try:
        trainer = Trainer(overrides={**TINY, "data": str(dataset), "epochs": 1,
                                     "project": str(tmp_path), "name": "frame_gap",
                                     "save_predictions": False, "model_args": {},
                                     "target": "frame_velocity", "rate": 200.0,
                                     "model": {"name": "frames",
                                               "arch": "test_frame_gap_model"}})
        trainer.train()
        assert trainer.model.input_spec.output_shape == (100, 2)
        gap = trainer.train_eval_gap(windows=64, repeats=1)
        assert gap["num_windows"] > 0 and math.isfinite(gap["ratio"]) and gap["ratio"] > 0
        assert len(gap["sequences"]) >= 1
    finally:
        MODELS.pop("test_frame_gap_model")


def test_oracle_ate_exposes_the_gap_floor(zero_model):
    """多秒 valid 缺口 + 参考位置跳变时，ate_oracle 必须把协议误差下限暴露出来。

    真实数据中（RIDI huayi_bag2 有 4.42 s 缺口）oracle ATE 甚至超过模型 ATE，
    因此 ate_oracle 是默认报表列，缺口序列的 ate 不能单独解读。
    """
    from inertial_benchmark.metrics import MAIN_METRICS

    assert "ate_oracle" in MAIN_METRICS
    predictor = Predictor(get_cfg({"model": zero_model, "device": "cpu", "eval_stride": 10}))
    clean = predictor.predict_sequence(make_sequence(duration=40.0, seed=7)).compute_metrics()
    assert clean["ate_oracle"] < 0.05

    seq = make_sequence(duration=40.0, seed=7)
    gap = slice(3000, 4000)  # 5 s 缺口，缺口后参考位置整体平移 5 m（跟踪重定位）
    seq.valid_pose[gap] = False
    seq.valid_imu[gap] = False
    seq.position[4000:] += np.array([5.0, 0.0, 0.0])
    res = predictor.predict_sequence(seq)
    m = res.compute_metrics()
    assert not res.window_valid.all()
    assert m["ate_oracle"] > 2.0 and math.isfinite(m["ate_oracle"])
    assert all(math.isfinite(m[k]) for k in ("ate", "rte", "pde", "plr", "vel_rmse"))
    assert "ate_oracle" in RunResult([res], cfg={}).metrics
    assert "ate_oracle=" in RunResult([res], cfg={}).summary()


def test_validation_loss_receives_model_input(dataset, tmp_path):
    """验证时的 batch 必须与训练一致（含 ``imu`` 与 ``mask``），否则用到它们的损失只能在训练中工作。

    ``LOSS_BATCH_KEYS`` 只断言键存在；这里用一个显式依赖 ``batch["imu"]``/``batch["mask"]``
    的损失，把 Predictor 的验证路径与 Trainer 的训练路径都跑一遍。
    """

    @register_model("test_input_loss_model")
    class InputLoss(BaseModel):
        def __init__(self, input_spec):
            super().__init__(input_spec)
            self.bias = torch.nn.Parameter(torch.zeros(input_spec.dims))

        def forward(self, imu):
            return {"vel": self.bias.expand(imu.shape[0], -1)}

        def loss(self, out, batch, epoch=0):
            # 显式依赖模型输入与掩码：Trainer 与 Predictor 都必须提供这两个键
            check_loss_batch(batch)
            scale = batch["imu"].abs().mean()
            valid = batch["mask"].to(out["vel"].dtype)
            value = masked_output_mean(((out["vel"] - batch["target"]) ** 2).sum(dim=-1),
                                       valid) + 0.0 * scale
            return value, {"mse": value.detach()}

    try:
        cfg = get_cfg({"model": {"name": "input_loss", "arch": "test_input_loss_model"},
                       "device": "cpu"})
        res = Predictor(cfg).predict_sequence(make_sequence(duration=8.0), collect_loss=True)
        assert res.loss is not None and math.isfinite(res.loss) and res.loss > 0
        # 训练路径也必须能跑（同一组键）
        trainer = Trainer(overrides={**TINY, "data": str(dataset), "epochs": 1,
                                     "project": str(tmp_path), "name": "input_loss",
                                     "save_predictions": False, "model_args": {},
                                     "model": {"name": "input_loss",
                                               "arch": "test_input_loss_model"}})
        assert math.isfinite(trainer.train()["ate"])
    finally:
        MODELS.pop("test_input_loss_model")


def test_trainer_lazy_val_does_not_cache_sequences(dataset, tmp_path):
    """``cache=false`` 时 val 划分也必须惰性读取（只保留路径），否则该开关名不副实。"""
    over = {**TINY, "data": str(dataset), "epochs": 1, "project": str(tmp_path),
            "name": "lazy", "cache": False, "save_predictions": False}
    trainer = Trainer(overrides=over)
    metrics = trainer.train()
    assert trainer.val_set.views == [] and trainer.val_set.lazy
    assert trainer.val_sources and all(isinstance(s, Path) for s in trainer.val_sources)
    assert math.isfinite(metrics["ate"])
    cached = Trainer(overrides={**over, "name": "cached", "cache": True})
    cached_metrics = cached.train()
    assert all(isinstance(s, SequenceView) for s in cached.val_sources)
    # 惰性与缓存路径必须给出完全相同的指标
    assert cached_metrics["ate"] == pytest.approx(metrics["ate"], rel=1e-9)


def test_budget_epochs_arithmetic_and_bounds():
    """等窗口预算的换算：向上取整，下界 1 轮，上界 ``budget_max_epochs``。"""
    assert budget_epochs(900, 300) == 3            # 整除
    assert budget_epochs(901, 300) == 4            # 向上取整：预算不足一轮也要跑完那一轮
    assert budget_epochs(1, 10**6) == 1            # 下界：至少一轮
    assert budget_epochs(10**9, 10, max_epochs=200) == 200   # 上界
    assert budget_epochs(10**9, 10, max_epochs=7) == 7
    for bad in ({"budget": 0}, {"budget": -1}):
        with pytest.raises(ValueError, match="train_windows_budget must be > 0"):
            budget_epochs(windows_per_epoch=10, **bad)
    with pytest.raises(ValueError, match="windows_per_epoch must be > 0"):
        budget_epochs(100, 0)
    with pytest.raises(ValueError, match="budget_max_epochs must be >= 1"):
        budget_epochs(100, 10, max_epochs=0)
    # resolve_budget 记录换算过程；预算为 0 表示关闭
    info = resolve_budget({"train_windows_budget": 1000, "budget_max_epochs": 3}, 300)
    assert info == {"train_windows_budget": 1000, "windows_per_epoch": 300,
                    "epochs_uncapped": 4, "epochs": 3, "budget_max_epochs": 3,
                    "planned_windows": 900}
    assert resolve_budget({"train_windows_budget": 0, "budget_max_epochs": 3}, 300) is None


def test_train_windows_budget_equalises_the_budget_across_dataset_sizes(tmp_path):
    """两个规模不同的数据集在同一预算下得到不同的 epoch 数，但看过的窗口数相当。

    这正是等窗口预算要解决的问题：按固定 epoch 数跑，大数据集会吃掉全部算力。
    """
    small = make_dataset(tmp_path / "small", n_train=4, duration=6.0)
    big = make_dataset(tmp_path / "big", n_train=4, duration=24.0)
    budget = 3000
    seen = {}
    for name, data in (("small", small), ("big", big)):
        over = {**TINY, "data": str(data), "train_windows_budget": budget,
                "budget_max_epochs": 200, "project": str(tmp_path), "name": name,
                "save_predictions": False}
        trainer = Trainer(overrides=over)
        trainer.train()
        windows = len(trainer.train_set)
        info = json.loads((tmp_path / "train" / name / "metrics.json").read_text())
        gap = info["train_budget"]
        assert gap["windows_per_epoch"] == windows
        assert gap["epochs"] == math.ceil(budget / windows) == trainer.args.epochs
        assert gap["planned_windows"] == gap["epochs"] * windows >= budget
        # 换算结果必须落在 args.yaml 里（审计：这个 run 到底跑了几轮）
        assert yaml_load(tmp_path / "train" / name / "args.yaml")["epochs"] == gap["epochs"]
        seen[name] = gap
    assert seen["big"]["windows_per_epoch"] > seen["small"]["windows_per_epoch"]
    assert seen["big"]["epochs"] < seen["small"]["epochs"]
    # 两个数据集看过的窗口数都在预算的一轮误差内（这才叫“等预算”）
    for gap in seen.values():
        assert budget <= gap["planned_windows"] < budget + gap["windows_per_epoch"]


def test_train_windows_budget_respects_the_epoch_cap(dataset, tmp_path, caplog):
    """预算需要的轮数超过 ``budget_max_epochs`` 时截断，并记下未截断的轮数与告警。"""
    over = {**TINY, "data": str(dataset), "train_windows_budget": 10**9,
            "budget_max_epochs": 1, "project": str(tmp_path), "name": "cap",
            "save_predictions": False}
    with caplog.at_level("WARNING", logger=LOGGER.name):
        trainer = Trainer(overrides=over)
        trainer.train()
    assert trainer.args.epochs == 1
    gap = json.loads((tmp_path / "train" / "cap" / "metrics.json").read_text())["train_budget"]
    assert gap["epochs"] == 1 and gap["epochs_uncapped"] > 1
    assert gap["planned_windows"] < gap["train_windows_budget"]
    assert any("budget_max_epochs=1 caps it" in r.message for r in caplog.records)
    # 预算关闭（缺省 0）时 epochs 原样生效，metrics.json 的 train_budget 为空
    plain = Trainer(overrides={**TINY, "data": str(dataset), "epochs": 1,
                               "project": str(tmp_path), "name": "plain",
                               "save_predictions": False})
    plain.train()
    meta = json.loads((tmp_path / "train" / "plain" / "metrics.json").read_text())
    assert plain.args.epochs == 1 and meta["train_budget"] == {} and meta["epochs"] == 1


def test_count_train_windows_matches_the_training_dataset(dataset):
    """规划用的窗口计数必须与训练时实际构建的数据集一致（否则 dry_run 的轮数是假的）。"""
    cfg = get_cfg({**TINY, "data": str(dataset), "train_windows_budget": 5000})
    trainer = Trainer(cfg=cfg)
    trainer.setup()
    try:
        assert count_train_windows(cfg) == len(trainer.train_set)
        assert trainer.args.epochs == budget_epochs(5000, len(trainer.train_set), 200)
    finally:
        trainer._close_log()


def test_sequence_models_ignore_the_window_budget(dataset, tmp_path):
    """序列级模型只在 train 划分上标定一次，不适用等窗口预算（epochs 不被改写）。"""
    @register_model("test_budget_sequence_model")
    class Calib(SequenceModel):
        def __init__(self, input_spec):
            super().__init__(input_spec)
            self.gain = torch.nn.Parameter(torch.ones(1))

        def calibrate(self, views, split="train"):
            return {"sequences": len(views)}

        def predict_sequence(self, seq, view, starts):
            return view.target_times(starts), view.targets(starts)

    try:
        model = {"name": "calib", "arch": "test_budget_sequence_model"}
        assert is_sequence_model(model) and not is_sequence_model("ronin_resnet18")
        over = {**{k: v for k, v in TINY.items() if k not in ("model", "model_args")},
                "model": model,
                "data": str(dataset), "epochs": 1, "train_windows_budget": 10**9,
                "project": str(tmp_path), "name": "seqmodel", "save_predictions": False}
        trainer = Trainer(overrides=over)
        trainer.train()
        assert trainer.args.epochs == 1 and trainer.budget == {}
        meta = json.loads((tmp_path / "train" / "seqmodel" / "metrics.json").read_text())
        assert meta["train_budget"] == {}
    finally:
        MODELS.pop("test_budget_sequence_model")


def test_benchmark_dry_run_plans_epochs_per_dataset(dataset, tmp_path):
    """``dry_run`` 必须在开跑之前给出每个 run 的 epoch 数，并且只计划 ``splits`` 指定的划分。"""
    plan = {"name": "b", "project": str(tmp_path), "models": ["ronin_resnet18"],
            "datasets": [str(dataset)], "seeds": [0], "splits": ["val"], "report": False,
            "overrides": {**TINY, "train_windows_budget": 5000, "budget_max_epochs": 200}}
    statuses = run_benchmark(plan, dry_run=True)
    assert len(statuses) == 1
    windows = count_train_windows(get_cfg({**TINY, "data": str(dataset)}))
    assert statuses[0]["epochs"] == budget_epochs(5000, windows, 200)
    assert statuses[0]["train_budget"]["windows_per_epoch"] == windows
    assert statuses[0]["splits"] == ["val"]   # 诚实协议：矩阵不计划 test


def test_benchmark_can_continue_after_a_failed_run(dataset, tmp_path):
    """``continue_on_error`` 让长跑矩阵不被一个坏组合拖垮，失败项进 benchmark.json。"""
    plan = {"name": "c", "project": str(tmp_path), "models": ["ronin_resnet18"],
            "datasets": [str(dataset), str(tmp_path / "missing")], "seeds": [0],
            "splits": ["val"], "report": False, "continue_on_error": True,
            "overrides": {**TINY, "epochs": 1, "save_predictions": False}}
    statuses = run_benchmark(plan)
    assert [s["train"] for s in statuses] == ["trained", "failed"]
    assert statuses[0]["epochs"] == 1 and statuses[0]["fitness"] == "mean ate"
    assert statuses[0]["val_fitness"] > 0 and statuses[0]["test"] is False
    assert statuses[0]["splits"] == {"val": "evaluated"}
    assert "error" in statuses[1]
    saved = json.loads((tmp_path / "c" / "benchmark.json").read_text())["runs"]
    assert len(saved) == 2 and saved[1]["error"] == statuses[1]["error"]
    with pytest.raises(FileNotFoundError):   # 缺省仍然直接抛出，不静默跳过
        run_benchmark({**plan, "name": "d", "continue_on_error": False})
