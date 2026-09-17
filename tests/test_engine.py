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
    reconstruct,
)
from inertial_benchmark.nn import MODELS, BaseModel, register_model  # noqa: E402
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
