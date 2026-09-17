import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

pytest.importorskip("torch")
pytest.importorskip("pandas")

from inertial_benchmark.cli import entrypoint  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FAKE = Path(__file__).with_name("fake_converter.py")


def write_raw(root: Path) -> Path:
    entries = [{"id": f"walk{k}", "seed": k, "group": f"subject{k % 3}", "split": "train",
                "duration": 12.0, "imu_rate": 100.0 if k % 2 else 200.0}
               for k in range(5)]
    entries += [
        {"id": "test_a", "seed": 31, "group": "subject9", "split": "test+test_unseen",
         "duration": 15.0, "imu_rate": 200.0},
        {"id": "test_b", "seed": 32, "group": "subject0", "split": "test+test_seen",
         "duration": 15.0, "imu_rate": 250.0, "pose_rate": 100.0},
    ]
    raw = root / "raw"
    raw.mkdir(parents=True)
    (raw / "spec.json").write_text(json.dumps({"sequences": entries}))
    return raw


def test_help_version_and_errors(capsys):
    assert entrypoint([]) == 0
    assert "usage: ipb" in capsys.readouterr().out
    assert entrypoint(["--version"]) == 0
    assert entrypoint(["bogus"]) == 2
    assert entrypoint(["train", "epoch=3"]) == 2
    assert entrypoint(["train", "epochs"]) == 2
    assert entrypoint(["convert", "dataset=x"]) == 2
    assert entrypoint(["report", "runs=/nonexistent", "colour=red"]) == 2


def test_cfg_and_info(capsys):
    assert entrypoint(["cfg", "model=ronin_resnet18", "recipe=official"]) == 0
    cfg = yaml.safe_load(capsys.readouterr().out)
    assert cfg["lr"] == pytest.approx(1e-4) and cfg["optimizer"] == "adam"
    assert entrypoint(["info"]) == 0
    info = yaml.safe_load(capsys.readouterr().out)
    assert "ronin_resnet18" in info["models"] and "ronin" in info["datasets"]
    assert entrypoint(["info", "model=ronin_resnet18", "flops=false"]) == 0
    assert yaml.safe_load(capsys.readouterr().out)["parameters"] == 4_634_882


def test_builtin_benchmark_plan(tmp_path, monkeypatch):
    from inertial_benchmark.engine.benchmark import load_plan

    monkeypatch.chdir(tmp_path)
    plan = load_plan("main")
    assert len(plan) == 8 * 3 and plan[0].label == "ronin_resnet18"
    assert plan[0].run_dir == Path("runs/benchmark/main/ronin_resnet18/ronin/seed0")
    assert entrypoint(["benchmark", "cfg=main", "dry_run=true"]) == 0
    assert not (tmp_path / "runs").exists()
    assert entrypoint(["benchmark", "cfg=missing_plan"]) == 2


def test_module_entry_point():
    out = subprocess.run([sys.executable, "-m", "inertial_benchmark", "version"],
                         capture_output=True, text=True, check=True,
                         env={"PYTHONPATH": str(ROOT / "src"), "PATH": ""})
    assert out.stdout.strip().startswith("1.")


@pytest.mark.slow
def test_end_to_end_smoke(tmp_path, capsys):
    """convert → check → train(ronin_resnet18, 2 epochs) → val → predict → benchmark → report。"""
    start = time.time()
    raw = write_raw(tmp_path)
    data = tmp_path / "ipb" / "fake"
    runs = tmp_path / "runs"
    assert entrypoint(["convert", "dataset=fake", f"source={raw}", f"output={data}",
                       f"converter={FAKE}", "workers=2"]) == 0
    assert entrypoint(["check", f"data={data}", "full=true", "hash=true",
                       f"save={tmp_path / 'check.json'}"]) == 0
    check = json.loads((tmp_path / "check.json").read_text())
    assert check["ok"] and check["splits"]["test_unseen"] == 1

    common = ["device=cpu", "workers=0", f"project={runs}", "exist_ok=true", "eval_stride=20"]
    assert entrypoint(["train", "model=ronin_resnet18", f"data={data}", "epochs=2",
                       "batch=32", "stride=40", "name=smoke", "plots=true",
                       "efficiency=false", *common]) == 0
    best = runs / "train" / "smoke" / "weights" / "best.pt"
    assert best.exists()
    rows = (runs / "train" / "smoke" / "results.csv").read_text().strip().splitlines()
    assert len(rows) == 3

    assert entrypoint(["val", f"model={best}", f"data={data}", "split=test", "name=smoke_test",
                       *common]) == 0
    metrics = json.loads((runs / "val" / "smoke_test" / "metrics.json").read_text())
    assert metrics["num_sequences"] == 2 and metrics["model"] == "ronin_resnet18"
    assert metrics["metrics"]["ate"] > 0 and metrics["efficiency"]["params"] == 4_634_882
    assert (runs / "val" / "smoke_test" / "plots" / "trajectories.png").exists()

    assert entrypoint(["predict", f"model={best}", f"source={data / 'sequences' / 'test_a.h5'}",
                       "name=smoke_pred", *common]) == 0
    assert (runs / "predict" / "smoke_pred" / "predictions" / "test_a.npz").exists()

    bench = tmp_path / "bench.yaml"
    bench.write_text(yaml.safe_dump({
        "name": "smoke",
        "project": str(runs / "benchmark"),
        "models": [
            {"label": "tiny", "model": "ronin_resnet18",
             "overrides": {"model_args": {"group_sizes": [1, 1], "base_plane": 8,
                                          "fc_dim": 16, "trans_planes": 4}}},
            {"label": "resnet18_trained", "model": str(best)},
        ],
        "datasets": [str(data)],
        "seeds": [0, 1],
        "overrides": {"epochs": 1, "batch": 32, "stride": 40, "eval_stride": 40,
                      "device": "cpu", "workers": 0,
                      "plots": False, "efficiency": False, "save_predictions": False},
        "report": False,
    }))
    assert entrypoint(["benchmark", f"cfg={bench}", "dry_run=true"]) == 0
    assert entrypoint(["benchmark", f"cfg={bench}"]) == 0
    bench_root = runs / "benchmark" / "smoke"
    for seed in (0, 1):
        for split in ("test", "test_seen", "test_unseen"):
            assert (bench_root / "tiny" / "fake" / f"seed{seed}" / split / "metrics.json").exists()
    status = json.loads((bench_root / "benchmark.json").read_text())["runs"]
    assert [s["train"] for s in status] == ["trained", "trained", "pretrained checkpoint",
                                            "pretrained checkpoint"]
    # 再次运行：全部跳过
    capsys.readouterr()
    assert entrypoint(["benchmark", f"cfg={bench}", "report=true"]) == 0
    status = json.loads((bench_root / "benchmark.json").read_text())["runs"]
    assert status[0]["train"] == "skipped (done)"
    assert set(status[0]["splits"].values()) == {"skipped (done)"}

    report = tmp_path / "report"
    assert entrypoint(["report", f"runs={bench_root}", f"out={report}", "n_boot=100"]) == 0
    summary = (report / "summary.md").read_text()
    assert "tiny" in summary and "resnet18_trained" in summary and "test_unseen" in summary
    assert (report / "summary.tex").exists() and (report / "wilcoxon.csv").exists()
    elapsed = time.time() - start
    print(f"end-to-end smoke finished in {elapsed:.1f}s")
