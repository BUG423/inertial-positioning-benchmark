import math

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from inertial_benchmark.engine.results import RunResult, SequenceResult  # noqa: E402
from inertial_benchmark.utils.reports import (  # noqa: E402
    bootstrap_ci,
    build_report,
    load_runs,
    paired_wilcoxon,
    summarize,
    to_latex,
    to_markdown,
)

ATE = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def write_run(root, model, seed, offset, dataset="syn", split="test"):
    seqs = []
    for k, ate in enumerate(ATE):
        s = SequenceResult(sequence_id=f"q{k}", group_id=f"g{k // 2}")
        s.metrics = {"ate": ate + offset + 0.1 * seed, "rte": 2 * ate, "plr": 1.0,
                     "num_windows": 10}
        seqs.append(s)
    run = RunResult(seqs, dataset=dataset, split=split, model=model, cfg={"seed": seed},
                    efficiency={"params": 1000 * (1 + offset), "flops": 5e6})
    return run.save(root / model / f"seed{seed}" / split, predictions=False, plots=False)


@pytest.fixture()
def runs(tmp_path):
    for seed in (0, 1):
        write_run(tmp_path, "A", seed, 0.0)
        write_run(tmp_path, "B", seed, 1.0)
    return tmp_path


def test_bootstrap_ci():
    assert bootstrap_ci([1.0, 2.0], ["g", "g"]) == (1.5, 1.5)
    lo, hi = bootstrap_ci(np.arange(20.0), np.repeat(np.arange(10), 2))
    assert lo < 9.5 < hi
    again = bootstrap_ci(np.arange(20.0), np.repeat(np.arange(10), 2))
    assert (lo, hi) == again
    assert all(math.isnan(v) for v in bootstrap_ci([math.nan], ["a"]))


def test_summary_statistics(runs):
    df = load_runs(runs)
    assert len(df) == 24 and set(df["model"]) == {"A", "B"}
    summary = summarize(df, ["ate", "rte"])
    a = summary[summary["model"] == "A"].iloc[0]
    assert a["n_sequences"] == 6 and a["n_seeds"] == 2
    assert a["ate_mean"] == pytest.approx(np.mean(ATE) + 0.05)
    assert a["ate_seed_std"] == pytest.approx(np.std([0.0, 0.1], ddof=1) + 0, abs=1e-9)
    assert a["ate_median"] == pytest.approx(np.median(ATE) + 0.05)
    assert a["ate_ci_lo"] <= a["ate_mean"] <= a["ate_ci_hi"]
    assert a["params"] == 1000


def test_paired_wilcoxon(runs):
    df = load_runs(runs)
    res = paired_wilcoxon(df, "ate", "A", "B", "syn", "test")
    assert res["n"] == 6 and res["median_diff"] == pytest.approx(-1.0)
    assert res["p_value"] == pytest.approx(2 / 64)
    same = paired_wilcoxon(df, "rte", "A", "B", "syn", "test")
    assert same["p_value"] == 1.0
    few = paired_wilcoxon(df, "ate", "A", "B", "syn", "test", min_pairs=10)
    assert math.isnan(few["p_value"])


def test_tables(runs):
    summary = summarize(load_runs(runs), ["ate", "rte"])
    md = to_markdown(summary, ["ate", "rte"])
    assert "| dataset | split | model | n | ATE (m) | RTE (m) |" in md
    assert "3.550 ± 0.071" in md
    tex = to_latex(summary, ["ate"])
    assert "\\toprule" in tex and "3.550 $\\pm$ 0.071" in tex and "ATE (m)" in tex


def test_build_report(runs, tmp_path):
    out = tmp_path / "report"
    info = build_report(runs, out, metrics=["ate", "rte"], n_boot=200)
    for name in ("per_sequence.csv", "summary.csv", "summary.md", "summary.tex",
                 "wilcoxon.csv", "report.json"):
        assert (out / name).exists(), name
    assert info["models"] == ["A", "B"] and info["num_runs"] == 4
    assert any("pareto" in p for p in info["plots"])
    assert any("box_ate" in p for p in info["plots"])
    tests = pd.read_csv(out / "wilcoxon.csv")
    assert set(tests["metric"]) == {"ate", "rte"}
    with pytest.raises(FileNotFoundError):
        build_report(tmp_path / "empty", out)
