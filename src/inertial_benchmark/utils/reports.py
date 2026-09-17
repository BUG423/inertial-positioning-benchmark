"""结果汇总（定义见 ``docs/METRICS.md`` 第 5 节）：跨序列统计、分组 bootstrap、多种子聚合、
配对 Wilcoxon，导出 CSV / Markdown / LaTeX 表格与汇总图。依赖 pandas。
"""

from __future__ import annotations

import math
from itertools import combinations
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np

from ..metrics import MAIN_METRICS, display_name
from . import LOGGER, json_load, json_save

PathLike = Union[str, Path]
KEYS = ("dataset", "split", "model")


def _pd():
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise ImportError("reports need pandas: pip install 'inertial-positioning-benchmark"
                          "[train]'") from exc
    return pd


def find_runs(root: PathLike) -> list:
    """``root`` 下所有同时含 ``metrics.json`` 与 ``sequences.csv`` 的评测目录（排除训练目录）。"""
    runs = []
    for path in sorted(Path(root).rglob("metrics.json")):
        if (path.parent / "sequences.csv").exists():
            meta = json_load(path)
            if meta.get("mode", "val") == "val":
                runs.append(path.parent)
    return runs


def load_runs(root: PathLike):
    """把评测结果展开为逐序列表：``dataset, split, model, seed, run, sequence_id, group_id, …``。"""
    pd = _pd()
    frames = []
    for run in find_runs(root):
        meta = json_load(run / "metrics.json")
        df = pd.read_csv(run / "sequences.csv")
        if "skipped" in df:
            df = df[df["skipped"].isna() | (df["skipped"] == "")]
        eff = meta.get("efficiency") or {}
        df.insert(0, "run", str(run))
        df.insert(0, "seed", meta.get("seed", 0))
        df.insert(0, "model", meta.get("model", "unknown"))
        df.insert(0, "split", meta.get("split", "unknown"))
        df.insert(0, "dataset", meta.get("dataset", "unknown"))
        for key in ("params", "flops"):
            df[key] = eff.get(key, np.nan)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=list(KEYS) + ["seed", "run", "sequence_id", "group_id"])
    df = pd.concat(frames, ignore_index=True)
    df["group_id"] = df.get("group_id", df["sequence_id"]).astype(str)
    df["sequence_id"] = df["sequence_id"].astype(str)
    return df


def bootstrap_ci(values: Sequence[float], groups: Optional[Sequence[str]] = None,
                 n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple:
    """按组有放回重采样的均值置信区间；组为空时按单个样本重采样。"""
    values = np.asarray(values, float)
    groups = np.asarray(groups if groups is not None else np.arange(len(values))).astype(str)
    ok = np.isfinite(values)
    values, groups = values[ok], groups[ok]
    if len(values) == 0:
        return (math.nan, math.nan)
    unique, inverse = np.unique(groups, return_inverse=True)
    if len(unique) == 1:
        m = float(values.mean())
        return (m, m)
    sums = np.bincount(inverse, weights=values, minlength=len(unique))
    counts = np.bincount(inverse, minlength=len(unique)).astype(float)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(unique), size=(n_boot, len(unique)))
    means = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


def summarize(df, metrics: Iterable[str] = MAIN_METRICS, n_boot: int = 2000):
    """按 (dataset, split, model) 汇总：跨序列均值/中位数/标准差、分组 CI、种子间标准差。"""
    pd = _pd()
    metrics = [m for m in metrics if m in df.columns]
    rows = []
    for key, part in df.groupby(list(KEYS), sort=True):
        # 逐序列先在种子间平均，再跨序列统计
        seq = part.groupby(["sequence_id", "group_id"], as_index=False)[metrics].mean()
        per_seed = part.groupby("seed")[metrics].mean()
        row = dict(zip(KEYS, key))
        row.update(n_sequences=len(seq), n_seeds=int(part["seed"].nunique()),
                   params=part["params"].iloc[0] if "params" in part else np.nan,
                   flops=part["flops"].iloc[0] if "flops" in part else np.nan)
        for m in metrics:
            v = seq[m].to_numpy(float)
            finite = v[np.isfinite(v)]
            row[f"{m}_mean"] = float(np.nanmean(per_seed[m])) if len(finite) else np.nan
            row[f"{m}_median"] = float(np.median(finite)) if len(finite) else np.nan
            row[f"{m}_std"] = float(np.std(finite, ddof=1)) if len(finite) > 1 else np.nan
            row[f"{m}_seed_std"] = float(np.nanstd(per_seed[m], ddof=1)) \
                if per_seed[m].notna().sum() > 1 else np.nan
            lo, hi = bootstrap_ci(v, seq["group_id"], n_boot=n_boot)
            row[f"{m}_ci_lo"], row[f"{m}_ci_hi"] = lo, hi
            row[f"{m}_n"] = int(len(finite))
        rows.append(row)
    return pd.DataFrame(rows)


def paired_wilcoxon(df, metric: str, model_a: str, model_b: str, dataset: str,
                    split: str, min_pairs: int = 5) -> dict:
    """同一数据集/划分上两模型的逐序列配对 Wilcoxon 检验（值先在种子间平均）。"""
    from scipy.stats import wilcoxon

    part = df[(df["dataset"] == dataset) & (df["split"] == split)]
    a = part[part["model"] == model_a].groupby("sequence_id")[metric].mean()
    b = part[part["model"] == model_b].groupby("sequence_id")[metric].mean()
    both = a.to_frame("a").join(b.to_frame("b"), how="inner").dropna()
    diff = (both["a"] - both["b"]).to_numpy(float)
    out = {"dataset": dataset, "split": split, "metric": metric, "model_a": model_a,
           "model_b": model_b, "n": int(len(diff)),
           "median_diff": float(np.median(diff)) if len(diff) else math.nan,
           "statistic": math.nan, "p_value": math.nan}
    if len(diff) >= min_pairs:
        if np.all(diff == 0):
            out.update(statistic=0.0, p_value=1.0)
        else:
            res = wilcoxon(both["a"], both["b"], zero_method="wilcox", alternative="two-sided")
            out.update(statistic=float(res.statistic), p_value=float(res.pvalue))
    return out


def all_pairwise_tests(df, metrics: Iterable[str] = ("ate", "rte")):
    pd = _pd()
    rows = []
    for (dataset, split), part in df.groupby(["dataset", "split"]):
        models = sorted(part["model"].unique())
        for a, b in combinations(models, 2):
            for m in metrics:
                if m in df.columns:
                    rows.append(paired_wilcoxon(df, m, a, b, dataset, split))
    return pd.DataFrame(rows)


def _fmt(value: float, digits: int = 3) -> str:
    return "–" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def _cell(row, m: str, digits: int, pm: str) -> str:
    mean = row[f"{m}_mean"]
    if row["n_seeds"] > 1 and np.isfinite(row[f"{m}_seed_std"]):
        return f"{_fmt(mean, digits)} {pm} {_fmt(row[f'{m}_seed_std'], digits)}"
    return _fmt(mean, digits)


def table_rows(summary, metrics: Sequence[str], digits: int = 3, pm: str = "±") -> tuple:
    metrics = [m for m in metrics if f"{m}_mean" in summary.columns]
    header = ["dataset", "split", "model", "n"] + [display_name(m) for m in metrics]
    rows = []
    for _, r in summary.iterrows():
        rows.append([str(r["dataset"]), str(r["split"]), str(r["model"]),
                     str(int(r["n_sequences"]))] + [_cell(r, m, digits, pm) for m in metrics])
    return header, rows


def to_markdown(summary, metrics: Sequence[str] = MAIN_METRICS, digits: int = 3) -> str:
    header, rows = table_rows(summary, metrics, digits)
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * 3 + ["---:"] * (len(header) - 3)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def _tex(text: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"),
                 ("#", r"\#")):
        text = text.replace(a, b)
    return text


def to_latex(summary, metrics: Sequence[str] = MAIN_METRICS, digits: int = 3) -> str:
    """booktabs 风格表格；多种子时单元格为 ``均值 $\\pm$ 种子标准差``。"""
    header, rows = table_rows(summary, metrics, digits, pm="$\\pm$")
    spec = "lll" + "r" * (len(header) - 3)
    out = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule",
           " & ".join(_tex(h) for h in header) + " \\\\", "\\midrule"]
    out += [" & ".join(_tex(c) if "$" not in c else c for c in r) + " \\\\" for r in rows]
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


def make_plots(df, summary, out: Path, metric: str = "ate") -> list:
    """每个 (dataset, split) 的箱线图、逐序列误差 CDF、长度比散点，以及全局帕累托图。"""
    from .plotting import (
        plot_boxplot,
        plot_error_cdf,
        plot_length_ratio,
        plot_pareto,
    )

    files = []
    models = sorted(df["model"].unique())
    if len(models) > 8:
        LOGGER.warning("report: more than 8 models; per-model plots are skipped")
        return files
    for (dataset, split), part in df.groupby(["dataset", "split"]):
        seq = part.groupby(["model", "sequence_id"], as_index=False).mean(numeric_only=True)
        values = {m: seq.loc[seq["model"] == m, metric].to_numpy() for m in models
                  if (seq["model"] == m).any()}
        tag = f"{dataset}_{split}"
        files.append(out / f"box_{metric}_{tag}.png")
        plot_boxplot(values, files[-1], ylabel=display_name(metric))
        files.append(out / f"cdf_{metric}_{tag}.png")
        plot_error_cdf(values, files[-1], xlabel=f"per-sequence {display_name(metric)}")
        if {"plr", "distance_m"} <= set(seq.columns):
            pts = {m: (seq.loc[seq["model"] == m, "distance_m"],
                       seq.loc[seq["model"] == m, "plr"]) for m in values}
            files.append(out / f"plr_{tag}.png")
            plot_length_ratio(pts, files[-1])
        sub = summary[(summary["dataset"] == dataset) & (summary["split"] == split)]
        if f"{metric}_mean" in sub and sub["params"].notna().any():
            files.append(out / f"pareto_params_{metric}_{tag}.png")
            plot_pareto(list(sub["model"]), sub["params"], sub[f"{metric}_mean"], files[-1],
                        xlabel="parameters", ylabel=display_name(metric))
    return files


def build_report(runs: PathLike, out: PathLike, metrics: Sequence[str] = MAIN_METRICS,
                 plots: bool = True, n_boot: int = 2000) -> dict:
    """汇总 ``runs`` 下全部评测结果，写出 CSV / Markdown / LaTeX 表格与图。"""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df = load_runs(runs)
    if df.empty:
        raise FileNotFoundError(f"no evaluation results (metrics.json + sequences.csv) "
                                f"under {runs}")
    # 所有浮点列都是逐序列指标（含 duration_s / distance_m），效率指标单独处理
    all_metrics = [c for c in df.columns
                   if df[c].dtype.kind == "f" and c not in ("params", "flops")]
    summary = summarize(df, all_metrics, n_boot=n_boot)
    tests = all_pairwise_tests(df, [m for m in ("ate", "rte") if m in df.columns])
    df.to_csv(out / "per_sequence.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    tests.to_csv(out / "wilcoxon.csv", index=False)
    (out / "summary.md").write_text(to_markdown(summary, metrics), encoding="utf-8")
    (out / "summary.tex").write_text(to_latex(summary, metrics), encoding="utf-8")
    files = make_plots(df, summary, out) if plots else []
    info = {"runs": str(runs), "num_runs": int(df["run"].nunique()),
            "num_rows": int(len(df)), "datasets": sorted(df["dataset"].unique()),
            "models": sorted(df["model"].unique()),
            "files": sorted(str(p.relative_to(out)) for p in out.iterdir()),
            "plots": [str(p) for p in files]}
    json_save(out / "report.json", info)
    LOGGER.info(f"report: {info['num_runs']} runs → {out}")
    LOGGER.info("\n" + to_markdown(summary, metrics))
    return info
