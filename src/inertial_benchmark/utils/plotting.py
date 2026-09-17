"""统一风格的论文图（matplotlib，面向对象接口，不依赖 pyplot 状态）。

配色：固定顺序的分类色（颜色跟随实体、不按排名循环），参考轨迹用主墨色，辅助线用灰色；
细线（1.5 pt）、发丝网格、≥2 个系列时总有图例。散点类图一张最多 3 个系列，超过则分面。
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Union

import numpy as np

PathLike = Union[str, Path]

# 分类色（已通过色觉缺陷分离校验的默认顺序），不得循环使用
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7",
           "#e34948")
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SCATTER_MAX_SERIES = 3
RC = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "lines.linewidth": 1.5,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
    "font.size": 8,
    "font.family": "sans-serif",
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
}


@contextmanager
def plot_style():
    import matplotlib as mpl

    with mpl.rc_context(RC):
        yield


def color_map(labels: Iterable[str]) -> dict:
    """实体 → 颜色（按首次出现顺序分配）；超过 8 个时报错，应合并为 Other 或分面。"""
    labels = list(dict.fromkeys(labels))
    if len(labels) > len(PALETTE):
        raise ValueError(f"{len(labels)} series exceed the {len(PALETTE)}-color palette; "
                         "fold extra series into 'Other' or use small multiples")
    return {label: PALETTE[i] for i, label in enumerate(labels)}


def _figure(nrows: int = 1, ncols: int = 1, size: tuple = (3.4, 3.0), **kw):
    from matplotlib.figure import Figure

    fig = Figure(figsize=(size[0] * ncols, size[1] * nrows), constrained_layout=True)
    axes = fig.subplots(nrows, ncols, squeeze=False, **kw)
    return fig, axes


def _save(fig, path: Optional[PathLike]) -> None:
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)


def _grid_shape(n: int, max_cols: int = 4) -> tuple:
    cols = min(max_cols, max(n, 1))
    return int(math.ceil(n / cols)), cols


def draw_trajectory(ax, pred: np.ndarray, gt: np.ndarray, valid: Optional[np.ndarray] = None,
                    oracle: Optional[np.ndarray] = None, title: str = "",
                    legend: bool = True) -> None:
    """在一个坐标轴上叠加参考（墨色）、oracle（灰色虚线）与预测（蓝色）水平轨迹。"""
    gt = np.array(gt[:, :2], dtype=float)
    if valid is not None:
        gt[~np.asarray(valid, bool)] = np.nan  # 无效段断开
    ax.plot(gt[:, 0], gt[:, 1], color=INK, lw=1.5, label="reference")
    if oracle is not None:
        ax.plot(oracle[:, 0], oracle[:, 1], color=MUTED, lw=1.0, ls="--", label="oracle")
    ax.plot(pred[:, 0], pred[:, 1], color=PALETTE[0], lw=1.5, label="prediction")
    start = np.flatnonzero(np.isfinite(gt[:, 0]))
    if len(start):
        ax.plot(*gt[start[0]], "o", ms=5, color=INK, mec=SURFACE, mew=1.5, zorder=5)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    if title:
        ax.set_title(title, loc="left")
    if legend:
        ax.legend(loc="best")


def plot_trajectory(pred: np.ndarray, gt: np.ndarray, valid: Optional[np.ndarray] = None,
                    oracle: Optional[np.ndarray] = None, title: str = "",
                    path: Optional[PathLike] = None):
    with plot_style():
        fig, axes = _figure(size=(4.0, 3.6))
        draw_trajectory(axes[0, 0], pred, gt, valid, oracle, title)
        _save(fig, path)
    return fig


def plot_trajectory_grid(items: Sequence[Mapping], path: Optional[PathLike] = None,
                         max_n: int = 12):
    """小多图：``items`` 每项含 ``pred``、``gt``、可选 ``valid``/``oracle``/``title``。"""
    items = list(items)[:max_n]
    rows, cols = _grid_shape(len(items))
    with plot_style():
        fig, axes = _figure(rows, cols, size=(2.6, 2.4))
        for k, ax in enumerate(axes.flat):
            if k >= len(items):
                ax.set_visible(False)
                continue
            it = items[k]
            draw_trajectory(ax, it["pred"], it["gt"], it.get("valid"), it.get("oracle"),
                            it.get("title", ""), legend=(k == 0))
        _save(fig, path)
    return fig


def plot_error_cdf(errors: Mapping[str, np.ndarray], path: Optional[PathLike] = None,
                   xlabel: str = "position error (m)"):
    """每个系列一条经验 CDF；同一实体在不同图中颜色一致。"""
    colors = color_map(errors)
    with plot_style():
        fig, axes = _figure(size=(3.6, 2.8))
        ax = axes[0, 0]
        for label, values in errors.items():
            arr = np.asarray(values, float)
            v = np.sort(arr[np.isfinite(arr)])
            if len(v) == 0:
                continue
            ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=colors[label],
                    label=label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("fraction")
        ax.set_ylim(0, 1.02)
        ax.set_xlim(left=0)
        if len(errors) > 1:
            ax.legend(loc="lower right")
        elif errors:
            ax.set_title(next(iter(errors)), loc="left")
        _save(fig, path)
    return fig


def plot_error_over_time(curves: Sequence[Mapping], path: Optional[PathLike] = None,
                         max_n: int = 12):
    """小多图：每个面板一条序列的位置误差随时间曲线（``t``、``error``、``title``）。"""
    curves = list(curves)[:max_n]
    rows, cols = _grid_shape(len(curves))
    with plot_style():
        fig, axes = _figure(rows, cols, size=(2.6, 1.8), sharey=True)
        for k, ax in enumerate(axes.flat):
            if k >= len(curves):
                ax.set_visible(False)
                continue
            c = curves[k]
            ax.plot(c["t"], c["error"], color=PALETTE[0], lw=1.2)
            ax.set_title(c.get("title", ""), loc="left")
            ax.set_xlabel("time (s)")
            if k % cols == 0:
                ax.set_ylabel("position error (m)")
            ax.set_ylim(bottom=0)
        _save(fig, path)
    return fig


def plot_boxplot(values: Mapping[str, np.ndarray], path: Optional[PathLike] = None,
                 ylabel: str = "ATE (m)"):
    """每个实体一个箱体（中位线为墨色，箱体为实体色的浅色填充）。"""
    labels = list(values)
    colors = color_map(labels)
    arrays = [np.asarray(values[k], float) for k in labels]
    data = [a[np.isfinite(a)] for a in arrays]
    with plot_style():
        fig, axes = _figure(size=(max(2.4, 0.7 * len(labels) + 1.2), 2.8))
        ax = axes[0, 0]
        bp = ax.boxplot(data, patch_artist=True, widths=0.5, showfliers=True,
                        medianprops={"color": INK, "lw": 1.5},
                        whiskerprops={"color": INK_2, "lw": 1.0},
                        capprops={"color": INK_2, "lw": 1.0},
                        flierprops={"marker": "o", "ms": 3, "mfc": MUTED, "mec": SURFACE})
        for patch, label in zip(bp["boxes"], labels):
            patch.set_facecolor(colors[label])
            patch.set_alpha(0.35)
            patch.set_edgecolor(colors[label])
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=30 if len(labels) > 3 else 0, ha="right"
                           if len(labels) > 3 else "center")
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        ax.grid(axis="x", visible=False)
        _save(fig, path)
    return fig


def plot_length_ratio(points: Mapping[str, tuple], path: Optional[PathLike] = None):
    """路径长度比 PLR 对参考路径长度的散点；``points[label] = (length_m, plr)``。

    超过 3 个系列时按系列分面（散点的全部两两组合无法保证色觉可分）。
    """
    labels = list(points)
    colors = color_map(labels)
    facets = [labels] if len(labels) <= SCATTER_MAX_SERIES else [[k] for k in labels]
    rows, cols = _grid_shape(len(facets))
    with plot_style():
        fig, axes = _figure(rows, cols, size=(3.2, 2.6), sharex=True, sharey=True)
        for k, ax in enumerate(axes.flat):
            if k >= len(facets):
                ax.set_visible(False)
                continue
            for label in facets[k]:
                length, plr = (np.asarray(v, float) for v in points[label])
                ax.scatter(length, plr, s=24, color=colors[label], edgecolors=SURFACE,
                           linewidths=1.0, label=label, zorder=3)
            ax.axhline(1.0, color=INK_2, lw=1.0, zorder=2)
            ax.set_xlabel("reference length (m)")
            ax.set_ylabel("PLR")
            if len(facets[k]) > 1:
                ax.legend(loc="best")
            else:
                ax.set_title(facets[k][0], loc="left")
        _save(fig, path)
    return fig


def pareto_front(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """最小化 x 与 y 的帕累托前沿索引（按 x 升序）。"""
    order = np.argsort(x, kind="stable")
    front, best = [], math.inf
    for i in order:
        if y[i] < best:
            front.append(i)
            best = y[i]
    return np.asarray(front, dtype=int)


def plot_pareto(labels: Sequence[str], x: Sequence[float], y: Sequence[float],
                path: Optional[PathLike] = None, xlabel: str = "parameters",
                ylabel: str = "ATE (m)", logx: bool = True):
    """效率–精度散点：单一系列（蓝色）+ 直接标注模型名，帕累托前沿用灰色阶梯线连接。"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    with plot_style():
        fig, axes = _figure(size=(3.8, 2.8))
        ax = axes[0, 0]
        if ok.any():
            idx = np.flatnonzero(ok)
            front = idx[pareto_front(x[ok], y[ok])]
            ax.step(x[front], y[front], where="post", color=MUTED, lw=1.0, zorder=2,
                    label="Pareto front")
            ax.scatter(x[ok], y[ok], s=36, color=PALETTE[0], edgecolors=SURFACE,
                       linewidths=1.5, zorder=3)
            for i in idx:
                ax.annotate(labels[i], (x[i], y[i]), xytext=(4, 4), textcoords="offset points",
                            fontsize=7, color=INK_2)
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        _save(fig, path)
    return fig


CURVE_COLUMNS = ("train/loss", "val/loss", "lr", "val/ate", "val/rte", "val/d_rte_10m",
                 "val/vel_rmse", "val/dir_err_mean", "val/plr")


def plot_training_curves(csv_path: PathLike, path: Optional[PathLike] = None,
                         columns: Optional[Sequence[str]] = None):
    """``results.csv`` 小多图：每个面板一个量（训练/验证损失同面板）。"""
    import csv

    from matplotlib.ticker import MaxNLocator

    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None

    def col(name: str) -> np.ndarray:
        return np.array([float(r[name]) if r.get(name) not in (None, "") else np.nan
                         for r in rows])

    epoch = col("epoch") + 1  # 展示为从 1 开始
    wanted = columns or CURVE_COLUMNS
    names = [c for c in wanted if c in rows[0]]
    panels = []
    if "train/loss" in names or "val/loss" in names:
        panels.append(("loss", [c for c in ("train/loss", "val/loss") if c in names]))
    panels += [(c, [c]) for c in names if c not in ("train/loss", "val/loss")]
    rows_n, cols_n = _grid_shape(len(panels))
    with plot_style():
        fig, axes = _figure(rows_n, cols_n, size=(2.6, 2.0))
        for k, ax in enumerate(axes.flat):
            if k >= len(panels):
                ax.set_visible(False)
                continue
            title, cols = panels[k]
            colors = color_map(cols)
            for c in cols:
                v = col(c)
                ax.plot(epoch[np.isfinite(v)], v[np.isfinite(v)], color=colors[c],
                        marker="o", ms=3, mec=SURFACE, label=c.split("/")[0])
            ax.set_title(title, loc="left")
            ax.set_xlabel("epoch")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            if len(cols) > 1:
                ax.legend(loc="best")
        _save(fig, path)
    return fig
