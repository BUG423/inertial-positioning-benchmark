"""结果对象：``Trajectory``（逐帧轨迹）、``SequenceResult``（一条序列的全部输出与指标）、
``RunResult``（一次评测的聚合结果与文件写出）。
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import numpy as np

from ..metrics import MAIN_METRICS, path_length, sequence_metrics
from ..utils import LOGGER, json_save, to_builtin

PathLike = Union[str, Path]


@dataclass
class Trajectory:
    """统一时间轴上的预测轨迹。"""

    t: np.ndarray  # (N,)
    pos: np.ndarray  # (N, D)
    vel: Optional[np.ndarray] = None  # (N, D) 逐帧速度（积分输入）
    sequence_id: str = ""

    def __len__(self) -> int:
        return len(self.t)

    def length(self, resolution: Optional[float] = 1.0) -> float:
        rate = 1.0 / float(np.median(np.diff(self.t))) if len(self.t) > 1 else 200.0
        return path_length(self.pos, None, rate, resolution, self.pos.shape[1])

    def save(self, path: PathLike) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"t": self.t, "pos": self.pos}
        if self.vel is not None:
            arrays["vel"] = self.vel
        np.savez_compressed(path, sequence_id=np.array(self.sequence_id), **arrays)
        return path

    @classmethod
    def load(cls, path: PathLike) -> "Trajectory":
        with np.load(path) as f:
            return cls(t=f["t"], pos=f["pos"], vel=f["vel"] if "vel" in f else None,
                       sequence_id=str(f["sequence_id"]))

    def plot(self, path: Optional[PathLike] = None, gt: Optional[np.ndarray] = None):
        from ..utils.plotting import plot_trajectory

        return plot_trajectory(self.pos, gt if gt is not None else np.full_like(self.pos, np.nan),
                               title=self.sequence_id, path=path)


ARRAYS = ("t", "pos_pred", "pos_gt", "pos_oracle", "valid_pose", "t_window", "starts",
          "vel_pred", "vel_target", "window_valid", "logstd")
OUTPUT_PREFIX = "out_"  # 其他逐窗口模型输出在 .npz 中的键前缀


@dataclass
class SequenceResult:
    """一条序列的推理结果（世界系）与指标。"""

    sequence_id: str
    dataset: str = "unknown"
    group_id: str = "unknown"
    t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    pos_pred: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    pos_gt: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    pos_oracle: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    valid_pose: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    t_window: np.ndarray = field(default_factory=lambda: np.zeros(0))
    starts: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    vel_pred: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    vel_target: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    window_valid: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    logstd: Optional[np.ndarray] = None
    # 模型的其他逐窗口输出（键 → (K, ...)），保持模型输出的视图坐标系，未做无效窗口插值
    outputs: dict = field(default_factory=dict)
    frame: str = "gravity_world"
    rate: float = 200.0
    loss: Optional[float] = None
    skipped: Optional[str] = None
    metrics: dict = field(default_factory=dict)

    @property
    def trajectory(self) -> Trajectory:
        return Trajectory(self.t, self.pos_pred, sequence_id=self.sequence_id)

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if len(self.t) > 1 else 0.0

    @property
    def distance(self) -> float:
        return path_length(self.pos_gt, self.valid_pose, self.rate) if len(self.t) else 0.0

    def position_error(self, dims: int = 2) -> np.ndarray:
        """逐帧水平位置误差（无效参考处为 NaN）。"""
        err = np.linalg.norm(self.pos_pred[:, :dims] - self.pos_gt[:, :dims], axis=1)
        return np.where(self.valid_pose, err, np.nan)

    def compute_metrics(self, dims: int = 2, rte_delta: float = 60.0,
                        t_rte: Iterable[float] = (1.0, 10.0),
                        d_rte: Iterable[float] = (10.0,), min_speed: float = 0.2) -> dict:
        if self.skipped:
            self.metrics = {}
            return self.metrics
        dims = min(dims, self.pos_pred.shape[1])
        self.metrics = sequence_metrics(
            self.pos_pred, self.pos_gt, self.valid_pose, self.vel_pred, self.vel_target,
            self.window_valid, rate=self.rate, pos_oracle=self.pos_oracle, dims=dims,
            rte_delta=rte_delta, t_rte=t_rte, d_rte_distance=d_rte, min_speed=min_speed)
        if self.loss is not None:
            self.metrics["loss"] = self.loss
        return self.metrics

    def to_dict(self) -> dict:
        """一行 ``sequences.csv``。"""
        row = {"sequence_id": self.sequence_id, "group_id": self.group_id,
               "duration_s": self.duration, "distance_m": self.distance,
               "skipped": self.skipped or ""}
        row.update(self.metrics)
        return row

    def save(self, path: PathLike) -> Path:
        """保存为 ``.npz``（数组 + ``meta`` JSON 字符串；其他模型输出的键加 ``out_`` 前缀）。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {k: getattr(self, k) for k in ARRAYS if getattr(self, k) is not None}
        arrays.update({OUTPUT_PREFIX + k: np.asarray(v) for k, v in self.outputs.items()})
        meta = {"sequence_id": self.sequence_id, "dataset": self.dataset,
                "group_id": self.group_id, "frame": self.frame, "rate": self.rate,
                "loss": self.loss, "skipped": self.skipped, "metrics": self.metrics}
        np.savez_compressed(path, meta=np.array(json.dumps(to_builtin(meta))), **arrays)
        return path

    @classmethod
    def load(cls, path: PathLike) -> "SequenceResult":
        with np.load(path) as f:
            meta = json.loads(str(f["meta"]))
            arrays = {k: f[k] for k in ARRAYS if k in f}
            outputs = {k[len(OUTPUT_PREFIX):]: f[k] for k in f.files
                       if k.startswith(OUTPUT_PREFIX)}
        metrics = {k: (math.nan if v is None else v) for k, v in meta.pop("metrics").items()}
        return cls(**meta, **arrays, outputs=outputs, metrics=metrics)

    def plot(self, path: Optional[PathLike] = None):
        from ..utils.plotting import plot_trajectory

        title = self.sequence_id
        if "ate" in self.metrics:
            title += f"  ATE {self.metrics['ate']:.2f} m"
        return plot_trajectory(self.pos_pred, self.pos_gt, self.valid_pose, self.pos_oracle,
                               title, path)


def _stats(values: list) -> tuple:
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return math.nan, math.nan, math.nan, 0
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else math.nan
    return float(arr.mean()), float(np.median(arr)), std, int(len(arr))


@dataclass
class RunResult:
    """一次评测（一个模型 × 一个数据集划分）的全部序列结果。"""

    sequences: list
    dataset: str = "unknown"
    split: str = "unknown"
    model: str = "unknown"
    weights: Optional[str] = None
    cfg: dict = field(default_factory=dict)
    efficiency: dict = field(default_factory=dict)
    env: dict = field(default_factory=dict)
    save_dir: Optional[Path] = None

    def __len__(self) -> int:
        return len(self.sequences)

    @property
    def evaluated(self) -> list:
        return [s for s in self.sequences if not s.skipped]

    def aggregate(self) -> dict:
        keys: list = []
        for s in self.evaluated:
            keys += [k for k in s.metrics if k not in keys]
        out = {"mean": {}, "median": {}, "std": {}, "count": {}}
        for k in keys:
            mean, median, std, n = _stats([s.metrics.get(k) for s in self.evaluated])
            out["mean"][k], out["median"][k], out["std"][k], out["count"][k] = mean, median, \
                std, n
        # 窗口级损失按窗口数加权
        weights = [(s.metrics.get("loss"), s.metrics.get("num_windows", 0))
                   for s in self.evaluated if s.metrics.get("loss") is not None]
        total = sum(w for _, w in weights)
        if total:
            out["mean"]["loss"] = float(sum(v * w for v, w in weights) / total)
        return out

    @property
    def metrics(self) -> dict:
        """各指标跨序列均值（``fitness`` 的来源）。"""
        return self.aggregate()["mean"]

    def __getitem__(self, key: str) -> float:
        return self.metrics[key]

    def to_dict(self) -> dict:
        from ..cfg import protocol_from_cfg

        agg = self.aggregate()
        cfg = self.cfg
        return {
            "mode": "val",
            "created_utc": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat(),
            "model": self.model,
            "weights": self.weights,
            "dataset": self.dataset,
            "split": self.split,
            "seed": cfg.get("seed"),
            "dataset_fingerprint": (self.env.get("dataset") or {}).get("fingerprint"),
            "git": (self.env.get("git") or {}).get("commit"),
            "protocol": protocol_from_cfg(cfg),
            "num_sequences": len(self.evaluated),
            "num_skipped": len(self.sequences) - len(self.evaluated),
            "skipped": {s.sequence_id: s.skipped for s in self.sequences if s.skipped},
            "metrics": agg["mean"],
            "median": agg["median"],
            "std": agg["std"],
            "count": agg["count"],
            "efficiency": self.efficiency,
        }

    def summary(self, keys: Iterable[str] = MAIN_METRICS) -> str:
        m = self.metrics
        parts = [f"{k}={m[k]:.4g}" for k in keys if k in m and m[k] == m[k]]
        return (f"{self.model} on {self.dataset}/{self.split} "
                f"({len(self.evaluated)} seq): " + ", ".join(parts))

    def save(self, save_dir: Optional[PathLike] = None, predictions: bool = True,
             plots: bool = True, max_plots: int = 12) -> Path:
        """写出 ``metrics.json``、``sequences.csv``、``predictions/*.npz``、``plots/*.png``。"""
        save_dir = Path(save_dir or self.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir = save_dir
        rows = [s.to_dict() for s in self.sequences]
        header: list = []
        for r in rows:
            header += [k for k in r if k not in header]
        with open(save_dir / "sequences.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: _csv_value(r.get(k)) for k in header})
        json_save(save_dir / "metrics.json", self.to_dict())
        if predictions:
            for s in self.evaluated:
                s.save(save_dir / "predictions" / f"{s.sequence_id}.npz")
        if plots and self.evaluated:
            try:
                self.plot(save_dir / "plots", max_plots)
            except Exception as exc:  # noqa: BLE001 - 绘图失败不影响结果文件
                LOGGER.warning(f"plotting failed: {exc}")
        return save_dir

    def plot(self, out: PathLike, max_plots: int = 12) -> list:
        from ..utils import plotting

        out = Path(out)
        seqs = sorted(self.evaluated, key=lambda s: s.sequence_id)
        files = []
        for s in seqs[:max_plots]:
            files.append(out / f"traj_{s.sequence_id}.png")
            s.plot(files[-1])
        items = [{"pred": s.pos_pred, "gt": s.pos_gt, "valid": s.valid_pose,
                  "oracle": s.pos_oracle, "title": s.sequence_id} for s in seqs]
        files.append(out / "trajectories.png")
        plotting.plot_trajectory_grid(items, files[-1], max_plots)
        errors = np.concatenate([s.position_error() for s in seqs])
        files.append(out / "error_cdf.png")
        plotting.plot_error_cdf({self.model: errors}, files[-1])
        curves = [{"t": s.t, "error": s.position_error(), "title": s.sequence_id} for s in seqs]
        files.append(out / "error_over_time.png")
        plotting.plot_error_over_time(curves, files[-1], max_plots)
        files.append(out / "ate_box.png")
        plotting.plot_boxplot({self.model: [s.metrics.get("ate", math.nan) for s in seqs]},
                              files[-1])
        files.append(out / "length_ratio.png")
        plotting.plot_length_ratio({self.model: ([s.distance for s in seqs],
                                                 [s.metrics.get("plr", math.nan) for s in seqs])},
                                   files[-1])
        return files


def _csv_value(v: Any) -> Any:
    if isinstance(v, float) and not math.isfinite(v):
        return ""
    return v
