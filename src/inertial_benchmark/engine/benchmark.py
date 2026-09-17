"""基准矩阵：模型 × 数据集 × 种子，逐项训练并在测试划分上评测，已完成项自动跳过。

YAML 示例::

    name: main
    project: runs/benchmark          # 输出到 <project>/<name>/<label>/<dataset>/seed<k>/
    models:
      - ronin_resnet18                 # 名称 / 模型 YAML / checkpoint
      - {label: ronin_official, model: ronin_resnet18, overrides: {recipe: official}}
    datasets: [ronin, ridi]            # 或 {data: ..., overrides: {...}}
    seeds: [0, 1, 2]
    splits: [val]                      # 缺省使用数据集 YAML 的 test_splits；[val] = 只评 val
    overrides: {train_windows_budget: 3.2e+7, device: 0}
    report: true                       # 结束后汇总到 <project>/<name>/report
    continue_on_error: false           # true 时单项失败只记录并继续

完成标志：``train/metrics.json``（训练完成）与 ``<split>/metrics.json``（该划分评测完成）。
训练中断（有 ``last.pt`` 无 ``metrics.json``）时自动断点续训。

``overrides.train_windows_budget``（见 ``budget.py``）让每个 run 固定“看过的训练窗口总数”，
epoch 数由框架按数据集规模换算——窗口数差一个数量级的数据集之间这才是等算力预算。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from ..cfg import INPUT_KEYS, ConfigError, get_cfg, is_checkpoint
from ..data.manifest import resolve_dataset
from ..utils import CFG_DIR, LOGGER, json_load, json_save, to_builtin, yaml_load

PathLike = Union[str, Path]
# 基准 YAML 允许的顶层键
PLAN_KEYS = ("name", "project", "models", "datasets", "seeds", "splits", "overrides", "report",
             "continue_on_error")
# 评测时不需要的训练专用键
TRAIN_ONLY = ("epochs", "train_windows_budget", "budget_max_epochs",
              "batch", "lr", "optimizer", "momentum", "weight_decay", "scheduler",
              "lr_final", "step_size", "gamma", "plateau_patience", "warmup_epochs", "grad_clip",
              "patience", "val_interval", "fitness", "fitness_stat", "save_period", "augment",
              "stride",
              "resume", "pretrained", "recipe", "loss", "loss_switch_epoch", "model_args")


@dataclass
class BenchmarkItem:
    label: str
    model: Any
    data: str
    seed: int
    overrides: dict = field(default_factory=dict)
    splits: Optional[list] = None
    root: Path = Path(".")

    @property
    def run_dir(self) -> Path:
        return self.root / self.label / Path(str(self.data)).name / f"seed{self.seed}"


def _entry(item: Any, key: str) -> tuple:
    if isinstance(item, Mapping):
        value = item.get(key)
        if value is None:
            raise ConfigError(f"benchmark entry {item} lacks '{key}'")
        label = item.get("label") or Path(str(value)).stem
        return label, value, dict(item.get("overrides") or {})
    return Path(str(item)).stem, item, {}


def resolve_benchmark_cfg(cfg: Union[PathLike, Mapping]) -> Union[Path, Mapping]:
    """YAML 路径；不存在时依次尝试 ``cfg/benchmarks/<name>.yaml``（``cfg=main`` 的简写）。"""
    if isinstance(cfg, Mapping):
        return cfg
    path = Path(cfg)
    if path.exists():
        return path
    candidate = CFG_DIR / "benchmarks" / f"{path.stem}.yaml"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"benchmark config {cfg} not found (also looked for {candidate})")


def load_plan(cfg: Union[PathLike, Mapping], overrides: Optional[dict] = None) -> list:
    """解析基准 YAML，返回 :class:`BenchmarkItem` 列表（模型 × 数据集 × 种子）。"""
    cfg = resolve_benchmark_cfg(cfg)
    spec = dict(yaml_load(cfg)) if not isinstance(cfg, Mapping) else dict(cfg)
    unknown = set(spec) - set(PLAN_KEYS)
    if unknown:
        raise ConfigError(f"unknown benchmark keys: {sorted(unknown)}")
    name = spec.get("name") or (Path(str(cfg)).stem if not isinstance(cfg, Mapping) else "bench")
    common = {**(spec.get("overrides") or {}), **(overrides or {})}
    # 输出根目录：命令行的 project= 优先于 YAML（同一份矩阵可以写到别的盘而不改配置）
    root = Path(common.get("project") or spec.get("project") or "runs/benchmark") / name
    items = []
    for m in spec.get("models") or []:
        label, model, m_over = _entry(m, "model")
        for d in spec.get("datasets") or []:
            _, data, d_over = _entry(d, "data")
            for seed in spec.get("seeds") or [0]:
                over = {**common, **m_over, **d_over, "seed": int(seed)}
                get_cfg({**over, "model": model, "data": data})  # 提前发现配置错误
                items.append(BenchmarkItem(label, model, data, int(seed), over,
                                           spec.get("splits"), root))
    if not items:
        raise ConfigError("benchmark needs at least one model and one dataset")
    return items


def plan_item(item: BenchmarkItem) -> dict:
    """``dry_run`` 用的计划条目：run 目录、每轮窗口数与等窗口预算换算出的 epoch 数。

    epoch 数在这里就能算出来（惰性扫描 train 划分的有效窗口数，不载入序列），因此
    ``dry_run=true`` 能在真正开跑之前给出“每个 run 跑几轮”。
    """
    from .budget import count_train_windows, is_sequence_model, resolve_budget

    cfg = get_cfg({**item.overrides, "model": item.model, "data": item.data})
    spec = resolve_dataset(item.data)
    status = {"label": item.label, "data": str(item.data), "seed": item.seed,
              "dir": str(item.run_dir), "train": "planned", "epochs": int(cfg.epochs)}
    sequence_model = not is_checkpoint(item.model) and is_sequence_model(item.model)
    if sequence_model:
        status["train"] = "planned (calibrate once)"
    if int(cfg.train_windows_budget or 0) and not sequence_model:
        try:
            windows = count_train_windows(cfg, spec=spec)
            budget = resolve_budget(cfg, windows) or {}
            status.update(epochs=budget.get("epochs", cfg.epochs), train_budget=budget)
        except Exception as exc:  # noqa: BLE001 - 规划失败不应阻断其余条目的计划
            status["epochs"] = None
            status["error"] = f"{type(exc).__name__}: {exc}"
    status["splits"] = list(item.splits or spec.test_splits)
    return status


def train_summary(train_dir: Path) -> dict:
    """从已完成的训练目录读出审计用摘要：epoch 数、等窗口预算、选模标量与 train/eval 比值。"""
    args_file, metrics_file = train_dir / "args.yaml", train_dir / "metrics.json"
    out: dict = {}
    if not metrics_file.exists():
        return out
    meta = json_load(metrics_file)
    args = yaml_load(args_file) if args_file.exists() else {}
    key = args.get("fitness", "ate")
    stat = args.get("fitness_stat", "mean")
    values = meta.get("metrics" if stat == "mean" else stat) or {}
    out["epochs"] = meta.get("epochs", args.get("epochs"))
    out["best_epoch"] = meta.get("best_epoch")
    out["fitness"] = f"{stat} {key}"
    out["val_fitness"] = values.get(key)
    out["train_budget"] = meta.get("train_budget") or {}
    gap = meta.get("train_eval_gap") or {}
    if gap.get("ratio") is not None:
        out["train_eval_gap"] = gap["ratio"]
    return out


def run_item(item: BenchmarkItem) -> dict:
    """训练（或跳过/续训）并在各测试划分上评测一项。"""
    from .trainer import Trainer
    from .validator import Validator

    status = {"label": item.label, "data": str(item.data), "seed": item.seed,
              "dir": str(item.run_dir), "train": None, "splits": {}}
    train_dir = item.run_dir / "train"
    weights = train_dir / "weights" / "best.pt"
    if is_checkpoint(item.model):
        weights, status["train"] = Path(item.model), "pretrained checkpoint"
    elif (train_dir / "metrics.json").exists() and weights.exists():
        status["train"] = "skipped (done)"
    else:
        over = {**item.overrides, "model": item.model, "data": item.data}
        last = train_dir / "weights" / "last.pt"
        if last.exists():
            over = {"resume": str(last), **{k: item.overrides[k] for k in ("device", "workers")
                                           if k in item.overrides}}
            status["train"] = "resumed"
        else:
            status["train"] = "trained"
        Trainer(overrides=over, save_dir=train_dir).train()

    spec = resolve_dataset(item.data)
    splits = item.splits or spec.test_splits
    for split in splits:
        out = item.run_dir / split
        if (out / "metrics.json").exists():
            status["splits"][split] = "skipped (done)"
            continue
        if not spec.has_split(split):
            LOGGER.warning(f"benchmark: {spec.name} has no split {split!r}; skipped")
            status["splits"][split] = "missing split"
            continue
        over = {k: v for k, v in item.overrides.items()
                if k not in TRAIN_ONLY and k not in INPUT_KEYS}
        cfg = get_cfg({**over, "model": str(weights), "data": item.data, "split": split,
                       "mode": "val"})
        Validator(cfg, save_dir=out, label=item.label)()
        status["splits"][split] = "evaluated"
    status.update(train_summary(train_dir))
    status["test"] = any(s.startswith("test") and v == "evaluated"
                         for s, v in status["splits"].items())
    return status


def _log_record(event: str, payload: Mapping) -> None:
    """一行结构化进度记录（便于 ``grep`` 与自动监控）；空字段不写出。"""
    body = {k: v for k, v in payload.items() if v is not None}
    LOGGER.info(f"benchmark {event}: " + json.dumps(to_builtin(body), ensure_ascii=False,
                                                    sort_keys=False))


def run_benchmark(cfg: Union[PathLike, Mapping], overrides: Optional[dict] = None,
                  dry_run: bool = False, report: Optional[bool] = None,
                  continue_on_error: Optional[bool] = None) -> list:
    """执行完整矩阵；返回每项的状态，并写出 ``<root>/benchmark.json``。

    每个 run 的开始与结束各写一行结构化日志（``benchmark run start`` /
    ``benchmark run done``），含数据集、模型、种子、epoch 数、用时、val 选模标量与是否评测
    过 test。``continue_on_error=true``（或 YAML 同名键）时，单个 run 失败只记录进
    ``benchmark.json`` 并继续后续条目——长跑矩阵不应被一个组合拖垮；返回的状态里带
    ``error`` 字段，CLI 以退出码 1 报出。
    """
    cfg = resolve_benchmark_cfg(cfg)
    items = load_plan(cfg, overrides)
    root = items[0].root
    spec = dict(yaml_load(cfg)) if not isinstance(cfg, Mapping) else dict(cfg)
    tolerate = bool(spec.get("continue_on_error", False) if continue_on_error is None
                    else continue_on_error)
    LOGGER.info(f"benchmark: {len(items)} runs → {root}")
    statuses = []
    for k, item in enumerate(items, 1):
        head = {"i": k, "n": len(items), "model": item.label, "dataset": str(item.data),
                "seed": item.seed, "dir": str(item.run_dir)}
        if dry_run:
            status = plan_item(item)
            _log_record("run planned", {**head, "epochs": status.get("epochs"),
                                        "splits": status.get("splits")})
            statuses.append(status)
            continue
        _log_record("run start", head)
        t0 = time.time()
        try:
            status = run_item(item)
        except Exception as exc:  # noqa: BLE001 - 见 docstring：长跑矩阵可选择跳过失败项
            if not tolerate:
                raise
            status = {"label": item.label, "data": str(item.data), "seed": item.seed,
                      "dir": str(item.run_dir), "train": "failed", "splits": {},
                      "error": f"{type(exc).__name__}: {exc}"}
            LOGGER.error(f"benchmark: {item.label} × {item.data} × seed {item.seed} failed: "
                         f"{type(exc).__name__}: {exc}")
        status["seconds"] = round(time.time() - t0, 1)
        statuses.append(status)
        json_save(root / "benchmark.json", {"plan": spec, "runs": statuses})
        _log_record("run done", {**head, "train": status.get("train"),
                                 "epochs": status.get("epochs"),
                                 "seconds": status["seconds"],
                                 "fitness": status.get("fitness"),
                                 "val_fitness": status.get("val_fitness"),
                                 "test": status.get("test"),
                                 "splits": status.get("splits"),
                                 "error": status.get("error")})
    do_report = spec.get("report", True) if report is None else report
    if do_report and not dry_run:
        from ..utils.reports import build_report

        build_report(root, root / "report")
    return statuses
