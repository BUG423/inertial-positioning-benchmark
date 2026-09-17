"""基准矩阵：模型 × 数据集 × 种子，逐项训练并在测试划分上评测，已完成项自动跳过。

YAML 示例::

    name: main
    project: runs/benchmark          # 输出到 <project>/<name>/<label>/<dataset>/seed<k>/
    models:
      - ronin_resnet18                 # 名称 / 模型 YAML / checkpoint
      - {label: ronin_official, model: ronin_resnet18, overrides: {recipe: official}}
    datasets: [ronin, ridi]            # 或 {data: ..., overrides: {...}}
    seeds: [0, 1, 2]
    splits: null                       # 缺省使用数据集 YAML 的 test_splits
    overrides: {epochs: 100, device: 0}
    report: true                       # 结束后汇总到 <project>/<name>/report

完成标志：``train/metrics.json``（训练完成）与 ``<split>/metrics.json``（该划分评测完成）。
训练中断（有 ``last.pt`` 无 ``metrics.json``）时自动断点续训。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from ..cfg import INPUT_KEYS, ConfigError, get_cfg, is_checkpoint
from ..data.manifest import resolve_dataset
from ..utils import CFG_DIR, LOGGER, json_save, yaml_load

PathLike = Union[str, Path]
# 评测时不需要的训练专用键
TRAIN_ONLY = ("epochs", "batch", "lr", "optimizer", "momentum", "weight_decay", "scheduler",
              "lr_final", "step_size", "gamma", "plateau_patience", "warmup_epochs", "grad_clip",
              "patience", "val_interval", "fitness", "save_period", "augment", "stride",
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
    unknown = set(spec) - {"name", "project", "models", "datasets", "seeds", "splits",
                           "overrides", "report"}
    if unknown:
        raise ConfigError(f"unknown benchmark keys: {sorted(unknown)}")
    name = spec.get("name") or (Path(str(cfg)).stem if not isinstance(cfg, Mapping) else "bench")
    root = Path(spec.get("project", "runs/benchmark")) / name
    common = {**(spec.get("overrides") or {}), **(overrides or {})}
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
    return status


def run_benchmark(cfg: Union[PathLike, Mapping], overrides: Optional[dict] = None,
                  dry_run: bool = False, report: Optional[bool] = None) -> list:
    """执行完整矩阵；返回每项的状态，并写出 ``<root>/benchmark.json``。"""
    cfg = resolve_benchmark_cfg(cfg)
    items = load_plan(cfg, overrides)
    root = items[0].root
    spec = dict(yaml_load(cfg)) if not isinstance(cfg, Mapping) else dict(cfg)
    LOGGER.info(f"benchmark: {len(items)} runs → {root}")
    statuses = []
    for k, item in enumerate(items, 1):
        LOGGER.info(f"[{k}/{len(items)}] {item.label} × {item.data} × seed {item.seed}")
        if dry_run:
            statuses.append({"label": item.label, "data": str(item.data), "seed": item.seed,
                             "dir": str(item.run_dir), "train": "planned"})
            continue
        t0 = time.time()
        status = run_item(item)
        status["seconds"] = round(time.time() - t0, 1)
        statuses.append(status)
        json_save(root / "benchmark.json", {"plan": spec, "runs": statuses})
    do_report = spec.get("report", True) if report is None else report
    if do_report and not dry_run:
        from ..utils.reports import build_report

        build_report(root, root / "report")
    return statuses
