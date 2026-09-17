"""训练循环：种子、AMP、梯度裁剪、优化器/调度器、逐轮验证、best/last 保存、断点续训、早停、
``results.csv``、回调与日志。训练过程只使用 train 与 val 划分。
"""

from __future__ import annotations

import csv
import datetime as _dt
import math
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from .. import __version__
from ..cfg import ConfigError, get_cfg
from ..data.build import build_dataloader, build_dataset
from ..data.manifest import resolve_dataset
from ..metrics import METRIC_INFO
from ..utils import LOGGER, add_file_handler, json_save, to_builtin, yaml_save
from ..utils.callbacks import add_callback, default_callbacks, run_callbacks
from ..utils.checks import collect_env
from ..utils.files import increment_path, latest_file
from ..utils.torch_utils import (
    EarlyStopping,
    build_optimizer,
    build_scheduler,
    de_parallel,
    load_checkpoint,
    model_info,
    save_checkpoint,
    seed_everything,
    select_device,
)
from .predictor import load_model
from .validator import Validator

RESUME_KEEP = ("device", "workers", "resume", "verbose")  # 续训时允许修改的键


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # torch < 2.3
        return torch.cuda.amp.GradScaler(enabled=enabled)


class Trainer:
    """训练器。``Trainer(overrides={...}).train()`` 返回最终（best 权重在 val 上的）指标。"""

    def __init__(self, cfg: Any = None, overrides: Optional[dict] = None,
                 callbacks: Optional[dict] = None, save_dir: Optional[Path] = None) -> None:
        overrides = dict(overrides or {})
        overrides["mode"] = "train"
        self.resume_ckpt = None
        if overrides.get("resume"):
            self.resume_ckpt = self._resume_path(overrides)
            ckpt = load_checkpoint(self.resume_ckpt)
            saved = dict(ckpt["cfg"])
            saved.update({k: overrides[k] for k in RESUME_KEEP if k in overrides})
            overrides = saved
            save_dir = save_dir or self.resume_ckpt.parent.parent
        self.args = cfg if cfg is not None else get_cfg(overrides)
        if self.args.data is None:
            raise ConfigError("data is required for training (e.g. data=ronin)")
        self.save_dir = Path(save_dir) if save_dir else increment_path(
            Path(self.args.project) / "train" / (self.args.name or "exp"),
            bool(self.args.exist_ok))
        self.wdir = self.save_dir / "weights"
        self.last, self.best = self.wdir / "last.pt", self.wdir / "best.pt"
        self.csv = self.save_dir / "results.csv"
        self.callbacks = callbacks or default_callbacks()
        self.metrics: dict = {}
        self.epoch = 0
        self.start_epoch = 0
        self.model = None
        self.stopper = EarlyStopping(int(self.args.patience))

    # ------------------------------------------------------------------ 准备
    @staticmethod
    def _resume_path(overrides: dict) -> Path:
        value = overrides["resume"]
        if isinstance(value, str) and value.lower() not in ("true", "1"):
            return Path(value)
        project = Path(overrides.get("project", "runs")) / "train"
        if overrides.get("name"):
            return project / overrides["name"] / "weights" / "last.pt"
        return latest_file(project, "*/weights/last.pt")

    def add_callback(self, event: str, fn) -> None:
        add_callback(self.callbacks, event, fn)

    def setup(self) -> None:
        a = self.args
        run_callbacks(self.callbacks, "on_pretrain_routine_start", self)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.wdir.mkdir(parents=True, exist_ok=True)
        self._log_handler = add_file_handler(self.save_dir / "log.txt")
        seed_everything(int(a.seed), bool(a.deterministic))
        self.device = select_device(a.device)
        self.amp = bool(a.amp) and self.device.type == "cuda"
        self.spec = resolve_dataset(a.data)

        self.train_set = build_dataset(a, "train", training=True, spec=self.spec)
        if len(self.train_set) == 0:
            raise RuntimeError(f"training split of {self.spec.name} has no valid windows")
        self.train_loader = build_dataloader(
            self.train_set, int(a.batch), shuffle=True, workers=int(a.workers), seed=int(a.seed),
            drop_last=True, pin_memory=self.device.type == "cuda")
        self.val_set = build_dataset(a, "val", training=False, spec=self.spec)
        # cache=true 预先构建并保留视图（快）；cache=false 只保留路径，验证时逐条读取（省内存）
        self.val_sources = ([self.val_set.sequence_view(k)
                             for k in range(len(self.val_set.sequence_ids))]
                            if self.val_set.views else [lz.path for lz in self.val_set.lazy])

        self.model = load_model(a, self.device)
        info = model_info(self.model, self.model.input_spec.window, flops=False)
        LOGGER.info(f"model {self.model.model_cfg.get('name')} "
                    f"({type(self.model).__name__}): {info['parameters']:,} parameters, "
                    f"loss={self.model.loss_name}")
        self.optimizer = build_optimizer(self.model, a.optimizer, float(a.lr), float(a.momentum),
                                         float(a.weight_decay))
        self.scheduler = build_scheduler(self.optimizer, a)
        self.scaler = make_grad_scaler(self.amp)
        self.validator = Validator(a, callbacks=self.callbacks)
        if self.resume_ckpt is not None:
            self._load_resume_state()
        yaml_save(self.save_dir / "args.yaml", a.to_dict())
        self.env = collect_env(self.device, self.spec)
        json_save(self.save_dir / "env.json", self.env)
        run_callbacks(self.callbacks, "on_pretrain_routine_end", self)

    def _load_resume_state(self) -> None:
        ckpt = load_checkpoint(self.resume_ckpt, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler"):
            self.scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt.get("scaler"):
            self.scaler.load_state_dict(ckpt["scaler"])
        self.stopper.load_state_dict(ckpt.get("stopper", {}))
        self.start_epoch = int(ckpt["epoch"]) + 1
        if self.start_epoch >= int(self.args.epochs):
            LOGGER.warning(f"resume: checkpoint already finished {self.args.epochs} epochs")
        LOGGER.info(f"resuming from {self.resume_ckpt} at epoch {self.start_epoch}")
        # 丢弃 results.csv 中续训起点之后的记录，保证每轮一行
        if self.csv.exists():
            with open(self.csv, encoding="utf-8") as f:
                rows = [r for r in csv.DictReader(f) if int(r["epoch"]) < self.start_epoch]
            self._write_rows(rows)

    # ------------------------------------------------------------------ 训练
    def train_one_epoch(self, epoch: int) -> dict:
        a = self.args
        model = self.model
        model.train()
        self.train_set.set_epoch(epoch)
        total, count, items_sum = 0.0, 0, {}
        grad_clip = float(a.grad_clip)
        for batch in self.train_loader:
            self.batch = batch
            run_callbacks(self.callbacks, "on_train_batch_start", self)
            x = batch["imu"].to(self.device, non_blocking=True)
            y = batch["target"].to(self.device, non_blocking=True)
            with torch.autocast(self.device.type, enabled=self.amp):
                out = model(x)
            out = {k: v.float() if torch.is_tensor(v) else v for k, v in out.items()}
            loss, items = de_parallel(model).loss(out, {"target": y, "imu": x}, epoch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}: {loss.item()}")
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            if grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            n = x.shape[0]
            total += float(loss.detach()) * n
            count += n
            for k, v in items.items():
                items_sum[k] = items_sum.get(k, 0.0) + float(v) * n
            run_callbacks(self.callbacks, "on_train_batch_end", self)
        out = {"train/loss": total / max(count, 1)}
        out.update({f"train/{k}": v / max(count, 1) for k, v in items_sum.items()
                    if f"train/{k}" != "train/loss"})
        return out

    def validate(self, epoch: int) -> dict:
        result = self.validator(model=self.model, sources=self.val_sources, device=self.device,
                                epoch=epoch, dataset=self.spec, split="val")
        self.val_result = result
        return result.metrics

    def fitness(self, metrics: dict) -> float:
        key = self.args.fitness
        value = metrics.get(key)
        if value is None:
            raise ConfigError(f"fitness={key!r} is not a validation metric; "
                              f"available: {sorted(metrics)}")
        if METRIC_INFO.get(key, ("", "", True))[2] is not True:
            raise ConfigError(f"fitness={key!r} is not a lower-is-better metric")
        return float(value)

    def train(self) -> dict:
        self.setup()
        a = self.args
        epochs = int(a.epochs)
        LOGGER.info(f"train: {self.spec.name}, {len(self.train_set)} windows, "
                    f"{len(self.val_sources)} val sequences, epochs {self.start_epoch}->{epochs}, "
                    f"save_dir={self.save_dir}")
        run_callbacks(self.callbacks, "on_train_start", self)
        t_start = time.time()
        epoch = self.start_epoch - 1
        try:
            for epoch in range(self.start_epoch, epochs):
                self.epoch = epoch
                t0 = time.time()
                run_callbacks(self.callbacks, "on_train_epoch_start", self)
                row = {"epoch": epoch, **self.train_one_epoch(epoch)}
                row["lr"] = self.optimizer.param_groups[0]["lr"]
                run_callbacks(self.callbacks, "on_train_epoch_end", self)
                is_plateau = isinstance(self.scheduler,
                                        torch.optim.lr_scheduler.ReduceLROnPlateau)
                if not is_plateau:
                    self.scheduler.step()
                last_epoch = epoch == epochs - 1
                improved = False
                if (epoch + 1) % int(a.val_interval) == 0 or last_epoch:
                    val = self.validate(epoch)
                    row.update({f"val/{k}": v for k, v in val.items()
                                if not k.startswith("num_")})
                    fit = self.fitness(val)
                    row["fitness"] = fit
                    improved = self.stopper.update(epoch, fit)
                    if is_plateau:
                        self.scheduler.step(val.get("loss", fit))
                    self.metrics = val
                row["time"] = time.time() - t0
                self.save_model(epoch, improved)
                self._append_row(row)
                LOGGER.info(self._epoch_summary(row, improved))
                run_callbacks(self.callbacks, "on_fit_epoch_end", self)
                if self.stopper.should_stop(epoch):
                    break
            self.final_eval(epoch)
            LOGGER.info(f"train: done in {(time.time() - t_start) / 60:.2f} min, "
                        f"best {a.fitness}={self.stopper.best:.4f} "
                        f"(epoch {self.stopper.best_epoch}), weights → {self.best}")
            run_callbacks(self.callbacks, "on_train_end", self)
        finally:
            handler = getattr(self, "_log_handler", None)
            if handler is not None:
                LOGGER.removeHandler(handler)
                handler.close()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        return self.metrics

    def _epoch_summary(self, row: dict, improved: bool) -> str:
        parts = [f"epoch {row['epoch'] + 1}/{self.args.epochs}",
                 f"loss {row['train/loss']:.4f}"]
        for k in ("val/loss", "val/ate", "val/rte"):
            if k in row and row[k] == row[k]:
                parts.append(f"{k.replace('/', '_')} {row[k]:.4f}")
        parts.append(f"lr {row['lr']:.2e}")
        parts.append(f"{row['time']:.1f}s")
        return "  ".join(parts) + ("  *" if improved else "")

    # ------------------------------------------------------------------ 保存与记录
    def checkpoint(self, epoch: int) -> dict:
        model = de_parallel(self.model)
        return {
            "epoch": epoch,
            "best_fitness": self.stopper.best,
            "model": {k: v.detach().float().cpu() if v.is_floating_point() else v.cpu()
                      for k, v in model.state_dict().items()},
            "model_cfg": to_builtin(model.model_cfg),
            "model_name": str(model.model_cfg.get("name")),
            "input_spec": model.input_spec.to_dict(),
            "cfg": to_builtin(self.args.to_dict()),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "stopper": self.stopper.state_dict(),
            "metrics": to_builtin(self.metrics),
            "git": self.env.get("git"),
            "dataset_fingerprint": (self.env.get("dataset") or {}).get("fingerprint"),
            "date": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "version": __version__,
        }

    def save_model(self, epoch: int, improved: bool) -> None:
        ckpt = self.checkpoint(epoch)
        save_checkpoint(self.last, ckpt)
        if improved:
            save_checkpoint(self.best, ckpt)
        period = int(self.args.save_period)
        if period > 0 and (epoch + 1) % period == 0:
            save_checkpoint(self.wdir / f"epoch{epoch + 1}.pt", ckpt)
        run_callbacks(self.callbacks, "on_model_save", self)

    def _write_rows(self, rows: list) -> None:
        header: list = []
        for r in rows:
            header += [k for k in r if k not in header]
        with open(self.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: ("" if isinstance(v, float) and not math.isfinite(v) else v)
                                 for k, v in r.items()})

    def _append_row(self, row: dict) -> None:
        rows = []
        if self.csv.exists():
            with open(self.csv, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        rows.append(row)
        self._write_rows(rows)

    def train_eval_gap(self, windows: int = 1024, repeats: int = 4) -> dict:
        """val 窗口上 ``eval()`` 与 ``train()`` 两种模式的窗口损失比，用于发现 train/eval 失配。

        dropout 与 BatchNorm 让同一权重在两种模式下行为不同。理想情况下两个损失相当；
        比值远大于 1 说明**报出的指标不是这组权重真实的能力**，而是 train/eval 失配的产物。
        真实案例：``ronin_resnet18`` 的 ``Dropout(0.5) → Linear → ReLU`` 头在统一配方下训练
        若干轮后，eval 模式的速度幅值只有训练模式的三分之一（损失 0.155 对 0.020），
        轨迹指标因此严重虚高。

        在模型副本上计算，不改动权重、BN 统计与随机数状态。
        """
        import copy

        n_seq = len(self.val_set.sequence_ids)
        if n_seq == 0:
            return {}
        # 跨若干条 val 序列均匀取窗口，避免结论依赖单条序列
        views, used = [], min(4, n_seq)
        for k in range(used):
            v = self.val_sources[k]
            views.append(v if hasattr(v, "imu_windows") else self.val_set.sequence_view(k))
        per_view = max(int(windows) // used, 1)
        xs, ys = [], []
        for v in views:
            starts = v.starts(int(self.args.eval_stride))
            if len(starts) == 0:
                continue
            starts = starts[np.linspace(0, len(starts) - 1, min(per_view, len(starts))).astype(int)]
            xs.append(v.imu_windows(starts))
            ys.append(v.targets(starts))
        if not xs:
            return {}
        work = copy.deepcopy(de_parallel(self.model))
        x = torch.from_numpy(np.concatenate(xs)).to(self.device)
        y = torch.from_numpy(np.concatenate(ys)).to(self.device)
        losses = {}
        with torch.no_grad():
            for mode in ("eval", "train"):
                work.train(mode == "train")
                values = []
                for _ in range(int(repeats) if mode == "train" else 1):
                    with torch.autocast(self.device.type, enabled=self.amp):
                        out = work(x)
                    out = {k: v.float() for k, v in out.items() if torch.is_tensor(v)}
                    values.append(float(work.loss(out, {"target": y, "imu": x}, self.epoch)[0]))
                losses[mode] = sum(values) / len(values)
        gap = {"loss_eval": losses["eval"], "loss_train": losses["train"],
               "ratio": losses["eval"] / losses["train"] if losses["train"] > 0 else math.inf,
               "num_windows": int(len(x)),
               "sequences": [v.sequence_id for v in views]}
        if gap["ratio"] > 2.0:
            LOGGER.warning(
                f"train/eval mismatch: val window loss is {gap['ratio']:.1f}x higher in eval() "
                f"than in train() mode ({gap['loss_eval']:.4f} vs {gap['loss_train']:.4f}). "
                "Reported metrics understate these weights; check the model's dropout/BatchNorm "
                "and the speed_ratio / plr metrics.")
        return gap

    def final_eval(self, epoch: int) -> None:
        """用 best 权重在 val 上做最终评测并写出 ``metrics.json`` 等文件（不触碰 test）。"""
        if not self.best.exists():
            LOGGER.warning("no finite fitness was recorded; using last.pt as best.pt")
            save_checkpoint(self.best, load_checkpoint(self.last))
        ckpt = load_checkpoint(self.best, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.validate(int(ckpt["epoch"]))
        result = self.val_result
        result.weights = str(self.best)
        result.env = self.env
        if self.args.efficiency:
            result.efficiency = self.validator.efficiency(self.model, self.device)
        result.cfg = self.args.to_dict()
        payload = result.to_dict()
        payload.update(mode="train", best_epoch=int(ckpt["epoch"]), last_epoch=epoch)
        try:
            payload["train_eval_gap"] = self.train_eval_gap()
        except Exception as exc:  # noqa: BLE001 - 诊断失败不影响训练结果
            LOGGER.warning(f"train/eval gap check failed: {exc}")
        result.save(self.save_dir, predictions=bool(self.args.save_predictions),
                    plots=bool(self.args.plots), max_plots=int(self.args.max_plots))
        json_save(self.save_dir / "metrics.json", payload)
        self.metrics = result.metrics
        if self.args.plots and self.csv.exists():
            try:
                from ..utils.plotting import plot_training_curves

                plot_training_curves(self.csv, self.save_dir / "plots" / "results.png")
            except Exception as exc:  # noqa: BLE001 - 绘图失败不影响训练结果
                LOGGER.warning(f"plotting results.csv failed: {exc}")
        # best.pt 去掉优化器状态以减小体积（续训使用 last.pt）
        slim = dict(ckpt)
        for key in ("optimizer", "scheduler", "scaler"):
            slim[key] = None
        save_checkpoint(self.best, slim)
