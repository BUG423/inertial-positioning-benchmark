"""训练循环：种子、AMP、梯度裁剪、优化器/调度器、逐轮验证、best/last 保存、断点续训、早停、
``results.csv``、回调与日志。训练过程只使用 train 与 val 划分。
"""

from __future__ import annotations

import csv
import datetime as _dt
import math
import random
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from .. import __version__
from ..cfg import ConfigError, get_cfg, protocol_from_cfg
from ..data.build import build_dataloader, build_dataset
from ..data.manifest import resolve_dataset
from ..metrics import METRIC_INFO
from ..nn.base import SequenceModel
from ..utils import LOGGER, add_file_handler, json_save, to_builtin, yaml_save
from ..utils.callbacks import add_callback, default_callbacks, run_callbacks
from ..utils.checks import collect_env
from ..utils.files import increment_path, latest_file
from ..utils.torch_utils import (
    EarlyStopping,
    build_optimizer,
    build_scheduler,
    de_parallel,
    epoch_seed,
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

    def _close_log(self) -> None:
        """移除并关闭本次运行的文件日志句柄（可重复调用）。"""
        handler = getattr(self, "_log_handler", None)
        if handler is not None:
            self._log_handler = None
            LOGGER.removeHandler(handler)
            handler.close()

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
        self.val_views = [self.val_set.sequence_view(k)
                          for k in range(len(self.val_set.sequence_ids))]

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
        yaml_save(self.save_dir / "args.yaml",
                  {**a.to_dict(), "protocol": protocol_from_cfg(a)})
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
    def seed_epoch(self, epoch: int) -> None:
        """按 ``(seed, epoch)`` 重置采样与全局随机数状态。

        DataLoader 的 ``generator`` 在每轮 shuffle 时被推进，torch 全局 RNG 也被 dropout 等消耗，
        因此“第 e 轮的随机性”原本依赖于之前跑过多少轮：续训那一轮会重复第 0 轮的 shuffle 顺序。
        每轮开始时按轮次重新播种后，第 e 轮的随机性只由 ``(seed, e)`` 决定，
        “连续训练 N 轮”与“中断后续训到 N 轮”得到逐位相同的权重。
        """
        seed = epoch_seed(int(self.args.seed), epoch)
        generator = getattr(self.train_loader, "generator", None)
        if generator is not None:
            generator.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed % 2**32)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def train_one_epoch(self, epoch: int) -> dict:
        a = self.args
        model = self.model
        if isinstance(de_parallel(model), SequenceModel):
            return self.calibrate_once(epoch)
        model.train()
        self.seed_epoch(epoch)
        self.train_set.set_epoch(epoch)
        total, count, items_sum = 0.0, 0, {}
        grad_clip = float(a.grad_clip)
        for batch in self.train_loader:
            self.batch = batch
            run_callbacks(self.callbacks, "on_train_batch_start", self)
            x = batch["imu"].to(self.device, non_blocking=True)
            y = batch["target"].to(self.device, non_blocking=True)
            mask = batch["mask"].to(self.device, non_blocking=True)
            extra = {k: v.to(self.device, non_blocking=True)
                     for k, v in batch.get("extra", {}).items()}
            with torch.autocast(self.device.type, enabled=self.amp):
                out = model(x, extra) if extra else model(x)
            out = {k: v.float() if torch.is_tensor(v) else v for k, v in out.items()}
            loss, items = de_parallel(model).loss(
                out, {"target": y, "imu": x, "mask": mask, "extra": extra}, epoch)
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

    def calibrate_once(self, epoch: int) -> dict:
        """序列级模型没有梯度训练：只在 **train 划分**上拟合一次标定标量。

        标定量写入 ``model_cfg["calibration"]``（随 checkpoint 与 ``metrics.json`` 发布），
        之后与其他模型走同一套验证、选模与结果写出。
        """
        model = de_parallel(self.model)
        if epoch > self.start_epoch:
            return {"train/loss": math.nan}
        views = [self.train_set.sequence_view(k)
                 for k in range(len(self.train_set.sequence_ids))]
        stats = model.calibrate(views) or {}
        model.model_cfg["calibration"] = to_builtin(stats)
        LOGGER.info(f"calibrate: {model.model_cfg.get('name')} on {self.spec.name}/train "
                    f"({len(views)} sequences): {stats}")
        row = {"train/loss": math.nan}
        row.update({f"train/{k}": float(v) for k, v in stats.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)})
        return row

    def validate(self, epoch: int) -> dict:
        result = self.validator(model=self.model, sources=self.val_views, device=self.device,
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
        try:
            self.setup()
        except BaseException:
            self._close_log()  # setup 失败时也要摘掉文件日志句柄，否则句柄泄漏到后续运行
            raise
        a = self.args
        epochs = int(a.epochs)
        LOGGER.info(f"train: {self.spec.name}, {len(self.train_set)} windows, "
                    f"{len(self.val_views)} val sequences, epochs {self.start_epoch}->{epochs}, "
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
            self._close_log()
            device = getattr(self, "device", None)
            if device is not None and device.type == "cuda":
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
        """先写 ``best.pt`` 再写 ``last.pt``。

        顺序很重要：``last.pt`` 里保存了早停器状态（``best_epoch``/``best``），只有 best 先落盘，
        中断后从 ``last.pt`` 续训时记录的最优轮次才一定对应磁盘上的 ``best.pt``。
        """
        ckpt = self.checkpoint(epoch)
        if improved:
            save_checkpoint(self.best, ckpt)
        save_checkpoint(self.last, ckpt)
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
