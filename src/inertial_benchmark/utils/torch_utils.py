"""torch 工具：设备选择、随机种子、优化器/调度器、早停、checkpoint 读写、模型信息。"""

from __future__ import annotations

import math
import os
import random
import time
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
from torch import nn

from . import LOGGER

PathLike = Union[str, Path]
# 共享服务器上 torch 默认线程数等于核数，CPU 推理会严重争用；未显式设置时限制为 ≤ 8
NUM_THREADS = min(8, max(1, (os.cpu_count() or 1) - 1))
if "OMP_NUM_THREADS" not in os.environ:
    torch.set_num_threads(NUM_THREADS)


def select_device(device: Any = None, verbose: bool = True) -> torch.device:
    """``None``/``""`` 自动选择；``cpu``；``0`` / ``"0"`` / ``cuda:0``；多卡字符串只取第一张。"""
    text = "" if device is None else str(device).strip().lower().replace("cuda:", "")
    if text in ("", "auto"):
        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif text == "cpu":
        dev = torch.device("cpu")
    elif text == "mps":
        dev = torch.device("mps")
    else:
        first = text.split(",")[0]
        if not first.isdigit():
            raise ValueError(f"invalid device {device!r}")
        if not torch.cuda.is_available():
            raise ValueError(f"device={device!r} requested but CUDA is not available")
        if "," in text:
            LOGGER.warning(f"device={device}: multi-GPU is not supported, using cuda:{first}")
        index = int(first)
        if index >= torch.cuda.device_count():
            raise ValueError(f"device={device!r} but only {torch.cuda.device_count()} GPU(s) "
                             "are visible (CUDA_VISIBLE_DEVICES remaps indices)")
        dev = torch.device(f"cuda:{index}")
    if verbose:
        name = torch.cuda.get_device_name(dev) if dev.type == "cuda" else dev.type
        LOGGER.info(f"device: {dev} ({name}), torch {torch.__version__}")
    return dev


def epoch_seed(seed: int, epoch: int) -> int:
    """``(seed, epoch)`` → 稳定的 63 位种子（与轮次顺序无关，便于续训复现）。"""
    state = np.random.SeedSequence([int(seed), int(epoch)]).generate_state(1, dtype=np.uint64)
    return int(state[0] >> 1)


def seed_everything(seed: int = 0, deterministic: bool = True) -> None:
    """设置 python / numpy / torch 随机种子；``deterministic`` 时启用确定性算法（仅告警）。"""
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)


def de_parallel(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, (nn.DataParallel,
                                              nn.parallel.DistributedDataParallel)) else model


def build_optimizer(model: nn.Module, name: str, lr: float, momentum: float = 0.9,
                    weight_decay: float = 0.0) -> torch.optim.Optimizer:
    """权重（ndim>1）施加 weight decay，偏置与归一化层参数不衰减。"""
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (decay if p.ndim > 1 else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    name = name.lower()
    if name == "sgd":
        # Nesterov 要求动量 > 0（且 dampening = 0），momentum=0 时退化为普通 SGD
        return torch.optim.SGD(groups, lr=lr, momentum=momentum, nesterov=momentum > 0)
    if name == "adam":
        return torch.optim.Adam(groups, lr=lr, betas=(momentum, 0.999))
    if name == "adamw":
        return torch.optim.AdamW(groups, lr=lr, betas=(momentum, 0.999))
    raise ValueError(f"unknown optimizer {name!r}")


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Any):
    """按 epoch 更新的学习率调度器；``plateau`` 需要在验证后传入指标。"""
    epochs, warmup = int(cfg.epochs), int(cfg.warmup_epochs)
    name = cfg.scheduler
    if name == "plateau":
        if warmup:
            LOGGER.warning("warmup_epochs is ignored with scheduler=plateau")
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=float(cfg.gamma), patience=int(cfg.plateau_patience),
            eps=1e-12)

    def factor(epoch: int) -> float:
        if warmup and epoch < warmup:
            return (epoch + 1) / (warmup + 1)
        e = epoch - warmup
        if name == "cosine":
            span = max(epochs - warmup, 1)
            final = float(cfg.lr_final)
            return final + (1 - final) * 0.5 * (1 + math.cos(math.pi * min(e / span, 1.0)))
        if name == "step":
            return float(cfg.gamma) ** (e // max(int(cfg.step_size), 1))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class EarlyStopping:
    """fitness（越小越好）连续 ``patience`` 个 epoch 未改善时停止；``patience<=0`` 关闭。"""

    def __init__(self, patience: int = 30) -> None:
        self.patience = int(patience)
        self.best = math.inf
        self.best_epoch = -1

    def update(self, epoch: int, fitness: float) -> bool:
        """记录一次验证结果，返回是否刷新最优。"""
        if fitness is not None and math.isfinite(fitness) and fitness < self.best:
            self.best, self.best_epoch = float(fitness), int(epoch)
            return True
        return False

    def should_stop(self, epoch: int) -> bool:
        if self.patience <= 0 or self.best_epoch < 0:
            return False
        stop = epoch - self.best_epoch >= self.patience
        if stop:
            LOGGER.info(f"early stopping: no improvement for {self.patience} epochs "
                        f"(best {self.best:.4f} at epoch {self.best_epoch})")
        return stop

    def state_dict(self) -> dict:
        return {"patience": self.patience, "best": self.best, "best_epoch": self.best_epoch}

    def load_state_dict(self, state: dict) -> None:
        self.best = float(state.get("best", math.inf))
        self.best_epoch = int(state.get("best_epoch", -1))


_CKPT_CACHE: dict = {}


def load_checkpoint(path: PathLike, map_location: Any = "cpu") -> dict:
    """读取 checkpoint（只含张量与基本类型，``weights_only=True``）；按路径与修改时间缓存。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    key = (str(path.resolve()), path.stat().st_mtime_ns, str(map_location))
    if key not in _CKPT_CACHE:
        _CKPT_CACHE.clear()
        _CKPT_CACHE[key] = torch.load(path, map_location=map_location, weights_only=True)
    return _CKPT_CACHE[key]


def save_checkpoint(path: PathLike, ckpt: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def time_sync(device: Optional[torch.device] = None) -> float:
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def model_info(model: nn.Module, window: Optional[int] = None, channels: int = 6,
               flops: bool = True) -> dict:
    """参数量（总数/可训练）与单窗口 FLOPs。"""
    from ..metrics.efficiency import count_flops, count_params

    info = {"parameters": count_params(model),
            "trainable": count_params(model, trainable_only=True),
            "layers": sum(1 for _ in model.modules())}
    if flops and window:
        info.update(count_flops(model, (1, channels, window)))
    return info
