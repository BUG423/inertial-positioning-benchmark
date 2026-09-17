"""数据集与 DataLoader 构建（依赖 torch）。"""

from __future__ import annotations

import random
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..utils import LOGGER
from .dataset import InertialDataset
from .manifest import DatasetSpec, resolve_dataset
from .views import ViewConfig


def build_dataset(cfg: Any, split: str, training: Optional[bool] = None,
                  spec: Optional[DatasetSpec] = None) -> InertialDataset:
    """按配置为 ``split`` 构建窗口数据集；训练集使用 ``stride`` 与增强，其余用 ``eval_stride``。"""
    spec = spec or resolve_dataset(cfg.data)
    training = (split == "train") if training is None else training
    paths = spec.sequence_paths(split)
    view = ViewConfig.from_cfg(cfg)
    dataset = InertialDataset(
        paths,
        view,
        stride=cfg.stride if training else cfg.eval_stride,
        augment=cfg.augment if training else None,
        training=training,
        cache=bool(cfg.cache),
        seed=int(cfg.seed),
    )
    LOGGER.info(f"{spec.name}/{split}: {dataset}")
    return dataset


def seed_worker(worker_id: int) -> None:
    """DataLoader worker 初始化：由 torch 派生的种子同步 numpy/random。"""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def build_dataloader(dataset: InertialDataset, batch: int, shuffle: bool, workers: int = 0,
                     seed: int = 0, drop_last: bool = False,
                     pin_memory: bool = False) -> DataLoader:
    """可复现的 DataLoader：固定 ``generator`` 种子，并为 worker 设置 ``worker_init_fn``。"""
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    workers = int(min(workers, max(len(dataset) // max(batch, 1), 1))) if len(dataset) else 0
    return DataLoader(
        dataset,
        batch_size=int(batch),
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        drop_last=drop_last and len(dataset) > batch,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=False,
    )
