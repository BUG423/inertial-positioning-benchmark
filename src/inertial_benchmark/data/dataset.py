"""torch 数据集：多序列、惰性索引、可选内存缓存，跳过含无效样本的窗口。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Optional, Union
from typing import Sequence as Seq

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from ..utils import LOGGER
from ..utils.geometry import quat_rotate, rotate_z
from .augment import build_augmentations
from .format import Sequence, load_sequence
from .views import (
    SequenceView,
    ViewConfig,
    first_valid_index,
    read_window,
    window_valid_mask,
    yaw_offset,
)

SequenceLike = Union[str, Path, Sequence]


class _LazySource:
    """惰性模式下一条序列的轻量元数据；h5 句柄按进程打开。"""

    def __init__(self, path: Path, cfg: ViewConfig) -> None:
        self.path = Path(path)
        with h5py.File(self.path, "r") as f:
            attrs = dict(f.attrs)
            self.n = int(f["timestamp"].shape[0])
            self.valid = np.asarray(f["valid/imu"]) & np.asarray(f["valid/pose"])
            self.has_velocity = "pose/velocity" in f
            self.sequence_id = str(attrs.get("sequence_id", self.path.stem))
            self.group_id = str(attrs.get("group_id", self.sequence_id))
            rate = float(attrs.get("sample_rate_hz", cfg.rate))
            if abs(rate - cfg.rate) > 1e-6:
                raise ValueError(f"{self.path}: sample rate {rate} != {cfg.rate}")
            self.yaw_offset = None
            if cfg.orientation == "device":
                if "imu/orientation" not in f:
                    raise ValueError(f"{self.path}: orientation=device needs imu/orientation")
                k = first_valid_index(self.valid)
                self.yaw_offset = yaw_offset(f["pose/orientation"][k], f["imu/orientation"][k])
        self._handle = None
        self._pid = None

    def handle(self) -> h5py.File:
        if self._handle is None or self._pid != os.getpid():
            self._handle = h5py.File(self.path, "r")
            self._pid = os.getpid()
        return self._handle

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state["_handle"] = None  # h5 句柄不可跨进程
        return state


class InertialDataset(Dataset):
    """窗口数据集。

    Args:
        sources: 序列文件路径或已加载的 ``Sequence``。
        view: 任务视图配置。
        stride: 滑窗步长（样本）。
        augment: 增强配置（见 ``augment.py``）；仅 ``training=True`` 时生效。
        cache: True 时把序列载入内存并预先旋转（快）；False 时逐窗口从 h5 读取（省内存）。
        seed: 增强随机数种子；每个样本的随机数由 ``(seed, epoch, index)`` 决定，与 worker 数无关。
    """

    def __init__(
        self,
        sources: Iterable[SequenceLike],
        view: ViewConfig,
        *,
        stride: int = 10,
        augment: Optional[Seq[Any]] = None,
        training: bool = False,
        cache: bool = True,
        require_valid: bool = True,
        seed: int = 0,
    ) -> None:
        self.view = view
        self.stride = int(stride)
        self.training = training
        self.cache = cache
        self.seed = int(seed)
        self.epoch = 0
        self.require_valid = require_valid
        time_shift, self.augs = build_augmentations(augment if training else None, view.frame)
        self.max_shift = time_shift.resolve(self.stride) if time_shift else 0

        self.views: list = []
        self.lazy: list = []
        self.sequence_ids: list = []
        self.group_ids: list = []
        self.valid_masks: list = []
        index = []
        for k, src in enumerate(sources):
            if cache or isinstance(src, Sequence):
                seq = src if isinstance(src, Sequence) else load_sequence(src)
                sv = SequenceView(seq, view)
                self.views.append(sv)
                n, valid = len(seq), sv.valid
                self.sequence_ids.append(seq.sequence_id)
                self.group_ids.append(seq.group_id)
            else:
                lz = _LazySource(Path(src), view)
                self.lazy.append(lz)
                n, valid = lz.n, lz.valid
                self.sequence_ids.append(lz.sequence_id)
                self.group_ids.append(lz.group_id)
            self.valid_masks.append(valid)
            if n < view.window:
                LOGGER.warning(f"dataset: {self.sequence_ids[-1]} shorter than one window, skipped")
                continue
            starts = np.arange(0, n - view.window + 1, self.stride, dtype=np.int64)
            if require_valid:
                starts = starts[window_valid_mask(valid, starts, view.window)]
            index.append(np.stack([np.full(len(starts), k, dtype=np.int64), starts], axis=1))
        self.index = np.concatenate(index) if index else np.zeros((0, 2), dtype=np.int64)
        if self.views and self.lazy:
            raise ValueError("mixing cached and lazy sources is not supported")

    def __len__(self) -> int:
        return len(self.index)

    def set_epoch(self, epoch: int) -> None:
        """每轮调用一次，使增强随机数随轮次变化且可复现。"""
        self.epoch = int(epoch)

    def _length(self, k: int) -> int:
        return len(self.views[k]) if self.views else self.lazy[k].n

    def _shift(self, k: int, start: int, rng: np.random.Generator) -> int:
        if self.max_shift <= 0:
            return start
        cand = start + int(rng.integers(-self.max_shift, self.max_shift + 1))
        cand = min(max(cand, 0), self._length(k) - self.view.window)
        if self.require_valid and not window_valid_mask(self.valid_masks[k], np.array([cand]),
                                                        self.view.window)[0]:
            return start
        return cand

    def load_window(self, k: int, start: int) -> dict:
        """取一个未增强的窗口：``imu (6,T)``、``target (D,)``、``body_to_frame`` 回调。"""
        if self.views:
            sv = self.views[k]
            starts = np.array([start])
            return {"imu": sv.imu_windows(starts)[0], "target": sv.targets(starts)[0],
                    "body_to_frame": lambda v, s=start, sv=sv: sv.body_to_frame(s, v)}
        lz = self.lazy[k]
        w = read_window(lz.handle(), start, self.view, lz.yaw_offset, lz.has_velocity)
        q, yaw_end, frame = w["q"], w["yaw_end"], self.view.frame

        def to_frame(v: np.ndarray) -> np.ndarray:
            if frame == "body":
                return np.broadcast_to(v, (len(q), 3)).astype(np.float64)
            out = quat_rotate(q, v)
            return rotate_z(out, -yaw_end) if frame == "gravity_yaw_local" else out

        return {"imu": w["imu"], "target": w["target"], "body_to_frame": to_frame}

    def __getitem__(self, i: int) -> dict:
        k, start = (int(v) for v in self.index[i])
        rng = np.random.default_rng([self.seed, self.epoch, int(i)]) if self.training else None
        if rng is not None:
            start = self._shift(k, start, rng)
        sample = self.load_window(k, start)
        if rng is not None:
            sample["imu"] = sample["imu"].copy()
            sample["target"] = sample["target"].copy()
            for aug in self.augs:
                sample = aug(sample, rng)
        return {
            "imu": torch.from_numpy(np.ascontiguousarray(sample["imu"])),
            "target": torch.from_numpy(np.ascontiguousarray(sample["target"])),
            "seq": k,
            "start": start,
        }

    def sequence(self, k: int) -> Sequence:
        """第 ``k`` 条序列（惰性模式下临时加载）。"""
        return self.views[k].seq if self.views else load_sequence(self.lazy[k].path)

    def sequence_view(self, k: int) -> SequenceView:
        return self.views[k] if self.views else SequenceView(self.sequence(k), self.view)

    def __repr__(self) -> str:
        mode = "cached" if self.views else "lazy"
        return (f"InertialDataset({len(self.sequence_ids)} sequences, {len(self)} windows, "
                f"stride={self.stride}, {mode}, augment={self.augs}, shift={self.max_shift})")
