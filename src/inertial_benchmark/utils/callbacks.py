"""回调系统：事件名 → 回调列表；回调接收触发它的对象（Trainer / Validator / Predictor）。"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Optional

from . import LOGGER

EVENTS = (
    "on_pretrain_routine_start",
    "on_pretrain_routine_end",
    "on_train_start",
    "on_train_epoch_start",
    "on_train_batch_start",
    "on_train_batch_end",
    "on_train_epoch_end",
    "on_val_start",
    "on_val_end",
    "on_fit_epoch_end",
    "on_model_save",
    "on_train_end",
    "on_predict_start",
    "on_predict_end",
)


def default_callbacks() -> dict:
    return defaultdict(list, {event: [] for event in EVENTS})


def add_callback(callbacks: dict, event: str, fn: Callable[[Any], None]) -> None:
    if event not in EVENTS:
        raise KeyError(f"unknown callback event {event!r}; available: {EVENTS}")
    callbacks[event].append(fn)


def run_callbacks(callbacks: Optional[dict], event: str, obj: Any) -> None:
    """依次调用事件的回调；回调抛出的异常会记录后继续向上抛出（不静默吞掉）。"""
    for fn in (callbacks or {}).get(event, []):
        try:
            fn(obj)
        except Exception:
            LOGGER.error(f"callback {getattr(fn, '__name__', fn)} for {event} failed")
            raise
