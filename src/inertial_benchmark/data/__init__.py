"""数据层：IPB v1 序列格式、重采样、转换流水线、划分与任务视图。

本包的核心部分只依赖 numpy/h5py/scipy/pyyaml；torch 相关的 ``dataset``/``build``
需显式导入（``from inertial_benchmark.data.build import build_dataloader``）。
"""

from typing import Any

from .format import (
    Sequence,
    SequenceError,
    ValidationReport,
    check_sequence,
    load_sequence,
    save_sequence,
    validate,
)

# v0.1 旧接口：保持可导入（惰性，首次访问时发出 DeprecationWarning），新代码不依赖它们
_LEGACY = ("CanonicalSequence", "SequenceValidationError", "WindowDataset", "WindowSample")


def __getattr__(name: str) -> Any:
    if name in _LEGACY:
        from . import legacy

        return getattr(legacy, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CanonicalSequence",
    "Sequence",
    "SequenceError",
    "SequenceValidationError",
    "ValidationReport",
    "WindowDataset",
    "WindowSample",
    "check_sequence",
    "load_sequence",
    "save_sequence",
    "validate",
]
