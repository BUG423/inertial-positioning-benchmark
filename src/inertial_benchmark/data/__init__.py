"""数据层：IPB v1 序列格式、重采样、转换流水线、划分与任务视图。

本包的核心部分只依赖 numpy/h5py/scipy/pyyaml；torch 相关的 ``dataset``/``build``
需显式导入（``from inertial_benchmark.data.build import build_dataloader``）。
"""

from .format import (
    Sequence,
    SequenceError,
    ValidationReport,
    check_sequence,
    load_sequence,
    save_sequence,
    validate,
)

# v0.1 旧接口：保持可导入，新代码不依赖它们
from .legacy_v01 import CanonicalSequence, SequenceValidationError, WindowDataset, WindowSample

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
