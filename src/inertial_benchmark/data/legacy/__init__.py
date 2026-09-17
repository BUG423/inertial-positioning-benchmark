"""已弃用的旧接口（IPB v0.1）：保持可导入，新代码不要使用。

``CanonicalSequence`` / ``WindowDataset`` / ``WindowSample`` / ``SequenceValidationError``
来自 v0.1 的单时钟序列表示，已被 :mod:`inertial_benchmark.data.format`（``Sequence``、
``load_sequence``）与 :mod:`inertial_benchmark.data.views`（``SequenceView``）取代。
导入本子包会发出 :class:`DeprecationWarning`；它们只用于读取历史文件与旧工具链。
"""

from __future__ import annotations

import warnings

from .v01 import (
    CanonicalSequence,
    SequenceValidationError,
    WindowDataset,
    WindowSample,
)

warnings.warn(
    "inertial_benchmark.data.legacy (IPB v0.1: CanonicalSequence / WindowDataset / WindowSample) "
    "is deprecated; use inertial_benchmark.data.format.Sequence, load_sequence and "
    "inertial_benchmark.data.views.SequenceView instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "CanonicalSequence",
    "SequenceValidationError",
    "WindowDataset",
    "WindowSample",
]
