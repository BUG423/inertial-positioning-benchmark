"""IPB：惯性定位基准（Inertial Positioning Benchmark）。

公共 API::

    from inertial_benchmark import NIO, Sequence, load_sequence
    model = NIO("ronin_resnet18")
    model.train(data="ronin", epochs=40)

``NIO`` 依赖 torch，按需惰性导入；数据层不依赖 torch。
"""

from __future__ import annotations

__version__ = "1.0.0.dev0"

from .data import (
    Sequence,
    SequenceError,
    load_sequence,
    save_sequence,
    validate,
)

# v0.1 旧接口：惰性导入，首次访问时发出 DeprecationWarning（见 data/legacy）
_LEGACY = ("CanonicalSequence", "SequenceValidationError", "WindowDataset", "WindowSample")


def __getattr__(name: str):
    if name == "NIO":
        from .engine.model import NIO

        return NIO
    if name in _LEGACY:
        from . import data

        return getattr(data, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "NIO",
    "CanonicalSequence",
    "Sequence",
    "SequenceError",
    "SequenceValidationError",
    "WindowDataset",
    "WindowSample",
    "__version__",
    "load_sequence",
    "save_sequence",
    "validate",
]
