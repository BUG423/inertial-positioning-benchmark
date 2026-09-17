"""训练 / 验证 / 推理引擎。

``results`` 只依赖 numpy，可在没有 torch 的环境中导入；其余组件依赖 torch，按需惰性导入。
"""

from __future__ import annotations

import importlib

_LAZY = {
    "NIO": ".model",
    "Predictor": ".predictor",
    "Trainer": ".trainer",
    "Validator": ".validator",
    "RunResult": ".results",
    "SequenceResult": ".results",
    "Trajectory": ".results",
}


def __getattr__(name: str):
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = sorted(_LAZY)
