"""原始公开数据集 → IPB v1 转换器。每个数据集一个模块，接口见 ``base.py``。

本文件只负责按名字发现与加载转换器模块；具体转换器由各自的 ``<dataset>.py`` 提供。
"""

from __future__ import annotations

import importlib
import importlib.util
import pkgutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Union

REQUIRED_MEMBERS = ("NAME", "VERSION", "LICENSE", "official_splits", "iter_raw_sequences")
PLANNED = ("ronin", "ridi", "oxiod", "tlio", "idol", "rnin", "imunet", "pedlocdata", "iplab")


class ConverterError(ImportError):
    """转换器不存在或不满足 ``base.py`` 契约。"""


def available_converters() -> list:
    """当前已安装的转换器模块名（不含 ``base``）。"""
    names = [m.name for m in pkgutil.iter_modules(__path__) if not m.name.startswith("_")]
    return sorted(n for n in names if n != "base")


def load_converter(ref: Union[str, Path, ModuleType]) -> ModuleType:
    """按名字（``ronin``）、模块路径（``pkg.mod``）、文件路径或模块对象加载转换器并检查契约。"""
    if isinstance(ref, ModuleType):
        module = ref
    else:
        text = str(ref)
        path = Path(text)
        if text.endswith(".py") and path.exists():
            module = _load_file(path)
        elif "." in text:
            module = importlib.import_module(text)
        else:
            try:
                module = importlib.import_module(f"{__name__}.{text}")
            except ModuleNotFoundError as exc:
                if exc.name != f"{__name__}.{text}":
                    raise
                raise ConverterError(
                    f"unknown converter {text!r}; available: {available_converters()}"
                ) from exc
    missing = [m for m in REQUIRED_MEMBERS if not hasattr(module, m)]
    if missing:
        raise ConverterError(f"converter {module.__name__} lacks {', '.join(missing)}")
    return module


def converter_ref(module: ModuleType) -> tuple:
    """可跨进程传递的模块引用 ``(module_name, file_path)``。"""
    return module.__name__, getattr(module, "__file__", None)


def resolve_converter_ref(ref: tuple) -> ModuleType:
    """在子进程中还原 ``converter_ref`` 得到的模块（先按名字，再按文件）。"""
    name, file = ref
    if name in sys.modules:
        return sys.modules[name]
    try:
        return load_converter(name if "." in name else f"{__name__}.{name}")
    except ImportError:
        if file is None:
            raise
        return load_converter(Path(file))


def _load_file(path: Path) -> ModuleType:
    name = f"_ipb_converter_{path.stem}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ConverterError(f"cannot load converter from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
