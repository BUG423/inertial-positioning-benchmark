"""插件加载：导入外部模块，使其中的 ``@register_model`` / ``@register_augmentation`` 生效。

环境变量 ``IPB_PLUGINS`` 列出插件，逗号分隔；每项为可导入的模块名（例如 ``my_pkg.models``）
或 ``.py`` 文件路径。插件在首次需要查询注册表时自动导入（``import_models()``、解析增强配置），
每个插件只导入一次。本模块不导入 torch。
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Iterable, Optional

from .logger import LOGGER

ENV_VAR = "IPB_PLUGINS"
_LOADED: dict = {}


def plugin_names(value: Optional[str] = None) -> list:
    """解析插件列表（缺省读取 ``IPB_PLUGINS``）。"""
    text = os.environ.get(ENV_VAR, "") if value is None else value
    return [item.strip() for item in text.split(",") if item.strip()]


def _import(spec: str):
    if spec.endswith(".py"):
        path = Path(spec).expanduser().resolve()
        if not path.exists():
            raise ImportError(f"plugin file not found: {path}")
        name = f"ipb_plugin_{path.stem}_{abs(hash(str(path))) & 0xFFFFFF:06x}"
        module_spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[name] = module
        module_spec.loader.exec_module(module)
        return module
    return importlib.import_module(spec)


def load_plugins(names: Optional[Iterable[str]] = None) -> list:
    """导入插件模块（缺省为 ``IPB_PLUGINS`` 中列出的），返回本次涉及的模块列表。"""
    modules = []
    for spec in (plugin_names() if names is None else list(names)):
        if spec not in _LOADED:
            try:
                _LOADED[spec] = _import(spec)
            except Exception as exc:
                raise ImportError(f"cannot load plugin {spec!r} ({ENV_VAR}): {exc}") from exc
            LOGGER.debug(f"loaded plugin {spec}")
        modules.append(_LOADED[spec])
    return modules


__all__ = ["ENV_VAR", "load_plugins", "plugin_names"]
