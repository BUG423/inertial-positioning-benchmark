"""通用工具：路径常量、YAML/JSON 读写、日志。本模块不导入 torch。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Union

import numpy as np
import yaml

from .logger import LOGGER, add_file_handler, set_logging

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = PACKAGE_ROOT / "cfg"
DEFAULT_CFG_PATH = CFG_DIR / "default.yaml"
PathLike = Union[str, Path]


def datasets_dir() -> Path:
    """转换后数据集根目录：环境变量 ``IPB_DATASETS``，缺省 ``~/datasets/ipb``。"""
    return Path(os.environ.get("IPB_DATASETS", "~/datasets/ipb")).expanduser()


class IterableSimpleNamespace(SimpleNamespace):
    """可迭代、可转 dict 的命名空间，用于承载解析后的配置。"""

    def __iter__(self):
        return iter(vars(self).items())

    def __contains__(self, key: str) -> bool:
        return key in vars(self)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> dict:
        return dict(vars(self))

    def __str__(self) -> str:
        return "\n".join(f"{k}={v}" for k, v in self)


def to_builtin(obj: Any) -> Any:
    """把 numpy / Path 等对象递归转为 JSON/YAML 可序列化的内置类型。"""
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_builtin(obj.tolist())
    if isinstance(obj, np.generic):
        return to_builtin(obj.item())
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None  # JSON 不支持 NaN/Inf，统一写 null
    return obj


def yaml_load(path: PathLike) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def yaml_save(path: PathLike, data: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(to_builtin(data), f, sort_keys=False, allow_unicode=True)


def json_load(path: PathLike) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def json_save(path: PathLike, data: Any, indent: int = 2) -> None:
    """写 JSON（NaN 写为 null），先写临时文件再原子替换。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(to_builtin(data), f, indent=indent, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    os.replace(tmp, path)


__all__ = [
    "CFG_DIR",
    "DEFAULT_CFG_PATH",
    "LOGGER",
    "PACKAGE_ROOT",
    "IterableSimpleNamespace",
    "add_file_handler",
    "datasets_dir",
    "json_load",
    "json_save",
    "set_logging",
    "to_builtin",
    "yaml_load",
    "yaml_save",
]
