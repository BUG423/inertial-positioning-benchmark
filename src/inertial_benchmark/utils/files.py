"""文件与路径工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Union

PathLike = Union[str, Path]


def increment_path(path: PathLike, exist_ok: bool = False, mkdir: bool = False) -> Path:
    """``runs/train/exp`` 已存在时返回 ``exp2``、``exp3`` …（``exist_ok`` 时原样返回）。"""
    path = Path(path)
    if path.exists() and not exist_ok:
        base = path
        k = 2
        while (candidate := base.with_name(f"{base.name}{k}")).exists():
            k += 1
        path = candidate
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)
    return path


def latest_file(root: PathLike, pattern: str = "**/last.pt") -> Path:
    """``root`` 下最近修改的匹配文件。"""
    files = sorted(Path(root).glob(pattern), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"no {pattern} under {root}")
    return files[-1]
