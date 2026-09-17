"""日志：统一的 ``ipb`` logger，控制台输出简洁，运行目录内额外写文件。"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Union

LOGGING_NAME = "ipb"
VERBOSE = os.environ.get("IPB_VERBOSE", "1").lower() not in {"0", "false", "no"}


def set_logging(name: str = LOGGING_NAME, verbose: bool = VERBOSE) -> logging.Logger:
    """配置并返回项目 logger（幂等）。"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    if not any(getattr(h, "_ipb_console", False) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler._ipb_console = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def add_file_handler(path: Union[str, Path], logger: Optional[logging.Logger] = None):
    """把日志同时写入 ``path``；返回 handler，调用方负责在结束时移除。"""
    logger = logger or LOGGER
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return handler


LOGGER = set_logging()
