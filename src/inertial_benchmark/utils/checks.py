"""环境检查与运行环境记录（``env.json``）。"""

from __future__ import annotations

import datetime as _dt
import importlib
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from . import PACKAGE_ROOT


def git_info(path: Path = PACKAGE_ROOT) -> dict:
    """当前代码的 git 提交、分支与是否有未提交修改（非 git 目录时返回 unknown）。"""

    def run(*args: str) -> Optional[str]:
        try:
            out = subprocess.run(["git", "-C", str(path), *args], capture_output=True,
                                 text=True, timeout=10, check=True)
            return out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    commit = run("rev-parse", "HEAD")
    if commit is None:
        return {"commit": "unknown", "branch": "unknown", "dirty": None}
    status = run("status", "--porcelain", "--untracked-files=no", "--", str(PACKAGE_ROOT))
    return {"commit": commit, "branch": run("rev-parse", "--abbrev-ref", "HEAD") or "unknown",
            "dirty": bool(status)}


def package_version(name: str) -> Optional[str]:
    try:
        return getattr(importlib.import_module(name), "__version__", "unknown")
    except ImportError:
        return None


def check_torch() -> Any:
    """导入 torch，失败时给出安装提示。"""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - CI 中总是安装 torch
        raise ImportError("this command needs PyTorch: pip install 'inertial-positioning-"
                          "benchmark[train]'") from exc
    return torch


def collect_env(device: Any = None, dataset: Any = None) -> dict:
    """运行环境快照：代码版本、依赖版本、GPU、数据集指纹。"""
    from .. import __version__

    env = {
        "created_utc": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat(),
        "ipb_version": __version__,
        "git": git_info(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "argv": sys.argv,
        "packages": {n: package_version(n) for n in
                     ("numpy", "scipy", "h5py", "yaml", "torch", "pandas", "matplotlib")},
    }
    torch = sys.modules.get("torch")
    if torch is not None:
        env["torch"] = {"version": torch.__version__, "cuda": torch.version.cuda,
                        "cudnn": torch.backends.cudnn.version()
                        if torch.backends.cudnn.is_available() else None,
                        "num_threads": torch.get_num_threads()}
        if device is not None and getattr(device, "type", None) == "cuda":
            props = torch.cuda.get_device_properties(device)
            env["gpu"] = {"name": props.name, "memory_gb": round(props.total_memory / 2**30, 1),
                          "index": device.index}
    if dataset is not None:
        from ..data.manifest import MANIFEST, sha256_file

        manifest_path = dataset.root / MANIFEST
        env["dataset"] = {
            "name": dataset.name,
            "root": str(dataset.root),
            "fingerprint": dataset.fingerprint(),
            "manifest_sha256": sha256_file(manifest_path) if manifest_path.exists() else None,
            "converter": dataset.manifest().get("converter"),
        }
    return env
