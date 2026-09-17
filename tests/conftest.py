"""测试公共设置：限制 CPU 线程数，并把 tests/ 加入导入路径（synthetic、fake_converter）。"""

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("IPB_VERBOSE", "0")
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import torch

    torch.set_num_threads(4)
except ImportError:  # 核心数据层测试不需要 torch
    pass
