"""PDR 基线（步检测 + Weinberg 步长 + 设备航向），规格见 ``docs/algorithms/pdr.md``。"""

from .heading import BODY_AXES, resolve_axes, segment_heading, window_headings
from .model import PDR

__all__ = ["BODY_AXES", "PDR", "resolve_axes", "segment_heading", "window_headings"]
