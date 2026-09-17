"""NIO from Lie Events（Jayanth et al., RSS 2025）：SE(3) 李事件表示 + RoNIN / TLIO 骨干。

两个变体都使用**特权输入** ``init_velocity``（见 `docs/algorithms/nio_lie_events.md`）。
"""

from .model import EVENT_CHANNELS, LieEventModel, NIOLieEventsRoNIN, NIOLieEventsTLIO

__all__ = ["EVENT_CHANNELS", "LieEventModel", "NIOLieEventsRoNIN", "NIOLieEventsTLIO"]
