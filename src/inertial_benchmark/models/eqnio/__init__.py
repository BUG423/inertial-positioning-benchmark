"""EqNIO（Jayanth et al., ICLR 2025）：O(2) 规范帧 + 骨干网络。v1 只实现 RoNIN 骨干变体。"""

from .ronin import EqNIORoNIN

__all__ = ["EqNIORoNIN"]
