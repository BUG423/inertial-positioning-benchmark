"""算法实现：每个算法一个子包，导入即注册（``@register_model``）。"""

from . import ronin  # noqa: F401

__all__ = ["ronin"]
