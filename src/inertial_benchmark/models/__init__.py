"""算法实现：每个算法一个子包，导入即注册（``@register_model``）。"""

from . import deepils, eqnio, imunet, llio, rio, ronin, tinyodom, tlio  # noqa: F401

__all__ = ["deepils", "eqnio", "imunet", "llio", "rio", "ronin", "tinyodom", "tlio"]
