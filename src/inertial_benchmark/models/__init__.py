"""算法实现：每个算法一个子包，导入即注册（``@register_model``）。"""

from . import (  # noqa: F401
    deepils,
    eqnio,
    gnio,
    imunet,
    ionet,
    llio,
    mean_speed_heading,
    pdr,
    rio,
    ronin,
    tinyodom,
    tlio,
)

__all__ = [
    "deepils",
    "eqnio",
    "gnio",
    "imunet",
    "ionet",
    "llio",
    "mean_speed_heading",
    "pdr",
    "rio",
    "ronin",
    "tinyodom",
    "tlio",
]
