"""算法实现：每个算法一个子包，导入即注册（``@register_model``）。"""

from . import (  # noqa: F401
    deepils,
    dive,
    eqnio,
    gnio,
    imunet,
    ionet,
    llio,
    mean_speed_heading,
    nio_lie_events,
    pdr,
    rio,
    ronin,
    tinyodom,
    tlio,
    velobins,
)

__all__ = [
    "deepils",
    "dive",
    "eqnio",
    "gnio",
    "imunet",
    "ionet",
    "llio",
    "mean_speed_heading",
    "nio_lie_events",
    "pdr",
    "rio",
    "ronin",
    "tinyodom",
    "tlio",
    "velobins",
]
