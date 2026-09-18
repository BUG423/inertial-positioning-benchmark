"""iMoT（Nguyen et al., AAAI 2025）：变量 token 惯性运动 Transformer。

规格见 `docs/algorithms/imot.md`。
"""

from .model import (
    AdaptivePositionalEncoding,
    AdaptiveSpatialChannel,
    ConditionalCrossAttention,
    DecoderLayer,
    EncoderLayer,
    IMoT,
    ProgressiveSeriesDecoupler,
)

__all__ = [
    "AdaptivePositionalEncoding",
    "AdaptiveSpatialChannel",
    "ConditionalCrossAttention",
    "DecoderLayer",
    "EncoderLayer",
    "IMoT",
    "ProgressiveSeriesDecoupler",
]
