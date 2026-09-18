"""RNIN（RNIN-VIO 的惯性网络，Chen et al., ISMAR 2021）参考实现。"""

from .model import RNINDisplacementLoss, RNINResNetLSTM

__all__ = ["RNINDisplacementLoss", "RNINResNetLSTM"]
