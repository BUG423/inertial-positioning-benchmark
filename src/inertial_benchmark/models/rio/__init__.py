"""RIO（Cao et al., CVPR 2022）：旋转等变自监督训练 + 自适应测试时训练（paper-only）。"""

from .model import (
    RIOJointLoss,
    RIOResNet,
    negative_cosine,
    replace_batchnorm_with_groupnorm,
    rotate_imu_z,
    rotate_vector_z,
)
from .ttt import DEFAULT_ANGLES_DEG, AdaptiveTTT

__all__ = [
    "DEFAULT_ANGLES_DEG",
    "AdaptiveTTT",
    "RIOJointLoss",
    "RIOResNet",
    "negative_cosine",
    "replace_batchnorm_with_groupnorm",
    "rotate_imu_z",
    "rotate_vector_z",
]
