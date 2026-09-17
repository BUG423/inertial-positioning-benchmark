"""VeloBins（IMU-only 变体）：逐轴分箱速度回归（`docs/algorithms/velobins.md`）。"""

from .model import ModalityEncoder, VeloBins, VeloBinsLoss, tilted_gaussian_bins

__all__ = ["ModalityEncoder", "VeloBins", "VeloBinsLoss", "tilted_gaussian_bins"]
