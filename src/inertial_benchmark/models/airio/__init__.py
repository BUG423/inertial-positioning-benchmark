"""AirIO（Qiu et al., RA-L 2025）：机体系 IMU + 姿态编码（`docs/algorithms/airio.md`）。"""

from .model import AirIO, AirIOHuberCovLoss, CNNEncoder

__all__ = ["AirIO", "AirIOHuberCovLoss", "CNNEncoder"]
