"""神经网络组件：模型基类、注册表、输出头、损失与通用模块（依赖 torch）。"""

from .base import LOSS_BATCH_KEYS, BaseModel, InputSpec, SequenceModel, check_loss_batch
from .heads import GaussianHead, PolarHead, VelocityHead, build_head
from .losses import LOSSES, build_loss, gaussian_nll, mse, register_loss
from .registry import MODELS, build_model, import_models, list_models, register_model

__all__ = [
    "LOSSES",
    "LOSS_BATCH_KEYS",
    "MODELS",
    "BaseModel",
    "GaussianHead",
    "InputSpec",
    "PolarHead",
    "SequenceModel",
    "VelocityHead",
    "build_head",
    "check_loss_batch",
    "build_loss",
    "build_model",
    "gaussian_nll",
    "import_models",
    "list_models",
    "mse",
    "register_loss",
    "register_model",
]
