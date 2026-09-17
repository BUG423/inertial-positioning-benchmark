"""通用网络积木。"""

from .resnet1d import BasicBlock1D, ResNet1DBackbone, conv1d_output_length, init_resnet_weights

__all__ = ["BasicBlock1D", "ResNet1DBackbone", "conv1d_output_length", "init_resnet_weights"]
