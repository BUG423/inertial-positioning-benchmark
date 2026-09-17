"""LLIO-Net：ResMLP 风格的轻量惯性里程计网络（3D 位移 + 对角 log σ）。

论文：Y. Wang, J. Kuang, X. Niu, J. Liu, "LLIO: Lightweight Learned Inertial Odometer",
IEEE IoT Journal 10(3):2508–2518, 2023（DOI 10.1109/JIOT.2022.3214087）。
论文全文（IEEE / TechRxiv / figshare）在本机均不可得，规格卡中的训练配方只能按“用 LLIO-Net
替换 TLIO 的网络”从 TLIO 继承，并标注未核实。
官方仓库：https://github.com/i2Nav-WHU/LightweightLearnedInertialOdometer，核对提交 e225ab8f3080。
许可：GPL-3.0。本文件依据 `docs/algorithms/llio.md` 规格卡从零编写（洁净室），未阅读、未复制
官方代码；仓库只提供两个模型文件，没有数据、训练、EKF 与权重。
fidelity：official-code（网络）/ 训练配方继承 TLIO（未核实）。

结构（规格卡 §4，参数量 7,191,654 由单元测试锁定）：

* IPB 适配的无参数抽取 ``avg_pool1d(k=2, s=2)``：200 Hz×200 → 100 Hz×100（官方 README 注释的
  “1 s of imu output at 100 Hz”）；
* Feature Convert：``b c (l w) -> b l (w c)``（patch 内**时间优先**展平，``w = 25`` → 4 个 token、
  每个 150 维）→ ``Linear(150, 512)``；
* 6 个 ResMLP 层，每层两个子层，都是 ``PreAffinePostLayerScale``：
  ``A_out(x + s ⊙ fn(A_in(x)))``（仿射作用在残差相加**之后**，与原始 ResMLP 不同）；
  token-mixing 的 ``fn`` 为 ``Conv1d(4, 4, k1, bias=False)``（把 patch 轴当通道），
  channel-mixing 的 ``fn`` 为 ``Linear(512→1024, bias=False) → GELU → Dropout(0.2)
  → Linear(1024→512, bias=False)``；``s`` 初值 0.1，仿射 ``g=1, b=0``；
* 末端 ``Affine(512)`` → patch 轴均值 → 3 × ``[Linear(512, 512) → GELU → Dropout(0.5)]``
  → 并联 ``out_linear`` 与 ``out2_linear``（各 ``Linear(512, dims)``）分别给位移与 ``logstd``。

与官方的差异及理由（引用卡片）：

1. **抽取层**（卡 §6）：IPB 数据为 200 Hz，官方网络按 100 Hz×100 设计。模型入口做两点平均抽取，
   保持 1 s 时长、4 个 0.25 s patch 与全部参数形状；这与 IPB 重采样器的多相滤波不同，属于
   已知差异。备选（``patch_len=50`` 或 8 个 patch）会改变参数形状，不采用；
2. 回归 MLP 的 dropout 固定为 0.5：官方 ``PoolingMLPReg`` 的默认值，README 的 ``dropout: 0.2``
   只作用于特征提取器（卡 §10.1）；
3. ``_initialize`` 在官方从未被调用，因此全部使用 PyTorch 默认初始化（卡 §4、§10.4）；
4. 保留 3 维输出（``dims=3``）与 ``target=displacement``：``vel`` 键按 IPB 约定给出目标量本身
   （位移，单位 m），损失在位移尺度上计算，框架预测器再除以 ``(T−1)·dt = 0.995 s``（同 `tlio`）；
5. 损失与阶段切换继承 TLIO 官方实际行为（``nll_detach_then_nll``，卡 §5）——**未经 LLIO 论文核实**；
6. 不实现 EKF（v1 只评网络 + IPB 积分，卡 §6）；
7. 仓库中的 ``model_MLP.py::MLPCombineNet`` 不是 LLIO（README 未如此称呼，且默认参数无法实例化），
   不注册（卡 §16、§10.8）。
"""

from __future__ import annotations

import torch
from torch import nn

from ...nn.base import BaseModel, InputSpec
from ...nn.registry import register_model


class PatchFlatten(nn.Module):
    """``b c (l w) -> b l (w c)``：切成 ``l`` 个 patch，每个 patch 内按**时间优先**展平。"""

    def __init__(self, patch_len: int) -> None:
        super().__init__()
        self.patch_len = int(patch_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        if t % self.patch_len:
            raise ValueError(f"input length {t} is not a multiple of patch_len {self.patch_len}")
        x = x.reshape(b, c, t // self.patch_len, self.patch_len)
        return x.permute(0, 2, 3, 1).reshape(b, t // self.patch_len, self.patch_len * c)


class Affine(nn.Module):
    """逐特征仿射 ``x·g + b``，``g``、``b`` 形状 ``(1, 1, D)``，初值 1 与 0。"""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, 1, dim))
        self.b = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.g + self.b


class PreAffinePostLayerScale(nn.Module):
    """``A_out(x + s ⊙ fn(A_in(x)))``：前仿射、LayerScale 残差、**相加之后再仿射**。"""

    def __init__(self, dim: int, fn: nn.Module, init_eps: float = 0.1) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.full((1, 1, dim), float(init_eps)))
        self.affine = Affine(dim)
        self.affine_out = Affine(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.affine_out(x + self.scale * self.fn(self.affine(x)))


class ResMLPExtractor(nn.Module):
    """Feature Convert + ``layer_num`` 个 ResMLP 层 + 末端仿射。"""

    def __init__(self, in_channels: int, input_len: int, patch_len: int, feature_dim: int,
                 layer_num: int, expansion: int, dropout: float,
                 activation: nn.Module) -> None:
        super().__init__()
        patches = input_len // patch_len
        blocks: list = [PatchFlatten(patch_len), nn.Linear(patch_len * in_channels, feature_dim)]
        for _ in range(layer_num):
            token_mixing = PreAffinePostLayerScale(
                feature_dim, nn.Conv1d(patches, patches, 1, bias=False))
            channel_mixing = PreAffinePostLayerScale(feature_dim, nn.Sequential(
                nn.Linear(feature_dim, feature_dim * expansion, bias=False),
                activation,
                nn.Dropout(dropout),
                nn.Linear(feature_dim * expansion, feature_dim, bias=False),
            ))
            blocks.append(nn.Sequential(token_mixing, channel_mixing))
        blocks.append(Affine(feature_dim))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PoolingMLPReg(nn.Module):
    """patch 轴池化（均值/最大）+ ``layer_num`` 个 ``[Linear → 激活 → Dropout]`` + 两个输出头。"""

    def __init__(self, feature_dim: int, out_dim: int, layer_num: int = 3,
                 dropout: float = 0.5, pooling: str = "mean",
                 activation: nn.Module = None) -> None:
        super().__init__()
        if pooling not in ("mean", "max"):
            raise ValueError(f"unknown pooling {pooling!r}; use 'mean' or 'max'")
        self.pooling = pooling
        act = activation if activation is not None else nn.GELU()
        drop = nn.Dropout(dropout)   # 官方三个块共用同一个 Dropout 实例
        blocks: list = [nn.Identity()]
        for _ in range(layer_num):
            blocks.append(nn.Sequential(nn.Linear(feature_dim, feature_dim), act, drop))
        self.net = nn.Sequential(*blocks)
        self.out_linear = nn.Linear(feature_dim, out_dim)
        self.out2_linear = nn.Linear(feature_dim, out_dim)

    def forward(self, x: torch.Tensor) -> tuple:
        pooled = x.mean(dim=1) if self.pooling == "mean" else x.max(dim=1).values
        feature = self.net(pooled)
        return self.out_linear(feature), self.out2_linear(feature)


@register_model("llio")
class LLIONet(BaseModel):
    """LLIO-Net（官方 ``model_twolayer.py::TwoLayerModel`` 的 README 配置）。"""

    default_loss = "nll_detach_then_nll"

    def __init__(
        self,
        input_spec: InputSpec,
        decimation: int = 2,
        input_len: int = 100,
        patch_len: int = 25,
        feature_dim: int = 512,
        layer_num: int = 6,
        expansion: int = 2,
        dropout: float = 0.2,
        reg_layer_num: int = 3,
        reg_dropout: float = 0.5,
        pooling: str = "mean",
    ) -> None:
        super().__init__(input_spec)
        self.decimation = int(decimation)
        if input_spec.window != input_len * self.decimation:
            raise ValueError(
                f"llio expects window = input_len × decimation = "
                f"{input_len}×{self.decimation}; got window={input_spec.window}")
        if input_len % patch_len:
            raise ValueError(f"input_len {input_len} is not a multiple of patch_len {patch_len}")
        self.active_function = nn.GELU()   # 官方全网共享同一个 GELU 实例
        self.extractor = ResMLPExtractor(input_spec.num_channels, input_len, patch_len,
                                         feature_dim, layer_num, expansion, dropout,
                                         self.active_function)
        self.reg = PoolingMLPReg(feature_dim, input_spec.dims, reg_layer_num, reg_dropout,
                                 pooling, self.active_function)

    def forward(self, imu: torch.Tensor) -> dict:
        self.check_input(imu)
        x = imu
        if self.decimation > 1:
            x = nn.functional.avg_pool1d(x, self.decimation, self.decimation)
        displacement, logstd = self.reg(self.extractor(x))
        # vel 键按 IPB 约定给出“目标量”，即窗口位移（m）；预测器再换算为平均速度
        return {"vel": displacement, "logstd": logstd}
