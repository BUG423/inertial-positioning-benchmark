"""VeloBins（IMU-only 变体）：逐轴分箱速度回归 + 误差条件高斯标签。

论文：Maulana Bisyir Azhari, Seungwook Lee, Donghun Han, Sung Jun Park, David Hyunchul Shim,
"VeloBins: Learning Velocity and Its Uncertainty via Bins and Error-Conditioned Gaussian Labels
for Aerial Inertial Odometry", arXiv:2608.29720v1 [cs.RO], 2026-08-30。
官方仓库：**无**（摘要："The code will be released upon acceptance"）；许可：不适用（无代码）。
fidelity：`paper-only`——本文件依据 `docs/algorithms/velobins.md` 规格卡实现。

.. warning::

   **这是 IMU-only 变体，不是论文配置。** 官方输入除 IMU 外还有**逐旋翼转速（RPM）或 PWM
   指令**，行人数据集与 IPB 格式都没有这一模态，因此把执行器支路整条去掉（模态支路 3 → 2，
   每条宽度 16 → 24 以保持 ``D_h = 48``）。这**改变了结构**，论文 Table II 报告的收益
   （全部来自空中平台 + EKF 融合）**不适用于本变体**，只能用来验证“分箱 + 误差条件标签”相对
   同编码器回归基线的相对收益（卡 §6、§10-9）。

结构（卡 §4，参数量 **69,968** 为本卡推导值，由单元测试锁定）：

1. 逐模态 1D 卷积支路 ×2（``acc``、``gyro``）：``Conv1d(3, 24, k5, p2) → ReLU →
   Conv1d(24, 24, k5, p2) → ReLU``，拼接得到 ``d_model = 48`` 的 token 序列（6,576）；
2. ``TransformerEncoder(TransformerEncoderLayer(48, nhead=4, ff=192, batch_first=True), 2)``
   （56,544）；取**最后一个 token** 作为 ``h ∈ R^48``；
3. **分箱解码器**（可复用的 `nn.heads.BinsHead`，式 3–4，6,848）：``SPE(b) = sin(γ ⊙ PE(b))``、
   ``e_n = φ(SPE(b_n))``、``q_i = θ_i(h)``、``π_i = softmax_n(q_i·e_n)``；解码 ``v̂_i = Σ π b``
   （式 1）与 ``σ̂_i² = Σ π (b − v̂_i)²``（式 2）。解码器参数量**与箱数 N 无关**。

参数量说明：卡 §4.1 的逐行分解把 96 个参数（一层 ``TransformerEncoderLayer`` 的两组 LayerNorm
偏置/权重差额）记在了卷积行上（"6,672 + 56,448"），但**总计 69,968 与本实现逐位一致**；
官方形状（含 rotor 支路、W=16）同样逐位得到 68,128。论文报告约 76.6 k（编码器 69.8 k +
解码器 6.8 k），差异全部落在未公开的编码器细节上（卡 §10-1）。

损失（`velobins_bins`，式 5–10）：``L = Σ_i (λ_H·Huber_δ(v̂_i − v_i) + λ_KL·KL(q_i ‖ π_i))``，
``λ_H = λ_KL = 1``、``δ = 0.1 m/s``。标签 ``q_i`` 是**误差条件高斯**：
``σ_i = max(|v̂_i − v_i|, Δ/10)``（detach，不回传），先按式 6 在每个箱的支撑上做 CDF 离散化，
再按式 7 做指数倾斜使 ``Σ_n q_{i,n} b_n = v_i`` 精确成立（``η_i`` 用二分法在 autograd 之外
求解）。**全程没有 NLL 项**
——这正是论文的卖点之一（卡 §5.2）。

与论文/卡片的差异及理由：

1. **去掉 rotor/PWM 支路**（见上）：改变结构，必须在结果中标注为 IMU-only 变体；
2. 编码器细节（卷积核/通道/头数/FFN 宽度/位置编码/dropout）论文全未给出，沿用卡 §4.1 的参考
   配置并在模型 YAML 中显式声明；参数量对这些选择敏感（卡 §10-1）；
3. 采样率：论文是 1 s @100 Hz（100 个 token），IPB 是 1 s @200 Hz（200 个 token）。卷积与
   transformer 都与长度无关，``decimate=2`` 可复现 100 个 token 的时间分辨率，默认关闭（卡 §6）；
4. 通道置换 ``[gyro, acc] → [acc, gyro]``，保持官方模态语义（卡 §2）；
5. 目标时刻论文未定义，IPB 默认 ``target=avg_velocity``、``frame=body``、``dims=3``
   （DESIGN §3 规定 body 系必须 3 维）；``velocity_at_end`` 为备选（卡 §10-3）；
6. 分箱范围 ``[−R, R]``：论文按数据集取 ``R ≈ 1.1 × 训练集逐轴最大速度``。IPB 的诚实协议要求
   **只用训练划分**统计；本实现把 ``value_range`` 作为模型超参写入 ``args``/``args.yaml``，
   默认 5.0 m/s（行人速度的保守上界），换数据集时必须用训练划分重新统计并记录越界比例
   （卡 §3.1、§10-4）；
7. 解码方式 ``decode ∈ {expectation, argmax, topk_expectation}`` 可切换（对照“分箱表示”与
   “期望解码”各自的贡献，卡 §6）；``argmax``/``topk`` 的方差解码仍用完整分布；
8. 不实现 EKF（IPB v1 只评网络 + DESIGN §5 积分）：论文 Table II 全部为融合后结果（卡 §6）。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from ...nn.base import BaseModel, InputSpec
from ...nn.heads import BinsHead, bin_edges, normal_cdf
from ...nn.losses import expand_mask, masked_mean, masked_output_mean, mse, register_loss
from ...nn.registry import register_model


class ModalityEncoder(nn.Module):
    """单模态 1D 卷积支路：``Conv1d(c, W, k, p) → ReLU → Conv1d(W, W, k, p) → ReLU``。"""

    def __init__(self, in_channels: int, width: int, kernel_size: int = 5) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, width, kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(width, width, kernel_size, padding=pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.conv2(F.relu(self.conv1(x))))


def tilted_gaussian_bins(target: torch.Tensor, sigma: torch.Tensor, lower: torch.Tensor,
                         upper: torch.Tensor, centers: torch.Tensor, iterations: int = 50,
                         tol: float = 1e-8) -> torch.Tensor:
    """误差条件高斯标签（式 6–7），``target``/``sigma`` 形状 ``(..., )``，返回 ``(..., N)``。

    式 6：``q_n ∝ Φ((b_n⁺ − v)/σ) − Φ((b_n⁻ − v)/σ)``（HL-Gauss 直方图口径）；
    式 7：``q_n ← q_n·e^{η b_n}/Z``，``η`` 由均值约束 ``Σ_n q_n b_n = v`` 唯一确定
    （``Σ_n q_n e^{η b_n}(b_n − v)`` 对 ``η`` 单调递增，用二分法求根，与 autograd 无关）。

    全程在 ``no_grad`` 语义下使用：调用方负责把 ``target``/``sigma`` detach。
    """
    v = target.unsqueeze(-1)
    s = sigma.unsqueeze(-1).clamp_min(1e-12)
    q = (normal_cdf((upper - v) / s) - normal_cdf((lower - v) / s)).clamp_min(0.0)
    q = q / q.sum(-1, keepdim=True).clamp_min(1e-30)
    shifted = centers - v                                  # (..., N)
    # 二分区间：η 的符号与当前均值偏差相反；先按倍增法找到包住根的区间
    low = torch.full_like(target, -1.0)
    high = torch.full_like(target, 1.0)

    def moment(eta: torch.Tensor) -> torch.Tensor:
        weight = q * torch.exp((eta.unsqueeze(-1) * shifted).clamp(-60.0, 60.0))
        return (weight * shifted).sum(-1) / weight.sum(-1).clamp_min(1e-30)

    for _ in range(iterations):
        expand = moment(low) > 0
        if not bool(expand.any()):
            break
        low = torch.where(expand, low * 2.0, low)
    for _ in range(iterations):
        expand = moment(high) < 0
        if not bool(expand.any()):
            break
        high = torch.where(expand, high * 2.0, high)
    for _ in range(iterations):
        mid = 0.5 * (low + high)
        negative = moment(mid) < 0
        low = torch.where(negative, mid, low)
        high = torch.where(negative, high, mid)
        if float((high - low).abs().max()) < tol:
            break
    eta = 0.5 * (low + high)
    q = q * torch.exp((eta.unsqueeze(-1) * shifted).clamp(-60.0, 60.0))
    return q / q.sum(-1, keepdim=True).clamp_min(1e-30)


@register_loss("velobins_bins")
class VeloBinsLoss:
    """VeloBins 式 8–10：``λ_H·Huber_δ(v̂ − v) + λ_KL·KL(q ‖ π)``，**没有 NLL 项**。

    ``q`` 为误差条件高斯标签（式 5–7），``σ_i = max(|v̂_i − v_i|, Δ/10)``，构造时全程 detach。
    两项都对 3 个轴求和、再按逐输出掩码 ``mask`` 平均。
    """

    def __init__(self, huber_weight: float = 1.0, kl_weight: float = 1.0, delta: float = 0.1,
                 sigma_floor_ratio: float = 0.1) -> None:
        self.huber_weight = float(huber_weight)
        self.kl_weight = float(kl_weight)
        self.delta = float(delta)
        self.sigma_floor_ratio = float(sigma_floor_ratio)

    def labels(self, out: dict, target: torch.Tensor) -> torch.Tensor:
        head = out["head"]
        with torch.no_grad():
            error = (out["vel"].detach() - target).abs()
            sigma = error.clamp_min(self.sigma_floor_ratio * head.bin_width)
            centers = head.centers.to(target.dtype)
            lower, upper = bin_edges(centers, head.value_range)
            return tilted_gaussian_bins(target, sigma, lower, upper, centers)

    def __call__(self, out: dict, target: torch.Tensor, epoch: int = 0,
                 mask: Optional[torch.Tensor] = None) -> tuple:
        if "probs" not in out or "head" not in out:
            raise KeyError("velobins_bins needs the model to output 'probs' and 'head'")
        pred = out["vel"]
        huber = masked_output_mean(
            F.huber_loss(pred, target, reduction="none", delta=self.delta).sum(-1), mask)
        labels = self.labels(out, target)
        log_probs = torch.log(out["probs"].clamp_min(1e-30))
        terms = (labels * (torch.log(labels.clamp_min(1e-30)) - log_probs)).sum(-1)
        kl = masked_mean(terms.sum(-1, keepdim=True),
                         expand_mask(mask, terms.sum(-1, keepdim=True)))
        loss = self.huber_weight * huber + self.kl_weight * kl
        return loss, {"huber": huber.detach(), "kl": kl.detach(),
                      "mse": mse(pred, target, mask).detach()}


@register_model("velobins")
class VeloBins(BaseModel):
    """VeloBins IMU-only 变体：逐模态卷积 + 两层 transformer + 分箱解码器。"""

    default_loss = "velobins_bins"
    # ``probs`` 是 (B, dims, N) 的诊断量，不写进预测文件（``vel``/``logstd`` 照常保存）
    saved_outputs = ()

    def __init__(
        self,
        input_spec: InputSpec,
        width: int = 24,
        kernel_size: int = 5,
        d_model: int = 48,
        heads: int = 4,
        feedforward: int = 192,
        layers: int = 2,
        dropout: float = 0.1,
        bins: int = 512,
        value_range: float = 5.0,
        code_dim: int = 64,
        query_dim: int = 32,
        encoding: str = "spe",
        decode: str = "expectation",
        topk: int = 5,
        clamp_std: bool = True,
        decimate: int = 1,
    ) -> None:
        super().__init__(input_spec)
        if input_spec.output_layout != "window":
            raise ValueError("velobins predicts one velocity per window "
                             "(target=avg_velocity / velocity_at_end / displacement)")
        if input_spec.num_channels != 6:
            raise ValueError("velobins expects the fixed [gyro, acc] channels")
        if 2 * int(width) != int(d_model):
            raise ValueError(f"two modality branches of width {width} must give "
                             f"d_model={2 * int(width)}, got {d_model}")
        if int(decimate) < 1:
            raise ValueError(f"decimate must be >= 1, got {decimate}")
        self.decimate = int(decimate)
        # 官方模态顺序为 [acc, gyro]（IPB 输入是 [gyro, acc]，在 forward 中置换）
        self.acc_encoder = ModalityEncoder(3, int(width), int(kernel_size))
        self.gyro_encoder = ModalityEncoder(3, int(width), int(kernel_size))
        layer = nn.TransformerEncoderLayer(int(d_model), int(heads), int(feedforward),
                                           dropout=float(dropout), batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, int(layers))
        self.head = BinsHead(int(d_model), input_spec.dims, bins=int(bins),
                             value_range=float(value_range), code_dim=int(code_dim),
                             query_dim=int(query_dim), encoding=encoding, decode=decode,
                             topk=int(topk), clamp_std=bool(clamp_std))

    def encode(self, imu: torch.Tensor) -> torch.Tensor:
        """``(B, 6, T)`` → 最后一个 token 的特征 ``h (B, d_model)``。"""
        self.check_input(imu)
        if self.decimate > 1:
            imu = imu[..., ::self.decimate]
        # IPB 的 [gyro, acc] → 官方的 [acc, gyro] 两条支路
        tokens = torch.cat([self.acc_encoder(imu[:, 3:6]), self.gyro_encoder(imu[:, 0:3])], dim=1)
        return self.transformer(tokens.transpose(1, 2))[:, -1]

    def forward(self, imu: torch.Tensor) -> dict:
        return self.head(self.encode(imu))

    def loss(self, out: dict, batch: dict, epoch: int = 0) -> tuple:
        """把分箱头本身传给损失（构造误差条件高斯标签需要箱中心/边界/箱宽）。

        ``forward`` 的返回值里不能放非张量对象：Predictor 会先过滤掉它们，训练与验证两条路径
        才会看到不同的键（DESIGN §4 的损失契约）。因此在这里注入，两条路径都一致。
        """
        from ...nn.base import check_loss_batch
        from ...nn.losses import build_loss

        check_loss_batch(batch)
        fn = build_loss(self.loss_name, **self.loss_kwargs)
        return fn({**out, "head": self.head}, batch["target"], epoch, batch.get("mask"))
