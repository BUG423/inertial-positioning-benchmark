"""RIO 的自适应测试时训练（A-TTT，规格卡 `docs/algorithms/rio.md` §5.3）。

论文补充材料 Algorithm 1：对**同一序列**按时间顺序，每 128 个窗口成一批；用 M 个独立训练模型
（深度集成，冻结）的逐样本方差决定动作：

* ``min_i σ²_i < reset_threshold``（1e-4）→ **恢复**：参数回到 ``θ*``，Adam 状态重置；
* 否则 ``mean_i σ²_i < hold_threshold``（0.04）→ **保持**：参数不变；
* 否则 → 用 4 个固定共轭角（72°/144°/216°/288°）的自监督损失做 ``steps``（5）次 Adam 更新。

预测始终用在线模型（``eval`` 模式），不用集成均值。``σ²_i = mean_m ‖v^m_i − v̄_i‖²``（平方范数，
即协方差的迹；论文对向量的写法不明确，卡 §10.5）。

本类只依赖窗口张量流，不接触框架的预测器：IPB v1 的 ``Predictor`` 还没有“序列级有状态钩子”
（`docs/ALGORITHMS.md` §3 的待落地扩展），所以 A-TTT 暂时只能在脚本/测试中使用。使用时必须在
结果中标注“使用了测试时自适应（无标签，转导式推理）”，并且每条序列都要 ``reset()``。
"""

from __future__ import annotations

import copy
import math
from typing import Iterable, Optional, Sequence

import torch

DEFAULT_ANGLES_DEG = (72.0, 144.0, 216.0, 288.0)


class AdaptiveTTT:
    """A-TTT 状态机（``mode='adaptive'``）与朴素 TTT（``mode='naive'``，仅消融）。"""

    def __init__(
        self,
        model,
        ensemble: Sequence,
        lr: float = 1e-4,
        steps: int = 5,
        batch: int = 128,
        angles_deg: Sequence[float] = DEFAULT_ANGLES_DEG,
        hold_threshold: float = 0.04,
        reset_threshold: float = 1e-4,
        mode: str = "adaptive",
    ) -> None:
        if mode not in ("adaptive", "naive", "off"):
            raise ValueError(f"mode must be 'adaptive', 'naive' or 'off', got {mode!r}")
        self.model = model
        self.ensemble = list(ensemble)
        self.lr = float(lr)
        self.steps = int(steps)
        self.batch = int(batch)
        self.angles = [math.radians(a) for a in angles_deg]
        self.hold_threshold = float(hold_threshold)
        self.reset_threshold = float(reset_threshold)
        self.mode = mode
        self.initial_state = copy.deepcopy(model.state_dict())
        self.actions: list = []
        self.reset()

    # ------------------------------------------------------------------ 状态
    def reset(self) -> None:
        """回到 ``θ*`` 并重置优化器状态（每条新序列都必须调用）。"""
        self.model.load_state_dict(self.initial_state)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    @torch.no_grad()
    def ensemble_variance(self, imu: torch.Tensor) -> torch.Tensor:
        """逐样本方差 ``σ²_i = mean_m ‖v^m_i − v̄_i‖²``（集成为空时返回全零）。"""
        if not self.ensemble:
            return torch.zeros(imu.shape[0], device=imu.device)
        preds = []
        for member in self.ensemble:
            was_training = member.training
            member.eval()
            preds.append(member(imu)["vel"])
            member.train(was_training)
        stacked = torch.stack(preds)
        mean = stacked.mean(dim=0, keepdim=True)
        return ((stacked - mean) ** 2).sum(dim=-1).mean(dim=0)

    def decide(self, variance: torch.Tensor) -> str:
        """返回 ``"reset"`` / ``"hold"`` / ``"update"``。"""
        if self.mode == "off":
            return "hold"
        if self.mode == "naive":
            return "update"
        if float(variance.min()) < self.reset_threshold:
            return "reset"
        if float(variance.mean()) < self.hold_threshold:
            return "hold"
        return "update"

    # ------------------------------------------------------------------ 更新
    def ssl_loss(self, imu: torch.Tensor) -> torch.Tensor:
        """4 个固定共轭角的自监督损失之和（门控与训练时相同）。"""
        from ...nn.losses import build_loss

        loss_fn = build_loss(self.model.loss_name, **self.model.loss_kwargs)
        total = imu.new_zeros(())
        for angle in self.angles:
            phi = imu.new_full((imu.shape[0],), angle)
            out = self.model(imu, angle=phi)
            total = total + loss_fn.ssl(out, out["vel"].detach())
        return total

    def update(self, imu: torch.Tensor) -> None:
        """``steps`` 次 Adam 更新（梯度步用 ``train()`` 模式，卡 §5.3 的假设）。"""
        was_training = self.model.training
        self.model.train()
        for _ in range(self.steps):
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.ssl_loss(imu)
            loss.backward()
            self.optimizer.step()
        self.model.train(was_training)

    @torch.no_grad()
    def infer(self, imu: torch.Tensor) -> torch.Tensor:
        was_training = self.model.training
        self.model.eval()
        vel = self.model(imu)["vel"]
        self.model.train(was_training)
        return vel

    def step(self, imu: torch.Tensor) -> torch.Tensor:
        """处理一批窗口：决策 → （可能）更新 → 用在线模型预测。"""
        action = self.decide(self.ensemble_variance(imu))
        if action == "reset":
            self.reset()
        elif action == "update":
            self.update(imu)
        self.actions.append(action)
        return self.infer(imu)

    def run(self, windows: Iterable[torch.Tensor], batch: Optional[int] = None) -> torch.Tensor:
        """按时间顺序处理一条序列的窗口 ``(N, 6, T)``（或已分批的迭代器），返回 ``(N, D)``。"""
        self.reset()
        self.actions = []
        if isinstance(windows, torch.Tensor):
            size = int(batch or self.batch)
            chunks = [windows[i:i + size] for i in range(0, windows.shape[0], size)]
        else:
            chunks = list(windows)
        return torch.cat([self.step(chunk) for chunk in chunks], dim=0)
