"""推理与轨迹重建（DESIGN 第 5 节）。

1. 以 ``eval_stride`` 滑窗推理，窗口速度的时间戳为目标对应时刻（平均速度/位移为窗口中心，
   ``velocity_at_end`` 为窗口末端）；
2. 含无效样本的窗口照常推理，但其速度用相邻有效窗口按时间线性插值替换；
3. 逐帧速度为窗口速度的分段线性插值，首尾半窗内常值外推；
4. 梯形积分，起点锚定首个有效参考位置，不做任何对齐；
5. 同时积分窗口目标得到 oracle 轨迹。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional, Union

import numpy as np
import torch

from ..cfg import get_cfg, is_checkpoint
from ..data.format import Sequence, load_sequence
from ..data.views import SequenceView, ViewConfig
from ..nn.base import InputSpec
from ..nn.registry import build_model
from ..utils import LOGGER
from ..utils.callbacks import default_callbacks, run_callbacks
from ..utils.torch_utils import load_checkpoint, select_device
from .results import SequenceResult, Trajectory

PathLike = Union[str, Path]


def load_model(cfg: Any, device: Optional[torch.device] = None):
    """按配置构建模型。

    ``cfg.model`` 为 checkpoint 时加载其结构与权重；``cfg.pretrained`` 另行加载（非严格）权重。
    """
    if is_checkpoint(cfg.model):
        ckpt = load_checkpoint(cfg.model)
        model = build_model(cfg, InputSpec.from_dict(ckpt["input_spec"]), ckpt["model_cfg"])
        model.load_state_dict(ckpt["model"])
    else:
        model = build_model(cfg)
    if getattr(cfg, "pretrained", None):
        ckpt = load_checkpoint(cfg.pretrained)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        LOGGER.info(f"loaded pretrained weights from {cfg.pretrained} "
                    f"(missing {len(missing)}, unexpected {len(unexpected)})")
    return model.to(device) if device is not None else model


def spec_to_view(spec: Any) -> ViewConfig:
    """模型输入规格 → 任务视图配置（只取视图字段）。"""
    return ViewConfig(**{k: getattr(spec, k) for k in ViewConfig.__dataclass_fields__})


def integrate(t: np.ndarray, vel: np.ndarray) -> np.ndarray:
    """逐帧速度的梯形积分，``pos[0] = 0``。"""
    dt = np.diff(t)[:, None]
    steps = 0.5 * (vel[1:] + vel[:-1]) * dt
    return np.concatenate([np.zeros((1, vel.shape[1])), np.cumsum(steps, axis=0)], axis=0)


def fill_invalid_windows(t_window: np.ndarray, values: np.ndarray,
                         valid: np.ndarray) -> np.ndarray:
    """无效窗口的值用有效窗口按时间线性插值（两端常值外推）替换。"""
    if valid.all():
        return values
    out = values.copy()
    for c in range(values.shape[1]):
        out[~valid, c] = np.interp(t_window[~valid], t_window[valid], values[valid, c])
    return out


def reconstruct(t: np.ndarray, t_window: np.ndarray, vel_window: np.ndarray,
                anchor_index: int, anchor: np.ndarray) -> np.ndarray:
    """窗口速度 → 逐帧速度（分段线性 + 首尾常值外推）→ 梯形积分 → 锚定。"""
    vel = np.stack([np.interp(t, t_window, vel_window[:, c])
                    for c in range(vel_window.shape[1])], axis=1)
    pos = integrate(t, vel)
    return pos + (anchor[: pos.shape[1]] - pos[anchor_index])


class Predictor:
    """把模型应用到整条序列，输出 :class:`SequenceResult`。"""

    def __init__(self, cfg: Any = None, model: Optional[torch.nn.Module] = None,
                 device: Optional[torch.device] = None, callbacks: Optional[dict] = None,
                 overrides: Optional[dict] = None) -> None:
        self.args = cfg if cfg is not None else get_cfg(overrides)
        self.device = device or select_device(self.args.device, verbose=False)
        self.model = model if model is not None else load_model(self.args, self.device)
        self.model.to(self.device)
        spec = getattr(self.model, "input_spec", None)
        self.view_cfg = spec_to_view(spec) if spec is not None else ViewConfig.from_cfg(self.args)
        self.callbacks = callbacks or default_callbacks()
        self.amp = bool(self.args.amp) and self.device.type == "cuda"

    @torch.inference_mode()
    def infer(self, view: SequenceView, collect_loss: bool = False, epoch: int = 0) -> dict:
        """返回 ``starts``、``window_valid``、视图坐标系下的 ``vel``/``logstd`` 与损失累计。"""
        model = self.model
        was_training = model.training
        model.eval()
        starts = view.starts(int(self.args.eval_stride), require_valid=False)
        valid = view.window_valid(starts)
        outs: dict = {}
        loss_sum, loss_n = 0.0, 0
        batch = int(self.args.val_batch)
        for i in range(0, len(starts), batch):
            s = starts[i:i + batch]
            x = torch.from_numpy(view.imu_windows(s)).to(self.device, non_blocking=True)
            with torch.autocast(self.device.type, enabled=self.amp):
                out = model(x)
            out = {k: v.float() for k, v in out.items() if torch.is_tensor(v)}
            for k in ("vel", "logstd"):
                if k in out:
                    outs.setdefault(k, []).append(out[k].cpu().numpy())
            v = valid[i:i + batch]
            if collect_loss and v.any():
                idx = torch.from_numpy(np.flatnonzero(v)).to(self.device)
                target = torch.from_numpy(view.targets(s[v])).to(self.device)
                sub = {k: t.index_select(0, idx) for k, t in out.items() if t.shape[0] == len(s)}
                # 与 Trainer 相同的 batch 键（含模型输入），否则用到 batch["imu"] 的损失
                # 只能在训练中工作，验证时会 KeyError
                loss, _ = model.loss(sub, {"target": target, "imu": x.index_select(0, idx)},
                                     epoch)
                loss_sum += float(loss) * int(v.sum())
                loss_n += int(v.sum())
        model.train(was_training)
        result = {"starts": starts, "window_valid": valid, "loss_sum": loss_sum, "loss_n": loss_n}
        dims = self.view_cfg.dims
        for k in ("vel", "logstd"):
            result[k] = np.concatenate(outs[k]) if k in outs else (
                np.zeros((0, dims)) if k == "vel" else None)
        return result

    def predict_view(self, view: SequenceView, collect_loss: bool = False,
                     epoch: int = 0) -> SequenceResult:
        seq = view.seq
        res = SequenceResult(sequence_id=seq.sequence_id, dataset=seq.dataset,
                             group_id=seq.group_id, frame=self.view_cfg.frame,
                             rate=self.view_cfg.rate)
        res.t = seq.timestamp
        res.pos_gt = seq.position
        res.valid_pose = seq.valid_pose
        if len(seq) < self.view_cfg.window:
            res.skipped = f"shorter than one window ({len(seq)} < {self.view_cfg.window})"
            return res
        out = self.infer(view, collect_loss, epoch)
        starts, wvalid = out["starts"], out["window_valid"]
        anchors = np.flatnonzero(seq.valid_pose)
        if not wvalid.any() or len(anchors) == 0:
            res.skipped = "no valid window or no valid reference pose"
            return res
        t_window = view.target_times(starts)
        vel = view.to_world_velocity(out["vel"], starts)
        tgt = view.to_world_velocity(view.targets(starts), starts)
        vel = fill_invalid_windows(t_window, vel, wvalid)
        tgt = fill_invalid_windows(t_window, tgt, wvalid)
        i0 = int(anchors[0])
        res.pos_pred = reconstruct(seq.timestamp, t_window, vel, i0, seq.position[i0])
        res.pos_oracle = reconstruct(seq.timestamp, t_window, tgt, i0, seq.position[i0])
        res.t_window, res.starts, res.window_valid = t_window, starts, wvalid
        res.vel_pred, res.vel_target, res.logstd = vel, tgt, out["logstd"]
        if collect_loss and out["loss_n"]:
            res.loss = out["loss_sum"] / out["loss_n"]
        return res

    def predict_sequence(self, seq: Union[Sequence, PathLike], collect_loss: bool = False,
                         epoch: int = 0) -> SequenceResult:
        if not isinstance(seq, Sequence):
            seq = load_sequence(seq)
        return self.predict_view(SequenceView(seq, self.view_cfg), collect_loss, epoch)

    def __call__(self, source: Union[PathLike, Sequence, Iterable]) -> Any:
        """``source`` 为单个 h5/``Sequence`` 时返回 :class:`Trajectory`，目录或列表时返回列表。"""
        run_callbacks(self.callbacks, "on_predict_start", self)
        if isinstance(source, Sequence):
            items, single = [source], True
        elif isinstance(source, (str, Path)):
            path = Path(source)
            single = not path.is_dir()
            items = [path] if single else (sorted(path.glob("*.h5"))
                                            or sorted((path / "sequences").glob("*.h5")))
        else:
            items, single = list(source), False
        self.results = [self.predict_sequence(s) for s in items]
        run_callbacks(self.callbacks, "on_predict_end", self)
        trajs = [Trajectory(r.t, r.pos_pred, sequence_id=r.sequence_id) for r in self.results]
        return trajs[0] if single else trajs
