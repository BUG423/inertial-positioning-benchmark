"""验证/评测：窗口级损失 + Predictor 重建轨迹 + 全部指标。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable, Optional

import torch

from ..cfg import ConfigError, get_cfg, is_checkpoint
from ..data.format import Sequence, load_sequence
from ..data.manifest import DatasetSpec, resolve_dataset
from ..data.views import SequenceView
from ..utils import LOGGER, add_file_handler, json_save, yaml_save
from ..utils.callbacks import default_callbacks, run_callbacks
from ..utils.checks import collect_env
from ..utils.files import increment_path
from ..utils.torch_utils import load_checkpoint, select_device
from .predictor import Predictor
from .results import RunResult


def run_dir(args: Any, mode: str) -> Path:
    """``<project>/<mode>/<name>``，``name`` 缺省为 ``exp``，已存在时递增（除非 ``exist_ok``）。"""
    return increment_path(Path(args.project) / mode / (args.name or "exp"), bool(args.exist_ok))


class Validator:
    """在一个数据集划分上评测模型。

    训练中由 Trainer 调用（传入模型与已缓存的序列视图，不写文件）；独立使用时从配置加载
    checkpoint，逐条读取序列并把结果写入 ``save_dir``。
    """

    def __init__(self, cfg: Any = None, save_dir: Optional[Path] = None,
                 callbacks: Optional[dict] = None, overrides: Optional[dict] = None,
                 label: Optional[str] = None) -> None:
        self.args = cfg if cfg is not None else get_cfg(overrides)
        self.save_dir = Path(save_dir) if save_dir else None
        self.label = label  # 结果中的模型名（缺省取模型配置名）
        self.callbacks = callbacks or default_callbacks()
        self.result: Optional[RunResult] = None
        self.metrics: dict = {}

    def metric_kwargs(self) -> dict:
        a = self.args
        return {"dims": int(a.metric_dims), "rte_delta": float(a.rte_delta),
                "t_rte": list(a.t_rte), "d_rte": list(a.d_rte),
                "min_speed": float(a.min_speed)}

    def evaluate(self, predictor: Predictor, sources: Iterable, epoch: int = 0,
                 dataset: str = "unknown", split: str = "val") -> RunResult:
        """逐条推理并计算指标；``sources`` 为 ``SequenceView``、``Sequence`` 或文件路径。"""
        results = []
        for src in sources:
            if isinstance(src, SequenceView):
                res = predictor.predict_view(src, collect_loss=True, epoch=epoch)
            else:
                seq = src if isinstance(src, Sequence) else load_sequence(src)
                res = predictor.predict_sequence(seq, collect_loss=True, epoch=epoch)
            res.compute_metrics(**self.metric_kwargs())
            if res.skipped:
                LOGGER.warning(f"val: {res.sequence_id} skipped ({res.skipped})")
            results.append(res)
        model_cfg = getattr(predictor.model, "model_cfg", {}) or {}
        return RunResult(results, dataset=dataset, split=split,
                         model=str(model_cfg.get("name", self.args.model)),
                         cfg=self.args.to_dict() if hasattr(self.args, "to_dict") else {})

    def __call__(self, model: Optional[torch.nn.Module] = None, sources: Optional[list] = None,
                 device: Optional[torch.device] = None, epoch: int = 0,
                 dataset: Optional[DatasetSpec] = None, split: Optional[str] = None) -> RunResult:
        run_callbacks(self.callbacks, "on_val_start", self)
        standalone = sources is None
        if dataset is None and self.args.data is None:
            raise ConfigError("data is required for evaluation (e.g. data=ronin)")
        spec = dataset or resolve_dataset(self.args.data)
        split = split or self.args.split
        device = device or select_device(self.args.device, verbose=standalone)
        predictor = Predictor(self.args, model=model, device=device)
        handler = None
        if standalone:
            if self.save_dir is None:
                self.save_dir = run_dir(self.args, "val")
            self.save_dir.mkdir(parents=True, exist_ok=True)
            handler = add_file_handler(self.save_dir / "log.txt")
            sources = spec.sequence_paths(split)
            LOGGER.info(f"val: {self.args.model} on {spec.name}/{split} "
                        f"({len(sources)} sequences) → {self.save_dir}")
        try:
            t0 = time.time()
            result = self.evaluate(predictor, sources, epoch, spec.name, split)
            if self.label:
                result.model = self.label
            if standalone:
                result.weights = str(self.args.model)
                if is_checkpoint(self.args.model):
                    # 报告按训练种子聚合：记录 checkpoint 的训练种子与训练配置来源
                    train_cfg = load_checkpoint(self.args.model).get("cfg", {})
                    result.cfg["seed"] = train_cfg.get("seed", result.cfg.get("seed"))
                    result.cfg["recipe"] = train_cfg.get("recipe", result.cfg.get("recipe"))
                result.env = collect_env(device, spec)
                if self.args.efficiency:
                    result.efficiency = self.efficiency(predictor.model, device)
                yaml_save(self.save_dir / "args.yaml", self.args.to_dict())
                json_save(self.save_dir / "env.json", result.env)
                result.save(self.save_dir, predictions=bool(self.args.save_predictions),
                            plots=bool(self.args.plots), max_plots=int(self.args.max_plots))
                LOGGER.info(result.summary() + f" [{time.time() - t0:.1f}s]")
        finally:
            if handler is not None:
                LOGGER.removeHandler(handler)
                handler.close()
        self.result = result
        self.metrics = result.metrics
        run_callbacks(self.callbacks, "on_val_end", self)
        return result

    def efficiency(self, model: torch.nn.Module, device: torch.device) -> dict:
        from ..metrics.efficiency import efficiency_metrics

        spec = model.input_spec
        devices = ["cpu"] + ([str(device)] if device.type == "cuda" else [])
        try:
            return efficiency_metrics(model, spec.window, spec.num_channels, devices)
        except Exception as exc:  # noqa: BLE001 - 效率统计失败不影响精度结果
            LOGGER.warning(f"efficiency metrics failed: {exc}")
            return {}
