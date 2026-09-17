"""``NIO`` 门面：统一的模型入口（模型名 / 模型 YAML / checkpoint）。

    from inertial_benchmark import NIO
    model = NIO("ronin_resnet18")
    model.train(data="ronin", epochs=40, device=0)
    metrics = model.val(data="ronin", split="test")
    traj = model.predict("path/to/sequence.h5")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

from ..cfg import get_cfg, is_checkpoint
from ..utils import LOGGER
from ..utils.callbacks import add_callback, default_callbacks

PathLike = Union[str, Path]


class NIO:
    """神经惯性里程计模型门面：``train`` / ``val`` / ``predict`` / ``info`` / ``benchmark``。"""

    def __init__(self, model: Union[PathLike, dict] = "ronin_resnet18", **overrides: Any) -> None:
        from .predictor import load_model

        if isinstance(model, Path):
            model = str(model)
        self.overrides = {"model": model, **overrides}
        self.callbacks = default_callbacks()
        cfg = get_cfg(self.overrides)
        self.model = load_model(cfg)
        self.ckpt_path: Optional[str] = model if is_checkpoint(model) else None
        self.trainer = None
        self.metrics: dict = {}

    def _merge(self, mode: str, kwargs: dict) -> dict:
        """合并构造参数与调用参数；``None`` 原样传给 ``get_cfg`` 校验（不再静默丢弃）。"""
        return {**self.overrides, **kwargs, "mode": mode}

    def __repr__(self) -> str:
        name = self.model.model_cfg.get("name", type(self.model).__name__)
        return f"NIO({name}, params={self.model.num_params:,}, ckpt={self.ckpt_path})"

    def add_callback(self, event: str, fn) -> None:
        add_callback(self.callbacks, event, fn)

    # ------------------------------------------------------------------ 训练与评测
    def train(self, trainer: Optional[type] = None, **kwargs: Any) -> dict:
        """训练；若当前模型来自 checkpoint 且未指定 ``resume``，则以其权重作为 ``pretrained``。

        ``trainer`` 可传入 :class:`Trainer` 的子类以定制训练流程（缺省为 ``Trainer``）。
        """
        from .trainer import Trainer

        trainer = trainer or Trainer
        if not (isinstance(trainer, type) and issubclass(trainer, Trainer)):
            raise TypeError(f"trainer must be a Trainer subclass, got {trainer!r}")
        overrides = self._merge("train", kwargs)
        if self.ckpt_path and not overrides.get("resume") and "pretrained" not in kwargs:
            overrides["pretrained"] = self.ckpt_path
        self.trainer = trainer(overrides=overrides, callbacks=self.callbacks)
        self.metrics = self.trainer.train()
        self.model = self.trainer.model
        self.ckpt_path = str(self.trainer.best)
        self.overrides["model"] = self.ckpt_path
        return self.metrics

    def val(self, **kwargs: Any):
        """在 ``split``（默认 val）上评测，返回 :class:`RunResult` 并写出结果目录。"""
        from .validator import Validator

        overrides = self._merge("val", kwargs)
        validator = Validator(get_cfg(overrides), callbacks=self.callbacks)
        result = validator(model=self.model)
        self.metrics = result.metrics
        return result

    def predict(self, source: Any, **kwargs: Any):
        """对 h5 序列文件 / ``Sequence`` / 目录推理，返回 :class:`Trajectory`（或列表）。

        ``save=True`` 时把轨迹与结果写到 ``<project>/predict/<name>``。
        """
        from .predictor import Predictor
        from .validator import run_dir

        save = bool(kwargs.pop("save", False))
        cfg = get_cfg(self._merge("predict", kwargs))
        predictor = Predictor(cfg, model=self.model, callbacks=self.callbacks)
        out = predictor(source)
        if save:
            save_dir = run_dir(cfg, "predict")
            for res in predictor.results:
                res.compute_metrics(int(cfg.metric_dims), float(cfg.rte_delta), cfg.t_rte,
                                    cfg.d_rte, float(cfg.min_speed))
                res.save(save_dir / "predictions" / f"{res.sequence_id}.npz")
                if cfg.plots and not res.skipped:
                    res.plot(save_dir / "plots" / f"traj_{res.sequence_id}.png")
            LOGGER.info(f"predict: {len(predictor.results)} sequence(s) → {save_dir}")
            self.predict_dir = save_dir
        self.results = predictor.results
        return out

    # ------------------------------------------------------------------ 信息
    def info(self, verbose: bool = True, flops: bool = True) -> dict:
        from ..utils.torch_utils import model_info

        spec = self.model.input_spec
        info = {"name": self.model.model_cfg.get("name"),
                "arch": self.model.model_cfg.get("arch"),
                "class": type(self.model).__name__,
                "checkpoint": self.ckpt_path,
                "input_spec": spec.to_dict(),
                "loss": self.model.loss_name,
                "args": self.model.model_cfg.get("args", {}),
                **model_info(self.model, spec.window, spec.num_channels, flops,
                             input_shape=spec.input_shape, extra=spec.dummy_extra())}
        for key in ("paper", "code", "license", "commit"):
            if key in self.model.model_cfg:
                info[key] = self.model.model_cfg[key]
        if verbose:
            LOGGER.info("\n".join(f"{k}: {v}" for k, v in info.items()))
        return info

    def benchmark(self, device: Any = None, runs: int = 30) -> dict:
        """效率评测：参数量、FLOPs、单窗口 CPU（及 CUDA）延迟。"""
        from ..metrics.efficiency import efficiency_metrics
        from ..utils.torch_utils import select_device

        dev = select_device(device, verbose=False)
        devices = ["cpu"] + ([str(dev)] if dev.type == "cuda" else [])
        spec = self.model.input_spec
        return efficiency_metrics(self.model, spec.window, spec.num_channels, devices, runs,
                                  input_shape=spec.input_shape, extra=spec.dummy_extra())
