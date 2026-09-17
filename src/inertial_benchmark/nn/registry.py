"""模型注册与构建：``@register_model(name)`` 与 ``build_model(cfg)``。"""

from __future__ import annotations

import importlib
from typing import Any, Mapping, Optional

MODELS: dict = {}


def register_model(*names: str):
    """类装饰器：以一个或多个名字注册模型类（名字即模型 YAML 中的 ``arch``）。"""

    def decorator(cls):
        for name in names:
            if name in MODELS and MODELS[name] is not cls:
                raise KeyError(f"model name {name!r} already registered by {MODELS[name]}")
            MODELS[name] = cls
        return cls

    return decorator


def import_models() -> None:
    """导入 ``inertial_benchmark.models`` 以触发全部注册。"""
    importlib.import_module("inertial_benchmark.models")


def list_models() -> list:
    import_models()
    return sorted(MODELS)


def build_model(cfg: Any, input_spec: Optional[Any] = None,
                model_cfg: Optional[Mapping[str, Any]] = None):
    """根据配置构建模型。

    ``cfg`` 为 ``get_cfg`` 的结果（或含相同字段的 dict）；``model_cfg`` 缺省时由 ``cfg.model``
    解析。模型构造签名为 ``Model(input_spec, **args)``，``args = model_cfg.args ∪ cfg.model_args``。
    """
    from ..cfg import load_model_cfg
    from .base import InputSpec

    get = cfg.get if hasattr(cfg, "get") else (lambda k, d=None: getattr(cfg, k, d))
    model_cfg = dict(model_cfg or load_model_cfg(get("model")))
    import_models()
    arch = model_cfg.get("arch", model_cfg.get("name"))
    if arch not in MODELS:
        raise KeyError(f"model arch {arch!r} is not registered; available: {sorted(MODELS)}")
    spec = input_spec if input_spec is not None else InputSpec.from_cfg(cfg)
    args = {**(model_cfg.get("args") or {}), **(get("model_args") or {})}
    model = MODELS[arch](spec, **args)
    model_cfg["args"] = args
    # 损失：用户覆盖 > 模型配置（checkpoint 中保存的是训练时实际使用的损失）
    if get("loss"):
        from .losses import SWITCH_LOSSES

        loss = get("loss")
        kwargs = {"switch_epoch": int(get("loss_switch_epoch", 10))} \
            if loss in SWITCH_LOSSES else {}
    else:
        loss = model_cfg.get("loss")
        kwargs = dict(model_cfg.get("loss_kwargs") or {})
    if loss:
        model.set_loss(loss, **kwargs)
    model_cfg["loss"] = model.loss_name
    model_cfg["loss_kwargs"] = dict(model.loss_kwargs)
    model.model_cfg = model_cfg
    return model
