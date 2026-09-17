"""配置系统：``default.yaml`` + 模型 YAML（input、recipes）+ 用户覆盖，并做键名与类型校验。"""

from __future__ import annotations

import difflib
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import yaml

from ..utils import CFG_DIR, DEFAULT_CFG_PATH, IterableSimpleNamespace, yaml_load

PathLike = Union[str, Path]

# 定义模型输入的键：使用 checkpoint 时必须与训练时一致
INPUT_KEYS = ("window", "frame", "orientation", "remove_gravity", "target", "dims", "rate")
CHOICES = {
    "mode": ("train", "val", "predict", "benchmark"),
    "optimizer": ("sgd", "adam", "adamw"),
    "scheduler": ("none", "cosine", "step", "plateau"),
    "frame": ("gravity_world", "body", "gravity_yaw_local"),
    "orientation": ("reference", "device"),
    "target": ("avg_velocity", "displacement", "velocity_at_end"),
    "dims": (2, 3),
    "metric_dims": (2, 3),
    "recipe": ("official", "unified"),
    "loss": (None, "mse", "gaussian_nll", "mse_then_nll"),
}
# 允许多种类型的键
FLEX_TYPES = {"resume": (bool, str), "device": (str, int, type(None)), "model": (str, dict)}
# 缺省为空、但取值应为字符串的键（CLI 可能把 name=1 解析成整数）
STR_KEYS = ("name", "data", "split", "source", "pretrained", "project", "model")


class ConfigError(ValueError):
    """配置键或取值非法。"""


@lru_cache(maxsize=None)
def _default_dict(path: str) -> dict:
    return yaml_load(path)


def load_default(path: PathLike = DEFAULT_CFG_PATH) -> dict:
    return dict(_default_dict(str(path)))


def model_yaml_names() -> list:
    return sorted(p.stem for p in (CFG_DIR / "models").glob("*.yaml"))


def is_checkpoint(model: Any) -> bool:
    return isinstance(model, (str, Path)) and str(model).endswith((".pt", ".pth"))


def load_model_cfg(model: Union[str, Path, Mapping, None]) -> dict:
    """解析模型描述：YAML 名 / YAML 路径 / checkpoint / dict，返回模型配置字典。

    模型配置字段：``name``、``arch``（注册名，缺省同 name）、``args``、``input``、``loss``、
    ``recipes``，以及论文/仓库/许可等元信息。
    """
    if model is None:
        raise ConfigError("model is required (e.g. model=ronin_resnet18)")
    if isinstance(model, Mapping):
        cfg = dict(model)
    elif is_checkpoint(model):
        from ..utils.torch_utils import load_checkpoint

        cfg = dict(load_checkpoint(model)["model_cfg"])
    else:
        text = str(model)
        path = Path(text).expanduser()
        if text.endswith((".yaml", ".yml")):
            if not path.exists() and (CFG_DIR / "models" / path.name).exists():
                path = CFG_DIR / "models" / path.name
            cfg = yaml_load(path)
        elif (CFG_DIR / "models" / f"{text}.yaml").exists():
            cfg = yaml_load(CFG_DIR / "models" / f"{text}.yaml")
        else:
            from ..nn.registry import MODELS, import_models

            import_models()
            if text not in MODELS:
                raise ConfigError(f"unknown model {text!r}; YAML configs: {model_yaml_names()}, "
                                  f"registered: {sorted(MODELS)}")
            cfg = {"name": text}
    cfg.setdefault("name", Path(str(model)).stem if not isinstance(model, Mapping) else "model")
    cfg.setdefault("arch", cfg["name"])
    cfg.setdefault("args", {})
    cfg.setdefault("input", {})
    cfg.setdefault("recipes", {})
    return cfg


def _suggest(key: str, keys: Any) -> str:
    close = difflib.get_close_matches(key, list(keys), n=3, cutoff=0.6)
    return f" (did you mean {', '.join(close)}?)" if close else ""


def check_keys(overrides: Mapping[str, Any], defaults: Mapping[str, Any]) -> None:
    unknown = [k for k in overrides if k not in defaults]
    if unknown:
        msg = "; ".join(f"'{k}'{_suggest(k, defaults)}" for k in unknown)
        raise ConfigError(f"unknown config key(s): {msg}. See `ipb cfg` for all keys.")


def _coerce(key: str, value: Any, default: Any) -> Any:
    if key in STR_KEYS and isinstance(value, (int, float, Path)) and not isinstance(value, bool):
        return str(value)
    if value is None or (default is None and key not in FLEX_TYPES):
        return value
    if key in FLEX_TYPES:
        if not isinstance(value, FLEX_TYPES[key]):
            raise ConfigError(f"{key}={value!r} must be one of {FLEX_TYPES[key]}")
        return value
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
    elif isinstance(default, int):
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
    elif isinstance(default, float):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    elif isinstance(default, str):
        if isinstance(value, str):
            return value
    elif isinstance(default, list):
        if isinstance(value, (list, tuple)):
            return list(value)
    elif isinstance(default, dict):
        if isinstance(value, Mapping):
            return dict(value)
    raise ConfigError(f"{key}={value!r} has type {type(value).__name__}, "
                      f"expected {type(default).__name__}")


def check_cfg(cfg: dict, defaults: Mapping[str, Any]) -> dict:
    """类型检查（必要时做无损转换）与取值范围检查。"""
    out = {k: _coerce(k, v, defaults.get(k)) for k, v in cfg.items()}
    for key, allowed in CHOICES.items():
        if key in out and out[key] not in allowed:
            raise ConfigError(f"{key}={out[key]!r} not in {allowed}")
    for key in ("epochs", "batch", "window", "stride", "eval_stride", "val_interval", "val_batch"):
        if key in out and out[key] is not None and out[key] < 1:
            raise ConfigError(f"{key} must be >= 1, got {out[key]}")
    if out.get("frame") == "body" and out.get("dims") != 3:
        raise ConfigError("frame=body requires dims=3 (targets are expressed in the device frame)")
    return out


def get_cfg(overrides: Optional[Mapping[str, Any]] = None,
            cfg: PathLike = DEFAULT_CFG_PATH) -> IterableSimpleNamespace:
    """合并默认配置、模型配置与覆盖项，返回命名空间。

    ``overrides`` 中出现未知键会报错并给出相近键名；checkpoint 模型的输入规格不可被改写。
    """
    defaults = load_default(cfg)
    overrides = dict(overrides or {})
    check_keys(overrides, defaults)
    merged = dict(defaults)
    model = overrides.get("model", defaults.get("model"))
    if model is not None:
        mcfg = load_model_cfg(model)
        check_keys(mcfg["input"], defaults)
        merged.update(mcfg["input"])
        recipe = overrides.get("recipe", defaults["recipe"])
        recipes = mcfg["recipes"]
        if recipe not in recipes and recipe != "unified":
            raise ConfigError(f"model {mcfg['name']} has no recipe {recipe!r} "
                              f"(available: {sorted(recipes) or ['unified']})")
        recipe_cfg = dict(recipes.get(recipe) or {})
        check_keys(recipe_cfg, defaults)
        merged.update(recipe_cfg)
        if is_checkpoint(model):
            from ..utils.torch_utils import load_checkpoint

            spec = load_checkpoint(model)["input_spec"]
            for key in INPUT_KEYS:
                if key in overrides and key in spec and overrides[key] != spec[key]:
                    raise ConfigError(f"{key}={overrides[key]!r} conflicts with checkpoint "
                                      f"input spec {key}={spec[key]!r}")
            merged.update({k: spec[k] for k in INPUT_KEYS if k in spec})
    merged.update(overrides)
    return IterableSimpleNamespace(**check_cfg(merged, defaults))


_NUMBER = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


def parse_value(text: str) -> Any:
    """CLI 值解析：布尔、None、整数、浮点、YAML 列表/字典，其余为字符串。"""
    s = text.strip()
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null", ""):
        return None
    if _NUMBER.match(s):
        return int(s) if re.match(r"^[+-]?\d+$", s) else float(s)
    if s[:1] in "[{":
        return _normalize(yaml.safe_load(s))
    return s


def _normalize(obj: Any) -> Any:
    """修正 PyYAML 把 ``1e-3`` 之类当作字符串的问题。"""
    if isinstance(obj, list):
        return [_normalize(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _normalize(v) for k, v in obj.items()}
    if isinstance(obj, str) and _NUMBER.match(obj):
        return parse_value(obj)
    return obj


def parse_key_value(args: list) -> dict:
    """``["epochs=2", "augment=[random_yaw]"]`` → ``{"epochs": 2, "augment": ["random_yaw"]}``。"""
    out: dict = {}
    for arg in args:
        if "=" not in arg:
            raise ConfigError(f"argument {arg!r} is not in key=value form")
        key, value = arg.split("=", 1)
        key = key.strip().lstrip("-").replace("-", "_")
        if not key:
            raise ConfigError(f"empty key in {arg!r}")
        out[key] = parse_value(value)
    return out


def cfg_to_dict(cfg: Any) -> dict:
    return cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)


__all__ = [
    "CHOICES",
    "INPUT_KEYS",
    "ConfigError",
    "cfg_to_dict",
    "check_cfg",
    "get_cfg",
    "load_default",
    "load_model_cfg",
    "parse_key_value",
    "parse_value",
]
