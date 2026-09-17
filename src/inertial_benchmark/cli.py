"""命令行入口 ``ipb``：``ipb <command> key=value ...``（详见 ``docs/CLI.md``）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable, Optional

import yaml

from . import __version__
from .cfg import ConfigError, get_cfg, load_default, parse_key_value
from .utils import LOGGER, datasets_dir, json_save, set_logging, to_builtin

USAGE = f"""IPB {__version__} — inertial positioning benchmark

usage: ipb <command> [key=value ...]

commands:
  convert    dataset=ronin source=/raw/ronin [output=...] [workers=8] [only=[a,b]]
  check      data=ronin [full=true] [hash=true] [save=report.json]
  train      model=ronin_resnet18 data=ronin epochs=40 device=0 ...
  val        model=runs/train/exp/weights/best.pt data=ronin split=test
  predict    model=... source=seq.h5|dir [save=true]
  benchmark  cfg=benchmarks/main.yaml [dry_run=true] [report=false] [key=value overrides]
  report     runs=runs/benchmark/main [out=reports/main] [metrics=[ate,rte]] [plots=true]
  info       [model=...] [data=...]   (no arguments: list models, datasets, converters)
  cfg        [key=value ...]          (print the resolved configuration)
  version

Training/evaluation keys are documented in `ipb cfg`; see docs/CLI.md.
"""

CONVERT_KEYS = {"dataset", "source", "output", "only", "workers", "overwrite", "converter",
                "compression", "gap_threshold", "min_duration", "val_fraction", "seed"}


def _require(kv: dict, *keys: str) -> None:
    missing = [k for k in keys if kv.get(k) in (None, "")]
    if missing:
        raise ConfigError(f"missing required argument(s): {', '.join(missing)}")


def _check_keys(kv: dict, allowed: set, command: str) -> None:
    unknown = sorted(set(kv) - allowed)
    if unknown:
        raise ConfigError(f"'ipb {command}' does not accept {unknown}; allowed: {sorted(allowed)}")


def _as_list(value) -> Optional[list]:
    """把 CLI 值统一成字符串列表（标量也接受，避免 ``TypeError``）。"""
    if value is None:
        return None
    if isinstance(value, str):
        return [v for v in value.split(",") if v]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


def cmd_convert(kv: dict) -> int:
    from .data.convert import convert_dataset, default_workers

    _check_keys(kv, CONVERT_KEYS, "convert")
    _require(kv, "dataset", "source")
    name = kv.pop("dataset")
    workers = kv.pop("workers", None)
    manifest = convert_dataset(
        name, kv.pop("source"), kv.pop("output", None) or datasets_dir() / name,
        only=_as_list(kv.pop("only", None)),
        workers=default_workers() if workers is None else int(workers), **kv)
    return 0 if manifest["sequences"] else 1


def cmd_check(kv: dict) -> int:
    from .data.manifest import check_dataset

    _check_keys(kv, {"data", "full", "hash", "save"}, "check")
    _require(kv, "data")
    report = check_dataset(kv["data"], full=bool(kv.get("full", False)),
                           verify_hash=bool(kv.get("hash", False)))
    for e in report["errors"]:
        LOGGER.warning(f"  error: {e}")
    for w in report["warnings"][:50]:
        LOGGER.info(f"  warning: {w}")
    stats = report["statistics"]
    if stats:
        LOGGER.info(f"  {stats.get('num_sequences')} sequences, "
                    f"{stats.get('total_duration_h', 0):.2f} h, "
                    f"{stats.get('total_distance_m', 0) / 1000:.2f} km, splits {report['splits']}")
    if kv.get("save"):
        json_save(kv["save"], report)
    return 0 if report["ok"] else 1


def _nio(kv: dict):
    from .engine.model import NIO

    model = kv.pop("model", load_default()["model"])
    return NIO(model)


def cmd_train(kv: dict) -> int:
    get_cfg(kv)  # 先校验，避免构建模型后才报错
    nio = _nio(kv)
    nio.train(**kv)
    return 0


def cmd_val(kv: dict) -> int:
    get_cfg(kv)
    nio = _nio(kv)
    nio.val(**kv)
    return 0


def cmd_predict(kv: dict) -> int:
    save = bool(kv.pop("save", True))
    _require(kv, "source")
    source = kv.pop("source")
    get_cfg(kv)
    nio = _nio(kv)
    trajs = nio.predict(source, save=save, **kv)
    trajs = trajs if isinstance(trajs, list) else [trajs]
    for t in trajs:
        LOGGER.info(f"{t.sequence_id}: {len(t)} samples, path length {t.length():.1f} m")
    return 0


def cmd_benchmark(kv: dict) -> int:
    from .engine.benchmark import run_benchmark

    _require(kv, "cfg")
    cfg = kv.pop("cfg")
    dry_run = bool(kv.pop("dry_run", False))
    report = kv.pop("report", None)
    statuses = run_benchmark(cfg, overrides=kv, dry_run=dry_run, report=report)
    for s in statuses:
        LOGGER.info(json.dumps(to_builtin(s), ensure_ascii=False))
    return 0


def cmd_report(kv: dict) -> int:
    from .metrics import MAIN_METRICS
    from .utils.reports import build_report

    _check_keys(kv, {"runs", "out", "metrics", "plots", "n_boot"}, "report")
    _require(kv, "runs")
    out = kv.get("out") or Path(kv["runs"]) / "report"
    build_report(kv["runs"], out, metrics=_as_list(kv.get("metrics")) or MAIN_METRICS,
                 plots=bool(kv.get("plots", True)), n_boot=int(kv.get("n_boot", 2000)))
    return 0


def cmd_info(kv: dict) -> int:
    _check_keys(kv, {"model", "data", "flops"}, "info")
    if not kv:
        from .cfg import model_yaml_names
        from .data.converters import PLANNED, available_converters
        from .data.manifest import dataset_yaml_names

        print(yaml.safe_dump({
            "version": __version__,
            "models": model_yaml_names(),
            "datasets": dataset_yaml_names(),
            "converters": available_converters(),
            "converters_planned": list(PLANNED),
            "datasets_dir": str(datasets_dir()),
        }, sort_keys=False, allow_unicode=True))
        return 0
    if "model" in kv:
        from .engine.model import NIO

        info = NIO(kv["model"]).info(verbose=False, flops=bool(kv.get("flops", True)))
        print(yaml.safe_dump(to_builtin(info), sort_keys=False, allow_unicode=True))
    if "data" in kv:
        from .data.manifest import resolve_dataset

        spec = resolve_dataset(kv["data"])
        manifest = spec.manifest()
        print(yaml.safe_dump(to_builtin({
            **spec.to_dict(),
            "converted": bool(manifest),
            "converter": manifest.get("converter"),
            "fingerprint": manifest.get("fingerprint"),
            "splits_available": manifest.get("splits"),
            "statistics": {k: v for k, v in manifest.get("statistics", {}).items()
                           if k != "by_split"},
        }), sort_keys=False, allow_unicode=True))
    return 0


def cmd_cfg(kv: dict) -> int:
    cfg = get_cfg(kv)
    print(yaml.safe_dump(to_builtin(cfg.to_dict()), sort_keys=False, allow_unicode=True))
    return 0


def cmd_version(kv: dict) -> int:
    print(__version__)
    return 0


COMMANDS: dict = {
    "convert": cmd_convert,
    "check": cmd_check,
    "train": cmd_train,
    "val": cmd_val,
    "predict": cmd_predict,
    "benchmark": cmd_benchmark,
    "report": cmd_report,
    "info": cmd_info,
    "cfg": cmd_cfg,
    "version": cmd_version,
}


def entrypoint(argv: Optional[list] = None) -> int:
    """``ipb`` 主函数；返回进程退出码（0 成功，1 检查未通过，2 参数/配置错误）。"""
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(USAGE)
        return 0
    if argv[0] in ("-V", "--version"):
        argv[0] = "version"
    command: Optional[Callable] = COMMANDS.get(argv[0])
    if command is None:
        print(f"unknown command {argv[0]!r}\n\n{USAGE}", file=sys.stderr)
        return 2
    try:
        kv = parse_key_value(argv[1:])
        if kv.get("verbose") is False:
            set_logging(verbose=False)
        return int(command(kv) or 0)
    except (ConfigError, FileNotFoundError, KeyError, ImportError) as exc:
        LOGGER.error(f"ipb {argv[0]}: {type(exc).__name__}: {exc}")
        return 2


def main() -> None:  # pragma: no cover - 由 console_scripts 调用
    sys.exit(entrypoint())


if __name__ == "__main__":  # pragma: no cover
    main()
