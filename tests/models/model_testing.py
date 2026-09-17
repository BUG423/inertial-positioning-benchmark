"""算法移植测试的公共工具：夹具读取、参数形状、确定性/反传检查与端到端冒烟。"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import torch

from inertial_benchmark import NIO

TESTS = Path(__file__).resolve().parents[1]
FIXTURES = TESTS / "fixtures" / "algorithms"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def param_shapes(model: torch.nn.Module) -> list:
    """按 ``named_parameters()`` 注册顺序的参数形状列表。"""
    return [list(p.shape) for _, p in model.named_parameters()]


def shape_multiset(shapes) -> Counter:
    return Counter(tuple(s) for s in shapes)


def assert_deterministic_eval(model: torch.nn.Module, x: torch.Tensor, key: str = "vel") -> None:
    model.eval()
    with torch.no_grad():
        a, b = model(x)[key], model(x)[key]
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def assert_backprop(model: torch.nn.Module, x: torch.Tensor, target: torch.Tensor,
                    epoch: int = 100) -> None:
    """一次前向 + 模型自身损失 + 反传：损失有限，且（几乎）所有参数都拿到有限梯度。"""
    model.train()
    model.zero_grad(set_to_none=True)
    out = model(x)
    loss, _ = model.loss(out, {"target": target, "imu": x}, epoch)
    assert loss.ndim == 0 and math.isfinite(loss.item())
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"parameters without gradient: {missing[:5]}"
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())


FAST = {"device": "cpu", "workers": 0, "batch": 16, "stride": 40, "eval_stride": 40,
        "efficiency": False, "plots": False, "save_predictions": False, "exist_ok": True}


def run_end_to_end(name: str, data: Path, project: Path, **overrides) -> dict:
    """``NIO(name).train(epochs=1)`` + ``val``（CPU、合成数据），返回 val 指标。"""
    kwargs = {**FAST, **overrides}
    model = NIO(name, **kwargs)
    model.train(data=str(data), epochs=1, project=str(project), name="train")
    result = model.val(data=str(data), split="val", project=str(project), name="val")
    metrics = result.metrics
    assert math.isfinite(metrics["ate"]) and math.isfinite(metrics["vel_rmse"])
    assert (project / "train" / "train" / "weights" / "best.pt").exists()
    assert (project / "val" / "val" / "metrics.json").exists()
    return metrics
