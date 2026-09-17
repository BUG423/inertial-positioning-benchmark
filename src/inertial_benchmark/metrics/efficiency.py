"""效率指标：参数量、FLOPs（乘加次数，batch=1 的单窗口）、单窗口推理延迟。"""

from __future__ import annotations

import copy
import statistics
import time
from typing import Optional, Sequence

import torch
from torch import nn

FLOP_BACKENDS = ("auto", "thop", "fvcore", "hooks")


def count_params(model: nn.Module, trainable_only: bool = False) -> int:
    return int(sum(p.numel() for p in model.parameters()
                   if p.requires_grad or not trainable_only))


def _rnn_flops(m: nn.RNNBase, inp: torch.Tensor) -> float:
    """整批输入的 RNN 乘加次数（门控线性变换 + 逐元素门控）。"""
    gates = {"LSTM": 4, "GRU": 3}.get(m.mode, 1)
    if inp.dim() == 2:
        steps, batch = inp.shape[0], 1
    elif m.batch_first:
        batch, steps = inp.shape[0], inp.shape[1]
    else:
        steps, batch = inp.shape[0], inp.shape[1]
    dirs = 2 if m.bidirectional else 1
    h, total, in_size = m.hidden_size, 0.0, m.input_size
    for _ in range(m.num_layers):
        per_step = gates * (in_size * h + h * h + (2 * h if m.bias else 0))
        per_step += h * (3 if m.mode == "LSTM" else 2)
        total += dirs * steps * per_step
        in_size = h * dirs
    return total * batch


def hook_flops(model: nn.Module, x: torch.Tensor) -> float:
    """用前向钩子粗算整批输入的乘加次数：Conv / Linear / 归一化 / RNN / 多头注意力。"""
    total = [0.0]

    def conv_hook(m, inp, out):
        kernel = m.in_channels // m.groups
        for k in m.kernel_size:
            kernel *= k
        total[0] += out.numel() * kernel + (out.numel() if m.bias is not None else 0)

    def linear_hook(m, inp, out):
        rows = out.numel() / m.out_features
        bias = m.out_features if m.bias is not None else 0
        total[0] += rows * (m.in_features * m.out_features + bias)

    def norm_hook(m, inp, out):
        total[0] += 2 * out.numel()

    def rnn_hook(m, inp, out):
        total[0] += _rnn_flops(m, inp[0])

    def mha_hook(m, inp, out):
        q = inp[0]
        if q.dim() == 2:
            batch, length = 1, q.shape[0]
        else:
            batch, length = (q.shape[0], q.shape[1]) if m.batch_first else (q.shape[1],
                                                                              q.shape[0])
        e = m.embed_dim
        total[0] += batch * (4 * length * e * e + length * length * e)

    hooks = ((nn.modules.conv._ConvNd, conv_hook), (nn.Linear, linear_hook),
             ((nn.modules.batchnorm._BatchNorm, nn.LayerNorm), norm_hook),
             (nn.RNNBase, rnn_hook), (nn.MultiheadAttention, mha_hook))
    handles = []
    for m in model.modules():
        for types, fn in hooks:
            if isinstance(m, types):
                handles.append(m.register_forward_hook(fn))
                break
    try:
        with torch.no_grad():
            model(x)
    finally:
        for h in handles:
            h.remove()
    return total[0]


def count_flops(model: nn.Module, input_shape: Sequence[int] = (1, 6, 200),
                backend: str = "auto") -> dict:
    """返回 ``{"flops": 乘加次数, "flops_backend": 实际使用的后端}``（batch=1）。

    ``auto`` 依次尝试 thop、fvcore，都不可用时使用本模块的钩子粗算；不同后端的计数约定
    略有差异，因此结果中总是记录后端名。
    """
    if backend not in FLOP_BACKENDS:
        raise ValueError(f"backend must be one of {FLOP_BACKENDS}")
    params = list(model.parameters())
    device = params[0].device if params else torch.device("cpu")
    dummy = torch.zeros(*input_shape, device=device)
    candidates = ["thop", "fvcore", "hooks"] if backend == "auto" else [backend]
    for name in candidates:
        work = copy.deepcopy(model).eval()
        try:
            if name == "thop":
                import thop  # type: ignore

                macs, _ = thop.profile(work, inputs=(dummy,), verbose=False)
                flops = float(macs)
            elif name == "fvcore":
                from fvcore.nn import FlopCountAnalysis  # type: ignore

                analysis = FlopCountAnalysis(work, dummy)
                analysis.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
                flops = float(analysis.total())
            else:
                flops = hook_flops(work, dummy)
        except ImportError:
            continue
        return {"flops": int(round(flops / input_shape[0])), "flops_backend": name}
    raise RuntimeError("no FLOP counting backend available")  # pragma: no cover


def measure_latency(model: nn.Module, input_shape: Sequence[int] = (1, 6, 200),
                    device: Optional[torch.device] = None, warmup: int = 10,
                    runs: int = 50) -> dict:
    """单窗口（batch=1）前向延迟的中位数（毫秒）。"""
    device = device or next(model.parameters()).device
    work = copy.deepcopy(model).to(device).eval()
    x = torch.randn(*input_shape, device=device)
    times = []
    with torch.inference_mode():
        for i in range(warmup + runs):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
            if start is not None:
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                work(x)
                end.record()
                torch.cuda.synchronize(device)
                elapsed = start.elapsed_time(end)
            else:
                t0 = time.perf_counter()
                work(x)
                elapsed = (time.perf_counter() - t0) * 1000.0
            if i >= warmup:
                times.append(elapsed)
    return {"latency_ms": float(statistics.median(times)), "device": str(device),
            "threads": torch.get_num_threads() if device.type == "cpu" else None}


def efficiency_metrics(model: nn.Module, window: int, channels: int = 6,
                       devices: Sequence[str] = ("cpu",), runs: int = 30) -> dict:
    """汇总效率指标：``params``、``flops``、``latency_ms_<device>``。"""
    out = {"params": count_params(model)}
    out.update(count_flops(model, (1, channels, window)))
    for dev in devices:
        d = torch.device(dev)
        if d.type == "cuda" and not torch.cuda.is_available():
            continue
        lat = measure_latency(model, (1, channels, window), d, runs=runs)
        out[f"latency_ms_{d.type}"] = lat["latency_ms"]
    return out
