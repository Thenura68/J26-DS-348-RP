"""Edge-deployment profiling helpers (parameter ratio, file size, latency).

Run latency measurements on CPU (optionally with torch.set_num_threads(1..4)) — numbers from
a Colab GPU say nothing about a Raspberry Pi or ARM Cortex device.
"""

from __future__ import annotations

import os
import statistics
import time
from collections.abc import Callable

from torch import nn


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def trainable_ratio(adapter: nn.Module, backbone: nn.Module) -> float:
    """Adapter parameters / (adapter + backbone parameters). Target for C3: < 0.05."""
    a = count_parameters(adapter)
    total = a + count_parameters(backbone)
    return a / total if total else 0.0


def file_size_kb(path: str | os.PathLike) -> float:
    return os.path.getsize(path) / 1024.0


def time_call(fn: Callable[[], object], n_runs: int = 50, warmup: int = 5) -> dict:
    """Wall-clock latency of ``fn()`` in milliseconds."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return {
        "mean_ms": statistics.fmean(times),
        "median_ms": statistics.median(times),
        "p95_ms": times[int(0.95 * (len(times) - 1))],
        "n_runs": n_runs,
    }
