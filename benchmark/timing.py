"""Synchronized CPU and CUDA benchmark timing helpers."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class TimingSummary:
    """Raw latency samples and selected percentiles, in milliseconds."""

    samples_ms: list[float]
    p50_ms: float
    p90_ms: float
    p99_ms: float


def _percentile(samples: list[float], quantile: float) -> float:
    ordered = sorted(samples)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def benchmark_callable(
    function: Callable[[], Any],
    *,
    device: torch.device,
    warmup: int,
    repetitions: int,
) -> TimingSummary:
    """Benchmark a callable with synchronized wall time or CUDA Events.

    CUDA is synchronized before measurement and every stop event is explicitly
    synchronized before its elapsed time is read.
    """

    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if repetitions < 1:
        raise ValueError("repetitions must be positive")

    samples: list[float] = []
    if device.type == "cuda":
        # The device context matters for indexed devices such as cuda:1: Events
        # and the operators under test must be recorded on the same device.
        with torch.cuda.device(device):
            for _ in range(warmup):
                function()
            torch.cuda.synchronize(device)
            for _ in range(repetitions):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                function()
                end.record()
                end.synchronize()
                samples.append(float(start.elapsed_time(end)))
    else:
        for _ in range(warmup):
            function()
        for _ in range(repetitions):
            start_ns = time.perf_counter_ns()
            function()
            elapsed_ns = time.perf_counter_ns() - start_ns
            samples.append(elapsed_ns / 1_000_000.0)

    return TimingSummary(
        samples_ms=samples,
        p50_ms=_percentile(samples, 0.50),
        p90_ms=_percentile(samples, 0.90),
        p99_ms=_percentile(samples, 0.99),
    )
