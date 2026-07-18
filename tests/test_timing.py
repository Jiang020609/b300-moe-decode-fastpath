"""Small tests for benchmark statistics and configuration."""

from __future__ import annotations

import torch

from benchmark.timing import benchmark_callable


def test_cpu_timer_returns_raw_samples_and_percentiles() -> None:
    result = benchmark_callable(lambda: sum(range(10)), device=torch.device("cpu"), warmup=1, repetitions=5)
    assert len(result.samples_ms) == 5
    assert all(sample >= 0 for sample in result.samples_ms)
    assert min(result.samples_ms) <= result.p50_ms <= result.p90_ms <= result.p99_ms <= max(
        result.samples_ms
    )
