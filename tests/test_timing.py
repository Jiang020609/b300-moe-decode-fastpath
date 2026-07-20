"""Small tests for benchmark statistics and configuration."""

from __future__ import annotations

import csv

import torch

from benchmark.benchmark import run
from benchmark.timing import benchmark_callable


def test_cpu_timer_returns_raw_samples_and_percentiles() -> None:
    result = benchmark_callable(
        lambda: sum(range(10)), device=torch.device("cpu"), warmup=1, repetitions=5
    )
    assert len(result.samples_ms) == 5
    assert all(sample >= 0 for sample in result.samples_ms)
    assert min(result.samples_ms) <= result.p50_ms <= result.p90_ms <= result.p99_ms <= max(
        result.samples_ms
    )


def test_benchmark_writes_raw_and_summary_csv(tmp_path) -> None:
    config = {
        "tokens": [1],
        "experts": [4],
        "top_k": [2],
        "hidden_size": 7,
        "intermediate_size": 11,
        "workloads": ["uniform"],
        "seed": 5,
        "warmup": 0,
        "repetitions": 2,
        "dtype": "float32",
    }
    raw_path = tmp_path / "raw.csv"
    summary_path = tmp_path / "summary.csv"
    completed = run(
        config,
        torch.device("cpu"),
        raw_path,
        backend="torch",
        stages=("dispatch", "combine"),
        summary_output=summary_path,
    )
    assert completed == 1
    with raw_path.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    with summary_path.open(newline="", encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))
    assert len(raw_rows) == 4
    assert {row["stage"] for row in raw_rows} == {"dispatch", "combine"}
    assert len(summary_rows) == 2
    assert all(row["samples"] == "2" for row in summary_rows)
