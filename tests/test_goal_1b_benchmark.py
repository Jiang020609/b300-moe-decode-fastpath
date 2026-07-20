"""CPU tests for the Goal 1B benchmark and CSV contract."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from benchmark import benchmark_goal_1b


QUICK_CONFIG = Path(__file__).resolve().parents[1] / "configs/quick.yaml"
B300_CONFIG = Path(__file__).resolve().parents[1] / "configs/b300.yaml"


def _tiny_kwargs() -> dict[str, object]:
    return {
        "tokens": [2],
        "experts": [4],
        "top_k": [1],
        "workloads": ["uniform"],
        "hidden_size": 8,
        "intermediate_size": 12,
        "warmup": 0,
        "repetitions": 2,
    }


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_load_config_normalizes_workload_aliases() -> None:
    config = benchmark_goal_1b.load_config(QUICK_CONFIG)
    assert config["workloads"] == ["uniform", "hotspot", "zipf"]


def test_tiny_cpu_torch_benchmark_writes_exact_stage_summary(tmp_path: Path) -> None:
    output = tmp_path / "goal_1b.csv"
    completed = benchmark_goal_1b.run(
        benchmark_goal_1b.load_config(QUICK_CONFIG),
        "cpu",
        output,
        backends=("torch",),
        **_tiny_kwargs(),
    )

    assert completed == 1
    rows = _read_rows(output)
    assert [row["stage"] for row in rows] == list(benchmark_goal_1b.SUMMARY_STAGES)
    assert len(rows) == 8
    assert {row["actual_backend"] for row in rows} == {"torch"}
    assert {row["requested_backend"] for row in rows} == {"torch"}
    assert {row["backend_set_requested"] for row in rows} == {"torch"}
    assert {row["architecture"] for row in rows} == {"cpu"}
    assert {row["quant_mode"] for row in rows} == {"none"}
    assert {row["used_fallback"] for row in rows} == {"false"}
    assert {row["gpu_name"] for row in rows} == {"CPU"}
    assert {row["gpu_capability"] for row in rows} == {"unavailable"}
    assert {row["measurement"] for row in rows} == {"steady_state"}
    assert {row["timing_method"] for row in rows} == {"perf_counter_ns"}
    assert {row["weights_prepared_outside_timing"] for row in rows} == {"true"}
    assert {row["cold_start_scope"] for row in rows} == {
        "first_forward_after_input_setup"
    }
    assert {row["wrapper_compiled_architecture"] for row in rows} == {
        "unavailable"
    }
    assert {row["samples"] for row in rows} == {"2"}
    assert all(float(row["cold_start_us"]) > 0 for row in rows)
    assert all(float(row["p50_us"]) >= 0 for row in rows)
    quantization = next(row for row in rows if row["stage"] == "quantization_us")
    assert quantization["stage_status"] == "not_applicable"
    assert float(quantization["p99_us"]) == 0.0


def test_compiled_backend_fails_unless_skip_is_explicit(tmp_path: Path) -> None:
    output = tmp_path / "unavailable.csv"
    with pytest.raises(benchmark_goal_1b.BackendUnavailableError, match="CUDA device"):
        benchmark_goal_1b.run(
            benchmark_goal_1b.load_config(QUICK_CONFIG),
            "cpu",
            output,
            backends=("cutlass_bf16",),
            dtype="bfloat16",
            **_tiny_kwargs(),
        )
    assert not output.exists()


def test_target_config_refuses_accidental_cpu_allocation(tmp_path: Path) -> None:
    config = benchmark_goal_1b.load_config(B300_CONFIG)
    assert config["requires_cuda"] is True
    with pytest.raises(
        benchmark_goal_1b.BackendUnavailableError, match="requires --device cuda"
    ):
        benchmark_goal_1b.run(config, "cpu", tmp_path / "never-created.csv")


def test_explicit_skip_keeps_truthful_torch_rows(tmp_path: Path, capsys) -> None:
    output = tmp_path / "skip.csv"
    completed = benchmark_goal_1b.run(
        benchmark_goal_1b.load_config(QUICK_CONFIG),
        "cpu",
        output,
        backends=("torch", "cutlass_bf16"),
        skip_unavailable=True,
        **_tiny_kwargs(),
    )

    assert completed == 1
    assert "SKIP cutlass_bf16" in capsys.readouterr().out
    rows = _read_rows(output)
    assert len(rows) == 8
    assert {row["actual_backend"] for row in rows} == {"torch"}


def test_cli_shape_and_repeat_overrides_keep_quick_run_tiny(tmp_path: Path) -> None:
    output = tmp_path / "cli.csv"
    status = benchmark_goal_1b.main(
        [
            "--config",
            str(QUICK_CONFIG),
            "--device",
            "cpu",
            "--backend",
            "torch",
            "--tokens",
            "1",
            "--experts",
            "4",
            "--top-k",
            "1",
            "--workloads",
            "hotspot",
            "--hidden-size",
            "8",
            "--intermediate-size",
            "12",
            "--warmup",
            "0",
            "--repeats",
            "1",
            "--output",
            str(output),
        ]
    )

    assert status == 0
    rows = _read_rows(output)
    assert len(rows) == 8
    assert {row["workload"] for row in rows} == {"hotspot"}
    assert {row["samples"] for row in rows} == {"1"}
