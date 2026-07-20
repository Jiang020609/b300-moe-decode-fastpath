"""Benchmark torch and CUDA-extension Local-MoE dispatch/combine paths."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Literal

import torch
import yaml

# Make direct invocation (``python benchmark/benchmark.py``) work from any cwd.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from baseline.routing import generate_router_logits, topk_routing
from baseline.torch_moe import grouped_expert_ffn, make_expert_weights
from benchmark.timing import TimingSummary, benchmark_callable
from fastpath import (
    combine_tokens,
    cuda_extension_available,
    cuda_extension_error,
    dispatch_tokens,
)

Backend = Literal["torch", "cuda_ext"]
ALL_STAGES = ("dispatch", "expert_ffn", "combine", "total")
RAW_CSV_FIELDS = (
    "backend",
    "device",
    "dtype",
    "workload",
    "num_tokens",
    "num_experts",
    "top_k",
    "hidden_size",
    "intermediate_size",
    "stage",
    "iteration",
    "latency_ms",
)
SUMMARY_CSV_FIELDS = (
    "backend",
    "device",
    "dtype",
    "workload",
    "num_tokens",
    "num_experts",
    "top_k",
    "hidden_size",
    "intermediate_size",
    "stage",
    "samples",
    "p50_ms",
    "p90_ms",
    "p99_ms",
)
DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _positive_int_list(config: dict[str, Any], key: str) -> list[int]:
    value = config.get(key)
    if not isinstance(value, list) or not value or any(
        not isinstance(item, int) or item < 1 for item in value
    ):
        raise ValueError(f"config field {key!r} must be a non-empty list of positive integers")
    return value


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate a benchmark YAML file."""

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("config root must be a mapping")
    for key in ("tokens", "experts", "top_k"):
        _positive_int_list(config, key)
    for key in ("hidden_size", "intermediate_size", "seed", "warmup", "repetitions"):
        if not isinstance(config.get(key), int):
            raise ValueError(f"config field {key!r} must be an integer")
    if config["hidden_size"] < 1 or config["intermediate_size"] < 1:
        raise ValueError("hidden_size and intermediate_size must be positive")
    if config["warmup"] < 0 or config["repetitions"] < 1:
        raise ValueError("warmup must be non-negative and repetitions must be positive")
    if any(top_k not in (1, 2, 4, 8) for top_k in config["top_k"]):
        raise ValueError("all configured top_k values must be 1, 2, 4, or 8")
    if any(experts < 4 or experts > 256 for experts in config["experts"]):
        raise ValueError("all configured expert counts must be in [4, 256]")
    workloads = config.get("workloads")
    valid_workloads = {"uniform", "skewed", "hot_expert", "zipf", "hotspot"}
    if not isinstance(workloads, list) or not workloads or not set(workloads) <= valid_workloads:
        raise ValueError(f"workloads must be a non-empty subset of {sorted(valid_workloads)}")
    dtype_name = config.get("dtype", "float32")
    if dtype_name not in DTYPES:
        raise ValueError(f"dtype must be one of {sorted(DTYPES)}")
    config["dtype"] = dtype_name
    requires_cuda = config.get("requires_cuda", False)
    if not isinstance(requires_cuda, bool):
        raise ValueError("requires_cuda must be a boolean when present")
    config["requires_cuda"] = requires_cuda
    return config


def _resolve_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("device must be 'cpu', 'cuda', or an indexed CUDA device")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    return device


def _parse_stages(value: str) -> tuple[str, ...]:
    stages = tuple(part.strip() for part in value.split(",") if part.strip())
    if not stages:
        raise argparse.ArgumentTypeError("--stages must select at least one stage")
    unknown = set(stages) - set(ALL_STAGES)
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown stages {sorted(unknown)}; choose from {list(ALL_STAGES)}"
        )
    if len(set(stages)) != len(stages):
        raise argparse.ArgumentTypeError("--stages must not contain duplicates")
    return stages


def _benchmark_shape(
    *,
    num_tokens: int,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    workload: str,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
    backend: Backend,
    stages: tuple[str, ...],
    warmup: int,
    repetitions: int,
) -> dict[str, TimingSummary]:
    # Input generation and extension import happen before every timed callable.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    hidden_states = torch.randn(
        num_tokens, hidden_size, generator=generator, dtype=torch.float32
    ).to(device=device, dtype=dtype)
    router_logits = generate_router_logits(
        num_tokens,
        num_experts,
        top_k,
        workload,  # type: ignore[arg-type]
        seed + 1,
        device=device,
        dtype=dtype,
    )
    weights = make_expert_weights(
        num_experts,
        hidden_size,
        intermediate_size,
        seed=seed + 2,
        device=device,
        dtype=dtype,
    )
    expert_ids, routing_weights = topk_routing(router_logits, top_k)
    expert_ids = expert_ids.contiguous()
    routing_weights = routing_weights.contiguous()
    dispatch_result = dispatch_tokens(
        hidden_states, expert_ids, num_experts, backend=backend
    )
    expert_outputs = grouped_expert_ffn(
        dispatch_result.permuted_hidden, weights, dispatch_result.expert_offsets
    )

    def total_path() -> torch.Tensor:
        current_dispatch = dispatch_tokens(
            hidden_states,
            expert_ids,
            num_experts,
            backend=backend,
            validate_values=False,
        )
        current_expert_outputs = grouped_expert_ffn(
            current_dispatch.permuted_hidden,
            weights,
            current_dispatch.expert_offsets,
        )
        return combine_tokens(
            current_expert_outputs,
            current_dispatch.assignment_to_permuted,
            routing_weights,
            backend=backend,
            validate_values=False,
        )

    callables = {
        "dispatch": lambda: dispatch_tokens(
            hidden_states,
            expert_ids,
            num_experts,
            backend=backend,
            validate_values=False,
        ),
        "expert_ffn": lambda: grouped_expert_ffn(
            dispatch_result.permuted_hidden, weights, dispatch_result.expert_offsets
        ),
        "combine": lambda: combine_tokens(
            expert_outputs,
            dispatch_result.assignment_to_permuted,
            routing_weights,
            backend=backend,
            validate_values=False,
        ),
        "total": total_path,
    }
    return {
        stage: benchmark_callable(
            callables[stage], device=device, warmup=warmup, repetitions=repetitions
        )
        for stage in stages
    }


def _summary_path(raw_output: Path) -> Path:
    return raw_output.with_name(f"{raw_output.stem}_summary{raw_output.suffix or '.csv'}")


def run(
    config: dict[str, Any],
    device: torch.device,
    output: Path,
    *,
    backend: Backend = "torch",
    stages: tuple[str, ...] = ALL_STAGES,
    summary_output: Path | None = None,
    warmup: int | None = None,
    repetitions: int | None = None,
) -> int:
    """Run configured cases and write raw samples plus percentile summaries."""

    if bool(config.get("requires_cuda", False)) and device.type != "cuda":
        raise ValueError(
            "this target-scale configuration requires a CUDA device; refusing "
            "the multi-GiB CPU allocation"
        )

    if backend == "cuda_ext":
        if device.type != "cuda":
            raise ValueError("cuda_ext backend requires --device cuda")
        # Resolve the extension once, explicitly outside all timed regions.
        if not cuda_extension_available():
            raise RuntimeError(
                "cuda_ext backend requested, but fastpath._C is unavailable: "
                f"{cuda_extension_error()}"
            )
    selected_warmup = config["warmup"] if warmup is None else warmup
    selected_repetitions = config["repetitions"] if repetitions is None else repetitions
    if selected_warmup < 0 or selected_repetitions < 1:
        raise ValueError("warmup must be non-negative and repetitions must be positive")

    summary_output = _summary_path(output) if summary_output is None else summary_output
    if output.resolve() == summary_output.resolve():
        raise ValueError("raw and summary CSV paths must be different")
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    completed_cases = 0
    dtype = DTYPES[config["dtype"]]

    with (
        output.open("w", newline="", encoding="utf-8") as raw_handle,
        summary_output.open("w", newline="", encoding="utf-8") as summary_handle,
    ):
        raw_writer = csv.DictWriter(raw_handle, fieldnames=RAW_CSV_FIELDS)
        summary_writer = csv.DictWriter(summary_handle, fieldnames=SUMMARY_CSV_FIELDS)
        raw_writer.writeheader()
        summary_writer.writeheader()
        for num_tokens in config["tokens"]:
            for num_experts in config["experts"]:
                for top_k in config["top_k"]:
                    if top_k > num_experts:
                        raise ValueError(
                            f"configured top_k={top_k} exceeds num_experts={num_experts}"
                        )
                    for workload in config["workloads"]:
                        summaries = _benchmark_shape(
                            num_tokens=num_tokens,
                            num_experts=num_experts,
                            top_k=top_k,
                            hidden_size=config["hidden_size"],
                            intermediate_size=config["intermediate_size"],
                            workload=workload,
                            seed=config["seed"],
                            dtype=dtype,
                            device=device,
                            backend=backend,
                            stages=stages,
                            warmup=selected_warmup,
                            repetitions=selected_repetitions,
                        )
                        completed_cases += 1
                        common = {
                            "backend": backend,
                            "device": str(device),
                            "dtype": config["dtype"],
                            "workload": workload,
                            "num_tokens": num_tokens,
                            "num_experts": num_experts,
                            "top_k": top_k,
                            "hidden_size": config["hidden_size"],
                            "intermediate_size": config["intermediate_size"],
                        }
                        label = f"T={num_tokens} E={num_experts} K={top_k} {workload}"
                        for stage, summary in summaries.items():
                            print(
                                f"{label:38s} {stage:11s} "
                                f"P50={summary.p50_ms:9.4f} ms "
                                f"P90={summary.p90_ms:9.4f} ms "
                                f"P99={summary.p99_ms:9.4f} ms"
                            )
                            for iteration, latency_ms in enumerate(summary.samples_ms):
                                raw_writer.writerow(
                                    common
                                    | {
                                        "stage": stage,
                                        "iteration": iteration,
                                        "latency_ms": f"{latency_ms:.9f}",
                                    }
                                )
                            summary_writer.writerow(
                                common
                                | {
                                    "stage": stage,
                                    "samples": len(summary.samples_ms),
                                    "p50_ms": f"{summary.p50_ms:.9f}",
                                    "p90_ms": f"{summary.p90_ms:.9f}",
                                    "p99_ms": f"{summary.p99_ms:.9f}",
                                }
                            )
    print(
        f"Wrote {completed_cases} cases to raw={output} and summary={summary_output}"
    )
    return completed_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="YAML shape configuration")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or e.g. cuda:1")
    parser.add_argument(
        "--backend", choices=("torch", "cuda_ext"), default="torch", help="dispatch/combine backend"
    )
    parser.add_argument(
        "--stages",
        type=_parse_stages,
        default=ALL_STAGES,
        help="comma-separated subset of dispatch,expert_ffn,combine,total",
    )
    parser.add_argument("--warmup", type=int, help="override config warmup")
    parser.add_argument("--repeats", type=int, help="override config repetitions")
    parser.add_argument("--output", type=Path, required=True, help="raw CSV destination")
    parser.add_argument(
        "--summary-output", type=Path, help="summary CSV destination (default: <output>_summary.csv)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        load_config(args.config),
        _resolve_device(args.device),
        args.output,
        backend=args.backend,
        stages=args.stages,
        summary_output=args.summary_output,
        warmup=args.warmup,
        repetitions=args.repeats,
    )


if __name__ == "__main__":
    main()
