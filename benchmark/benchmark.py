"""Command-line benchmark for the pure-PyTorch Local-MoE reference."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

# Make direct invocation (``python benchmark/benchmark.py``) work from any cwd.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from baseline.routing import generate_router_logits, group_tokens, topk_routing
from baseline.torch_moe import (
    combine_expert_outputs,
    grouped_expert_ffn,
    local_moe,
    make_expert_weights,
    permute_hidden_states,
)
from benchmark.timing import TimingSummary, benchmark_callable

STAGES = ("topk_routing", "token_grouping", "expert_ffn", "expert_combine", "total_moe")
CSV_FIELDS = (
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


def _positive_int_list(config: dict[str, Any], key: str) -> list[int]:
    value = config.get(key)
    if not isinstance(value, list) or not value or any(not isinstance(x, int) or x < 1 for x in value):
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
    workloads = config.get("workloads")
    valid_workloads = {"uniform", "skewed", "hot_expert"}
    if not isinstance(workloads, list) or not workloads or not set(workloads) <= valid_workloads:
        raise ValueError(f"workloads must be a non-empty subset of {sorted(valid_workloads)}")
    return config


def _resolve_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("device must be 'cpu', 'cuda', or an indexed CUDA device")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    return device


def _benchmark_shape(
    *,
    num_tokens: int,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    workload: str,
    seed: int,
    device: torch.device,
    warmup: int,
    repetitions: int,
) -> dict[str, TimingSummary]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    hidden_states = torch.randn(num_tokens, hidden_size, generator=generator).to(device)
    router_logits = generate_router_logits(
        num_tokens, num_experts, top_k, workload, seed + 1, device=device
    )
    weights = make_expert_weights(
        num_experts,
        hidden_size,
        intermediate_size,
        seed=seed + 2,
        device=device,
    )

    # Prepare dependencies outside each isolated stage's timed region.
    topk_indices, topk_weights = topk_routing(router_logits, top_k)
    routing = group_tokens(topk_indices, topk_weights, num_experts)
    permuted_states = permute_hidden_states(hidden_states, routing)
    grouped_outputs = grouped_expert_ffn(permuted_states, weights, routing.expert_offsets)

    callables = {
        "topk_routing": lambda: topk_routing(router_logits, top_k),
        "token_grouping": lambda: group_tokens(topk_indices, topk_weights, num_experts),
        "expert_ffn": lambda: grouped_expert_ffn(permuted_states, weights, routing.expert_offsets),
        "expert_combine": lambda: combine_expert_outputs(grouped_outputs, routing, num_tokens),
        "total_moe": lambda: local_moe(hidden_states, router_logits, weights, top_k),
    }
    return {
        stage: benchmark_callable(
            callables[stage], device=device, warmup=warmup, repetitions=repetitions
        )
        for stage in STAGES
    }


def run(config: dict[str, Any], device: torch.device, output: Path) -> int:
    """Run every configured shape/workload and write one CSV row per raw sample."""

    output.parent.mkdir(parents=True, exist_ok=True)
    completed_shapes = 0
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for num_tokens in config["tokens"]:
            for num_experts in config["experts"]:
                for top_k in config["top_k"]:
                    if top_k > num_experts:
                        raise ValueError(f"configured top_k={top_k} exceeds num_experts={num_experts}")
                    for workload in config["workloads"]:
                        summaries = _benchmark_shape(
                            num_tokens=num_tokens,
                            num_experts=num_experts,
                            top_k=top_k,
                            hidden_size=config["hidden_size"],
                            intermediate_size=config["intermediate_size"],
                            workload=workload,
                            seed=config["seed"],
                            device=device,
                            warmup=config["warmup"],
                            repetitions=config["repetitions"],
                        )
                        completed_shapes += 1
                        label = f"T={num_tokens} E={num_experts} K={top_k} {workload}"
                        for stage, summary in summaries.items():
                            print(
                                f"{label:38s} {stage:15s} "
                                f"P50={summary.p50_ms:9.4f} ms "
                                f"P90={summary.p90_ms:9.4f} ms "
                                f"P99={summary.p99_ms:9.4f} ms"
                            )
                            for iteration, latency_ms in enumerate(summary.samples_ms):
                                writer.writerow(
                                    {
                                        "device": str(device),
                                        "dtype": "float32",
                                        "workload": workload,
                                        "num_tokens": num_tokens,
                                        "num_experts": num_experts,
                                        "top_k": top_k,
                                        "hidden_size": config["hidden_size"],
                                        "intermediate_size": config["intermediate_size"],
                                        "stage": stage,
                                        "iteration": iteration,
                                        "latency_ms": f"{latency_ms:.9f}",
                                    }
                                )
    print(f"Wrote raw samples for {completed_shapes} shape/workload cases to {output}")
    return completed_shapes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="YAML shape configuration")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or e.g. cuda:1")
    parser.add_argument("--output", type=Path, required=True, help="destination CSV path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(load_config(args.config), _resolve_device(args.device), args.output)


if __name__ == "__main__":
    main()
