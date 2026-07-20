from __future__ import annotations

import argparse
import csv
import gc
import math
import statistics
from pathlib import Path
from typing import Callable

import torch


M_VALUES = (1, 2, 4, 8, 16, 32, 64)
E_VALUES = (8, 64)

OPERATIONS = {
    "gate_up": (4096, 28672),
    "down": (14336, 4096),
}


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute percentile of empty list")

    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def balanced_counts(m: int, e: int) -> list[int]:
    counts = [0] * e
    for row in range(m):
        counts[row % e] += 1
    return counts


def measure_us(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    repetitions: int,
) -> tuple[dict[str, float], torch.Tensor]:
    last_output: torch.Tensor | None = None

    for _ in range(warmup):
        last_output = fn()

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples: list[float] = []

    for _ in range(repetitions):
        start.record()
        last_output = fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)) * 1000.0)

    if last_output is None:
        raise RuntimeError("benchmark function produced no output")

    stats = {
        "p50_us": percentile(samples, 0.50),
        "p90_us": percentile(samples, 0.90),
        "p99_us": percentile(samples, 0.99),
        "mean_us": statistics.fmean(samples),
        "min_us": min(samples),
        "max_us": max(samples),
    }
    return stats, last_output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    if args.warmup < 1:
        raise ValueError("warmup must be >= 1")
    if args.repetitions < 10:
        raise ValueError("repetitions must be >= 10")
    if not hasattr(torch, "_grouped_mm"):
        raise RuntimeError("torch._grouped_mm is unavailable")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    print("torch:", torch.__version__)
    print("torch_cuda:", torch.version.cuda)
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    print("warmup:", args.warmup)
    print("repetitions:", args.repetitions)
    print("M_VALUES:", M_VALUES)
    print("E_VALUES:", E_VALUES)
    print("OPERATIONS:", OPERATIONS)
    print()

    rows: list[dict[str, object]] = []
    total_cases = (
        len(M_VALUES)
        * len(E_VALUES)
        * len(OPERATIONS)
        * 2
    )
    completed_cases = 0

    with torch.inference_mode():
        for operation, (k, n) in OPERATIONS.items():
            for experts in E_VALUES:
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.reset_peak_memory_stats(device)

                # Match the real project layout:
                # contiguous [E,N,K] storage, then transpose to a
                # column-major [E,K,N] view.
                weight_storage = torch.empty(
                    (experts, n, k),
                    device=device,
                    dtype=dtype,
                )
                weight_storage.zero_()
                weights = weight_storage.transpose(1, 2)

                if tuple(weights.shape) != (experts, k, n):
                    raise RuntimeError(
                        f"bad weight shape: {tuple(weights.shape)}"
                    )

                print(
                    f"allocated operation={operation} E={experts} "
                    f"K={k} N={n} "
                    f"storage_gib="
                    f"{weight_storage.numel() * weight_storage.element_size() / 2**30:.3f} "
                    f"stride={weights.stride()}",
                    flush=True,
                )

                for m in M_VALUES:
                    counts = balanced_counts(m, experts)
                    offsets = torch.tensor(
                        counts,
                        device=device,
                        dtype=torch.int32,
                    ).cumsum(0, dtype=torch.int32)

                    activations = torch.ones(
                        (m, k),
                        device=device,
                        dtype=dtype,
                    )
                    matmul_output = torch.empty(
                        (m, n),
                        device=device,
                        dtype=dtype,
                    )

                    active_experts = sum(
                        1 for count in counts if count > 0
                    )
                    repeated_offsets = sum(
                        1
                        for left, right in zip(
                            offsets[:-1].tolist(),
                            offsets[1:].tolist(),
                        )
                        if left == right
                    )

                    def grouped_mm_fn() -> torch.Tensor:
                        return torch._grouped_mm(
                            activations,
                            weights,
                            offs=offsets,
                        )

                    def matmul_fn() -> torch.Tensor:
                        start_row = 0

                        for expert_index, count in enumerate(counts):
                            end_row = start_row + count

                            if count > 0:
                                torch.matmul(
                                    activations[start_row:end_row],
                                    weights[expert_index],
                                    out=matmul_output[start_row:end_row],
                                )

                            start_row = end_row

                        return matmul_output

                    for backend, fn in (
                        ("grouped_mm", grouped_mm_fn),
                        ("matmul", matmul_fn),
                    ):
                        stats, output = measure_us(
                            fn,
                            warmup=args.warmup,
                            repetitions=args.repetitions,
                        )

                        if tuple(output.shape) != (m, n):
                            raise RuntimeError(
                                f"{backend} returned bad shape "
                                f"{tuple(output.shape)}, expected {(m, n)}"
                            )

                        p50_us = stats["p50_us"]
                        flop_count = 2.0 * m * k * n
                        tflops = flop_count / (p50_us * 1.0e6)

                        completed_cases += 1

                        row: dict[str, object] = {
                            "operation": operation,
                            "backend": backend,
                            "m": m,
                            "experts": experts,
                            "active_experts": active_experts,
                            "repeated_offsets": repeated_offsets,
                            "k": k,
                            "n": n,
                            "dtype": "bfloat16",
                            "device": "cuda:0",
                            "architecture": "sm_103a",
                            "layout": "transpose_view_EKN",
                            **stats,
                            "tflops_p50": tflops,
                        }
                        rows.append(row)

                        print(
                            f"case={completed_cases:02d}/{total_cases} "
                            f"op={operation:7s} "
                            f"E={experts:2d} "
                            f"M={m:2d} "
                            f"backend={backend:10s} "
                            f"P50={p50_us:10.3f} us "
                            f"P90={stats['p90_us']:10.3f} us "
                            f"P99={stats['p99_us']:10.3f} us "
                            f"TFLOPS={tflops:8.3f}",
                            flush=True,
                        )

                peak_gib = (
                    torch.cuda.max_memory_allocated(device) / 2**30
                )
                print(
                    f"peak_allocated_gib operation={operation} "
                    f"E={experts}: {peak_gib:.3f}",
                    flush=True,
                )

                del weights
                del weight_storage
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                gc.collect()

    args.csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "operation",
        "backend",
        "m",
        "experts",
        "active_experts",
        "repeated_offsets",
        "k",
        "n",
        "dtype",
        "device",
        "architecture",
        "layout",
        "p50_us",
        "p90_us",
        "p99_us",
        "mean_us",
        "min_us",
        "max_us",
        "tflops_p50",
    ]

    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    indexed = {
        (
            str(row["operation"]),
            int(row["experts"]),
            int(row["m"]),
            str(row["backend"]),
        ): row
        for row in rows
    }

    summary_rows: list[dict[str, object]] = []

    for operation in OPERATIONS:
        for experts in E_VALUES:
            for m in M_VALUES:
                grouped = indexed[
                    (operation, experts, m, "grouped_mm")
                ]
                matmul = indexed[
                    (operation, experts, m, "matmul")
                ]

                grouped_p50 = float(grouped["p50_us"])
                matmul_p50 = float(matmul["p50_us"])

                summary_rows.append(
                    {
                        "operation": operation,
                        "m": m,
                        "experts": experts,
                        "active_experts": grouped["active_experts"],
                        "grouped_mm_p50_us": grouped_p50,
                        "matmul_p50_us": matmul_p50,
                        "grouped_mm_speedup": (
                            matmul_p50 / grouped_p50
                        ),
                    }
                )

    with args.summary.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "operation",
                "m",
                "experts",
                "active_experts",
                "grouped_mm_p50_us",
                "matmul_p50_us",
                "grouped_mm_speedup",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print()
    print(f"WROTE_RAW_ROWS={len(rows)} path={args.csv}")
    print(
        f"WROTE_SUMMARY_ROWS={len(summary_rows)} "
        f"path={args.summary}"
    )
    print("STEP4_02_GEMM_ONLY=PASS")


if __name__ == "__main__":
    main()
