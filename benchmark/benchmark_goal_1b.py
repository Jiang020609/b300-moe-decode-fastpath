"""Goal 1B staged microbenchmark for torch and exact CUTLASS backends.

Examples::

    python benchmark/benchmark_goal_1b.py --config configs/quick.yaml
    python benchmark/benchmark_goal_1b.py --config configs/quick.yaml \
        --tokens 1 --experts 4 --top-k 1 --workloads uniform \
        --warmup 0 --repeats 1

Input and expert-weight construction happens before cold-start and steady-state
timing.  CUDA cases use :func:`benchmark.timing.benchmark_callable`, which
records synchronized CUDA Events; CPU cases use its high-resolution wall clock.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

import torch
import torch.nn.functional as F
import yaml

# Support direct invocation from any current working directory.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from baseline.routing import generate_router_logits, topk_routing
from baseline.torch_moe import (
    combine_expert_outputs,
    make_expert_weights,
    permute_hidden_states,
)
from benchmark.timing import TimingSummary, benchmark_callable
from fastpath.b300_moe import B300MoEWorkspace, b300_moe_forward
from fastpath.gemm_dispatch import (
    GemmPolicy,
    normalize_gemm_policy,
    per_expert_gemm_out,
    select_gemm_strategy,
)
from fastpath.reference import build_routing_metadata

Backend = Literal["torch", "cutlass_bf16", "cutlass_nvfp4"]
BACKENDS: tuple[Backend, ...] = ("torch", "cutlass_bf16", "cutlass_nvfp4")
SUMMARY_STAGES = (
    "routing_us",
    "permutation_us",
    "quantization_us",
    "gate_up_gemm_us",
    "swiglu_us",
    "down_gemm_us",
    "combine_us",
    "total_us",
)
WORKLOAD_ALIASES = {
    "uniform": "uniform",
    "hotspot": "hotspot",
    "hot_expert": "hotspot",
    "zipf": "zipf",
    "skewed": "zipf",
}
DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
CSV_FIELDS = (
    "backend_set_requested",
    "requested_backend",
    "actual_backend",
    "device",
    "gpu_name",
    "gpu_capability",
    "architecture",
    "dtype",
    "workload",
    "num_tokens",
    "num_experts",
    "top_k",
    "hidden_size",
    "intermediate_size",
    "quant_mode",
    "used_fallback",
    "gemm_policy",
    "gate_up_gemm_strategy",
    "down_gemm_strategy",
    "stage",
    "stage_status",
    "measurement",
    "samples",
    "p50_us",
    "p90_us",
    "p99_us",
    "cold_start_us",
    "cold_start_scope",
    "warmup",
    "repetitions",
    "timing_method",
    "weights_prepared_outside_timing",
    "torch_version",
    "torch_cuda_version",
    "torch_arch_list",
    "cuda_toolkit_version",
    "cuda_toolkit_version_source",
    "cutlass_version",
    "cutlass_version_source",
    "cutlass_path",
    "wrapper_compiled_architecture",
    "grouped_gemm_provider",
    "grouped_gemm_provider_architecture",
    "pytorch_cxx_version",
)


class BackendUnavailableError(RuntimeError):
    """An exact requested backend cannot execute in the active environment."""


@dataclass(frozen=True)
class StageSummary:
    samples_us: tuple[float, ...]
    p50_us: float
    p90_us: float
    p99_us: float
    status: str = "measured"


@dataclass(frozen=True)
class BenchmarkInputs:
    hidden_states: torch.Tensor
    router_logits: torch.Tensor
    expert_indices: torch.Tensor
    expert_weights: torch.Tensor
    gate_up_weight: torch.Tensor
    down_weight: torch.Tensor
    num_experts: int
    top_k: int
    intermediate_size: int


@dataclass(frozen=True)
class CaseResult:
    cold_start_us: float
    metadata: Mapping[str, object]
    stages: Mapping[str, StageSummary]
    gemm_strategies: Mapping[str, str]


def _positive_int_list(config: Mapping[str, Any], key: str) -> list[int]:
    value = config.get(key)
    if not isinstance(value, list) or not value or any(
        not isinstance(item, int) or isinstance(item, bool) or item < 1
        for item in value
    ):
        raise ValueError(f"config field {key!r} must be a non-empty list of positive integers")
    return list(value)


def _canonical_workloads(values: object) -> list[str]:
    if not isinstance(values, list) or not values:
        raise ValueError("workloads must be a non-empty list")
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str) or value not in WORKLOAD_ALIASES:
            raise ValueError(
                "workloads must use uniform/hotspot/zipf "
                "(hot_expert and skewed are accepted aliases)"
            )
        canonical = WORKLOAD_ALIASES[value]
        if canonical not in normalized:
            normalized.append(canonical)
    return normalized


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate a Goal 1B benchmark configuration."""

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")
    config: dict[str, Any] = dict(raw)
    for key in ("tokens", "experts", "top_k"):
        config[key] = _positive_int_list(config, key)
    if any(value not in (1, 2, 4, 8) for value in config["top_k"]):
        raise ValueError("all top_k values must be one of 1, 2, 4, or 8")
    if any(value > 256 for value in config["experts"]):
        raise ValueError("expert counts greater than 256 are not supported")
    for key in ("hidden_size", "intermediate_size"):
        value = config.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"config field {key!r} must be a positive integer")
    for key in ("seed", "warmup", "repetitions"):
        if not isinstance(config.get(key), int) or isinstance(config.get(key), bool):
            raise ValueError(f"config field {key!r} must be an integer")
    if config["warmup"] < 0 or config["repetitions"] < 1:
        raise ValueError("warmup must be non-negative and repetitions must be positive")
    dtype_name = config.get("dtype", "float32")
    if dtype_name not in DTYPES:
        raise ValueError(f"dtype must be one of {sorted(DTYPES)}")
    config["dtype"] = dtype_name
    config["workloads"] = _canonical_workloads(config.get("workloads"))
    requires_cuda = config.get("requires_cuda", False)
    if not isinstance(requires_cuda, bool):
        raise ValueError("requires_cuda must be a boolean when present")
    config["requires_cuda"] = requires_cuda
    return config


def _resolve_device(requested: str | torch.device) -> torch.device:
    device = torch.device(requested)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("device must be cpu, cuda, or an indexed CUDA device")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
    return device


def _validate_override(values: Sequence[int] | None, name: str) -> list[int] | None:
    if values is None:
        return None
    result = list(values)
    if not result or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 1
        for value in result
    ):
        raise ValueError(f"{name} override must contain positive integers")
    return result


def _make_inputs(
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
) -> BenchmarkInputs:
    # All random initialization and device transfers are outside every timer.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    hidden_states = torch.randn(
        (num_tokens, hidden_size), generator=generator, dtype=torch.float32
    ).to(device=device, dtype=dtype)
    router_logits = generate_router_logits(
        num_tokens,
        num_experts,
        top_k,
        workload,  # type: ignore[arg-type]
        seed + 1,
        device=device,
        dtype=dtype,
    ).contiguous()
    weights = make_expert_weights(
        num_experts,
        hidden_size,
        intermediate_size,
        seed=seed + 2,
        device=device,
        dtype=dtype,
    )
    gate_up_weight = torch.cat((weights.gate, weights.up), dim=1).contiguous()
    down_weight = weights.down.contiguous()
    expert_indices, expert_weights = topk_routing(router_logits, top_k)
    return BenchmarkInputs(
        hidden_states=hidden_states.contiguous(),
        router_logits=router_logits,
        expert_indices=expert_indices.contiguous(),
        expert_weights=expert_weights.contiguous(),
        gate_up_weight=gate_up_weight,
        down_weight=down_weight,
        num_experts=num_experts,
        top_k=top_k,
        intermediate_size=intermediate_size,
    )


def _to_us(summary: TimingSummary) -> StageSummary:
    return StageSummary(
        samples_us=tuple(value * 1000.0 for value in summary.samples_ms),
        p50_us=summary.p50_ms * 1000.0,
        p90_us=summary.p90_ms * 1000.0,
        p99_us=summary.p99_ms * 1000.0,
    )


def _zero_summary(repetitions: int, reason: str = "not_applicable") -> StageSummary:
    return StageSummary(
        samples_us=(0.0,) * repetitions,
        p50_us=0.0,
        p90_us=0.0,
        p99_us=0.0,
        status=reason,
    )


def _time_stage(
    function: Callable[[], object],
    *,
    device: torch.device,
    warmup: int,
    repetitions: int,
) -> StageSummary:
    return _to_us(
        benchmark_callable(
            function,
            device=device,
            warmup=warmup,
            repetitions=repetitions,
        )
    )


def _torch_stage_callables(
    inputs: BenchmarkInputs,
    total: Callable[[], object],
) -> Mapping[str, Callable[[], object]]:
    routing = build_routing_metadata(
        inputs.expert_indices, inputs.expert_weights, inputs.num_experts
    )
    permuted = permute_hidden_states(inputs.hidden_states, routing)
    boundaries = tuple(int(value) for value in routing.expert_offsets.cpu().tolist())

    def permutation() -> torch.Tensor:
        current = build_routing_metadata(
            inputs.expert_indices, inputs.expert_weights, inputs.num_experts
        )
        return permute_hidden_states(inputs.hidden_states, current)

    def gate_up_gemm() -> torch.Tensor:
        output = inputs.hidden_states.new_empty(
            (permuted.shape[0], 2 * inputs.intermediate_size)
        )
        for expert, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
            if start != end:
                output[start:end] = F.linear(
                    permuted[start:end], inputs.gate_up_weight[expert]
                )
        return output

    gate_up_output = gate_up_gemm()

    def swiglu() -> torch.Tensor:
        gate = gate_up_output[:, : inputs.intermediate_size]
        up = gate_up_output[:, inputs.intermediate_size :]
        return F.silu(gate) * up

    swiglu_output = swiglu()

    def down_gemm() -> torch.Tensor:
        output = inputs.hidden_states.new_empty(permuted.shape)
        for expert, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
            if start != end:
                output[start:end] = F.linear(
                    swiglu_output[start:end], inputs.down_weight[expert]
                )
        return output

    expert_output = down_gemm()
    return {
        "routing_us": lambda: topk_routing(inputs.router_logits, inputs.top_k),
        "permutation_us": permutation,
        "gate_up_gemm_us": gate_up_gemm,
        "swiglu_us": swiglu,
        "down_gemm_us": down_gemm,
        "combine_us": lambda: combine_expert_outputs(
            expert_output, routing, inputs.hidden_states.shape[0]
        ),
        "total_us": total,
    }


def _compiled_stage_callables(
    inputs: BenchmarkInputs,
    workspace: B300MoEWorkspace,
    total: Callable[[], object],
    gemm_policy: GemmPolicy = "auto",
) -> tuple[Mapping[str, Callable[[], object]], dict[str, str]]:
    extension = importlib.import_module("fastpath._C")
    buffers = workspace.buffers
    assignments = inputs.hidden_states.shape[0] * inputs.top_k
    counts = buffers["expert_counts"][: inputs.num_experts]
    offsets = buffers["expert_offsets"][: inputs.num_experts + 1]
    assignment_to_permuted = buffers["reverse_mapping"][:assignments]
    permuted_to_assignment = buffers["permutation"][:assignments]
    extension.routing_metadata_out(
        inputs.expert_indices,
        inputs.num_experts,
        counts,
        offsets,
        assignment_to_permuted,
        permuted_to_assignment,
    )
    offsets_i32 = buffers["expert_offsets_i32"][: inputs.num_experts]
    offsets_i32.copy_(offsets[1:])

    # Stage timing must exercise the same GEMM strategy the requested policy
    # selects end-to-end; the host offsets sync happens here, outside the
    # timed region, exactly like the end-to-end fast path resolves it once
    # per forward.  The fused kernel consumes device offsets directly, so it
    # needs no host copy at all.
    assignments_total = int(assignments)
    gate_up_strategy = "grouped"
    down_strategy = "grouped"
    host_offsets: list[int] | None = None
    if gemm_policy == "fused":
        if not hasattr(extension, "fused_grouped_gemm_bf16_out"):
            raise BackendUnavailableError(
                "gemm_policy='fused' requires fastpath._C.fused_grouped_gemm_bf16_out; "
                "rebuild the Goal 1B extension from this source tree"
            )
        gate_up_strategy = "fused"
        down_strategy = "fused"
    elif gemm_policy != "grouped":
        host_offsets = offsets.tolist()
        active_experts = sum(
            1
            for expert in range(inputs.num_experts)
            if host_offsets[expert + 1] > host_offsets[expert]
        )
        gate_up_strategy = select_gemm_strategy(
            "gate_up",
            total_rows=assignments_total,
            active_experts=active_experts,
            policy=gemm_policy,
        )
        down_strategy = select_gemm_strategy(
            "down",
            total_rows=assignments_total,
            active_experts=active_experts,
            policy=gemm_policy,
        )

    permuted = buffers["permuted_hidden"][:assignments]
    gate_up_output = buffers["gate_up_output"][:assignments]
    swiglu_output = buffers["swiglu_output"][:assignments]
    expert_output = buffers["expert_output"][:assignments]
    combined = torch.empty_like(inputs.hidden_states)
    packed_gate_up = inputs.gate_up_weight.transpose(1, 2)
    packed_down = inputs.down_weight.transpose(1, 2)

    def permutation() -> torch.Tensor:
        extension.routing_metadata_out(
            inputs.expert_indices,
            inputs.num_experts,
            counts,
            offsets,
            assignment_to_permuted,
            permuted_to_assignment,
        )
        extension.permute_out(
            inputs.hidden_states, permuted_to_assignment, inputs.top_k, permuted
        )
        return permuted

    return {
        "routing_us": lambda: topk_routing(inputs.router_logits, inputs.top_k),
        "permutation_us": permutation,
        "gate_up_gemm_us": (
            (
                lambda: extension.grouped_gemm_bf16_out(
                    permuted, packed_gate_up, offsets_i32, gate_up_output
                )
            )
            if gate_up_strategy == "grouped"
            else (
                lambda: extension.fused_grouped_gemm_bf16_out(
                    permuted, packed_gate_up, offsets, gate_up_output
                )
            )
            if gate_up_strategy == "fused"
            else (
                lambda: per_expert_gemm_out(
                    permuted, packed_gate_up, host_offsets, gate_up_output
                )
            )
        ),
        "swiglu_us": lambda: extension.swiglu_out(gate_up_output, swiglu_output),
        "down_gemm_us": (
            (
                lambda: extension.grouped_gemm_bf16_out(
                    swiglu_output, packed_down, offsets_i32, expert_output
                )
            )
            if down_strategy == "grouped"
            else (
                lambda: extension.fused_grouped_gemm_bf16_out(
                    swiglu_output, packed_down, offsets, expert_output
                )
            )
            if down_strategy == "fused"
            else (
                lambda: per_expert_gemm_out(
                    swiglu_output, packed_down, host_offsets, expert_output
                )
            )
        ),
        "combine_us": lambda: extension.combine_out(
            expert_output, assignment_to_permuted, inputs.expert_weights, combined
        ),
        "total_us": total,
    }, {
        "gate_up_gemm_strategy": gate_up_strategy,
        "down_gemm_strategy": down_strategy,
    }


def _benchmark_case(
    *,
    inputs: BenchmarkInputs,
    backend: Backend,
    device: torch.device,
    warmup: int,
    repetitions: int,
    gemm_policy: GemmPolicy = "auto",
) -> CaseResult:
    if backend != "torch" and device.type != "cuda":
        raise BackendUnavailableError(
            f"backend={backend!r} requires a CUDA device; no backend fallback is allowed"
        )
    if backend == "cutlass_bf16" and inputs.hidden_states.dtype != torch.bfloat16:
        raise BackendUnavailableError(
            "backend='cutlass_bf16' requires dtype=bfloat16; no dtype fallback is allowed"
        )

    quant_mode = {
        "torch": None,
        "cutlass_bf16": "bf16",
        "cutlass_nvfp4": "nvfp4",
    }[backend]
    workspace = B300MoEWorkspace(
        capacity_tokens=inputs.hidden_states.shape[0], device=device
    )

    def total() -> object:
        expert_indices, expert_weights = topk_routing(
            inputs.router_logits, inputs.top_k
        )
        return b300_moe_forward(
            inputs.hidden_states,
            expert_indices.contiguous(),
            expert_weights.contiguous(),
            inputs.gate_up_weight,
            inputs.down_weight,
            num_experts=inputs.num_experts,
            top_k=inputs.top_k,
            quant_mode=quant_mode,
            backend=backend,
            workspace=workspace,
            gemm_policy=gemm_policy,
        )

    try:
        cold = benchmark_callable(
            total, device=device, warmup=0, repetitions=1
        ).samples_ms[0] * 1000.0
        _, metadata = b300_moe_forward(
            inputs.hidden_states,
            inputs.expert_indices,
            inputs.expert_weights,
            inputs.gate_up_weight,
            inputs.down_weight,
            num_experts=inputs.num_experts,
            top_k=inputs.top_k,
            quant_mode=quant_mode,
            backend=backend,
            workspace=workspace,
            gemm_policy=gemm_policy,
            return_metadata=True,
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        if backend == "torch":
            raise
        raise BackendUnavailableError(
            f"backend={backend!r} is unavailable: {error}"
        ) from error

    actual_backend = metadata.get("backend")
    used_fallback = metadata.get("used_fallback")
    if actual_backend != backend or used_fallback is not False:
        raise BackendUnavailableError(
            f"backend={backend!r} returned backend={actual_backend!r}, "
            f"used_fallback={used_fallback!r}; refusing to benchmark it"
        )

    if backend == "torch":
        callables = _torch_stage_callables(inputs, total)
        # The dense torch backend never routes through the hybrid GEMM
        # dispatch, so strategies are reported truthfully as not applicable.
        gemm_strategies = {
            "gate_up_gemm_strategy": "not_applicable",
            "down_gemm_strategy": "not_applicable",
        }
    else:
        try:
            callables, gemm_strategies = _compiled_stage_callables(
                inputs, workspace, total, gemm_policy
            )
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
            raise BackendUnavailableError(
                f"backend={backend!r} staged benchmark setup failed: {error}"
            ) from error

    summaries: dict[str, StageSummary] = {}
    for stage in SUMMARY_STAGES:
        if stage == "quantization_us":
            # Torch and BF16 consume already-typed inputs. The current public
            # API refuses NVFP4 before this point unless real quantize/GEMM
            # entry points exist, so a zero must never be reported for NVFP4.
            summaries[stage] = _zero_summary(repetitions)
            continue
        summaries[stage] = _time_stage(
            callables[stage],
            device=device,
            warmup=warmup,
            repetitions=repetitions,
        )
    return CaseResult(
        cold_start_us=cold,
        metadata=metadata,
        stages=summaries,
        gemm_strategies=gemm_strategies,
    )


def _parse_version_header(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "unavailable"
    values: list[str] = []
    for component in ("MAJOR", "MINOR", "PATCH"):
        match = re.search(
            rf"^\s*#\s*define\s+CUTLASS_{component}\s+(\d+)\b",
            text,
            flags=re.MULTILINE,
        )
        if match is None:
            return "unavailable"
        values.append(match.group(1))
    return ".".join(values)


def _toolchain_metadata(device: torch.device) -> dict[str, str]:
    cuda_home_value = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if not cuda_home_value:
        try:
            cpp_extension = importlib.import_module("torch.utils.cpp_extension")
            cuda_home_value = getattr(cpp_extension, "CUDA_HOME", None)
        except (ImportError, OSError):
            cuda_home_value = None
    cuda_toolkit_version = "unavailable"
    if cuda_home_value:
        cuda_home = Path(str(cuda_home_value)).expanduser()
        nvcc = next(
            (
                cuda_home / "bin" / name
                for name in ("nvcc", "nvcc.exe")
                if (cuda_home / "bin" / name).is_file()
            ),
            None,
        )
        if nvcc is not None:
            try:
                completed = subprocess.run(
                    [str(nvcc), "--version"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=15,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError):
                completed = None
            if completed is not None and completed.returncode == 0:
                match = re.search(r"\brelease\s+(\d+\.\d+(?:\.\d+)?)", completed.stdout)
                if match:
                    cuda_toolkit_version = match.group(1)

    cutlass_value = os.environ.get("CUTLASS_PATH", "").strip()
    cutlass_path = (
        str(Path(cutlass_value).expanduser().resolve()) if cutlass_value else "unavailable"
    )
    cutlass_version = (
        _parse_version_header(Path(cutlass_path) / "include/cutlass/version.h")
        if cutlass_path != "unavailable"
        else "unavailable"
    )
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        capability = torch.cuda.get_device_capability(device)
        gpu_capability = f"{capability[0]}.{capability[1]}"
        torch_arch_list = ",".join(torch.cuda.get_arch_list())
    else:
        gpu_name = "CPU"
        gpu_capability = "unavailable"
        torch_arch_list = "unavailable"
    return {
        "gpu_name": gpu_name,
        "gpu_capability": gpu_capability,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda or "unavailable",
        "torch_arch_list": torch_arch_list,
        "cuda_toolkit_version": cuda_toolkit_version,
        "cutlass_version": cutlass_version,
        "cutlass_path": cutlass_path,
    }


def _cudart_version_text(value: object) -> str:
    """Render CUDA's integer CUDART_VERSION macro without guessing a toolkit."""

    if not isinstance(value, int) or isinstance(value, bool) or value < 1000:
        return "unavailable"
    major = value // 1000
    minor = (value % 1000) // 10
    patch = value % 10
    return f"{major}.{minor}" if patch == 0 else f"{major}.{minor}.{patch}"


def _normalize_backends(backends: Sequence[str] | str) -> tuple[Backend, ...]:
    values = (backends,) if isinstance(backends, str) else tuple(backends)
    if not values:
        raise ValueError("at least one backend must be selected")
    normalized: list[Backend] = []
    for value in values:
        if value not in BACKENDS:
            raise ValueError(f"unknown backend {value!r}; choose from {BACKENDS}")
        if value not in normalized:
            normalized.append(value)  # type: ignore[arg-type]
    return tuple(normalized)


def run(
    config: Mapping[str, Any],
    device: torch.device | str,
    output: Path,
    *,
    backends: Sequence[str] | str = ("torch",),
    tokens: Sequence[int] | None = None,
    experts: Sequence[int] | None = None,
    top_k: Sequence[int] | None = None,
    workloads: Sequence[str] | None = None,
    hidden_size: int | None = None,
    intermediate_size: int | None = None,
    dtype: str | None = None,
    warmup: int | None = None,
    repetitions: int | None = None,
    gemm_policy: str = "auto",
    skip_unavailable: bool = False,
) -> int:
    """Run selected cases and write one percentile-summary CSV."""

    selected_device = _resolve_device(device)
    selected_gemm_policy = normalize_gemm_policy(gemm_policy)
    if bool(config.get("requires_cuda", False)) and selected_device.type != "cuda":
        raise BackendUnavailableError(
            "this target-scale configuration requires --device cuda; refusing "
            "the multi-GiB CPU allocation"
        )
    selected_backends = _normalize_backends(backends)
    selected_tokens = _validate_override(tokens, "tokens") or list(config["tokens"])
    selected_experts = _validate_override(experts, "experts") or list(config["experts"])
    selected_top_k = _validate_override(top_k, "top_k") or list(config["top_k"])
    if any(value not in (1, 2, 4, 8) for value in selected_top_k):
        raise ValueError("top_k values must be one of 1, 2, 4, or 8")
    if any(value > 256 for value in selected_experts):
        raise ValueError("expert counts greater than 256 are not supported")
    selected_workloads = (
        _canonical_workloads(list(workloads))
        if workloads is not None
        else list(config["workloads"])
    )
    selected_hidden = config["hidden_size"] if hidden_size is None else hidden_size
    selected_intermediate = (
        config["intermediate_size"] if intermediate_size is None else intermediate_size
    )
    if selected_hidden < 1 or selected_intermediate < 1:
        raise ValueError("hidden_size and intermediate_size must be positive")
    dtype_name = config["dtype"] if dtype is None else dtype
    if dtype_name not in DTYPES:
        raise ValueError(f"dtype must be one of {sorted(DTYPES)}")
    selected_warmup = config["warmup"] if warmup is None else warmup
    selected_repetitions = config["repetitions"] if repetitions is None else repetitions
    if selected_warmup < 0 or selected_repetitions < 1:
        raise ValueError("warmup must be non-negative and repetitions must be positive")

    environment = _toolchain_metadata(selected_device)
    rows: list[dict[str, object]] = []
    completed_cases = 0
    skipped_backends: set[str] = set()
    torch_dtype = DTYPES[dtype_name]

    for backend in selected_backends:
        if backend != "torch" and selected_device.type != "cuda":
            error = BackendUnavailableError(
                f"backend={backend!r} requires a CUDA device; pass --skip-unavailable "
                "to skip it explicitly"
            )
            if not skip_unavailable:
                raise error
            print(f"SKIP {backend}: {error}")
            skipped_backends.add(backend)
            continue
        if backend == "cutlass_bf16" and torch_dtype != torch.bfloat16:
            error = BackendUnavailableError(
                "backend='cutlass_bf16' requires dtype=bfloat16; pass --dtype bfloat16"
            )
            if not skip_unavailable:
                raise error
            print(f"SKIP {backend}: {error}")
            skipped_backends.add(backend)
            continue

        for num_tokens in selected_tokens:
            for num_experts in selected_experts:
                for routed_top_k in selected_top_k:
                    if routed_top_k > num_experts:
                        raise ValueError(
                            f"top_k={routed_top_k} exceeds num_experts={num_experts}"
                        )
                    for workload in selected_workloads:
                        case_inputs = _make_inputs(
                            num_tokens=num_tokens,
                            num_experts=num_experts,
                            top_k=routed_top_k,
                            hidden_size=selected_hidden,
                            intermediate_size=selected_intermediate,
                            workload=workload,
                            seed=config["seed"],
                            dtype=torch_dtype,
                            device=selected_device,
                        )
                        try:
                            result = _benchmark_case(
                                inputs=case_inputs,
                                backend=backend,
                                device=selected_device,
                                warmup=selected_warmup,
                                repetitions=selected_repetitions,
                                gemm_policy=selected_gemm_policy,
                            )
                        except BackendUnavailableError as error:
                            if not skip_unavailable:
                                raise
                            print(f"SKIP {backend}: {error}")
                            skipped_backends.add(backend)
                            break

                        completed_cases += 1
                        metadata = result.metadata
                        raw_build = metadata.get("build")
                        build_metadata = (
                            raw_build if isinstance(raw_build, Mapping) else {}
                        )
                        case_environment = dict(environment)
                        if build_metadata:
                            case_environment["cuda_toolkit_version"] = (
                                _cudart_version_text(
                                    build_metadata.get("cuda_runtime_version")
                                )
                            )
                            external_cutlass = build_metadata.get(
                                "external_cutlass_headers_version"
                            )
                            case_environment["cutlass_version"] = (
                                str(external_cutlass)
                                if external_cutlass is not None
                                else "unavailable"
                            )
                        label = (
                            f"backend={backend} T={num_tokens} E={num_experts} "
                            f"K={routed_top_k} {workload}"
                        )
                        print(f"{label} cold_start={result.cold_start_us:.3f} us")
                        common: dict[str, object] = {
                            "backend_set_requested": ",".join(selected_backends),
                            "requested_backend": backend,
                            "actual_backend": metadata["backend"],
                            "device": str(selected_device),
                            "gpu_name": environment["gpu_name"],
                            "gpu_capability": environment["gpu_capability"],
                            "architecture": metadata["architecture"],
                            "dtype": dtype_name,
                            "workload": workload,
                            "num_tokens": num_tokens,
                            "num_experts": num_experts,
                            "top_k": routed_top_k,
                            "hidden_size": selected_hidden,
                            "intermediate_size": selected_intermediate,
                            "quant_mode": metadata["quant_mode"],
                            "used_fallback": str(bool(metadata["used_fallback"])).lower(),
                            "gemm_policy": selected_gemm_policy,
                            "gate_up_gemm_strategy": result.gemm_strategies[
                                "gate_up_gemm_strategy"
                            ],
                            "down_gemm_strategy": result.gemm_strategies[
                                "down_gemm_strategy"
                            ],
                            "measurement": "steady_state",
                            "cold_start_us": f"{result.cold_start_us:.9f}",
                            "cold_start_scope": "first_forward_after_input_setup",
                            "warmup": selected_warmup,
                            "repetitions": selected_repetitions,
                            "timing_method": (
                                "cuda_events"
                                if selected_device.type == "cuda"
                                else "perf_counter_ns"
                            ),
                            "weights_prepared_outside_timing": "true",
                            **case_environment,
                            "cuda_toolkit_version_source": (
                                "extension_cudart_header"
                                if build_metadata
                                else "runtime_environment"
                            ),
                            "cutlass_version_source": (
                                "extension_external_headers"
                                if build_metadata
                                else "runtime_environment"
                            ),
                            "wrapper_compiled_architecture": build_metadata.get(
                                "wrapper_compiled_architecture", "unavailable"
                            ),
                            "grouped_gemm_provider": build_metadata.get(
                                "bf16_grouped_gemm_provider", "unavailable"
                            ),
                            "grouped_gemm_provider_architecture": build_metadata.get(
                                "grouped_gemm_provider_architecture", "unavailable"
                            ),
                            "pytorch_cxx_version": build_metadata.get(
                                "pytorch_cxx_version", "unavailable"
                            ),
                        }
                        for stage in SUMMARY_STAGES:
                            summary = result.stages[stage]
                            print(
                                f"  {stage:20s} P50={summary.p50_us:10.3f} us "
                                f"P90={summary.p90_us:10.3f} us "
                                f"P99={summary.p99_us:10.3f} us"
                            )
                            rows.append(
                                common
                                | {
                                    "stage": stage,
                                    "stage_status": summary.status,
                                    "samples": len(summary.samples_us),
                                    "p50_us": f"{summary.p50_us:.9f}",
                                    "p90_us": f"{summary.p90_us:.9f}",
                                    "p99_us": f"{summary.p99_us:.9f}",
                                }
                            )
                    if backend in skipped_backends:
                        break
                if backend in skipped_backends:
                    break
            if backend in skipped_backends:
                break

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {completed_cases} cases ({len(rows)} stage summaries) to {output}")
    return completed_cases


def _parse_int_csv(value: str) -> list[int]:
    try:
        parsed = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not parsed or any(item < 1 for item in parsed):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return parsed


def _parse_workload_csv(value: str) -> list[str]:
    parsed = [part.strip() for part in value.split(",") if part.strip()]
    try:
        return _canonical_workloads(parsed)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _parse_backend_csv(value: str) -> tuple[Backend, ...]:
    parsed = [part.strip() for part in value.split(",") if part.strip()]
    if parsed == ["all"]:
        return BACKENDS
    try:
        return _normalize_backends(parsed)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or e.g. cuda:1")
    parser.add_argument(
        "--backend",
        type=_parse_backend_csv,
        default=("torch",),
        help="torch, cutlass_bf16, cutlass_nvfp4, all, or a comma-separated list",
    )
    parser.add_argument("--skip-unavailable", action="store_true")
    parser.add_argument("--tokens", type=_parse_int_csv)
    parser.add_argument("--experts", type=_parse_int_csv)
    parser.add_argument("--top-k", type=_parse_int_csv)
    parser.add_argument("--workloads", type=_parse_workload_csv)
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--intermediate-size", type=int)
    parser.add_argument("--dtype", choices=tuple(DTYPES))
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument(
        "--gemm-policy",
        choices=("grouped", "auto", "per_expert", "fused"),
        default="auto",
        help="Goal 1C GEMM strategy for compiled backends; "
        "'auto' is the B300-validated hybrid default, 'grouped' is the "
        "Goal 1B baseline, 'fused' is the opt-in device-offset kernel",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "results/goal_1b_benchmark.csv",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    try:
        run(
            load_config(arguments.config),
            arguments.device,
            arguments.output,
            backends=arguments.backend,
            tokens=arguments.tokens,
            experts=arguments.experts,
            top_k=arguments.top_k,
            workloads=arguments.workloads,
            hidden_size=arguments.hidden_size,
            intermediate_size=arguments.intermediate_size,
            dtype=arguments.dtype,
            warmup=arguments.warmup,
            repetitions=arguments.repeats,
            gemm_policy=arguments.gemm_policy,
            skip_unavailable=arguments.skip_unavailable,
        )
    except BackendUnavailableError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
