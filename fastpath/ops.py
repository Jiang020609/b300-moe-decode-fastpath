"""Unified Python API for Local-MoE dispatch and combine operations."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import Literal

import torch

Backend = Literal["torch", "cuda", "cuda_ext"]
_SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_CUDA_EXTENSION: ModuleType | None = None
_CUDA_EXTENSION_ERROR: BaseException | None = None
_CUDA_EXTENSION_CHECKED = False


@dataclass(frozen=True)
class DispatchResult:
    """Expert-major token assignments and their bidirectional mappings.

    Assignment IDs are token-major: ``token_id * top_k + slot_id``.
    ``assignment_to_permuted[a]`` gives assignment ``a``'s expert-major row,
    while ``permuted_to_assignment`` is its exact inverse.
    """

    permuted_hidden: torch.Tensor
    expert_counts: torch.Tensor
    expert_offsets: torch.Tensor
    assignment_to_permuted: torch.Tensor
    permuted_to_assignment: torch.Tensor


def _load_cuda_extension() -> ModuleType | None:
    global _CUDA_EXTENSION, _CUDA_EXTENSION_CHECKED, _CUDA_EXTENSION_ERROR
    if not _CUDA_EXTENSION_CHECKED:
        _CUDA_EXTENSION_CHECKED = True
        try:
            _CUDA_EXTENSION = importlib.import_module("fastpath._C")
        except (ImportError, OSError) as error:
            _CUDA_EXTENSION_ERROR = error
    return _CUDA_EXTENSION


def cuda_extension_available() -> bool:
    """Return whether the compiled ``fastpath._C`` module can be imported."""

    return _load_cuda_extension() is not None


def cuda_extension_error() -> str | None:
    """Return the CUDA extension import error, if one occurred."""

    _load_cuda_extension()
    return None if _CUDA_EXTENSION_ERROR is None else str(_CUDA_EXTENSION_ERROR)


def _normalize_backend(backend: Backend) -> Literal["torch", "cuda_ext"]:
    if backend == "torch":
        return "torch"
    if backend in ("cuda", "cuda_ext"):
        return "cuda_ext"
    raise ValueError("backend must be 'torch', 'cuda', or 'cuda_ext'")


def _require_extension() -> ModuleType:
    extension = _load_cuda_extension()
    if extension is None:
        detail = cuda_extension_error() or "extension module was not built"
        raise RuntimeError(
            "CUDA extension backend requested, but fastpath._C is unavailable. "
            "Build it in a CUDA development environment with "
            "`MOE_BUILD_CUDA=1 python -m pip install -e .`. "
            f"Import error: {detail}"
        )
    return extension


def _validate_dispatch_inputs(
    hidden_states: torch.Tensor,
    expert_ids: torch.Tensor,
    num_experts: int,
    backend: Literal["torch", "cuda_ext"],
    validate_values: bool,
) -> tuple[int, int]:
    if hidden_states.ndim != 2:
        raise ValueError("hidden_states must have shape [num_tokens, hidden_size]")
    if expert_ids.ndim != 2:
        raise ValueError("expert_ids must have shape [num_tokens, top_k]")
    num_tokens, hidden_size = hidden_states.shape
    if num_tokens < 1 or hidden_size < 1:
        raise ValueError("num_tokens and hidden_size must be positive")
    if expert_ids.shape[0] != num_tokens:
        raise ValueError("hidden_states and expert_ids must have the same num_tokens")
    top_k = expert_ids.shape[1]
    if top_k not in (1, 2, 4, 8):
        raise ValueError(f"top_k must be one of 1, 2, 4, or 8, got {top_k}")
    if not isinstance(num_experts, int) or isinstance(num_experts, bool):
        raise TypeError("num_experts must be an integer")
    if not 4 <= num_experts <= 256:
        raise ValueError(f"num_experts must be in [4, 256], got {num_experts}")
    if hidden_states.dtype not in _SUPPORTED_DTYPES:
        raise TypeError("hidden_states dtype must be float32, float16, or bfloat16")
    if expert_ids.dtype != torch.long:
        raise TypeError("expert_ids must have dtype torch.int64")
    if hidden_states.device != expert_ids.device:
        raise ValueError("hidden_states and expert_ids must be on the same device")
    if not hidden_states.is_contiguous() or not expert_ids.is_contiguous():
        raise ValueError("hidden_states and expert_ids must be contiguous")
    if backend == "cuda_ext":
        if hidden_states.device.type != "cuda":
            raise ValueError("cuda_ext backend requires CUDA tensors")
        if hidden_states.requires_grad:
            raise ValueError("cuda_ext backend does not support backward")
    if validate_values:
        in_range = (expert_ids >= 0).all() & (expert_ids < num_experts).all()
        if expert_ids.device.type == "cuda":
            torch._assert_async(in_range, "expert_ids contains an out-of-range expert index")
        elif not bool(in_range):
            raise ValueError("expert_ids contains an out-of-range expert index")
    return top_k, num_tokens * top_k


def _torch_dispatch(
    hidden_states: torch.Tensor, expert_ids: torch.Tensor, num_experts: int
) -> DispatchResult:
    num_tokens, top_k = expert_ids.shape
    flat_experts = expert_ids.reshape(-1)
    permuted_to_assignment = torch.argsort(flat_experts, stable=True)
    assignment_to_permuted = torch.empty_like(permuted_to_assignment)
    assignment_to_permuted[permuted_to_assignment] = torch.arange(
        permuted_to_assignment.numel(), device=expert_ids.device
    )
    expert_counts = torch.bincount(flat_experts, minlength=num_experts)
    expert_offsets = torch.cat(
        [expert_counts.new_zeros(1), expert_counts.cumsum(dim=0)]
    )
    token_ids = torch.div(permuted_to_assignment, top_k, rounding_mode="floor")
    permuted_hidden = hidden_states.index_select(0, token_ids)
    return DispatchResult(
        permuted_hidden=permuted_hidden,
        expert_counts=expert_counts,
        expert_offsets=expert_offsets,
        assignment_to_permuted=assignment_to_permuted,
        permuted_to_assignment=permuted_to_assignment,
    )


def dispatch_tokens(
    hidden_states: torch.Tensor,
    expert_ids: torch.Tensor,
    num_experts: int,
    *,
    backend: Backend = "torch",
    validate_values: bool = True,
) -> DispatchResult:
    """Group token assignments by expert using the selected backend.

    ``validate_values=False`` is intended only for benchmark loops whose inputs
    were already validated outside the timed region.
    """

    normalized_backend = _normalize_backend(backend)
    _validate_dispatch_inputs(
        hidden_states,
        expert_ids,
        num_experts,
        normalized_backend,
        validate_values,
    )
    if normalized_backend == "torch":
        return _torch_dispatch(hidden_states, expert_ids, num_experts)

    outputs = _require_extension().dispatch(hidden_states, expert_ids, num_experts)
    if len(outputs) != 5:
        raise RuntimeError("fastpath._C.dispatch returned an invalid result")
    return DispatchResult(*outputs)


def _validate_combine_inputs(
    expert_outputs: torch.Tensor,
    assignment_to_permuted: torch.Tensor,
    routing_weights: torch.Tensor,
    backend: Literal["torch", "cuda_ext"],
    validate_values: bool,
) -> tuple[int, int, int]:
    if expert_outputs.ndim != 2:
        raise ValueError("expert_outputs must have shape [num_assignments, hidden_size]")
    if assignment_to_permuted.ndim != 1:
        raise ValueError("assignment_to_permuted must have shape [num_assignments]")
    if routing_weights.ndim != 2:
        raise ValueError("routing_weights must have shape [num_tokens, top_k]")
    num_tokens, top_k = routing_weights.shape
    num_assignments = num_tokens * top_k
    if num_tokens < 1 or expert_outputs.shape[1] < 1:
        raise ValueError("num_tokens and hidden_size must be positive")
    if top_k not in (1, 2, 4, 8):
        raise ValueError(f"top_k must be one of 1, 2, 4, or 8, got {top_k}")
    if expert_outputs.shape[0] != num_assignments:
        raise ValueError("expert_outputs first dimension must equal num_tokens * top_k")
    if assignment_to_permuted.numel() != num_assignments:
        raise ValueError("assignment_to_permuted length must equal num_tokens * top_k")
    if expert_outputs.dtype not in _SUPPORTED_DTYPES:
        raise TypeError("expert_outputs dtype must be float32, float16, or bfloat16")
    if routing_weights.dtype != expert_outputs.dtype:
        raise TypeError("routing_weights and expert_outputs must have the same dtype")
    if assignment_to_permuted.dtype != torch.long:
        raise TypeError("assignment_to_permuted must have dtype torch.int64")
    tensors = (expert_outputs, assignment_to_permuted, routing_weights)
    if any(tensor.device != expert_outputs.device for tensor in tensors):
        raise ValueError("all combine inputs must be on the same device")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("all combine inputs must be contiguous")
    if backend == "cuda_ext":
        if expert_outputs.device.type != "cuda":
            raise ValueError("cuda_ext backend requires CUDA tensors")
        if expert_outputs.requires_grad or routing_weights.requires_grad:
            raise ValueError("cuda_ext backend does not support backward")
    if validate_values:
        non_negative = (assignment_to_permuted >= 0).all()
        in_bounds = (assignment_to_permuted < num_assignments).all()
        if assignment_to_permuted.device.type == "cuda":
            torch._assert_async(
                non_negative & in_bounds,
                "assignment_to_permuted contains an out-of-range position",
            )
        elif not bool(non_negative & in_bounds):
            raise ValueError("assignment_to_permuted contains an out-of-range position")
        ordered = torch.sort(assignment_to_permuted).values
        expected = torch.arange(num_assignments, device=assignment_to_permuted.device)
        is_permutation = (ordered == expected).all()
        if assignment_to_permuted.device.type == "cuda":
            torch._assert_async(
                is_permutation, "assignment_to_permuted must be a permutation"
            )
        elif not bool(is_permutation):
            raise ValueError("assignment_to_permuted must be a permutation")
    return num_tokens, top_k, num_assignments


def combine_tokens(
    expert_outputs: torch.Tensor,
    assignment_to_permuted: torch.Tensor,
    routing_weights: torch.Tensor,
    *,
    backend: Backend = "torch",
    validate_values: bool = True,
) -> torch.Tensor:
    """Gather expert rows by assignment and compute each token's weighted sum.

    ``validate_values=False`` is intended only for benchmark loops whose mapping
    was produced by dispatch before timing.
    """

    normalized_backend = _normalize_backend(backend)
    num_tokens, top_k, _ = _validate_combine_inputs(
        expert_outputs,
        assignment_to_permuted,
        routing_weights,
        normalized_backend,
        validate_values,
    )
    if normalized_backend == "torch":
        assignment_order = expert_outputs.index_select(0, assignment_to_permuted)
        weighted = assignment_order.reshape(num_tokens, top_k, -1).float()
        weighted = weighted * routing_weights.float().unsqueeze(-1)
        return weighted.sum(dim=1).to(expert_outputs.dtype)
    return _require_extension().combine(
        expert_outputs, assignment_to_permuted, routing_weights
    )
