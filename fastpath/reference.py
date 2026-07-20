"""Pure-PyTorch reference for explicitly routed SwiGLU MoE execution.

Unlike :func:`baseline.torch_moe.local_moe`, this module consumes routing
indices and weights directly.  In particular, routing weights are never
normalized or otherwise rewritten.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Mapping, overload

import torch
import torch.nn.functional as F

from baseline.routing import RoutingResult, group_tokens
from baseline.torch_moe import (
    ExpertWeights,
    combine_expert_outputs,
    grouped_expert_ffn,
    permute_hidden_states,
)

SUPPORTED_TOP_K = (1, 2, 4, 8)
SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


@dataclass(frozen=True)
class RoutedMoEShape:
    """Validated dimensions for the public Goal 1B tensor layout."""

    num_tokens: int
    num_experts: int
    top_k: int
    hidden_size: int
    intermediate_size: int

    @property
    def num_assignments(self) -> int:
        return self.num_tokens * self.top_k


def _require_tensor(name: str, value: object) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    return value


def _validate_positive_int(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_routing_inputs(
    expert_indices: torch.Tensor,
    expert_weights: torch.Tensor,
    num_experts: int,
    top_k: int | None = None,
) -> tuple[int, int]:
    _require_tensor("expert_indices", expert_indices)
    _require_tensor("expert_weights", expert_weights)
    num_experts = _validate_positive_int("num_experts", num_experts)

    if expert_indices.ndim != 2:
        raise ValueError("expert_indices must have shape [num_tokens, top_k]")
    if expert_weights.ndim != 2:
        raise ValueError("expert_weights must have shape [num_tokens, top_k]")
    if expert_weights.shape != expert_indices.shape:
        raise ValueError("expert_indices and expert_weights must have the same shape")
    num_tokens, routed_top_k = expert_indices.shape
    if num_tokens < 1:
        raise ValueError("num_tokens must be positive")
    if routed_top_k not in SUPPORTED_TOP_K:
        raise ValueError(
            f"top_k must be one of {SUPPORTED_TOP_K}, got {routed_top_k}"
        )
    if routed_top_k > num_experts:
        raise ValueError("top_k must not exceed num_experts")
    if top_k is not None:
        top_k = _validate_positive_int("top_k", top_k)
        if top_k != routed_top_k:
            raise ValueError(
                "top_k must match expert_indices.shape[1] and expert_weights.shape[1]"
            )
    if expert_indices.dtype != torch.int64:
        raise TypeError("expert_indices must have dtype torch.int64")
    if expert_weights.dtype not in SUPPORTED_DTYPES:
        raise TypeError(
            "expert_weights dtype must be float32, float16, or bfloat16"
        )
    if expert_indices.device != expert_weights.device:
        raise ValueError("expert_indices and expert_weights must be on the same device")
    if not expert_indices.is_contiguous() or not expert_weights.is_contiguous():
        raise ValueError("expert_indices and expert_weights must be contiguous")

    in_range = (expert_indices >= 0).all() & (expert_indices < num_experts).all()
    if expert_indices.device.type == "cuda":
        torch._assert_async(
            in_range, "expert_indices contains an out-of-range expert index"
        )
    elif not bool(in_range):
        raise ValueError("expert_indices contains an out-of-range expert index")
    return num_tokens, routed_top_k


def validate_routed_moe_inputs(
    hidden_states: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_weights: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    num_experts: int,
    top_k: int,
) -> RoutedMoEShape:
    """Validate the complete Goal 1B tensor contract and return its dimensions."""

    tensors = {
        "hidden_states": _require_tensor("hidden_states", hidden_states),
        "expert_indices": _require_tensor("expert_indices", expert_indices),
        "expert_weights": _require_tensor("expert_weights", expert_weights),
        "gate_up_weight": _require_tensor("gate_up_weight", gate_up_weight),
        "down_weight": _require_tensor("down_weight", down_weight),
    }
    num_experts = _validate_positive_int("num_experts", num_experts)
    top_k = _validate_positive_int("top_k", top_k)
    num_tokens, routed_top_k = _validate_routing_inputs(
        expert_indices, expert_weights, num_experts, top_k
    )

    if hidden_states.ndim != 2:
        raise ValueError("hidden_states must have shape [num_tokens, hidden_size]")
    if hidden_states.shape[0] != num_tokens:
        raise ValueError(
            "hidden_states, expert_indices, and expert_weights must have the same num_tokens"
        )
    hidden_size = hidden_states.shape[1]
    if hidden_size < 1:
        raise ValueError("hidden_size must be positive")
    if gate_up_weight.ndim != 3:
        raise ValueError("gate_up_weight must have shape [E, 2I, H]")
    if gate_up_weight.shape[0] != num_experts:
        raise ValueError("gate_up_weight.shape[0] must equal num_experts")
    if gate_up_weight.shape[2] != hidden_size:
        raise ValueError("gate_up_weight hidden dimension must match hidden_states")
    gate_up_size = gate_up_weight.shape[1]
    if gate_up_size < 2 or gate_up_size % 2:
        raise ValueError("gate_up_weight dimension 1 must be a positive even size 2I")
    intermediate_size = gate_up_size // 2
    expected_down_shape = (num_experts, hidden_size, intermediate_size)
    if down_weight.shape != expected_down_shape:
        raise ValueError(
            "down_weight must have shape [E, H, I] compatible with gate_up_weight; "
            f"expected {expected_down_shape}, got {tuple(down_weight.shape)}"
        )

    floating_tensors = (
        hidden_states,
        expert_weights,
        gate_up_weight,
        down_weight,
    )
    if hidden_states.dtype not in SUPPORTED_DTYPES:
        raise TypeError(
            "hidden_states dtype must be float32, float16, or bfloat16"
        )
    if any(tensor.dtype != hidden_states.dtype for tensor in floating_tensors[1:]):
        raise TypeError("all floating-point inputs must have the same dtype")
    if any(tensor.device != hidden_states.device for tensor in tensors.values()):
        raise ValueError("all inputs must be on the same device")
    if any(not tensor.is_contiguous() for tensor in tensors.values()):
        raise ValueError("all inputs must be contiguous")

    return RoutedMoEShape(
        num_tokens=num_tokens,
        num_experts=num_experts,
        top_k=routed_top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )


def build_routing_metadata(
    expert_indices: torch.Tensor,
    expert_weights: torch.Tensor,
    num_experts: int,
) -> RoutingResult:
    """Build Goal 1A-compatible stable routing metadata from explicit routes."""

    _validate_routing_inputs(expert_indices, expert_weights, num_experts)
    return group_tokens(expert_indices, expert_weights, num_experts)


def _route_through_workspace(
    routing: RoutingResult,
    buffers: Mapping[str, torch.Tensor],
) -> RoutingResult:
    """Copy routing metadata into capacity-backed workspace views."""

    assignments = routing.permutation.numel()
    experts = routing.expert_counts.numel()
    views = {
        "permutation": buffers["permutation"][:assignments],
        "reverse_mapping": buffers["reverse_mapping"][:assignments],
        "permuted_token_indices": buffers["permuted_token_indices"][:assignments],
        "permuted_expert_indices": buffers["permuted_expert_indices"][:assignments],
        "permuted_weights": buffers["permuted_weights"][:assignments],
        "expert_counts": buffers["expert_counts"][:experts],
        "expert_offsets": buffers["expert_offsets"][: experts + 1],
    }
    for name, view in views.items():
        view.copy_(getattr(routing, name))
    return replace(routing, **views)


def _routed_moe_reference_with_routing(
    hidden_states: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_weights: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    num_experts: int,
    top_k: int,
    buffers: Mapping[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, RoutingResult]:
    """Internal validated reference runner with optional reusable buffers."""

    shape = validate_routed_moe_inputs(
        hidden_states,
        expert_indices,
        expert_weights,
        gate_up_weight,
        down_weight,
        num_experts=num_experts,
        top_k=top_k,
    )
    routing = build_routing_metadata(expert_indices, expert_weights, num_experts)

    # The unbuffered path deliberately uses Goal 1A's exact staged reference.
    # It also preserves autograd for callers that use the explicit reference.
    if buffers is None:
        weights = ExpertWeights(
            gate=gate_up_weight[:, : shape.intermediate_size],
            up=gate_up_weight[:, shape.intermediate_size :],
            down=down_weight,
        )
        permuted_states = permute_hidden_states(hidden_states, routing)
        grouped_outputs = grouped_expert_ffn(
            permuted_states, weights, routing.expert_offsets
        )
        return (
            combine_expert_outputs(grouped_outputs, routing, shape.num_tokens),
            routing,
        )

    routing = _route_through_workspace(routing, buffers)
    assignments = shape.num_assignments
    permuted_states = buffers["permuted_hidden"][:assignments]
    permuted_states.copy_(hidden_states.index_select(0, routing.permuted_token_indices))
    gate_up_output = buffers["gate_up_output"][:assignments]
    swiglu_output = buffers["swiglu_output"][:assignments]
    grouped_outputs = buffers["expert_output"][:assignments]

    for expert in range(shape.num_experts):
        start = int(routing.expert_offsets[expert].item())
        end = int(routing.expert_offsets[expert + 1].item())
        if start == end:
            continue
        gate_up_output[start:end].copy_(
            F.linear(permuted_states[start:end], gate_up_weight[expert])
        )
        gate = gate_up_output[start:end, : shape.intermediate_size]
        up = gate_up_output[start:end, shape.intermediate_size :]
        swiglu_output[start:end].copy_(F.silu(gate) * up)
        grouped_outputs[start:end].copy_(
            F.linear(swiglu_output[start:end], down_weight[expert])
        )

    output = combine_expert_outputs(grouped_outputs, routing, shape.num_tokens)
    return output, routing


@overload
def routed_moe_reference(
    hidden_states: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_weights: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    num_experts: int,
    top_k: int,
    return_routing: Literal[False] = False,
) -> torch.Tensor: ...


@overload
def routed_moe_reference(
    hidden_states: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_weights: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    num_experts: int,
    top_k: int,
    return_routing: Literal[True],
) -> tuple[torch.Tensor, RoutingResult]: ...


def routed_moe_reference(
    hidden_states: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_weights: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    num_experts: int,
    top_k: int,
    return_routing: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, RoutingResult]:
    """Execute explicitly routed per-expert SwiGLU with pure PyTorch.

    ``gate_up_weight[:, :I]`` is the gate projection and
    ``gate_up_weight[:, I:]`` is the up projection.  Routing weights are used
    exactly as supplied, including when they do not sum to one.
    """

    if not isinstance(return_routing, bool):
        raise TypeError("return_routing must be a bool")
    output, routing = _routed_moe_reference_with_routing(
        hidden_states,
        expert_indices,
        expert_weights,
        gate_up_weight,
        down_weight,
        num_experts=num_experts,
        top_k=top_k,
    )
    return (output, routing) if return_routing else output


__all__ = [
    "RoutedMoEShape",
    "SUPPORTED_DTYPES",
    "SUPPORTED_TOP_K",
    "build_routing_metadata",
    "routed_moe_reference",
    "validate_routed_moe_inputs",
]
