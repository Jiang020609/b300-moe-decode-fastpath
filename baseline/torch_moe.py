"""Readable, pure-PyTorch local MoE reference implementations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .routing import RoutingResult, group_tokens, topk_routing


@dataclass(frozen=True)
class ExpertWeights:
    """Per-expert SwiGLU weights, stored in PyTorch linear layout [E, out, in]."""

    gate: torch.Tensor  # [E, I, H]
    up: torch.Tensor  # [E, I, H]
    down: torch.Tensor  # [E, H, I]


@dataclass(frozen=True)
class LocalMoEMetadata:
    """Routing metadata returned by :func:`local_moe`."""

    routing: RoutingResult


def make_expert_weights(
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    *,
    seed: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> ExpertWeights:
    """Create deterministic, reasonably scaled random expert weights."""

    if min(num_experts, hidden_size, intermediate_size) < 1:
        raise ValueError("all weight dimensions must be positive")
    if not dtype.is_floating_point:
        raise TypeError("dtype must be floating point")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    scale_in = hidden_size**-0.5
    scale_down = intermediate_size**-0.5

    def sample(shape: tuple[int, ...], scale: float) -> torch.Tensor:
        return (torch.randn(shape, generator=generator) * scale).to(device=device, dtype=dtype)

    return ExpertWeights(
        gate=sample((num_experts, intermediate_size, hidden_size), scale_in),
        up=sample((num_experts, intermediate_size, hidden_size), scale_in),
        down=sample((num_experts, hidden_size, intermediate_size), scale_down),
    )


def _validate_inputs(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    weights: ExpertWeights,
    top_k: int,
) -> tuple[int, int, int]:
    if hidden_states.ndim != 2:
        raise ValueError(f"hidden_states must have shape [T, H], got {tuple(hidden_states.shape)}")
    if router_logits.ndim != 2:
        raise ValueError(f"router_logits must have shape [T, E], got {tuple(router_logits.shape)}")
    num_tokens, hidden_size = hidden_states.shape
    if router_logits.shape[0] != num_tokens:
        raise ValueError("hidden_states and router_logits must have the same num_tokens")
    num_experts = router_logits.shape[1]
    if not 1 <= top_k <= num_experts:
        raise ValueError(f"top_k must be in [1, {num_experts}], got {top_k}")
    if not hidden_states.is_floating_point() or not router_logits.is_floating_point():
        raise TypeError("hidden_states and router_logits must use floating-point dtypes")
    if hidden_states.device != router_logits.device:
        raise ValueError("hidden_states and router_logits must be on the same device")
    if hidden_states.dtype != router_logits.dtype:
        raise ValueError("hidden_states and router_logits must have the same dtype")

    tensors = (weights.gate, weights.up, weights.down)
    if any(t.device != hidden_states.device for t in tensors):
        raise ValueError("all expert weights and inputs must be on the same device")
    if any(t.dtype != hidden_states.dtype for t in tensors):
        raise ValueError("all expert weights and inputs must have the same dtype")
    if weights.gate.ndim != 3 or weights.up.shape != weights.gate.shape:
        raise ValueError("gate and up must have identical shape [E, I, H]")
    if weights.gate.shape[0] != num_experts or weights.gate.shape[2] != hidden_size:
        raise ValueError("gate/up shape is incompatible with router_logits or hidden_states")
    intermediate_size = weights.gate.shape[1]
    if weights.down.shape != (num_experts, hidden_size, intermediate_size):
        raise ValueError("down must have shape [E, H, I] compatible with gate/up")
    return num_tokens, num_experts, hidden_size


def permute_hidden_states(hidden_states: torch.Tensor, routing: RoutingResult) -> torch.Tensor:
    """Gather one hidden-state row for each assignment in expert-major order."""

    return hidden_states.index_select(0, routing.permuted_token_indices)


def grouped_expert_ffn(
    permuted_states: torch.Tensor,
    weights: ExpertWeights,
    expert_offsets: torch.Tensor,
) -> torch.Tensor:
    """Apply each expert's two-linear SwiGLU FFN to its contiguous token slice."""

    output = torch.empty_like(permuted_states)
    num_experts = weights.gate.shape[0]
    for expert in range(num_experts):
        start = int(expert_offsets[expert].item())
        end = int(expert_offsets[expert + 1].item())
        if start == end:
            continue
        states = permuted_states[start:end]
        gated = F.silu(F.linear(states, weights.gate[expert]))
        activated = gated * F.linear(states, weights.up[expert])
        output[start:end] = F.linear(activated, weights.down[expert])
    return output


def combine_expert_outputs(
    grouped_outputs: torch.Tensor,
    routing: RoutingResult,
    num_tokens: int,
) -> torch.Tensor:
    """Weight grouped assignment outputs and sum them in original token order."""

    weighted = grouped_outputs * routing.permuted_weights.unsqueeze(-1)
    output = grouped_outputs.new_zeros((num_tokens, grouped_outputs.shape[-1]))
    output.index_add_(0, routing.permuted_token_indices, weighted)
    return output


def local_moe(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    weights: ExpertWeights,
    top_k: int,
) -> tuple[torch.Tensor, LocalMoEMetadata]:
    """Run top-k routing, grouping, per-expert SwiGLU, and weighted combine."""

    num_tokens, num_experts, _ = _validate_inputs(hidden_states, router_logits, weights, top_k)
    topk_indices, topk_weights = topk_routing(router_logits, top_k)
    routing = group_tokens(topk_indices, topk_weights, num_experts)
    permuted_states = permute_hidden_states(hidden_states, routing)
    grouped_outputs = grouped_expert_ffn(permuted_states, weights, routing.expert_offsets)
    output = combine_expert_outputs(grouped_outputs, routing, num_tokens)
    return output, LocalMoEMetadata(routing=routing)


def naive_local_moe(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    weights: ExpertWeights,
    top_k: int,
) -> torch.Tensor:
    """Direct token/rank-loop reference intended for correctness checks only."""

    num_tokens, _, hidden_size = _validate_inputs(hidden_states, router_logits, weights, top_k)
    topk_indices, topk_weights = topk_routing(router_logits, top_k)
    output = hidden_states.new_zeros((num_tokens, hidden_size))
    for token in range(num_tokens):
        for rank in range(top_k):
            expert = int(topk_indices[token, rank].item())
            state = hidden_states[token : token + 1]
            activated = F.silu(F.linear(state, weights.gate[expert])) * F.linear(
                state, weights.up[expert]
            )
            expert_output = F.linear(activated, weights.down[expert]).squeeze(0)
            output[token] += topk_weights[token, rank] * expert_output
    return output
