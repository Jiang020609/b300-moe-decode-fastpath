"""Routing and deterministic routing-workload helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

Workload = Literal["uniform", "skewed", "hot_expert", "zipf", "hotspot"]


@dataclass(frozen=True)
class RoutingResult:
    """Top-k decisions and the metadata needed to group and ungroup tokens.

    ``reverse_mapping`` indexes a tensor in grouped order and restores the
    original flattened assignment order (token-major, then top-k rank).
    ``expert_offsets[e]:expert_offsets[e + 1]`` is expert ``e``'s slice.
    """

    topk_indices: torch.Tensor
    topk_weights: torch.Tensor
    permutation: torch.Tensor
    reverse_mapping: torch.Tensor
    permuted_token_indices: torch.Tensor
    permuted_expert_indices: torch.Tensor
    permuted_weights: torch.Tensor
    expert_counts: torch.Tensor
    expert_offsets: torch.Tensor


def _validate_logits(router_logits: torch.Tensor, top_k: int) -> None:
    if router_logits.ndim != 2:
        raise ValueError(
            f"router_logits must have shape [num_tokens, num_experts], got "
            f"{tuple(router_logits.shape)}"
        )
    num_tokens, num_experts = router_logits.shape
    if num_tokens < 1 or num_experts < 1:
        raise ValueError("router_logits must contain at least one token and expert")
    if not router_logits.is_floating_point():
        raise TypeError("router_logits must use a floating-point dtype")
    if not 1 <= top_k <= num_experts:
        raise ValueError(f"top_k must be in [1, {num_experts}], got {top_k}")


def topk_routing(router_logits: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Select experts and normalize the selected logits with a softmax."""

    _validate_logits(router_logits, top_k)
    selected_logits, topk_indices = torch.topk(router_logits, top_k, dim=-1)
    # Softmax in fp32 avoids avoidable loss of precision for fp16/bf16 routing.
    topk_weights = torch.softmax(selected_logits.float(), dim=-1).to(router_logits.dtype)
    return topk_indices, topk_weights


def group_tokens(
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
) -> RoutingResult:
    """Create an expert-major stable permutation for flattened token assignments."""

    if topk_indices.ndim != 2 or topk_weights.shape != topk_indices.shape:
        raise ValueError("topk_indices and topk_weights must have the same 2-D shape")
    if topk_indices.dtype != torch.long:
        raise TypeError("topk_indices must have dtype torch.long")
    if num_experts < 1:
        raise ValueError("num_experts must be positive")
    if topk_indices.numel() == 0:
        raise ValueError("routing tensors must not be empty")
    if torch.any(topk_indices < 0) or torch.any(topk_indices >= num_experts):
        raise ValueError("topk_indices contains an out-of-range expert index")

    num_tokens, top_k = topk_indices.shape
    flat_experts = topk_indices.reshape(-1)
    flat_weights = topk_weights.reshape(-1)
    flat_tokens = torch.arange(num_tokens, device=topk_indices.device).repeat_interleave(top_k)

    # stable=True gives deterministic token/rank order inside each expert.
    permutation = torch.argsort(flat_experts, stable=True)
    reverse_mapping = torch.empty_like(permutation)
    reverse_mapping[permutation] = torch.arange(permutation.numel(), device=permutation.device)
    expert_counts = torch.bincount(flat_experts, minlength=num_experts)
    expert_offsets = torch.cat(
        [torch.zeros(1, dtype=torch.long, device=flat_experts.device), expert_counts.cumsum(0)]
    )
    return RoutingResult(
        topk_indices=topk_indices,
        topk_weights=topk_weights,
        permutation=permutation,
        reverse_mapping=reverse_mapping,
        permuted_token_indices=flat_tokens[permutation],
        permuted_expert_indices=flat_experts[permutation],
        permuted_weights=flat_weights[permutation],
        expert_counts=expert_counts,
        expert_offsets=expert_offsets,
    )


def _sample_distinct_experts(
    probabilities: torch.Tensor, num_tokens: int, top_k: int, generator: torch.Generator
) -> torch.Tensor:
    """Sample top-k distinct experts per token on CPU for cross-device repeatability."""

    rows = probabilities.expand(num_tokens, -1)
    return torch.multinomial(rows, top_k, replacement=False, generator=generator)


def generate_router_logits(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    workload: Workload,
    seed: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate deterministic logits with a controlled top-k load pattern.

    Randomness is generated on CPU and copied to ``device``, so a seed defines
    the same routing targets on CPU and CUDA. The workloads are qualitative:
    uniform cycles through experts, skewed uses a long-tailed distribution, and
    hot_expert heavily favors a small leading set of experts.
    """

    if num_tokens < 1 or num_experts < 1:
        raise ValueError("num_tokens and num_experts must be positive")
    if not 1 <= top_k <= num_experts:
        raise ValueError(f"top_k must be in [1, {num_experts}], got {top_k}")
    if workload not in ("uniform", "skewed", "hot_expert", "zipf", "hotspot"):
        raise ValueError(f"unknown workload {workload!r}")
    if not dtype.is_floating_point:
        raise TypeError("dtype must be floating point")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(num_tokens, num_experts, generator=generator, dtype=torch.float32) * 0.05

    normalized_workload = {
        "zipf": "skewed",
        "hotspot": "hot_expert",
    }.get(workload, workload)

    if normalized_workload == "uniform":
        token_ids = torch.arange(num_tokens).unsqueeze(1)
        ranks = torch.arange(top_k).unsqueeze(0)
        targets = (token_ids * top_k + ranks) % num_experts
    elif normalized_workload == "skewed":
        probabilities = torch.arange(1, num_experts + 1, dtype=torch.float32).reciprocal()
        probabilities /= probabilities.sum()
        targets = _sample_distinct_experts(probabilities, num_tokens, top_k, generator)
    else:
        # Keep at least top_k hot experts so most *assignments* (not merely most
        # tokens) can land in the hot set while expert choices remain distinct.
        hot_count = min(num_experts, max(top_k, max(1, num_experts // 8)))
        hot_probability = 0.9
        probabilities = torch.full((num_experts,), (1.0 - hot_probability) / max(1, num_experts - hot_count))
        probabilities[:hot_count] = hot_probability / hot_count
        if hot_count == num_experts:
            probabilities.fill_(1.0 / num_experts)
        targets = _sample_distinct_experts(probabilities, num_tokens, top_k, generator)

    # Descending rank biases make the selected ordering deterministic as well.
    rank_bias = torch.linspace(4.0, 3.0, top_k).expand(num_tokens, -1)
    logits = noise.scatter(1, targets, rank_bias)
    return logits.to(device=device, dtype=dtype)
