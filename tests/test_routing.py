"""Tests for routing, grouping metadata, and synthetic workloads."""

from __future__ import annotations

import pytest
import torch

from baseline.routing import generate_router_logits, group_tokens, topk_routing


@pytest.mark.parametrize("top_k", [1, 2, 4])
def test_routing_weights_are_normalized(top_k: int) -> None:
    logits = generate_router_logits(32, 8, top_k, "skewed", seed=17)
    indices, weights = topk_routing(logits, top_k)
    assert indices.shape == (32, top_k)
    assert weights.shape == (32, top_k)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(32), atol=1e-6, rtol=1e-6)
    assert torch.all(weights >= 0)


def test_permutation_and_reverse_mapping_round_trip() -> None:
    logits = generate_router_logits(13, 7, 4, "uniform", seed=8)
    indices, weights = topk_routing(logits, 4)
    routing = group_tokens(indices, weights, 7)

    original_assignments = torch.arange(indices.numel()).reshape(indices.shape)
    grouped = original_assignments.reshape(-1)[routing.permutation]
    restored = grouped[routing.reverse_mapping]
    torch.testing.assert_close(restored, original_assignments.reshape(-1))
    assert routing.expert_offsets.shape == (8,)
    assert routing.expert_offsets[0].item() == 0
    assert routing.expert_offsets[-1].item() == indices.numel()
    torch.testing.assert_close(routing.expert_counts.sum(), torch.tensor(indices.numel()))
    assert torch.all(routing.permuted_expert_indices[:-1] <= routing.permuted_expert_indices[1:])


@pytest.mark.parametrize("workload", ["uniform", "skewed", "hot_expert"])
def test_workload_is_seed_reproducible(workload: str) -> None:
    first = generate_router_logits(64, 16, 2, workload, seed=99)
    second = generate_router_logits(64, 16, 2, workload, seed=99)
    different = generate_router_logits(64, 16, 2, workload, seed=100)
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    assert not torch.equal(first, different)


def test_uniform_workload_balances_assignments() -> None:
    logits = generate_router_logits(64, 8, 2, "uniform", seed=3)
    indices, _ = topk_routing(logits, 2)
    counts = torch.bincount(indices.reshape(-1), minlength=8)
    assert int(counts.max() - counts.min()) <= 1


def test_skewed_and_hot_workloads_are_imbalanced() -> None:
    skewed, _ = topk_routing(generate_router_logits(2048, 16, 1, "skewed", seed=4), 1)
    skewed_counts = torch.bincount(skewed.reshape(-1), minlength=16)
    assert skewed_counts[0] > 5 * skewed_counts[-1]

    hot, _ = topk_routing(generate_router_logits(2048, 16, 1, "hot_expert", seed=4), 1)
    hot_counts = torch.bincount(hot.reshape(-1), minlength=16)
    assert hot_counts[:2].sum().float() / hot_counts.sum() > 0.8


@pytest.mark.parametrize(
    ("shape", "top_k", "message"),
    [((2, 3, 4), 1, "shape"), ((2, 3), 4, "top_k")],
)
def test_invalid_routing_inputs(shape: tuple[int, ...], top_k: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        topk_routing(torch.randn(shape), top_k)


def test_unknown_workload_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown workload"):
        generate_router_logits(2, 4, 1, "not-a-workload", seed=1)  # type: ignore[arg-type]
