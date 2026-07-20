"""Routing-contract tests for the Goal 1B explicit-route API."""

from __future__ import annotations

import pytest
import torch

from fastpath import build_routing_metadata


@pytest.mark.parametrize("top_k", [1, 2, 4, 8])
def test_goal_1b_routing_is_stable_and_invertible(top_k: int) -> None:
    num_tokens, num_experts = 7, 8
    indices = torch.stack(
        [(torch.arange(num_tokens) * 3 + rank) % num_experts for rank in range(top_k)],
        dim=1,
    ).long()
    # Deliberately neither normalized nor restricted to probabilities.
    weights = torch.arange(1, indices.numel() + 1, dtype=torch.float32).reshape_as(
        indices
    )
    routing = build_routing_metadata(indices, weights, num_experts)

    assignments = torch.arange(indices.numel())
    expected_permutation = torch.argsort(indices.reshape(-1), stable=True)
    torch.testing.assert_close(routing.permutation, expected_permutation)
    torch.testing.assert_close(
        routing.permutation[routing.reverse_mapping], assignments
    )
    torch.testing.assert_close(
        routing.reverse_mapping[routing.permutation], assignments
    )
    torch.testing.assert_close(
        routing.permuted_weights, weights.reshape(-1)[expected_permutation]
    )

    expected_counts = torch.bincount(indices.reshape(-1), minlength=num_experts)
    torch.testing.assert_close(routing.expert_counts, expected_counts)
    torch.testing.assert_close(
        routing.expert_offsets,
        torch.cat([expected_counts.new_zeros(1), expected_counts.cumsum(0)]),
    )
    assert int(routing.expert_offsets[-1]) == num_tokens * top_k


def test_goal_1b_routing_allows_repeated_routes_and_empty_experts() -> None:
    indices = torch.tensor([[3, 3], [3, 3], [3, 3]], dtype=torch.int64)
    weights = torch.tensor([[2.0, -0.5], [0.0, 4.0], [1.5, 0.25]])
    routing = build_routing_metadata(indices, weights, num_experts=8)

    assert routing.expert_counts.tolist() == [0, 0, 0, 6, 0, 0, 0, 0]
    assert routing.expert_offsets.tolist() == [0, 0, 0, 0, 6, 6, 6, 6, 6]
    torch.testing.assert_close(routing.permutation, torch.arange(6))
    torch.testing.assert_close(routing.permuted_weights, weights.reshape(-1))


def test_goal_1b_routing_rejects_bad_contracts() -> None:
    indices = torch.zeros(3, 2, dtype=torch.int64)
    weights = torch.ones(3, 2)
    with pytest.raises(TypeError, match="int64"):
        build_routing_metadata(indices.to(torch.int32), weights, 4)
    with pytest.raises(ValueError, match="same shape"):
        build_routing_metadata(indices, weights[:, :1], 4)
    with pytest.raises(ValueError, match="out-of-range"):
        build_routing_metadata(indices.fill_(4), weights, 4)
    with pytest.raises(ValueError, match="top_k"):
        build_routing_metadata(torch.zeros(2, 3, dtype=torch.int64), torch.ones(2, 3), 4)
    with pytest.raises(ValueError, match="contiguous"):
        build_routing_metadata(
            torch.zeros(2, 4, dtype=torch.int64)[:, ::2],
            torch.ones(2, 4)[:, ::2],
            4,
        )
