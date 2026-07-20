"""CPU tests for the backend-neutral dispatch/combine API."""

from __future__ import annotations

import pytest
import torch

from fastpath import combine_tokens, dispatch_tokens


def _assert_dispatch_semantics(
    hidden_states: torch.Tensor,
    expert_ids: torch.Tensor,
    num_experts: int,
) -> None:
    result = dispatch_tokens(hidden_states, expert_ids, num_experts, backend="torch")
    num_tokens, top_k = expert_ids.shape
    num_assignments = num_tokens * top_k
    assignments = torch.arange(num_assignments, device=expert_ids.device)

    expected_counts = torch.bincount(expert_ids.reshape(-1), minlength=num_experts)
    torch.testing.assert_close(result.expert_counts, expected_counts)
    torch.testing.assert_close(
        result.expert_offsets,
        torch.cat([expected_counts.new_zeros(1), expected_counts.cumsum(0)]),
    )
    torch.testing.assert_close(
        result.assignment_to_permuted[result.permuted_to_assignment], assignments
    )
    torch.testing.assert_close(
        result.permuted_to_assignment[result.assignment_to_permuted], assignments
    )
    grouped_experts = expert_ids.reshape(-1)[result.permuted_to_assignment]
    assert torch.all(grouped_experts[:-1] <= grouped_experts[1:])
    expected_hidden = hidden_states[
        torch.div(result.permuted_to_assignment, top_k, rounding_mode="floor")
    ]
    torch.testing.assert_close(result.permuted_hidden, expected_hidden)


@pytest.mark.parametrize("top_k", [1, 2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_torch_dispatch_and_combine(top_k: int, dtype: torch.dtype) -> None:
    generator = torch.Generator().manual_seed(41)
    hidden = torch.randn(9, 13, generator=generator).to(dtype)
    expert_ids = torch.stack(
        [(torch.arange(9) * top_k + slot) % 8 for slot in range(top_k)], dim=1
    ).long()
    _assert_dispatch_semantics(hidden, expert_ids, num_experts=8)

    dispatch = dispatch_tokens(hidden, expert_ids, 8, backend="torch")
    expert_outputs = dispatch.permuted_hidden * 0.75
    routing_weights = torch.softmax(
        torch.randn(9, top_k, generator=generator), dim=-1
    ).to(dtype)
    actual = combine_tokens(
        expert_outputs,
        dispatch.assignment_to_permuted,
        routing_weights,
        backend="torch",
    )
    assignment_order = expert_outputs[dispatch.assignment_to_permuted]
    expected = (
        assignment_order.reshape(9, top_k, 13).float()
        * routing_weights.float().unsqueeze(-1)
    ).sum(1).to(dtype)
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)


def test_empty_experts_tokens_one_and_alias_backend_validation() -> None:
    hidden = torch.randn(1, 7)
    expert_ids = torch.tensor([[2, 3]], dtype=torch.long)
    result = dispatch_tokens(hidden, expert_ids, 8)
    assert result.expert_counts.tolist() == [0, 0, 1, 1, 0, 0, 0, 0]
    with pytest.raises(ValueError, match="requires CUDA tensors"):
        dispatch_tokens(hidden, expert_ids, 8, backend="cuda")


def test_invalid_dispatch_inputs_are_rejected() -> None:
    hidden = torch.randn(4, 8)
    ids = torch.zeros(4, 2, dtype=torch.long)
    with pytest.raises(ValueError, match=r"\[4, 256\]"):
        dispatch_tokens(hidden, ids, 3)
    with pytest.raises(ValueError, match="top_k"):
        dispatch_tokens(hidden, torch.zeros(4, 3, dtype=torch.long), 4)
    with pytest.raises(ValueError, match="out-of-range"):
        dispatch_tokens(hidden, ids.fill_(4), 4)
    with pytest.raises(ValueError, match="contiguous"):
        dispatch_tokens(hidden[:, ::2], ids, 4)


def test_invalid_combine_mapping_is_rejected() -> None:
    outputs = torch.randn(4, 5)
    weights = torch.full((2, 2), 0.5)
    with pytest.raises(ValueError, match="permutation"):
        combine_tokens(outputs, torch.tensor([0, 0, 2, 3]), weights)
    with pytest.raises(ValueError, match="out-of-range"):
        combine_tokens(outputs, torch.tensor([0, 1, 2, 4]), weights)
