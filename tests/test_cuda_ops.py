"""CUDA extension correctness tests; skipped when CUDA or the build is absent."""

from __future__ import annotations

import pytest
import torch

from baseline.routing import generate_router_logits, topk_routing
from fastpath import combine_tokens, cuda_extension_available, dispatch_tokens

CUDA_READY = torch.cuda.is_available() and cuda_extension_available()
pytestmark = pytest.mark.skipif(
    not CUDA_READY, reason="CUDA runtime or compiled fastpath._C extension is unavailable"
)


def _dtype_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-5, 1e-5
    if dtype == torch.float16:
        return 2e-3, 2e-3
    return 1e-2, 1e-2


@pytest.mark.parametrize("top_k", [1, 2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cuda_dispatch_metadata_and_combine(top_k: int, dtype: torch.dtype) -> None:
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("device does not support bfloat16")
    num_tokens, num_experts, hidden_size = 17, 8, 37
    generator = torch.Generator().manual_seed(101)
    hidden = torch.randn(num_tokens, hidden_size, generator=generator).to(
        device="cuda", dtype=dtype
    )
    logits = generate_router_logits(
        num_tokens,
        num_experts,
        top_k,
        "hot_expert",
        seed=102,
        device="cuda",
        dtype=dtype,
    )
    expert_ids, routing_weights = topk_routing(logits, top_k)
    result = dispatch_tokens(hidden, expert_ids, num_experts, backend="cuda_ext")
    torch.cuda.synchronize()

    num_assignments = num_tokens * top_k
    assignments = torch.arange(num_assignments, device="cuda")
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
    expected_stable = torch.argsort(expert_ids.reshape(-1), stable=True)
    torch.testing.assert_close(result.permuted_to_assignment, expected_stable)
    expected_hidden = hidden[
        torch.div(result.permuted_to_assignment, top_k, rounding_mode="floor")
    ]
    torch.testing.assert_close(result.permuted_hidden, expected_hidden, atol=0, rtol=0)

    # Use position-dependent values so mapping errors cannot be hidden by the
    # duplicated input token rows produced by dispatch.
    expert_outputs = torch.randn(
        num_assignments, hidden_size, device="cuda", dtype=dtype
    )
    actual = combine_tokens(
        expert_outputs,
        result.assignment_to_permuted,
        routing_weights,
        backend="cuda_ext",
    )
    expected = combine_tokens(
        expert_outputs,
        result.assignment_to_permuted,
        routing_weights,
        backend="torch",
    )
    atol, rtol = _dtype_tolerances(dtype)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def test_cuda_tokens_one_and_empty_experts() -> None:
    hidden = torch.randn(1, 29, device="cuda", dtype=torch.float32)
    expert_ids = torch.tensor([[0, 1, 2, 3]], device="cuda", dtype=torch.long)
    result = dispatch_tokens(hidden, expert_ids, 256, backend="cuda_ext")
    torch.cuda.synchronize()
    assert result.expert_counts[:4].tolist() == [1, 1, 1, 1]
    assert result.expert_counts[4:].count_nonzero().item() == 0
    assert result.permuted_hidden.shape == (4, 29)


def test_cuda_rejects_noncontiguous_and_cpu_inputs() -> None:
    hidden = torch.randn(4, 10, device="cuda")
    expert_ids = torch.zeros(4, 1, device="cuda", dtype=torch.long)
    with pytest.raises(ValueError, match="contiguous"):
        dispatch_tokens(hidden[:, ::2], expert_ids, 4, backend="cuda_ext")
    with pytest.raises(ValueError, match="requires CUDA tensors"):
        dispatch_tokens(hidden.cpu(), expert_ids.cpu(), 4, backend="cuda_ext")


def test_cuda_ops_follow_current_stream() -> None:
    hidden = torch.randn(6, 19, device="cuda")
    expert_ids = torch.tensor(
        [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3]],
        device="cuda",
        dtype=torch.long,
    )
    routing_weights = torch.full((6, 2), 0.5, device="cuda")
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        dispatch = dispatch_tokens(hidden, expert_ids, 4, backend="cuda_ext")
        output = combine_tokens(
            dispatch.permuted_hidden,
            dispatch.assignment_to_permuted,
            routing_weights,
            backend="cuda_ext",
        )
    stream.synchronize()
    # Each assignment's expert output equals its source hidden row, so 0.5 +
    # 0.5 reconstructs hidden exactly if both kernels ran on the chosen stream.
    torch.testing.assert_close(output, hidden, atol=1e-6, rtol=1e-6)
