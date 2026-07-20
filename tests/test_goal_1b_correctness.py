"""CPU numerical contract for the Goal 1B MoE forward API."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from fastpath import b300_moe_forward, routed_moe_reference


def _case(
    *,
    num_tokens: int = 5,
    num_experts: int = 8,
    top_k: int = 2,
    hidden_size: int = 7,
    intermediate_size: int = 11,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(701 + top_k + num_tokens)
    hidden = torch.randn(num_tokens, hidden_size, generator=generator).to(dtype)
    indices = torch.stack(
        [
            (torch.arange(num_tokens) * max(1, top_k) + rank) % num_experts
            for rank in range(top_k)
        ],
        dim=1,
    ).long()
    routing_weights = (
        torch.randn(num_tokens, top_k, generator=generator) * 1.7 + 0.25
    ).to(dtype)
    gate = torch.randn(
        num_experts, intermediate_size, hidden_size, generator=generator
    ).to(dtype)
    up = torch.randn(
        num_experts, intermediate_size, hidden_size, generator=generator
    ).to(dtype)
    gate_up = torch.cat([gate, up], dim=1).contiguous()
    down = torch.randn(
        num_experts, hidden_size, intermediate_size, generator=generator
    ).to(dtype)
    return hidden, indices, routing_weights, gate_up, down


def _naive_explicit(
    hidden: torch.Tensor,
    indices: torch.Tensor,
    routing_weights: torch.Tensor,
    gate_up: torch.Tensor,
    down: torch.Tensor,
) -> torch.Tensor:
    intermediate_size = gate_up.shape[1] // 2
    output = hidden.new_zeros(hidden.shape)
    for token in range(hidden.shape[0]):
        state = hidden[token : token + 1]
        for rank in range(indices.shape[1]):
            expert = int(indices[token, rank])
            gate = F.linear(state, gate_up[expert, :intermediate_size])
            up = F.linear(state, gate_up[expert, intermediate_size:])
            expert_output = F.linear(F.silu(gate) * up, down[expert]).squeeze(0)
            output[token] += routing_weights[token, rank] * expert_output
    return output


@pytest.mark.parametrize("top_k", [1, 2, 4, 8])
def test_torch_backend_matches_independent_explicit_oracle(top_k: int) -> None:
    tensors = _case(top_k=top_k)
    actual = b300_moe_forward(
        *tensors,
        num_experts=8,
        top_k=top_k,
        quant_mode="none",
        backend="torch",
    )
    expected = _naive_explicit(*tensors)
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)


def test_reference_and_forward_preserve_unnormalized_weights() -> None:
    hidden, indices, _, gate_up, down = _case(num_tokens=3, top_k=2)
    weights = torch.tensor([[2.0, -0.5], [0.0, 3.25], [1.0, 1.0]])
    tensors = (hidden, indices, weights, gate_up, down)
    expected = _naive_explicit(*tensors)
    reference = routed_moe_reference(*tensors, num_experts=8, top_k=2)
    actual = b300_moe_forward(
        *tensors,
        num_experts=8,
        top_k=2,
        backend="torch",
        quant_mode="none",
    )
    torch.testing.assert_close(reference, expected, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)


def test_metadata_reports_actual_torch_execution() -> None:
    tensors = _case(num_tokens=1, top_k=4)
    output, metadata = b300_moe_forward(
        *tensors,
        num_experts=8,
        top_k=4,
        backend="torch",
        quant_mode="none",
        return_metadata=True,
    )
    assert output.shape == tensors[0].shape
    assert metadata["backend"] == "torch"
    assert metadata["architecture"] == "cpu"
    assert metadata["quant_mode"] == "none"
    assert metadata["used_fallback"] is False
    assert metadata["expert_counts"].sum().item() == 4
    assert metadata["expert_offsets"].shape == (9,)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_cpu_reference(dtype: torch.dtype) -> None:
    tensors = _case(num_tokens=3, top_k=2, dtype=dtype)
    quant_mode = "bf16" if dtype == torch.bfloat16 else "none"
    actual = b300_moe_forward(
        *tensors,
        num_experts=8,
        top_k=2,
        backend="torch",
        quant_mode=quant_mode,
    )
    expected = _naive_explicit(*tensors)
    torch.testing.assert_close(actual, expected, atol=0.2, rtol=0.02)


def test_torch_reference_preserves_autograd_without_workspace() -> None:
    hidden, indices, weights, gate_up, down = _case(num_tokens=2, top_k=2)
    hidden.requires_grad_(True)
    gate_up.requires_grad_(True)
    down.requires_grad_(True)
    output = b300_moe_forward(
        hidden,
        indices,
        weights,
        gate_up,
        down,
        num_experts=8,
        top_k=2,
        backend="torch",
        quant_mode="none",
    )
    output.square().sum().backward()
    assert hidden.grad is not None
    assert gate_up.grad is not None
    assert down.grad is not None


@pytest.mark.parametrize("num_tokens", [1, 2, 4, 8, 16, 32, 64])
def test_decode_token_count_matrix_runs_on_reference(num_tokens: int) -> None:
    tensors = _case(
        num_tokens=num_tokens,
        num_experts=8,
        top_k=2,
        hidden_size=3,
        intermediate_size=5,
    )
    actual = b300_moe_forward(
        *tensors,
        num_experts=8,
        top_k=2,
        backend="torch",
        quant_mode="none",
    )
    expected = _naive_explicit(*tensors)
    assert actual.shape == (num_tokens, 3)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize("num_experts", [8, 16, 64, 128])
def test_dynamic_expert_count_matrix_including_many_empty_experts(
    num_experts: int,
) -> None:
    tensors = _case(
        num_tokens=2,
        num_experts=num_experts,
        top_k=2,
        hidden_size=3,
        intermediate_size=5,
    )
    actual, metadata = b300_moe_forward(
        *tensors,
        num_experts=num_experts,
        top_k=2,
        backend="torch",
        quant_mode="none",
        return_metadata=True,
    )
    expected = _naive_explicit(*tensors)
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
    assert metadata["expert_counts"].shape == (num_experts,)
    assert int((metadata["expert_counts"] == 0).sum()) >= num_experts - 4


def test_random_routing_with_normalized_weights_matches_oracle() -> None:
    tensors = list(
        _case(
            num_tokens=16,
            num_experts=16,
            top_k=4,
            hidden_size=5,
            intermediate_size=9,
        )
    )
    generator = torch.Generator().manual_seed(4242)
    tensors[1] = torch.randint(0, 16, (16, 4), generator=generator)
    tensors[2] = torch.softmax(torch.randn(16, 4, generator=generator), dim=-1)
    inputs = tuple(tensors)
    actual = b300_moe_forward(
        *inputs,
        num_experts=16,
        top_k=4,
        backend="torch",
        quant_mode="none",
    )
    expected = _naive_explicit(*inputs)
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
