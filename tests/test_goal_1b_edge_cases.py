"""Validation, backend-selection, and workspace tests for Goal 1B."""

from __future__ import annotations

import pytest
import torch

from fastpath import B300MoEWorkspace, b300_moe_forward


def _case(
    *, num_tokens: int = 3, top_k: int = 2, hidden_size: int = 5
) -> tuple[torch.Tensor, ...]:
    num_experts, intermediate_size = 8, 7
    generator = torch.Generator().manual_seed(811 + num_tokens + top_k)
    hidden = torch.randn(num_tokens, hidden_size, generator=generator)
    indices = torch.stack(
        [(torch.arange(num_tokens) + rank) % num_experts for rank in range(top_k)],
        dim=1,
    ).long()
    routing_weights = torch.randn(num_tokens, top_k, generator=generator)
    gate_up = torch.randn(
        num_experts, 2 * intermediate_size, hidden_size, generator=generator
    )
    down = torch.randn(
        num_experts, hidden_size, intermediate_size, generator=generator
    )
    return hidden, indices, routing_weights, gate_up, down


def _forward(tensors: tuple[torch.Tensor, ...], **kwargs: object) -> torch.Tensor:
    return b300_moe_forward(
        *tensors,
        num_experts=8,
        top_k=tensors[1].shape[1],
        backend="torch",
        quant_mode="none",
        **kwargs,
    )


def test_strict_shape_dtype_contiguous_and_index_validation() -> None:
    hidden, indices, weights, gate_up, down = _case()
    with pytest.raises(ValueError, match="same num_tokens"):
        _forward((hidden[:-1], indices, weights, gate_up, down))
    with pytest.raises(TypeError, match="int64"):
        _forward((hidden, indices.int(), weights, gate_up, down))
    with pytest.raises(TypeError, match="same dtype"):
        _forward((hidden, indices, weights.half(), gate_up, down))
    with pytest.raises(ValueError, match="out-of-range"):
        _forward((hidden, indices.clone().fill_(8), weights, gate_up, down))
    with pytest.raises(ValueError, match="contiguous"):
        noncontiguous = torch.randn(hidden.shape[1], hidden.shape[0]).t()
        assert not noncontiguous.is_contiguous()
        _forward((noncontiguous, indices, weights, gate_up, down))
    with pytest.raises(ValueError, match="positive even"):
        _forward((hidden, indices, weights, gate_up[:, :-1], down))
    with pytest.raises(ValueError, match=r"\[E, H, I\]"):
        _forward((hidden, indices, weights, gate_up, down[:, :-1]))


def test_top_k_and_declared_dimensions_must_match() -> None:
    tensors = _case(top_k=2)
    with pytest.raises(ValueError, match="top_k must match"):
        b300_moe_forward(
            *tensors,
            num_experts=8,
            top_k=4,
            backend="torch",
            quant_mode="none",
        )
    with pytest.raises(TypeError, match="num_experts"):
        b300_moe_forward(
            *tensors,
            num_experts=True,
            top_k=2,
            backend="torch",
            quant_mode="none",
        )
    three_way = _case(top_k=2)
    bad_indices = torch.zeros(3, 3, dtype=torch.int64)
    bad_weights = torch.ones(3, 3)
    with pytest.raises(ValueError, match="top_k"):
        b300_moe_forward(
            three_way[0],
            bad_indices,
            bad_weights,
            three_way[3],
            three_way[4],
            num_experts=8,
            top_k=3,
            backend="torch",
            quant_mode="none",
        )


@pytest.mark.parametrize("backend", ["cutlass_bf16", "cutlass_nvfp4"])
def test_compiled_backend_on_cpu_is_rejected_without_fallback(backend: str) -> None:
    tensors = _case()
    quant_mode = "bf16" if backend == "cutlass_bf16" else "nvfp4"
    with pytest.raises(ValueError, match="requires CUDA tensors"):
        b300_moe_forward(
            *tensors,
            num_experts=8,
            top_k=2,
            backend=backend,
            quant_mode=quant_mode,
        )


def test_backend_and_quant_mode_cannot_silently_change() -> None:
    tensors = _case()
    with pytest.raises(ValueError, match="cannot execute NVFP4"):
        b300_moe_forward(
            *tensors,
            num_experts=8,
            top_k=2,
            backend="torch",
            quant_mode="nvfp4",
        )
    with pytest.raises(ValueError, match="requires quant_mode='bf16'"):
        b300_moe_forward(
            *tensors,
            num_experts=8,
            top_k=2,
            backend="cutlass_bf16",
            quant_mode="none",
        )


def test_workspace_reuses_grows_and_rebinds_shape() -> None:
    workspace = B300MoEWorkspace(capacity_tokens=4, device="cpu")
    first = _case(num_tokens=3, top_k=2)
    output1, metadata1 = b300_moe_forward(
        *first,
        num_experts=8,
        top_k=2,
        backend="torch",
        quant_mode="none",
        workspace=workspace,
        return_metadata=True,
    )
    first_ptrs = workspace.buffer_data_ptrs()
    assert workspace.allocation_count == 1
    assert workspace.reuse_count == 0
    assert workspace.capacity_tokens == 4
    assert workspace.capacity_assignments == 8
    assert workspace.capacity_bytes > 0
    assert workspace.last_stream is None

    output2, metadata2 = b300_moe_forward(
        *first,
        num_experts=8,
        top_k=2,
        backend="torch",
        quant_mode="none",
        workspace=workspace,
        return_metadata=True,
    )
    torch.testing.assert_close(output2, output1)
    assert workspace.buffer_data_ptrs() == first_ptrs
    assert workspace.allocation_count == 1
    assert workspace.reuse_count == 1
    assert metadata2["workspace"]["reused"] is True

    # Metadata owns its routing tensors; a later reuse cannot mutate it.
    counts_snapshot = metadata1["expert_counts"].clone()
    changed = list(first)
    changed[1] = torch.full_like(first[1], 7)
    b300_moe_forward(
        *changed,
        num_experts=8,
        top_k=2,
        backend="torch",
        quant_mode="none",
        workspace=workspace,
    )
    torch.testing.assert_close(metadata1["expert_counts"], counts_snapshot)

    grown = _case(num_tokens=6, top_k=2)
    _forward(grown, workspace=workspace)
    assert workspace.allocation_count == 2
    assert workspace.capacity_tokens >= 6

    reshaped = _case(num_tokens=2, top_k=1, hidden_size=9)
    _forward(reshaped, workspace=workspace)
    assert workspace.allocation_count == 3
    assert workspace.last_shape == (2, 8, 1, 9, 7)


def test_invalid_call_does_not_mutate_workspace_and_device_is_bound() -> None:
    workspace = B300MoEWorkspace(device="cpu")
    tensors = _case()
    bad = list(tensors)
    bad[1] = torch.full_like(tensors[1], 8)
    with pytest.raises(ValueError, match="out-of-range"):
        _forward(tuple(bad), workspace=workspace)
    assert workspace.allocation_count == 0
    assert workspace.reuse_count == 0
    with pytest.raises(ValueError, match="bound to device"):
        workspace.reserve(
            num_tokens=1,
            num_experts=8,
            top_k=1,
            hidden_size=4,
            intermediate_size=6,
            device="meta",
            dtype=torch.float32,
        )


def test_workspace_is_inference_only_and_return_flag_is_strict() -> None:
    tensors = list(_case())
    tensors[0].requires_grad_(True)
    workspace = B300MoEWorkspace()
    with pytest.raises(ValueError, match="inference-only"):
        _forward(tuple(tensors), workspace=workspace)
    assert workspace.allocation_count == 0
    with pytest.raises(TypeError, match="return_metadata"):
        b300_moe_forward(
            *_case(),
            num_experts=8,
            top_k=2,
            backend="torch",
            quant_mode="none",
            return_metadata=1,  # type: ignore[arg-type]
        )
