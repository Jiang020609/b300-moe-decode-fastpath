"""CPU tests for the Goal 1C hybrid GEMM dispatch policy and executor."""

from __future__ import annotations

import pytest
import torch

from fastpath import b300_moe_forward
from fastpath.gemm_dispatch import (
    normalize_gemm_policy,
    per_expert_gemm_out,
    select_gemm_strategy,
)


# ---------------------------------------------------------------------------
# Policy layer: pure decision logic against the measured B300 rules.
# ---------------------------------------------------------------------------


def test_forced_policies_are_honored_verbatim() -> None:
    for operation in ("gate_up", "down"):
        assert (
            select_gemm_strategy(
                operation, total_rows=64, active_experts=64, policy="grouped"
            )
            == "grouped"
        )
        assert (
            select_gemm_strategy(
                operation, total_rows=64, active_experts=64, policy="per_expert"
            )
            == "per_expert"
        )


def test_auto_gate_up_never_selects_grouped_in_measured_range() -> None:
    # Measured: gate_up matmul won all 14 shapes up to M=64 (best grouped
    # ratio 0.873x), so auto keeps the loop everywhere in that range.
    for rows in (1, 2, 8, 32, 64):
        assert (
            select_gemm_strategy(
                "gate_up", total_rows=rows, active_experts=min(rows, 64)
            )
            == "per_expert"
        )


def test_auto_down_crossover_matches_measurements() -> None:
    # Measured crossover: down, >=32 rows spread over >=32 active experts.
    assert (
        select_gemm_strategy("down", total_rows=32, active_experts=32)
        == "grouped"
    )
    assert (
        select_gemm_strategy("down", total_rows=64, active_experts=64)
        == "grouped"
    )
    # Hotspot routing concentrates rows on few experts: loop still wins.
    assert (
        select_gemm_strategy("down", total_rows=32, active_experts=8)
        == "per_expert"
    )
    assert (
        select_gemm_strategy("down", total_rows=16, active_experts=16)
        == "per_expert"
    )


def test_auto_beyond_measured_range_defers_to_grouped() -> None:
    for operation in ("gate_up", "down"):
        assert (
            select_gemm_strategy(operation, total_rows=256, active_experts=64)
            == "grouped"
        )


def test_policy_validation() -> None:
    with pytest.raises(ValueError, match="operation"):
        select_gemm_strategy("up", total_rows=1, active_experts=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="total_rows"):
        select_gemm_strategy("down", total_rows=1.0, active_experts=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="active_experts cannot exceed"):
        select_gemm_strategy("down", total_rows=2, active_experts=3)
    with pytest.raises(ValueError, match="gemm_policy"):
        normalize_gemm_policy("hybrid")
    with pytest.raises(TypeError, match="gemm_policy"):
        normalize_gemm_policy(None)


# ---------------------------------------------------------------------------
# Executor layer: numerical equivalence with per-expert oracle on CPU.
# ---------------------------------------------------------------------------


def _grouped_case(
    counts: list[int], k: int = 16, n: int = 24, seed: int = 1301
) -> tuple[torch.Tensor, torch.Tensor, list[int], torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    rows = sum(counts)
    num_experts = len(counts)
    inputs = torch.randn(rows, k, generator=generator)
    weights = torch.randn(num_experts, k, n, generator=generator)
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    out = torch.full((rows, n), float("nan"))
    return inputs, weights, offsets, out


def test_per_expert_gemm_matches_oracle_with_empty_experts() -> None:
    counts = [3, 0, 1, 0, 0, 5, 2, 0]
    inputs, weights, offsets, out = _grouped_case(counts)
    per_expert_gemm_out(inputs, weights, offsets, out)
    for expert, _ in enumerate(counts):
        start, end = offsets[expert], offsets[expert + 1]
        expected = inputs[start:end] @ weights[expert]
        assert torch.equal(out[start:end], expected)
    assert not torch.isnan(out).any()


def test_per_expert_gemm_consumes_transpose_view_like_grouped_path() -> None:
    # The orchestration hands over gate_up_weight.transpose(1, 2); verify the
    # loop works on that zero-copy non-contiguous view.
    counts = [2, 0, 4]
    inputs, _, offsets, out = _grouped_case(counts, k=8, n=12)
    public_weight = torch.randn(len(counts), 12, 8)  # public [E, N, K]
    packed = public_weight.transpose(1, 2)
    assert not packed.is_contiguous()
    per_expert_gemm_out(inputs, packed, offsets, out)
    for expert, _ in enumerate(counts):
        start, end = offsets[expert], offsets[expert + 1]
        expected = inputs[start:end] @ public_weight[expert].t()
        assert torch.allclose(out[start:end], expected)


def test_per_expert_gemm_validates_offsets_and_shapes() -> None:
    inputs, weights, offsets, out = _grouped_case([2, 2])
    with pytest.raises(ValueError, match="length num_experts"):
        per_expert_gemm_out(inputs, weights, offsets[:-1], out)
    with pytest.raises(ValueError, match="scan from 0"):
        per_expert_gemm_out(inputs, weights, [0, 2, 3], out)
    with pytest.raises(ValueError, match="non-decreasing"):
        # Intermediate offset overshoots total rows; must fail before any
        # matmul touches an out-of-range slice.
        per_expert_gemm_out(inputs, weights, [0, 5, 4], out)
    with pytest.raises(ValueError, match=r"out must be \[rows, N\]"):
        per_expert_gemm_out(inputs, weights, offsets, out[:, :-1])
    with pytest.raises(ValueError, match="feature dim"):
        per_expert_gemm_out(inputs[:, :-1], weights, offsets, out)


# ---------------------------------------------------------------------------
# API layer: gemm_policy is validated on every backend; torch backend
# computes densely and reports no gemm_dispatch metadata.
# ---------------------------------------------------------------------------


def _forward_case() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(1303)
    num_tokens, num_experts, top_k, hidden, inter = 3, 4, 2, 8, 6
    hidden_states = torch.randn(num_tokens, hidden, generator=generator)
    indices = torch.stack(
        [(torch.arange(num_tokens) + rank) % num_experts for rank in range(top_k)],
        dim=1,
    ).long()
    routing_weights = torch.randn(num_tokens, top_k, generator=generator)
    gate_up = torch.randn(num_experts, 2 * inter, hidden, generator=generator)
    down = torch.randn(num_experts, hidden, inter, generator=generator)
    return hidden_states, indices, routing_weights, gate_up, down


def test_forward_accepts_all_policies_on_torch_backend() -> None:
    tensors = _forward_case()
    baseline = b300_moe_forward(*tensors, num_experts=4, top_k=2)
    for policy in ("auto", "grouped", "per_expert"):
        output, metadata = b300_moe_forward(
            *tensors,
            num_experts=4,
            top_k=2,
            gemm_policy=policy,
            return_metadata=True,
        )
        assert torch.equal(output, baseline)
        assert "gemm_dispatch" not in metadata


def test_forward_rejects_unknown_policy() -> None:
    tensors = _forward_case()
    with pytest.raises(ValueError, match="gemm_policy"):
        b300_moe_forward(*tensors, num_experts=4, top_k=2, gemm_policy="fastest")
    with pytest.raises(TypeError, match="gemm_policy"):
        b300_moe_forward(*tensors, num_experts=4, top_k=2, gemm_policy=1)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# B300 layer: the compiled BF16 path must produce equivalent outputs under
# every policy and report the chosen strategies truthfully.
# ---------------------------------------------------------------------------


def _goal1b_ready() -> bool:
    from fastpath import cuda_extension_available

    if not torch.cuda.is_available() or not cuda_extension_available():
        return False
    if torch.cuda.get_device_capability() != (10, 3):
        return False
    try:
        import fastpath._C as extension

        return extension.build_info().get("goal") == "1B"
    except (ImportError, OSError, AttributeError, RuntimeError):
        return False


@pytest.mark.skipif(
    not _goal1b_ready(),
    reason="requires a compiled Goal 1B extension on a CC 10.3 B300 GPU",
)
@pytest.mark.parametrize("num_tokens", [1, 4, 16, 64])
def test_b300_policies_agree_and_report_dispatch(num_tokens: int) -> None:
    generator = torch.Generator().manual_seed(1307 + num_tokens)
    num_experts, top_k, hidden, inter = 64, 2, 256, 512
    device = torch.device("cuda")
    hidden_states = torch.randn(
        num_tokens, hidden, generator=generator
    ).to(device=device, dtype=torch.bfloat16)
    indices = (
        torch.randint(0, num_experts, (num_tokens, top_k), generator=generator)
        .long()
        .to(device)
    )
    routing_weights = (
        torch.rand(num_tokens, top_k, generator=generator)
        .to(device=device, dtype=torch.bfloat16)
    )
    gate_up = torch.randn(
        num_experts, 2 * inter, hidden, generator=generator
    ).to(device=device, dtype=torch.bfloat16) * 0.05
    down = torch.randn(
        num_experts, hidden, inter, generator=generator
    ).to(device=device, dtype=torch.bfloat16) * 0.05

    outputs: dict[str, torch.Tensor] = {}
    for policy in ("grouped", "per_expert", "auto"):
        output, metadata = b300_moe_forward(
            hidden_states,
            indices,
            routing_weights,
            gate_up.contiguous(),
            down.contiguous(),
            num_experts=num_experts,
            top_k=top_k,
            backend="cutlass_bf16",
            gemm_policy=policy,
            return_metadata=True,
        )
        outputs[policy] = output
        dispatch = metadata["gemm_dispatch"]
        assert dispatch["policy"] == policy
        assert dispatch["total_rows"] == num_tokens * top_k
        if policy == "grouped":
            assert dispatch["gate_up_strategy"] == "grouped"
            assert dispatch["down_strategy"] == "grouped"
            assert dispatch["active_experts"] is None
        else:
            assert dispatch["active_experts"] >= 1
        if policy == "per_expert":
            assert dispatch["gate_up_strategy"] == "per_expert"
            assert dispatch["down_strategy"] == "per_expert"

    # Both strategies consume identical weight memory; only kernel-level
    # accumulation order differs, so tolerances stay tight for BF16.
    assert torch.allclose(
        outputs["per_expert"].float(),
        outputs["grouped"].float(),
        atol=0.12,
        rtol=0.06,
    )
    assert torch.allclose(
        outputs["auto"].float(), outputs["grouped"].float(), atol=0.12, rtol=0.06
    )
