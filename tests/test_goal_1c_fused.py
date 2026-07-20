"""Goal 1C step 2: fused device-offset grouped GEMM strategy tests.

Layer 1 (CPU): policy plumbing for the new "fused" value.
Layer 2 (CPU): the torch backend stays dense and policy-agnostic.
Layer 3 (B300-gated): the compiled kernel must match a per-expert matmul
oracle directly, and the end-to-end fused forward must agree with the
validated grouped baseline while reporting its dispatch truthfully.
"""

from __future__ import annotations

import pytest
import torch

from fastpath import b300_moe_forward
from fastpath.gemm_dispatch import (
    normalize_gemm_policy,
    select_gemm_strategy,
)

# ---------------------------------------------------------------------------
# Policy layer
# ---------------------------------------------------------------------------


def test_normalize_accepts_fused() -> None:
    assert normalize_gemm_policy("fused") == "fused"


def test_forced_fused_policy_selects_fused_everywhere() -> None:
    for operation in ("gate_up", "down"):
        for rows, active in ((1, 1), (64, 32), (2048, 64)):
            assert (
                select_gemm_strategy(
                    operation, total_rows=rows, active_experts=active, policy="fused"
                )
                == "fused"
            )


def test_auto_never_selects_fused_before_b300_validation() -> None:
    # Iteration discipline: the fused kernel enters "auto" only after a B300
    # A/B run ranks it. If this assert fires because auto was upgraded on
    # purpose, move the fused expectations into the auto rule tests.
    for operation in ("gate_up", "down"):
        for rows in (1, 2, 8, 32, 64, 256):
            strategy = select_gemm_strategy(
                operation, total_rows=rows, active_experts=min(rows, 64)
            )
            assert strategy in ("grouped", "per_expert")


# ---------------------------------------------------------------------------
# torch backend layer
# ---------------------------------------------------------------------------


def _forward_case() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(1409)
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


def test_torch_backend_accepts_fused_policy() -> None:
    tensors = _forward_case()
    baseline = b300_moe_forward(*tensors, num_experts=4, top_k=2)
    output, metadata = b300_moe_forward(
        *tensors,
        num_experts=4,
        top_k=2,
        gemm_policy="fused",
        return_metadata=True,
    )
    assert torch.equal(output, baseline)
    assert "gemm_dispatch" not in metadata


# ---------------------------------------------------------------------------
# B300 layer
# ---------------------------------------------------------------------------


def _fused_ready() -> bool:
    from fastpath import cuda_extension_available

    if not torch.cuda.is_available() or not cuda_extension_available():
        return False
    if torch.cuda.get_device_capability() != (10, 3):
        return False
    try:
        import fastpath._C as extension

        return extension.build_info().get("goal") == "1B" and hasattr(
            extension, "fused_grouped_gemm_bf16_out"
        )
    except (ImportError, OSError, AttributeError, RuntimeError):
        return False


requires_fused = pytest.mark.skipif(
    not _fused_ready(),
    reason="requires a Goal 1C step 2 extension build on a CC 10.3 B300 GPU",
)


@requires_fused
@pytest.mark.parametrize(
    "counts",
    [
        [1, 0, 0, 0, 0, 0, 0, 1],   # T=1-like: two active experts
        [0, 0, 0, 0, 0, 0, 0, 0],   # no assignments at all
        [3, 0, 1, 0, 0, 5, 2, 0],   # ragged with empty experts
        [16, 16, 16, 16, 0, 0, 0, 0],  # per-expert rows above kRowChunk
    ],
)
def test_fused_kernel_matches_matmul_oracle(counts: list[int]) -> None:
    import fastpath._C as extension

    device = torch.device("cuda")
    generator = torch.Generator().manual_seed(1411 + sum(counts))
    num_experts, k_dim, n_dim = len(counts), 64, 48
    rows = sum(counts)
    inputs = (
        torch.randn(rows, k_dim, generator=generator)
        .to(device=device, dtype=torch.bfloat16)
    )
    public_weight = (
        torch.randn(num_experts, n_dim, k_dim, generator=generator)
        .to(device=device, dtype=torch.bfloat16)
    )
    packed = public_weight.transpose(1, 2)  # zero-copy [E, K, N] view
    offsets = torch.tensor(
        [0] + list(torch.tensor(counts).cumsum(0)), dtype=torch.long, device=device
    )
    out = torch.full((rows, n_dim), float("nan"), device=device, dtype=torch.bfloat16)

    extension.fused_grouped_gemm_bf16_out(inputs, packed, offsets, out)

    for expert in range(num_experts):
        start, end = int(offsets[expert]), int(offsets[expert + 1])
        if end == start:
            continue
        expected = inputs[start:end].float() @ packed[expert].float()
        assert torch.allclose(
            out[start:end].float(), expected, atol=0.08, rtol=0.05
        ), f"expert {expert} mismatch"
    assert not torch.isnan(out).any()


@requires_fused
def test_fused_kernel_validates_inputs() -> None:
    import fastpath._C as extension

    device = torch.device("cuda")
    inputs = torch.zeros(2, 64, device=device, dtype=torch.bfloat16)
    packed = torch.zeros(2, 48, 64, device=device, dtype=torch.bfloat16).transpose(1, 2)
    out = torch.zeros(2, 48, device=device, dtype=torch.bfloat16)
    good_offsets = torch.tensor([0, 1, 2], dtype=torch.long, device=device)

    with pytest.raises(RuntimeError, match="int64"):
        extension.fused_grouped_gemm_bf16_out(
            inputs, packed, good_offsets.int(), out
        )
    with pytest.raises(RuntimeError, match="num_experts \\+ 1"):
        extension.fused_grouped_gemm_bf16_out(
            inputs, packed, good_offsets[:-1], out
        )
    with pytest.raises(RuntimeError, match="transpose view"):
        extension.fused_grouped_gemm_bf16_out(
            inputs, packed.contiguous(), good_offsets, out
        )


@requires_fused
@pytest.mark.parametrize("num_tokens", [1, 4, 16, 64])
def test_b300_fused_forward_matches_grouped(num_tokens: int) -> None:
    generator = torch.Generator().manual_seed(1413 + num_tokens)
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
    for policy in ("grouped", "fused"):
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

    # The fused path never syncs offsets to the host; its output must still
    # agree with the grouped baseline within BF16 accumulation tolerance.
    assert torch.allclose(
        outputs["fused"].float(), outputs["grouped"].float(), atol=0.12, rtol=0.06
    )
