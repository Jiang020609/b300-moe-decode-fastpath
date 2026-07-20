"""B300-only end-to-end tests for the real Goal 1B BF16 backend."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import yaml

from baseline.routing import generate_router_logits, topk_routing
from fastpath import B300MoEWorkspace, b300_moe_forward, cuda_extension_available


def _goal1b_ready() -> bool:
    if not torch.cuda.is_available() or not cuda_extension_available():
        return False
    if torch.cuda.get_device_capability() != (10, 3):
        return False
    try:
        import fastpath._C as extension

        return extension.build_info().get("goal") == "1B"
    except (ImportError, OSError, AttributeError, RuntimeError):
        return False


pytestmark = pytest.mark.skipif(
    not _goal1b_ready(),
    reason="requires a compiled Goal 1B extension on a CC 10.3 B300 GPU",
)


def _assert_close_with_diagnostics(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
    context: str,
) -> None:
    if torch.allclose(actual, expected, atol=atol, rtol=rtol):
        return
    difference = (actual.float() - expected.float()).abs()
    relative = difference / expected.float().abs().clamp_min(1e-6)
    pytest.fail(
        f"{context}; stage=end_to_end; shape={tuple(actual.shape)}; "
        f"max_abs_error={difference.max().item():.6g}; "
        f"mean_abs_error={difference.mean().item():.6g}; "
        f"max_relative_error={relative.max().item():.6g}"
    )


def _case(
    tokens: int,
    experts: int,
    top_k: int,
    workload: str,
    *,
    seed: int,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(seed)
    hidden_size, intermediate_size = 32, 48
    hidden = torch.randn(tokens, hidden_size, generator=generator).to(
        device="cuda", dtype=torch.bfloat16
    )
    logits = generate_router_logits(
        tokens,
        experts,
        top_k,
        workload,  # type: ignore[arg-type]
        seed + 1,
        device="cuda",
        dtype=torch.bfloat16,
    )
    indices, routing_weights = topk_routing(logits, top_k)
    gate_up = (
        torch.randn(experts, 2 * intermediate_size, hidden_size, generator=generator)
        * hidden_size**-0.5
    ).to(device="cuda", dtype=torch.bfloat16)
    down = (
        torch.randn(experts, hidden_size, intermediate_size, generator=generator)
        * intermediate_size**-0.5
    ).to(device="cuda", dtype=torch.bfloat16)
    return hidden, indices.contiguous(), routing_weights.contiguous(), gate_up, down


@pytest.mark.parametrize("tokens", [1, 2, 4, 8, 16, 32, 64])
@pytest.mark.parametrize("top_k", [1, 2, 4, 8])
@pytest.mark.parametrize("workload", ["uniform", "hotspot", "zipf"])
def test_cutlass_bf16_matches_explicit_torch_reference(
    tokens: int, top_k: int, workload: str
) -> None:
    experts = 16
    hidden, indices, weights, gate_up, down = _case(
        tokens, experts, top_k, workload, seed=1000 + tokens + top_k
    )
    workspace = B300MoEWorkspace(capacity_tokens=64, device="cuda")
    actual, metadata = b300_moe_forward(
        hidden,
        indices,
        weights,
        gate_up,
        down,
        num_experts=experts,
        top_k=top_k,
        backend="cutlass_bf16",
        workspace=workspace,
        return_metadata=True,
    )
    expected = b300_moe_forward(
        hidden,
        indices,
        weights,
        gate_up,
        down,
        num_experts=experts,
        top_k=top_k,
        backend="torch",
    )
    assert torch.isfinite(actual).all()
    _assert_close_with_diagnostics(
        actual,
        expected,
        atol=0.12,
        rtol=0.06,
        context=(
            f"backend=cutlass_bf16 workload={workload} "
            f"T={tokens} E={experts} K={top_k}"
        ),
    )
    assert metadata["backend"] == "cutlass_bf16"
    assert metadata["architecture"] == "sm_103a"
    assert metadata["quant_mode"] == "bf16"
    assert metadata["used_fallback"] is False


def test_empty_experts_unnormalized_weights_reuse_and_stream() -> None:
    hidden, _, _, gate_up, down = _case(8, 16, 4, "uniform", seed=2020)
    indices = torch.tensor([[0, 1, 2, 3]] * 8, device="cuda", dtype=torch.int64)
    weights = torch.tensor(
        [[1.0, -0.25, 0.5, 2.0]] * 8, device="cuda", dtype=torch.bfloat16
    )
    workspace = B300MoEWorkspace(capacity_tokens=8, device="cuda")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        first, first_metadata = b300_moe_forward(
            hidden,
            indices,
            weights,
            gate_up,
            down,
            num_experts=16,
            top_k=4,
            backend="cutlass_bf16",
            workspace=workspace,
            return_metadata=True,
        )
        second, second_metadata = b300_moe_forward(
            hidden,
            indices,
            weights,
            gate_up,
            down,
            num_experts=16,
            top_k=4,
            backend="cutlass_bf16",
            workspace=workspace,
            return_metadata=True,
        )
    stream.synchronize()
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    assert first_metadata["expert_counts"].tolist() == [8, 8, 8, 8] + [0] * 12
    assert second_metadata["workspace"]["reused"] is True


@pytest.mark.parametrize("experts", [8, 16, 64, 128])
def test_cutlass_bf16_dynamic_expert_count_matrix(experts: int) -> None:
    hidden, indices, weights, gate_up, down = _case(
        4, experts, 2, "uniform", seed=2500 + experts
    )
    actual = b300_moe_forward(
        hidden,
        indices,
        weights,
        gate_up,
        down,
        num_experts=experts,
        top_k=2,
        backend="cutlass_bf16",
        workspace=B300MoEWorkspace(capacity_tokens=4, device="cuda"),
    )
    expected = b300_moe_forward(
        hidden,
        indices,
        weights,
        gate_up,
        down,
        num_experts=experts,
        top_k=2,
        backend="torch",
    )
    _assert_close_with_diagnostics(
        actual,
        expected,
        atol=0.12,
        rtol=0.06,
        context=f"backend=cutlass_bf16 workload=uniform T=4 E={experts} K=2",
    )


def test_cutlass_single_expert_concentration_and_repeated_route() -> None:
    hidden, _, _, gate_up, down = _case(8, 16, 2, "uniform", seed=2727)
    indices = torch.full((8, 2), 5, device="cuda", dtype=torch.int64)
    weights = torch.tensor(
        [[0.25, 0.75]] * 8, device="cuda", dtype=torch.bfloat16
    )
    actual, metadata = b300_moe_forward(
        hidden,
        indices,
        weights,
        gate_up,
        down,
        num_experts=16,
        top_k=2,
        backend="cutlass_bf16",
        return_metadata=True,
    )
    expected = b300_moe_forward(
        hidden,
        indices,
        weights,
        gate_up,
        down,
        num_experts=16,
        top_k=2,
        backend="torch",
    )
    _assert_close_with_diagnostics(
        actual,
        expected,
        atol=0.12,
        rtol=0.06,
        context="backend=cutlass_bf16 workload=single_expert T=8 E=16 K=2",
    )
    assert metadata["expert_counts"].tolist() == [0] * 5 + [16] + [0] * 10


@pytest.mark.skipif(
    os.environ.get("MOE_RUN_B300_LARGE") != "1",
    reason="set MOE_RUN_B300_LARGE=1 to allocate the ~21 GiB BF16 target weights",
)
def test_b300_config_target_shape_when_explicitly_enabled() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs/b300.yaml"
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    experts = min(config["experts"])
    hidden_size = config["hidden_size"]
    intermediate_size = config["intermediate_size"]
    top_k = config["top_k"][0]
    assert (experts, hidden_size, intermediate_size, top_k) == (64, 4096, 14336, 2)

    generator = torch.Generator(device="cuda").manual_seed(9090)
    hidden = torch.empty(1, hidden_size, device="cuda", dtype=torch.bfloat16)
    hidden.normal_(generator=generator)
    indices = torch.tensor([[0, 1]], device="cuda", dtype=torch.int64)
    weights = torch.tensor([[0.6, 0.4]], device="cuda", dtype=torch.bfloat16)
    gate_up = torch.empty(
        experts,
        2 * intermediate_size,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    gate_up.normal_(0, hidden_size**-0.5, generator=generator)
    down = torch.empty(
        experts,
        hidden_size,
        intermediate_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    down.normal_(0, intermediate_size**-0.5, generator=generator)

    actual = b300_moe_forward(
        hidden,
        indices,
        weights,
        gate_up,
        down,
        num_experts=experts,
        top_k=top_k,
        backend="cutlass_bf16",
        workspace=B300MoEWorkspace(capacity_tokens=1, device="cuda"),
    )
    expected = b300_moe_forward(
        hidden,
        indices,
        weights,
        gate_up,
        down,
        num_experts=experts,
        top_k=top_k,
        backend="torch",
    )
    _assert_close_with_diagnostics(
        actual,
        expected,
        atol=0.12,
        rtol=0.06,
        context="backend=cutlass_bf16 workload=target_shape T=1 E=64 K=2",
    )


def test_nvfp4_request_never_runs_bf16_or_torch_fallback() -> None:
    hidden, indices, weights, gate_up, down = _case(2, 8, 2, "uniform", seed=3030)
    with pytest.raises(RuntimeError, match="NVFP4|nvfp4"):
        b300_moe_forward(
            hidden,
            indices,
            weights,
            gate_up,
            down,
            num_experts=8,
            top_k=2,
            backend="cutlass_nvfp4",
        )


def test_irregular_bf16_dimensions_fail_until_padding_is_implemented() -> None:
    hidden_size, intermediate_size, experts, top_k = 30, 46, 8, 2
    hidden = torch.randn(2, hidden_size, device="cuda", dtype=torch.bfloat16)
    indices = torch.tensor([[0, 1], [2, 3]], device="cuda", dtype=torch.int64)
    weights = torch.ones(2, top_k, device="cuda", dtype=torch.bfloat16)
    gate_up = torch.randn(
        experts,
        2 * intermediate_size,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    down = torch.randn(
        experts,
        hidden_size,
        intermediate_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    with pytest.raises(ValueError, match="divisible by 8"):
        b300_moe_forward(
            hidden,
            indices,
            weights,
            gate_up,
            down,
            num_experts=experts,
            top_k=top_k,
            backend="cutlass_bf16",
        )


def test_workspace_metadata_clone_is_ordered_before_cross_stream_reuse() -> None:
    hidden, _, _, gate_up, down = _case(4, 8, 1, "uniform", seed=4040)
    weights = torch.ones(4, 1, device="cuda", dtype=torch.bfloat16)
    first_indices = torch.zeros(4, 1, device="cuda", dtype=torch.int64)
    second_indices = torch.ones(4, 1, device="cuda", dtype=torch.int64)
    workspace = B300MoEWorkspace(capacity_tokens=4, device="cuda")
    first_stream = torch.cuda.Stream()
    second_stream = torch.cuda.Stream()

    with torch.cuda.stream(first_stream):
        _, first_metadata = b300_moe_forward(
            hidden,
            first_indices,
            weights,
            gate_up,
            down,
            num_experts=8,
            top_k=1,
            backend="cutlass_bf16",
            workspace=workspace,
            return_metadata=True,
        )
    with torch.cuda.stream(second_stream):
        b300_moe_forward(
            hidden,
            second_indices,
            weights,
            gate_up,
            down,
            num_experts=8,
            top_k=1,
            backend="cutlass_bf16",
            workspace=workspace,
        )

    first_stream.synchronize()
    second_stream.synchronize()
    assert first_metadata["expert_counts"].tolist() == [4] + [0] * 7
