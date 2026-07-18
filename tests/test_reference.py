"""Correctness tests for the staged and naive Local-MoE references."""

from __future__ import annotations

import pytest
import torch

from baseline.routing import generate_router_logits
from baseline.torch_moe import ExpertWeights, local_moe, make_expert_weights, naive_local_moe


def _inputs(
    *, num_tokens: int = 11, num_experts: int = 4, top_k: int = 2
) -> tuple[torch.Tensor, torch.Tensor, ExpertWeights]:
    generator = torch.Generator().manual_seed(23)
    hidden = torch.randn(num_tokens, 16, generator=generator)
    logits = generate_router_logits(num_tokens, num_experts, top_k, "skewed", seed=24)
    weights = make_expert_weights(num_experts, 16, 24, seed=25)
    return hidden, logits, weights


@pytest.mark.parametrize("top_k", [1, 2, 4])
def test_staged_reference_matches_naive(top_k: int) -> None:
    hidden, logits, weights = _inputs(top_k=top_k)
    actual, metadata = local_moe(hidden, logits, weights, top_k)
    expected = naive_local_moe(hidden, logits, weights, top_k)
    assert actual.shape == hidden.shape
    assert metadata.routing.topk_indices.shape == (hidden.shape[0], top_k)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_empty_experts_are_supported() -> None:
    hidden, _, weights = _inputs(num_tokens=9, num_experts=8, top_k=1)
    logits = torch.full((9, 8), -20.0)
    logits[:, 3] = 20.0
    actual, metadata = local_moe(hidden, logits, weights, top_k=1)
    expected = naive_local_moe(hidden, logits, weights, top_k=1)
    assert metadata.routing.expert_counts.tolist() == [0, 0, 0, 9, 0, 0, 0, 0]
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_severe_load_imbalance_is_supported() -> None:
    hidden, _, weights = _inputs(num_tokens=32, num_experts=4, top_k=2)
    logits = torch.full((32, 4), -20.0)
    logits[:, 0] = 10.0
    logits[:, 1] = 9.0
    actual, metadata = local_moe(hidden, logits, weights, top_k=2)
    expected = naive_local_moe(hidden, logits, weights, top_k=2)
    assert metadata.routing.expert_counts.tolist() == [32, 32, 0, 0]
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_shape_mismatch_and_invalid_top_k_are_clear() -> None:
    hidden, logits, weights = _inputs()
    with pytest.raises(ValueError, match="same num_tokens"):
        local_moe(hidden[:-1], logits, weights, top_k=2)
    with pytest.raises(ValueError, match="top_k"):
        local_moe(hidden, logits, weights, top_k=5)


def test_incompatible_weight_shape_is_rejected() -> None:
    hidden, logits, weights = _inputs()
    bad_weights = ExpertWeights(weights.gate, weights.up, weights.down[:, :-1])
    with pytest.raises(ValueError, match="down"):
        local_moe(hidden, logits, bad_weights, top_k=2)
