"""Pure PyTorch reference implementation for local mixture-of-experts."""

from .routing import RoutingResult, generate_router_logits, group_tokens, topk_routing
from .torch_moe import (
    ExpertWeights,
    LocalMoEMetadata,
    local_moe,
    make_expert_weights,
    naive_local_moe,
)

__all__ = [
    "ExpertWeights",
    "LocalMoEMetadata",
    "RoutingResult",
    "generate_router_logits",
    "group_tokens",
    "local_moe",
    "make_expert_weights",
    "naive_local_moe",
    "topk_routing",
]
