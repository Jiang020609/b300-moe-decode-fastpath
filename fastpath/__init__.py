"""Local-MoE reference, dispatch/combine, and Goal 1B forward APIs."""

from .b300_moe import (
    B300Backend,
    B300MoEMetadata,
    B300MoEWorkspace,
    QuantMode,
    b300_moe_forward,
)

from .gemm_dispatch import (
    GemmOperation,
    GemmPolicy,
    GemmStrategy,
    per_expert_gemm_out,
    select_gemm_strategy,
)

from .ops import (
    DispatchResult,
    combine_tokens,
    cuda_extension_available,
    cuda_extension_error,
    dispatch_tokens,
)
from .reference import (
    RoutedMoEShape,
    build_routing_metadata,
    routed_moe_reference,
    validate_routed_moe_inputs,
)

__all__ = [
    "B300Backend",
    "B300MoEMetadata",
    "B300MoEWorkspace",
    "DispatchResult",
    "GemmOperation",
    "GemmPolicy",
    "GemmStrategy",
    "QuantMode",
    "RoutedMoEShape",
    "b300_moe_forward",
    "build_routing_metadata",
    "combine_tokens",
    "cuda_extension_available",
    "cuda_extension_error",
    "dispatch_tokens",
    "per_expert_gemm_out",
    "routed_moe_reference",
    "select_gemm_strategy",
    "validate_routed_moe_inputs",
]
