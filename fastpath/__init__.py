"""Local-MoE reference, dispatch/combine, and Goal 1B forward APIs."""

from .b300_moe import (
    B300Backend,
    B300MoEMetadata,
    B300MoEWorkspace,
    QuantMode,
    b300_moe_forward,
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
    "QuantMode",
    "RoutedMoEShape",
    "b300_moe_forward",
    "build_routing_metadata",
    "combine_tokens",
    "cuda_extension_available",
    "cuda_extension_error",
    "dispatch_tokens",
    "routed_moe_reference",
    "validate_routed_moe_inputs",
]
