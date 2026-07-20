"""Goal 1B Python API for explicitly routed B300 MoE execution.

The dense ``torch`` backend is a correctness implementation.  CUTLASS
backends are selected only through dedicated extension entry points and never
fall back to the torch implementation.
"""

from __future__ import annotations

import importlib
import threading
from types import MappingProxyType, ModuleType
from typing import Literal, Mapping, TypedDict, overload

import torch

from .gemm_dispatch import (
    GemmPolicy,
    GemmStrategy,
    normalize_gemm_policy,
    per_expert_gemm_out,
    select_gemm_strategy,
)
from .reference import (
    SUPPORTED_DTYPES,
    RoutedMoEShape,
    _routed_moe_reference_with_routing,
    validate_routed_moe_inputs,
)

B300Backend = Literal["torch", "cutlass_bf16", "cutlass_nvfp4"]
QuantMode = Literal["none", "bf16", "nvfp4"]


class _B300MoEMetadataRequired(TypedDict):
    """Required metadata fields returned by :func:`b300_moe_forward`."""

    backend: str
    architecture: str
    quant_mode: str
    used_fallback: bool


class B300MoEMetadata(_B300MoEMetadataRequired, total=False):
    """Required fields plus optional routing/workspace/build diagnostics."""

    expert_counts: torch.Tensor
    expert_offsets: torch.Tensor
    permutation: torch.Tensor
    reverse_mapping: torch.Tensor
    workspace: dict[str, object]
    build: dict[str, object]
    gemm_dispatch: dict[str, object]


def _normalize_device(device: torch.device | str) -> torch.device:
    normalized = torch.device(device)
    if normalized.type == "cuda" and normalized.index is None:
        normalized = torch.device("cuda", torch.cuda.current_device())
    return normalized


class B300MoEWorkspace:
    """Capacity-backed scratch storage for the Goal 1B forward path.

    A workspace binds to its first device.  Compatible calls reuse allocations;
    larger token counts grow capacity, while layout changes reallocate.  CUDA
    reuse across streams is ordered with an event recorded after each use.
    Returned outputs and metadata never alias these scratch buffers.
    """

    _LONG_BUFFERS = {
        "expert_counts",
        "expert_offsets",
        "permutation",
        "reverse_mapping",
        "permuted_token_indices",
        "permuted_expert_indices",
    }

    def __init__(
        self,
        capacity_tokens: int = 0,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        if not isinstance(capacity_tokens, int) or isinstance(capacity_tokens, bool):
            raise TypeError("capacity_tokens must be an integer")
        if capacity_tokens < 0:
            raise ValueError("capacity_tokens must be non-negative")
        self._capacity_hint = capacity_tokens
        self._capacity_tokens = 0
        self._capacity_assignments = 0
        self._device = None if device is None else _normalize_device(device)
        self._dtype: torch.dtype | None = None
        self._layout_signature: tuple[object, ...] | None = None
        self._last_shape: tuple[int, int, int, int, int] | None = None
        self._buffers: dict[str, torch.Tensor] = {}
        self._allocation_count = 0
        self._reuse_count = 0
        self._last_reserve_reused = False
        self._last_stream: int | None = None
        self._completion_event: torch.cuda.Event | None = None
        self._lock = threading.RLock()

    @property
    def device(self) -> torch.device | None:
        return self._device

    @property
    def dtype(self) -> torch.dtype | None:
        return self._dtype

    @property
    def capacity_tokens(self) -> int:
        return max(self._capacity_hint, self._capacity_tokens)

    @property
    def capacity_assignments(self) -> int:
        return self._capacity_assignments

    @property
    def capacity_bytes(self) -> int:
        return sum(tensor.numel() * tensor.element_size() for tensor in self._buffers.values())

    @property
    def allocation_count(self) -> int:
        return self._allocation_count

    @property
    def reuse_count(self) -> int:
        return self._reuse_count

    @property
    def last_reserve_reused(self) -> bool:
        return self._last_reserve_reused

    @property
    def shape_signature(self) -> tuple[object, ...] | None:
        return self._layout_signature

    @property
    def last_shape(self) -> tuple[int, int, int, int, int] | None:
        """Last requested ``(T, E, K, H, I)`` shape."""

        return self._last_shape

    @property
    def last_stream(self) -> int | None:
        return self._last_stream

    @property
    def buffers(self) -> Mapping[str, torch.Tensor]:
        """Read-only mapping of scratch-buffer names to capacity tensors."""

        return MappingProxyType(self._buffers)

    def buffer_data_ptrs(self) -> dict[str, int]:
        """Return stable allocation identities for reuse tests and diagnostics."""

        return {name: tensor.data_ptr() for name, tensor in self._buffers.items()}

    def _wait_for_prior_use(self, device: torch.device) -> None:
        if self._completion_event is None:
            return
        if self._device == device:
            torch.cuda.current_stream(device).wait_event(self._completion_event)
        else:
            # Device rebinding is rejected, but keep this branch defensive for
            # workspaces constructed before a device is known.
            self._completion_event.synchronize()

    def _allocate(
        self,
        *,
        capacity_tokens: int,
        shape: RoutedMoEShape,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        assignments = capacity_tokens * shape.top_k
        specs: dict[str, tuple[int, ...]] = {
            "expert_counts": (shape.num_experts,),
            "expert_offsets": (shape.num_experts + 1,),
            "expert_offsets_i32": (shape.num_experts,),
            "permutation": (assignments,),
            "reverse_mapping": (assignments,),
            "permuted_token_indices": (assignments,),
            "permuted_expert_indices": (assignments,),
            "permuted_weights": (assignments,),
            "permuted_hidden": (assignments, shape.hidden_size),
            "gate_up_output": (assignments, 2 * shape.intermediate_size),
            "swiglu_output": (assignments, shape.intermediate_size),
            "expert_output": (assignments, shape.hidden_size),
        }
        self._buffers = {}
        for name, dims in specs.items():
            buffer_dtype = dtype
            if name in self._LONG_BUFFERS:
                buffer_dtype = torch.int64
            elif name == "expert_offsets_i32":
                buffer_dtype = torch.int32
            self._buffers[name] = torch.empty(
                dims, device=device, dtype=buffer_dtype
            )
        self._capacity_tokens = capacity_tokens
        self._capacity_assignments = assignments
        self._dtype = dtype
        self._allocation_count += 1

    def reserve(
        self,
        *,
        num_tokens: int,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        device: torch.device | str,
        dtype: torch.dtype,
        backend: B300Backend = "torch",
        quant_mode: QuantMode = "none",
    ) -> bool:
        """Ensure capacity and return whether the existing allocation was reused."""

        values = {
            "num_tokens": num_tokens,
            "num_experts": num_experts,
            "top_k": top_k,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
        }
        for name, value in values.items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if top_k not in (1, 2, 4, 8):
            raise ValueError("top_k must be one of (1, 2, 4, 8)")
        if top_k > num_experts:
            raise ValueError("top_k must not exceed num_experts")
        if dtype not in SUPPORTED_DTYPES:
            raise TypeError("dtype must be float32, float16, or bfloat16")
        if backend not in ("torch", "cutlass_bf16", "cutlass_nvfp4"):
            raise ValueError("unknown B300 backend")
        if quant_mode not in ("none", "bf16", "nvfp4"):
            raise ValueError("unknown quant_mode")

        normalized_device = _normalize_device(device)
        with self._lock:
            if self._device is not None and self._device != normalized_device:
                raise ValueError(
                    f"workspace is bound to device {self._device}, got {normalized_device}"
                )
            self._device = normalized_device
            shape = RoutedMoEShape(
                num_tokens=num_tokens,
                num_experts=num_experts,
                top_k=top_k,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
            )
            layout_signature = (
                normalized_device,
                dtype,
                num_experts,
                top_k,
                hidden_size,
                intermediate_size,
                backend,
                quant_mode,
            )
            needs_layout = self._layout_signature != layout_signature
            needs_growth = num_tokens > self._capacity_tokens
            needs_allocation = not self._buffers or needs_layout or needs_growth

            if normalized_device.type == "cuda":
                self._wait_for_prior_use(normalized_device)
                capturing = torch.cuda.is_current_stream_capturing()
                if self._completion_event is None:
                    if capturing:
                        raise RuntimeError(
                            "workspace event cannot be created during CUDA graph capture; "
                            "call reserve before capture"
                        )
                    self._completion_event = torch.cuda.Event(blocking=False)
                    # Force lazy CUDA event creation before a later graph
                    # capture. This does not attest that ATen grouped-MM's own
                    # allocations are capture-safe.
                    self._completion_event.record(
                        torch.cuda.current_stream(normalized_device)
                    )
                if needs_allocation and capturing:
                    raise RuntimeError(
                        "workspace capacity/layout cannot change during CUDA graph capture; "
                        "call reserve before capture"
                    )

            if needs_allocation:
                if needs_layout:
                    capacity = max(self._capacity_hint, num_tokens)
                else:
                    capacity = max(
                        self._capacity_hint,
                        num_tokens,
                        max(1, self._capacity_tokens * 2),
                    )
                self._allocate(
                    capacity_tokens=capacity,
                    shape=shape,
                    device=normalized_device,
                    dtype=dtype,
                )
                self._layout_signature = layout_signature
                reused = False
            else:
                self._reuse_count += 1
                reused = True
            self._last_shape = (
                num_tokens,
                num_experts,
                top_k,
                hidden_size,
                intermediate_size,
            )
            self._last_reserve_reused = reused
            if normalized_device.type == "cuda":
                stream = torch.cuda.current_stream(normalized_device)
                self._last_stream = int(stream.cuda_stream)
                for tensor in self._buffers.values():
                    tensor.record_stream(stream)
            else:
                self._last_stream = None
            return reused

    def _record_use(self) -> None:
        if self._device is None or self._device.type != "cuda":
            return
        stream = torch.cuda.current_stream(self._device)
        if self._completion_event is None:
            raise RuntimeError(
                "workspace completion event is missing; call reserve before use"
            )
        self._completion_event.record(stream)
        self._last_stream = int(stream.cuda_stream)
        for tensor in self._buffers.values():
            tensor.record_stream(stream)


def _normalize_backend(backend: object) -> B300Backend:
    if not isinstance(backend, str):
        raise TypeError("backend must be a string")
    if backend not in ("torch", "cutlass_bf16", "cutlass_nvfp4"):
        raise ValueError(
            "backend must be 'torch', 'cutlass_bf16', or 'cutlass_nvfp4'"
        )
    return backend  # type: ignore[return-value]


def _normalize_quant_mode(
    backend: B300Backend, quant_mode: object, dtype: torch.dtype
) -> QuantMode:
    if quant_mode is None:
        quant_mode = {
            "torch": "none",
            "cutlass_bf16": "bf16",
            "cutlass_nvfp4": "nvfp4",
        }[backend]
    if not isinstance(quant_mode, str):
        raise TypeError("quant_mode must be a string or None")
    if quant_mode not in ("none", "bf16", "nvfp4"):
        raise ValueError("quant_mode must be 'none', 'bf16', or 'nvfp4'")
    if backend == "torch":
        if quant_mode == "nvfp4":
            raise ValueError(
                "backend='torch' cannot execute NVFP4; select cutlass_nvfp4 explicitly"
            )
        if quant_mode == "bf16" and dtype != torch.bfloat16:
            raise TypeError("torch quant_mode='bf16' requires bfloat16 inputs")
    elif backend == "cutlass_bf16" and quant_mode != "bf16":
        raise ValueError("cutlass_bf16 requires quant_mode='bf16'")
    elif backend == "cutlass_nvfp4" and quant_mode != "nvfp4":
        raise ValueError("cutlass_nvfp4 requires quant_mode='nvfp4'")
    return quant_mode  # type: ignore[return-value]


def _architecture_for(device: torch.device) -> str:
    if device.type != "cuda":
        return device.type
    major, minor = torch.cuda.get_device_capability(device)
    return "sm_103a" if (major, minor) == (10, 3) else f"sm_{major}{minor}"


def _load_goal_1b_extension() -> ModuleType:
    try:
        return importlib.import_module("fastpath._C")
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "compiled Goal 1B backend requested, but fastpath._C is unavailable; "
            "build the CUDA extension for sm_103a"
        ) from error



_GROUPED_MM_RUNTIME_PROBED_DEVICES: set[int] = set()


def _verify_grouped_mm_runtime(device: torch.device) -> None:
    """Verify the PyTorch BF16 GroupMM provider on one CC 10.3 device."""

    device_index = (
        torch.cuda.current_device()
        if device.index is None
        else int(device.index)
    )

    if device_index in _GROUPED_MM_RUNTIME_PROBED_DEVICES:
        return

    grouped_mm = getattr(torch, "_grouped_mm", None)
    if not callable(grouped_mm):
        raise RuntimeError(
            "the active PyTorch build does not expose torch._grouped_mm; "
            "the audited BF16 GroupMM provider is unavailable"
        )

    probe_device = torch.device("cuda", device_index)

    try:
        with torch.no_grad():
            tokens_per_expert = 16
            expert_count = 2
            k = 64
            n = 64
            total_tokens = tokens_per_expert * expert_count

            activation_values = torch.arange(
                total_tokens * k,
                device=probe_device,
                dtype=torch.float32,
            ).reshape(total_tokens, k)
            activations = (
                ((activation_values % 23) - 11) / 8
            ).to(torch.bfloat16)

            weight_values = torch.arange(
                expert_count * k * n,
                device=probe_device,
                dtype=torch.float32,
            ).reshape(expert_count, k, n)
            weights = (
                ((weight_values % 19) - 9) / 8
            ).to(torch.bfloat16)

            offsets = torch.tensor(
                [tokens_per_expert, total_tokens],
                device=probe_device,
                dtype=torch.int32,
            )

            output = grouped_mm(
                activations,
                weights,
                offs=offsets,
            )

            reference = torch.cat(
                [
                    activations[:tokens_per_expert] @ weights[0],
                    activations[tokens_per_expert:] @ weights[1],
                ],
                dim=0,
            )

            torch.cuda.synchronize(device_index)

            if tuple(output.shape) != tuple(reference.shape):
                raise RuntimeError(
                    "provider returned shape "
                    f"{tuple(output.shape)}, expected "
                    f"{tuple(reference.shape)}"
                )

            if output.dtype != torch.bfloat16:
                raise RuntimeError(
                    "provider returned dtype "
                    f"{output.dtype}, expected torch.bfloat16"
                )

            if not bool(
                torch.allclose(
                    output.float(),
                    reference.float(),
                    rtol=5e-2,
                    atol=1e-1,
                )
            ):
                max_abs_diff = (
                    output.float() - reference.float()
                ).abs().max().item()
                raise RuntimeError(
                    "provider produced incorrect output; "
                    f"max_abs_diff={max_abs_diff}"
                )

    except Exception as error:
        raise RuntimeError(
            "PyTorch grouped-MM provider runtime verification failed "
            f"on cuda:{device_index}: "
            f"{type(error).__name__}: {error}"
        ) from error

    _GROUPED_MM_RUNTIME_PROBED_DEVICES.add(device_index)


def _resolve_compiled_backend(
    backend: B300Backend,
    hidden_states: torch.Tensor,
    expert_weights: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    shape: RoutedMoEShape,
) -> tuple[ModuleType, Mapping[str, object]]:
    """Validate hardware/build capabilities before allocating workspace."""

    if hidden_states.device.type != "cuda":
        raise ValueError(f"backend={backend!r} requires CUDA tensors")
    capability = torch.cuda.get_device_capability(hidden_states.device)
    if capability != (10, 3):
        raise RuntimeError(
            f"backend={backend!r} requires B300 compute capability (10, 3), "
            f"got {capability}"
        )
    _verify_grouped_mm_runtime(hidden_states.device)
    if backend == "cutlass_bf16" and hidden_states.dtype != torch.bfloat16:
        raise TypeError("cutlass_bf16 requires bfloat16 inputs and weights")
    if backend == "cutlass_bf16" and (
        shape.hidden_size % 8 != 0 or shape.intermediate_size % 8 != 0
    ):
        raise ValueError(
            "cutlass_bf16 requires hidden_size and intermediate_size divisible "
            "by 8 (16-byte BF16 alignment); padding is not implemented"
        )
    if shape.num_experts > 256:
        raise ValueError("compiled Goal 1B backends support at most 256 experts")
    if any(
        tensor.requires_grad
        for tensor in (hidden_states, expert_weights, gate_up_weight, down_weight)
    ):
        raise ValueError("compiled Goal 1B backends do not support autograd")

    extension = _load_goal_1b_extension()
    build_info_fn = getattr(extension, "build_info", None)
    if not callable(build_info_fn):
        raise RuntimeError(
            "fastpath._C is present but lacks the Goal 1B `build_info()` API"
        )
    try:
        build_info = build_info_fn()
    except Exception as error:
        raise RuntimeError("fastpath._C.build_info() failed") from error
    if not isinstance(build_info, dict):
        raise RuntimeError("fastpath._C.build_info() must return a dict")
    if build_info.get("goal") != "1B":
        raise RuntimeError("fastpath._C is not a Goal 1B build")
    if build_info.get("compiled_architecture") != "sm_103a":
        raise RuntimeError("fastpath._C was not compiled for sm_103a")
    if build_info.get("uses_fallback") is not False:
        raise RuntimeError("fastpath._C build_info did not attest uses_fallback=false")

    if backend == "cutlass_nvfp4":
        available_fn = getattr(extension, "nvfp4_grouped_gemm_available", None)
        if not callable(available_fn):
            raise RuntimeError(
                "fastpath._C lacks `nvfp4_grouped_gemm_available()`"
            )
        available = available_fn()
        if available is not True:
            reason_fn = getattr(
                extension, "nvfp4_grouped_gemm_unavailable_reason", None
            )
            reason = reason_fn() if callable(reason_fn) else "capability returned false"
            raise RuntimeError(f"cutlass_nvfp4 is unavailable: {reason}")
        # A real NVFP4 path also needs explicit quantize/scale-aware staged
        # entry points. Never substitute the BF16 grouped GEMM.
        required_nvfp4 = ("quantize_nvfp4_out", "grouped_gemm_nvfp4_out")
        missing = [name for name in required_nvfp4 if not callable(getattr(extension, name, None))]
        if missing:
            raise RuntimeError(
                "fastpath._C reports NVFP4 available but lacks required staged APIs: "
                + ", ".join(missing)
            )
        raise RuntimeError(
            "cutlass_nvfp4 staged scale/layout orchestration is not implemented; "
            "no fallback was used"
        )

    provider = build_info.get("bf16_grouped_gemm_provider")
    if provider != "at::cuda::detail::bf16bf16_grouped_mm (PyTorch binary)":
        raise RuntimeError(
            "fastpath._C did not identify the audited PyTorch GroupMM provider"
        )
    required_bf16 = (
        "routing_metadata",
        "routing_metadata_out",
        "permute_out",
        "grouped_gemm_bf16_out",
        "swiglu_out",
        "combine_out",
    )
    missing = [name for name in required_bf16 if not callable(getattr(extension, name, None))]
    if missing:
        raise RuntimeError(
            "fastpath._C Goal 1B build lacks required staged APIs: "
            + ", ".join(missing)
        )
    return extension, build_info


def _call_compiled_bf16(
    extension: ModuleType,
    hidden_states: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_weights: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    shape: RoutedMoEShape,
    workspace: B300MoEWorkspace,
    gemm_policy: GemmPolicy = "grouped",
) -> tuple[torch.Tensor, Mapping[str, torch.Tensor], dict[str, object]]:
    """Orchestrate the extension's caller-owned staged BF16 operations."""

    assignments = shape.num_assignments
    buffers = workspace.buffers
    routed = {
        "expert_counts": buffers["expert_counts"][: shape.num_experts],
        "expert_offsets": buffers["expert_offsets"][: shape.num_experts + 1],
        "reverse_mapping": buffers["reverse_mapping"][:assignments],
        "permutation": buffers["permutation"][:assignments],
    }
    try:
        extension.routing_metadata_out(
            expert_indices,
            shape.num_experts,
            routed["expert_counts"],
            routed["expert_offsets"],
            routed["reverse_mapping"],
            routed["permutation"],
        )
    except Exception as error:
        raise RuntimeError("fastpath._C.routing_metadata_out failed") from error
    offsets_i32 = buffers["expert_offsets_i32"][: shape.num_experts]
    offsets_i32.copy_(routed["expert_offsets"][1:])

    permuted_hidden = buffers["permuted_hidden"][:assignments]
    gate_up_output = buffers["gate_up_output"][:assignments]
    swiglu_output = buffers["swiglu_output"][:assignments]
    expert_output = buffers["expert_output"][:assignments]
    output = torch.empty_like(hidden_states, memory_format=torch.contiguous_format)

    # The grouped-mm wrapper requires a column-major [E,K,N] transpose view.
    # This is a zero-copy view of the public [E,N,K] weights, so mutations are
    # visible immediately and no stale packed-weight cache is possible.
    packed_gate_up = gate_up_weight.transpose(1, 2)
    packed_down = down_weight.transpose(1, 2)

    # Launch the permutation gather before any host-side dispatch decision so
    # the device stays busy while the offsets sync below blocks the CPU.
    try:
        extension.permute_out(
            hidden_states, routed["permutation"], shape.top_k, permuted_hidden
        )
    except Exception as error:
        raise RuntimeError("a staged Goal 1B BF16 extension operation failed") from error

    # Goal 1C hybrid dispatch: decide per GEMM stage between the grouped
    # kernel and a per-active-expert matmul loop. The loop needs host-side
    # offsets for Python slicing, which costs one device sync (overlapped
    # with the permute kernel launched above); the default "grouped" policy
    # skips that copy and is byte-for-byte the validated Goal 1B path.
    gate_up_strategy: GemmStrategy = "grouped"
    down_strategy: GemmStrategy = "grouped"
    host_offsets: list[int] | None = None
    active_experts: int | None = None
    if gemm_policy != "grouped":
        if hidden_states.is_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "per-expert GEMM dispatch requires host-side expert offsets "
                "and cannot run during CUDA graph capture; use "
                "gemm_policy='grouped'"
            )
        host_offsets = routed["expert_offsets"].tolist()
        active_experts = sum(
            1
            for expert in range(shape.num_experts)
            if host_offsets[expert + 1] > host_offsets[expert]
        )
        gate_up_strategy = select_gemm_strategy(
            "gate_up",
            total_rows=assignments,
            active_experts=active_experts,
            policy=gemm_policy,
        )
        down_strategy = select_gemm_strategy(
            "down",
            total_rows=assignments,
            active_experts=active_experts,
            policy=gemm_policy,
        )

    try:
        if gate_up_strategy == "grouped":
            extension.grouped_gemm_bf16_out(
                permuted_hidden, packed_gate_up, offsets_i32, gate_up_output
            )
        else:
            assert host_offsets is not None
            per_expert_gemm_out(
                permuted_hidden, packed_gate_up, host_offsets, gate_up_output
            )
        extension.swiglu_out(gate_up_output, swiglu_output)
        if down_strategy == "grouped":
            extension.grouped_gemm_bf16_out(
                swiglu_output, packed_down, offsets_i32, expert_output
            )
        else:
            assert host_offsets is not None
            per_expert_gemm_out(
                swiglu_output, packed_down, host_offsets, expert_output
            )
        extension.combine_out(
            expert_output, routed["reverse_mapping"], expert_weights, output
        )
    except Exception as error:
        raise RuntimeError("a staged Goal 1B BF16 extension operation failed") from error

    if not isinstance(output, torch.Tensor):
        raise RuntimeError("compiled BF16 orchestration must return a Tensor")
    if output.shape != hidden_states.shape:
        raise RuntimeError(
            f"compiled BF16 path returned shape {tuple(output.shape)}, "
            f"expected {tuple(hidden_states.shape)}"
        )
    if output.dtype != hidden_states.dtype or output.device != hidden_states.device:
        raise RuntimeError("compiled BF16 path returned the wrong dtype or device")
    if not output.is_contiguous():
        raise RuntimeError("compiled BF16 path returned non-contiguous output")
    dispatch_info: dict[str, object] = {
        "policy": gemm_policy,
        "gate_up_strategy": gate_up_strategy,
        "down_strategy": down_strategy,
        "total_rows": assignments,
        "active_experts": active_experts,
    }
    return output, routed, dispatch_info


def _metadata(
    *,
    backend: B300Backend,
    architecture: str,
    quant_mode: QuantMode,
    routing: object | None,
    workspace: B300MoEWorkspace | None,
    workspace_reused: bool,
    build_info: Mapping[str, object] | None,
) -> B300MoEMetadata:
    metadata: dict[str, object] = {
        "backend": backend,
        "architecture": architecture,
        "quant_mode": quant_mode,
        "used_fallback": False,
    }
    if routing is not None:
        # Clones keep metadata stable after a workspace is reused.
        def routing_tensor(name: str) -> torch.Tensor:
            if isinstance(routing, Mapping):
                value = routing[name]
            else:
                value = getattr(routing, name)
            if not isinstance(value, torch.Tensor):
                raise RuntimeError(f"routing metadata field {name!r} is not a Tensor")
            return value

        metadata.update(
            expert_counts=routing_tensor("expert_counts").clone(),
            expert_offsets=routing_tensor("expert_offsets").clone(),
            permutation=routing_tensor("permutation").clone(),
            reverse_mapping=routing_tensor("reverse_mapping").clone(),
        )
    if workspace is not None:
        metadata["workspace"] = {
            "capacity_tokens": workspace.capacity_tokens,
            "capacity_assignments": workspace.capacity_assignments,
            "capacity_bytes": workspace.capacity_bytes,
            "allocation_count": workspace.allocation_count,
            "reuse_count": workspace.reuse_count,
            "reused": workspace_reused,
            "shape_signature": workspace.shape_signature,
        }
    if build_info is not None:
        metadata["build"] = dict(build_info)
    return metadata  # type: ignore[return-value]


@overload
def b300_moe_forward(
    hidden_states: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_weights: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    num_experts: int,
    top_k: int,
    quant_mode: QuantMode | None = None,
    backend: B300Backend = "torch",
    workspace: B300MoEWorkspace | None = None,
    gemm_policy: GemmPolicy = "grouped",
    return_metadata: Literal[False] = False,
) -> torch.Tensor: ...


@overload
def b300_moe_forward(
    hidden_states: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_weights: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    num_experts: int,
    top_k: int,
    quant_mode: QuantMode | None = None,
    backend: B300Backend = "torch",
    workspace: B300MoEWorkspace | None = None,
    gemm_policy: GemmPolicy = "grouped",
    return_metadata: Literal[True],
) -> tuple[torch.Tensor, B300MoEMetadata]: ...


def b300_moe_forward(
    hidden_states: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_weights: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    num_experts: int,
    top_k: int,
    quant_mode: QuantMode | None = None,
    backend: B300Backend = "torch",
    workspace: B300MoEWorkspace | None = None,
    gemm_policy: GemmPolicy = "grouped",
    return_metadata: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, B300MoEMetadata]:
    """Run explicitly routed two-GEMM SwiGLU MoE execution.

    The ``torch`` backend is dense and does not emulate NVFP4.  Requesting a
    CUTLASS backend is an exact request: missing extension features, wrong
    hardware, or incompatible dtypes raise instead of falling back.

    ``gemm_policy`` selects the Goal 1C GEMM execution strategy on the
    compiled BF16 path: ``"grouped"`` (default, the validated Goal 1B
    behavior), ``"per_expert"`` (force the per-active-expert matmul loop), or
    ``"auto"`` (measured-data-driven hybrid dispatch).  The ``torch`` backend
    computes densely and ignores the policy beyond validation.
    """

    if not isinstance(return_metadata, bool):
        raise TypeError("return_metadata must be a bool")
    normalized_backend = _normalize_backend(backend)
    normalized_gemm_policy = normalize_gemm_policy(gemm_policy)
    shape = validate_routed_moe_inputs(
        hidden_states,
        expert_indices,
        expert_weights,
        gate_up_weight,
        down_weight,
        num_experts=num_experts,
        top_k=top_k,
    )
    normalized_quant_mode = _normalize_quant_mode(
        normalized_backend, quant_mode, hidden_states.dtype
    )
    if workspace is not None and not isinstance(workspace, B300MoEWorkspace):
        raise TypeError("workspace must be a B300MoEWorkspace or None")

    requires_grad = any(
        tensor.requires_grad
        for tensor in (hidden_states, expert_weights, gate_up_weight, down_weight)
    )
    if workspace is not None and requires_grad:
        raise ValueError(
            "B300MoEWorkspace is inference-only; omit workspace when autograd is required"
        )

    extension: ModuleType | None = None
    compiled_build_info: Mapping[str, object] | None = None
    active_workspace = workspace
    if normalized_backend != "torch":
        extension, compiled_build_info = _resolve_compiled_backend(
            normalized_backend,
            hidden_states,
            expert_weights,
            gate_up_weight,
            down_weight,
            shape,
        )
        # A caller-provided workspace is the steady-state path. An ephemeral
        # workspace keeps the optional API usable without pretending that its
        # allocations are cached across independent calls.
        if active_workspace is None:
            active_workspace = B300MoEWorkspace(device=hidden_states.device)

    workspace_reused = False
    lock = (
        active_workspace._lock
        if active_workspace is not None
        else threading.RLock()
    )
    with lock:
        if active_workspace is not None:
            workspace_reused = active_workspace.reserve(
                num_tokens=shape.num_tokens,
                num_experts=shape.num_experts,
                top_k=shape.top_k,
                hidden_size=shape.hidden_size,
                intermediate_size=shape.intermediate_size,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
                backend=normalized_backend,
                quant_mode=normalized_quant_mode,
            )

        routing = None
        gemm_dispatch_info: dict[str, object] | None = None
        if normalized_backend == "torch":
            buffers = None if active_workspace is None else active_workspace.buffers
            output, routing = _routed_moe_reference_with_routing(
                hidden_states,
                expert_indices,
                expert_weights,
                gate_up_weight,
                down_weight,
                num_experts=shape.num_experts,
                top_k=shape.top_k,
                buffers=buffers,
            )
        else:
            if extension is None or active_workspace is None:
                raise RuntimeError("compiled backend initialization is incomplete")
            output, routing, gemm_dispatch_info = _call_compiled_bf16(
                extension,
                hidden_states,
                expert_indices,
                expert_weights,
                gate_up_weight,
                down_weight,
                shape,
                active_workspace,
                normalized_gemm_policy,
            )
        # Clone routing metadata while the workspace lock is still held. On
        # CUDA the completion event must be recorded *after* these async copies,
        # otherwise another stream/thread could reuse scratch before the clones
        # have consumed it.
        result_metadata = None
        if return_metadata:
            result_metadata = _metadata(
                backend=normalized_backend,
                architecture=_architecture_for(hidden_states.device),
                quant_mode=normalized_quant_mode,
                routing=routing,
                workspace=active_workspace,
                workspace_reused=workspace_reused,
                build_info=compiled_build_info,
            )
            if gemm_dispatch_info is not None:
                result_metadata["gemm_dispatch"] = gemm_dispatch_info
        if active_workspace is not None:
            active_workspace._record_use()

    if not return_metadata:
        return output
    if result_metadata is None:
        raise RuntimeError("metadata construction was unexpectedly skipped")
    return output, result_metadata


__all__ = [
    "B300Backend",
    "B300MoEMetadata",
    "B300MoEWorkspace",
    "QuantMode",
    "b300_moe_forward",
]
