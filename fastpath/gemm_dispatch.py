"""Goal 1C hybrid GEMM dispatch for decode-sized MoE batches.

Motivation (measured on B300, 2026-07-20, see
``results/b300_goal1b_20260720/step4_02_gemm_only_speedup.csv``):

- Per-active-expert ``torch.matmul`` beat ``torch._grouped_mm`` in 26 of the
  28 decode-sized shapes (geomean speedup of grouped-MM was only 0.72-0.78).
- Grouped-MM crossed over only for the ``down`` projection with
  ``total_rows >= 32`` and ``>= 32`` active experts (1.036x / 1.051x).

The grouped kernel pays per-*declared*-group scheduling cost for all ``E``
groups (including empty ones), while a per-expert loop launches work only for
active experts.  At decode sizes each active expert holds one or two rows, so
the loop wins despite launching more kernels.

This module keeps the decision logic and the per-expert executor separate
from the orchestration in :mod:`fastpath.b300_moe` so both are unit-testable
on CPU.  The policy is measured-data-driven and intentionally conservative:
anything outside the measured decode range falls back to the grouped kernel.

Goal 1C step 2 adds the ``fused`` strategy: a repository-compiled kernel
(``csrc/fused_gemm_kernels.cu``) that reads the routing prefix sum on device
and computes every expert's tiny-M GEMM in one launch.  The A/B run in
``results/b300_goal1c_20260720`` showed the per-expert loop's host offsets
sync eating ~2/3 of its kernel-level win; ``fused`` removes that sync and
the per-expert launch overhead entirely.  It is opt-in (``policy="fused"``)
until B300 A/B data justifies promoting it into ``auto``.
"""

from __future__ import annotations

from typing import Literal, Sequence

import torch

GemmOperation = Literal["gate_up", "down"]
GemmStrategy = Literal["grouped", "per_expert", "fused"]
GemmPolicy = Literal["auto", "grouped", "per_expert", "fused"]

# Measured decode range: the GEMM-only benchmark covered total_rows (M) up to
# 64.  Beyond twice that we have no evidence, and the grouped kernel's fixed
# cost amortizes with M, so "auto" defers to grouped there.
_MEASURED_MAX_ROWS = 64

# Crossover measured only for the down projection: E=64 with all experts
# active and M=32 (1.036x) / M=64 (1.051x).  gate_up never crossed over
# (best case 0.873x at M=64).
_DOWN_GROUPED_MIN_ROWS = 32
_DOWN_GROUPED_MIN_ACTIVE_EXPERTS = 32


def normalize_gemm_policy(policy: object) -> GemmPolicy:
    if not isinstance(policy, str):
        raise TypeError("gemm_policy must be a string")
    if policy not in ("auto", "grouped", "per_expert", "fused"):
        raise ValueError(
            "gemm_policy must be 'auto', 'grouped', 'per_expert', or 'fused'"
        )
    return policy  # type: ignore[return-value]


def select_gemm_strategy(
    operation: GemmOperation,
    *,
    total_rows: int,
    active_experts: int,
    policy: GemmPolicy = "auto",
) -> GemmStrategy:
    """Choose the execution strategy for one grouped-GEMM stage.

    ``total_rows`` is the number of permuted assignment rows (``A = T * K``)
    and ``active_experts`` is the number of experts with at least one row.
    Forced policies are honored verbatim; ``auto`` applies the measured
    B300 decision rules documented in the module docstring.
    """

    if operation not in ("gate_up", "down"):
        raise ValueError("operation must be 'gate_up' or 'down'")
    for name, value in (("total_rows", total_rows), ("active_experts", active_experts)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer")
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    if active_experts > total_rows:
        raise ValueError("active_experts cannot exceed total_rows")
    normalized = normalize_gemm_policy(policy)

    if normalized == "grouped":
        return "grouped"
    if normalized == "per_expert":
        return "per_expert"
    if normalized == "fused":
        # Forced opt-in: the fused kernel is new in Goal 1C step 2 and stays
        # out of "auto" until a B300 A/B run ranks it against the two
        # validated strategies.
        return "fused"

    # Unmeasured territory (large prefill-like batches): stay on the
    # validated grouped path rather than extrapolating the loop's win.
    if total_rows > 2 * _MEASURED_MAX_ROWS:
        return "grouped"
    # The only measured crossover: down projection, many rows spread over
    # many active experts.
    if (
        operation == "down"
        and total_rows >= _DOWN_GROUPED_MIN_ROWS
        and active_experts >= _DOWN_GROUPED_MIN_ACTIVE_EXPERTS
    ):
        return "grouped"
    return "per_expert"


def per_expert_gemm_out(
    inputs: torch.Tensor,
    packed_weights: torch.Tensor,
    expert_offsets: Sequence[int],
    out: torch.Tensor,
) -> None:
    """Run one matmul per non-empty expert, writing into caller-owned ``out``.

    ``packed_weights`` is the same zero-copy ``[E, K, N]`` transpose view the
    grouped path consumes, so both strategies read identical weight memory
    and no repacking or weight cache can go stale.  ``expert_offsets`` are
    host-side exclusive-scan offsets of length ``E + 1`` (``offsets[e]`` to
    ``offsets[e + 1]`` is expert ``e``'s row range); they must already be on
    the host because slicing bounds are Python integers.

    Empty experts are skipped, which is exactly why this path wins at decode
    sizes.  Rows of ``out`` outside every range are left untouched, matching
    the grouped kernel's contract for zero-size groups.
    """

    if inputs.dim() != 2 or out.dim() != 2:
        raise ValueError("inputs and out must be 2-D [rows, features]")
    if packed_weights.dim() != 3:
        raise ValueError("packed_weights must be [E, K, N]")
    num_experts = packed_weights.shape[0]
    if len(expert_offsets) != num_experts + 1:
        raise ValueError("expert_offsets must have length num_experts + 1")
    rows = inputs.shape[0]
    if int(expert_offsets[0]) != 0 or int(expert_offsets[-1]) != rows:
        raise ValueError("expert_offsets must scan from 0 to inputs.shape[0]")
    if out.shape[0] != rows or out.shape[1] != packed_weights.shape[2]:
        raise ValueError("out must be [rows, N]")
    if inputs.shape[1] != packed_weights.shape[1]:
        raise ValueError("inputs feature dim must match packed_weights K dim")

    previous = 0
    for expert in range(num_experts):
        start = int(expert_offsets[expert])
        end = int(expert_offsets[expert + 1])
        if start != previous or end < start or end > rows:
            raise ValueError(
                "expert_offsets must be non-decreasing and within range"
            )
        previous = end
        if end == start:
            continue
        # Row slices of a contiguous 2-D buffer are themselves contiguous, so
        # matmul can write straight into the workspace without a temp copy.
        torch.matmul(
            inputs[start:end], packed_weights[expert], out=out[start:end]
        )


__all__ = [
    "GemmOperation",
    "GemmPolicy",
    "GemmStrategy",
    "normalize_gemm_policy",
    "per_expert_gemm_out",
    "select_gemm_strategy",
]
