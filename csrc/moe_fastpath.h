#pragma once

#include <torch/extension.h>

#include <cstdint>
#include <string>
#include <vector>

// Goal 1B device-only routing metadata. The mappings use the same directions
// as fastpath.DispatchResult and baseline.RoutingResult:
//   assignment_to_permuted: token-major assignment -> expert-major row
//   permuted_to_assignment: expert-major row -> token-major assignment
std::vector<torch::Tensor> routing_metadata_cuda(
    torch::Tensor expert_ids,
    int64_t num_experts);

void routing_metadata_out_cuda(
    torch::Tensor expert_ids,
    int64_t num_experts,
    torch::Tensor expert_counts,
    torch::Tensor expert_offsets,
    torch::Tensor assignment_to_permuted,
    torch::Tensor permuted_to_assignment);

void permute_out_cuda(
    torch::Tensor hidden_states,
    torch::Tensor permuted_to_assignment,
    int64_t top_k,
    torch::Tensor output);

void combine_out_cuda(
    torch::Tensor expert_outputs,
    torch::Tensor assignment_to_permuted,
    torch::Tensor routing_weights,
    torch::Tensor output);

void swiglu_out_cuda(torch::Tensor gate_up, torch::Tensor output);

// This wrapper calls PyTorch's CUDA bf16bf16_grouped_mm implementation. That
// implementation is backed by CUTLASS in CUDA-enabled PyTorch builds and,
// unlike the public allocating operator, accepts caller-owned output storage.
void grouped_gemm_bf16_out_cuda(
    torch::Tensor activations,
    torch::Tensor packed_weights,
    torch::Tensor expert_offsets_i32,
    torch::Tensor output);

// Goal 1C fused device-side grouped GEMM. Reads the int64 [E + 1] routing
// prefix sum directly on device (no host offsets sync) and computes every
// expert's tiny-M GEMM in a single launch. Optimized for decode shapes where
// the weight stream is the bandwidth bottleneck.
void fused_grouped_gemm_bf16_out_cuda(
    torch::Tensor activations,
    torch::Tensor packed_weights,
    torch::Tensor expert_offsets,
    torch::Tensor output);

bool nvfp4_grouped_gemm_available_cuda();
std::string nvfp4_grouped_gemm_unavailable_reason_cuda();
