#include "moe_fastpath.h"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr int kMaxBlocks = 65535;

int launch_blocks(int64_t work_items) {
  return static_cast<int>(
      std::min<int64_t>((work_items + kThreads - 1) / kThreads, kMaxBlocks));
}

__global__ void count_experts_kernel(
    const int64_t* __restrict__ expert_ids,
    int64_t* __restrict__ counts,
    int64_t assignments,
    int64_t experts) {
  for (int64_t assignment = blockIdx.x * blockDim.x + threadIdx.x;
       assignment < assignments;
       assignment += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t expert = expert_ids[assignment];
    if (expert >= 0 && expert < experts) {
      atomicAdd(reinterpret_cast<unsigned long long*>(counts + expert), 1ULL);
    }
  }
}

__global__ void exclusive_offsets_kernel(
    const int64_t* __restrict__ counts,
    int64_t* __restrict__ offsets,
    int64_t experts) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    int64_t running = 0;
    offsets[0] = 0;
    for (int64_t expert = 0; expert < experts; ++expert) {
      running += counts[expert];
      offsets[expert + 1] = running;
    }
  }
}

__global__ void stable_mapping_kernel(
    const int64_t* __restrict__ expert_ids,
    const int64_t* __restrict__ offsets,
    int64_t* __restrict__ assignment_to_permuted,
    int64_t* __restrict__ permuted_to_assignment,
    int64_t assignments,
    int64_t experts) {
  const int64_t expert = blockIdx.x * blockDim.x + threadIdx.x;
  if (expert >= experts) {
    return;
  }

  // Decode has very small A=T*K. One thread per expert is intentionally simple
  // and deterministic: assignments are visited in token-major/rank-minor order.
  int64_t position = offsets[expert];
  for (int64_t assignment = 0; assignment < assignments; ++assignment) {
    if (expert_ids[assignment] == expert) {
      assignment_to_permuted[assignment] = position;
      permuted_to_assignment[position] = assignment;
      ++position;
    }
  }
}

}  // namespace

void routing_metadata_out_cuda(
    torch::Tensor expert_ids,
    int64_t num_experts,
    torch::Tensor counts,
    torch::Tensor offsets,
    torch::Tensor assignment_to_permuted,
    torch::Tensor permuted_to_assignment) {
  TORCH_CHECK(
      expert_ids.is_cuda() && counts.is_cuda() && offsets.is_cuda() &&
          assignment_to_permuted.is_cuda() && permuted_to_assignment.is_cuda(),
      "routing tensors must be CUDA tensors");
  TORCH_CHECK(
      expert_ids.device() == counts.device() && expert_ids.device() == offsets.device() &&
          expert_ids.device() == assignment_to_permuted.device() &&
          expert_ids.device() == permuted_to_assignment.device(),
      "routing tensors must share a CUDA device");
  TORCH_CHECK(expert_ids.dim() == 2, "expert_ids must have shape [T, K]");
  TORCH_CHECK(
      expert_ids.scalar_type() == at::kLong && counts.scalar_type() == at::kLong &&
          offsets.scalar_type() == at::kLong &&
          assignment_to_permuted.scalar_type() == at::kLong &&
          permuted_to_assignment.scalar_type() == at::kLong,
      "routing tensors must be int64");
  TORCH_CHECK(
      expert_ids.is_contiguous() && counts.is_contiguous() && offsets.is_contiguous() &&
          assignment_to_permuted.is_contiguous() && permuted_to_assignment.is_contiguous(),
      "routing tensors must be contiguous");
  TORCH_CHECK(expert_ids.numel() > 0, "routing tensors must not be empty");
  TORCH_CHECK(num_experts > 0 && num_experts <= 256, "num_experts must be in [1, 256]");
  const int64_t top_k = expert_ids.size(1);
  TORCH_CHECK(
      top_k == 1 || top_k == 2 || top_k == 4 || top_k == 8,
      "top_k must be 1, 2, 4, or 8");
  TORCH_CHECK(counts.dim() == 1 && counts.numel() == num_experts, "expert_counts must have shape [E]");
  TORCH_CHECK(offsets.dim() == 1 && offsets.numel() == num_experts + 1, "expert_offsets must have shape [E+1]");
  TORCH_CHECK(assignment_to_permuted.dim() == 1 && assignment_to_permuted.numel() == expert_ids.numel(), "assignment_to_permuted must have shape [T*K]");
  TORCH_CHECK(permuted_to_assignment.dim() == 1 && permuted_to_assignment.numel() == expert_ids.numel(), "permuted_to_assignment must have shape [T*K]");

  const c10::cuda::CUDAGuard guard(expert_ids.device());
  counts.zero_();
  // Initialize defensively so invalid ids cannot leave uninitialized indices.
  assignment_to_permuted.fill_(-1);
  permuted_to_assignment.fill_(-1);
  const cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(expert_ids.get_device()).stream();

  count_experts_kernel<<<launch_blocks(expert_ids.numel()), kThreads, 0, stream>>>(
      expert_ids.data_ptr<int64_t>(), counts.data_ptr<int64_t>(), expert_ids.numel(), num_experts);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  exclusive_offsets_kernel<<<1, 1, 0, stream>>>(
      counts.data_ptr<int64_t>(), offsets.data_ptr<int64_t>(), num_experts);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  stable_mapping_kernel<<<1, kThreads, 0, stream>>>(
      expert_ids.data_ptr<int64_t>(),
      offsets.data_ptr<int64_t>(),
      assignment_to_permuted.data_ptr<int64_t>(),
      permuted_to_assignment.data_ptr<int64_t>(),
      expert_ids.numel(),
      num_experts);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<torch::Tensor> routing_metadata_cuda(
    torch::Tensor expert_ids,
    int64_t num_experts) {
  auto counts = at::empty({num_experts}, expert_ids.options());
  auto offsets = at::empty({num_experts + 1}, expert_ids.options());
  auto assignment_to_permuted = at::empty({expert_ids.numel()}, expert_ids.options());
  auto permuted_to_assignment = at::empty({expert_ids.numel()}, expert_ids.options());
  routing_metadata_out_cuda(
      expert_ids,
      num_experts,
      counts,
      offsets,
      assignment_to_permuted,
      permuted_to_assignment);
  return {counts, offsets, assignment_to_permuted, permuted_to_assignment};
}
