#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda.h>
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
    int64_t* __restrict__ expert_counts,
    int64_t num_assignments,
    int64_t num_experts) {
  for (int64_t assignment = blockIdx.x * blockDim.x + threadIdx.x;
       assignment < num_assignments;
       assignment += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t expert = expert_ids[assignment];
    if (expert >= 0 && expert < num_experts) {
      atomicAdd(
          reinterpret_cast<unsigned long long*>(expert_counts + expert), 1ULL);
    }
  }
}

__global__ void exclusive_offsets_kernel(
    const int64_t* __restrict__ expert_counts,
    int64_t* __restrict__ expert_offsets,
    int64_t num_experts) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    int64_t offset = 0;
    expert_offsets[0] = 0;
    for (int64_t expert = 0; expert < num_experts; ++expert) {
      offset += expert_counts[expert];
      expert_offsets[expert + 1] = offset;
    }
  }
}

__global__ void stable_assign_positions_kernel(
    const int64_t* __restrict__ expert_ids,
    const int64_t* __restrict__ expert_offsets,
    int64_t* __restrict__ assignment_to_permuted,
    int64_t* __restrict__ permuted_to_assignment,
    int64_t num_assignments,
    int64_t num_experts) {
  const int64_t expert = blockIdx.x * blockDim.x + threadIdx.x;
  if (expert >= num_experts) {
    return;
  }

  // One thread owns one expert and visits assignments in token-major order.
  // This deliberately favors the small-A decode regime: it is deterministic,
  // does not copy counts to the host, and exactly matches torch.argsort(...,
  // stable=True). Goal 1C may replace it with a stable radix-sort/scan kernel.
  int64_t position = expert_offsets[expert];
  for (int64_t assignment = 0; assignment < num_assignments; ++assignment) {
    if (expert_ids[assignment] == expert) {
      assignment_to_permuted[assignment] = position;
      permuted_to_assignment[position] = assignment;
      ++position;
    }
  }
}

template <typename scalar_t>
__global__ void gather_hidden_kernel(
    const scalar_t* __restrict__ hidden_states,
    const int64_t* __restrict__ assignment_to_permuted,
    scalar_t* __restrict__ permuted_hidden,
    int64_t num_assignments,
    int64_t hidden_size,
    int64_t top_k) {
  const int64_t work_items = num_assignments * hidden_size;
  for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
       linear < work_items;
       linear += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t assignment = linear / hidden_size;
    const int64_t hidden = linear % hidden_size;
    const int64_t token = assignment / top_k;
    const int64_t position = assignment_to_permuted[assignment];
    permuted_hidden[position * hidden_size + hidden] =
        hidden_states[token * hidden_size + hidden];
  }
}

void validate_dispatch(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_ids,
    int64_t num_experts) {
  TORCH_CHECK(hidden_states.is_cuda(), "hidden_states must be a CUDA tensor");
  TORCH_CHECK(expert_ids.is_cuda(), "expert_ids must be a CUDA tensor");
  TORCH_CHECK(
      hidden_states.device() == expert_ids.device(),
      "hidden_states and expert_ids must be on the same CUDA device");
  TORCH_CHECK(hidden_states.dim() == 2, "hidden_states must have shape [T, H]");
  TORCH_CHECK(expert_ids.dim() == 2, "expert_ids must have shape [T, K]");
  TORCH_CHECK(hidden_states.size(0) > 0, "num_tokens must be positive");
  TORCH_CHECK(hidden_states.size(1) > 0, "hidden_size must be positive");
  TORCH_CHECK(
      hidden_states.size(0) == expert_ids.size(0),
      "hidden_states and expert_ids must have the same num_tokens");
  const int64_t top_k = expert_ids.size(1);
  TORCH_CHECK(
      top_k == 1 || top_k == 2 || top_k == 4 || top_k == 8,
      "top_k must be 1, 2, 4, or 8");
  TORCH_CHECK(
      num_experts >= 4 && num_experts <= 256,
      "num_experts must be in [4, 256]");
  TORCH_CHECK(
      hidden_states.scalar_type() == at::kFloat ||
          hidden_states.scalar_type() == at::kHalf ||
          hidden_states.scalar_type() == at::kBFloat16,
      "hidden_states dtype must be float32, float16, or bfloat16");
  TORCH_CHECK(expert_ids.scalar_type() == at::kLong, "expert_ids must have dtype int64");
  TORCH_CHECK(hidden_states.is_contiguous(), "hidden_states must be contiguous");
  TORCH_CHECK(expert_ids.is_contiguous(), "expert_ids must be contiguous");
  TORCH_CHECK(!hidden_states.requires_grad(), "dispatch V0 does not support backward");
}

}  // namespace

std::vector<at::Tensor> dispatch_cuda(
    at::Tensor hidden_states,
    at::Tensor expert_ids,
    int64_t num_experts) {
  validate_dispatch(hidden_states, expert_ids, num_experts);
  const c10::cuda::CUDAGuard device_guard(hidden_states.device());

  const int64_t num_tokens = hidden_states.size(0);
  const int64_t hidden_size = hidden_states.size(1);
  const int64_t top_k = expert_ids.size(1);
  const int64_t num_assignments = num_tokens * top_k;
  const at::Tensor flat_experts = expert_ids.reshape({num_assignments});
  at::Tensor expert_counts = at::zeros({num_experts}, flat_experts.options());
  at::Tensor expert_offsets = at::empty({num_experts + 1}, flat_experts.options());
  at::Tensor assignment_to_permuted = at::empty_like(flat_experts);
  at::Tensor permuted_to_assignment = at::empty_like(flat_experts);
  const cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(hidden_states.get_device()).stream();

  count_experts_kernel<<<launch_blocks(num_assignments), kThreads, 0, stream>>>(
      flat_experts.data_ptr<int64_t>(),
      expert_counts.data_ptr<int64_t>(),
      num_assignments,
      num_experts);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  exclusive_offsets_kernel<<<1, 1, 0, stream>>>(
      expert_counts.data_ptr<int64_t>(),
      expert_offsets.data_ptr<int64_t>(),
      num_experts);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  stable_assign_positions_kernel<<<1, kThreads, 0, stream>>>(
      flat_experts.data_ptr<int64_t>(),
      expert_offsets.data_ptr<int64_t>(),
      assignment_to_permuted.data_ptr<int64_t>(),
      permuted_to_assignment.data_ptr<int64_t>(),
      num_assignments,
      num_experts);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  at::Tensor permuted_hidden = at::empty(
      {num_assignments, hidden_size}, hidden_states.options());
  const int64_t copy_items = num_assignments * hidden_size;
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      hidden_states.scalar_type(),
      "moe_dispatch_gather",
      [&] {
        gather_hidden_kernel<scalar_t><<<launch_blocks(copy_items), kThreads, 0, stream>>>(
            hidden_states.data_ptr<scalar_t>(),
            assignment_to_permuted.data_ptr<int64_t>(),
            permuted_hidden.data_ptr<scalar_t>(),
            num_assignments,
            hidden_size,
            top_k);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {
      permuted_hidden,
      expert_counts,
      expert_offsets,
      assignment_to_permuted,
      permuted_to_assignment};
}
