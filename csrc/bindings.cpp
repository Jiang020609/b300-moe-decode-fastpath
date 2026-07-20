#include <torch/extension.h>
#include <torch/version.h>

#include <cstdint>
#include <string>
#include <vector>

#ifdef MOE_GOAL1B_BUILD
#include "moe_fastpath.h"
#include <cutlass/version.h>
#include <cuda_runtime_api.h>
#endif

std::vector<torch::Tensor> dispatch_cuda(
    torch::Tensor hidden_states,
    torch::Tensor expert_ids,
    int64_t num_experts);

torch::Tensor combine_cuda(
    torch::Tensor expert_outputs,
    torch::Tensor assignment_to_permuted,
    torch::Tensor routing_weights);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.doc() = "Local-MoE CUDA dispatch, grouped GEMM, SwiGLU, and combine";
  module.def("dispatch", &dispatch_cuda, "Expert-major token dispatch (CUDA)");
  module.def("combine", &combine_cuda, "Gather-style expert combine (CUDA)");

#ifdef MOE_GOAL1B_BUILD
  module.def(
      "routing_metadata",
      &routing_metadata_cuda,
      "Stable expert counts, offsets, and mappings (CUDA, internal)");
  module.def(
      "routing_metadata_out",
      &routing_metadata_out_cuda,
      "Stable routing metadata into caller-owned storage (CUDA, internal)");
  module.def(
      "permute_out",
      &permute_out_cuda,
      "Stable token permutation into caller-owned storage (CUDA, internal)");
  module.def(
      "combine_out",
      &combine_out_cuda,
      "Routing-weighted combine into caller-owned storage (CUDA, internal)");
  module.def(
      "swiglu_out",
      &swiglu_out_cuda,
      "SwiGLU into caller-owned storage (CUDA, internal)");
  module.def(
      "grouped_gemm_bf16_out",
      &grouped_gemm_bf16_out_cuda,
      "ATen BF16 grouped GEMM into caller-owned storage (CUDA, internal)");
  module.def(
      "fused_grouped_gemm_bf16_out",
      &fused_grouped_gemm_bf16_out_cuda,
      "Goal 1C device-offset fused BF16 grouped GEMM (CUDA, internal)");
  module.def(
      "nvfp4_grouped_gemm_available",
      &nvfp4_grouped_gemm_available_cuda,
      "Whether a real SM103 NVFP4 grouped GEMM adapter was compiled");
  module.def(
      "nvfp4_grouped_gemm_unavailable_reason",
      &nvfp4_grouped_gemm_unavailable_reason_cuda,
      "Why the real SM103 NVFP4 grouped GEMM adapter is unavailable");

  module.def("build_info", []() {
    pybind11::dict info;
    const std::string external_cutlass_version =
        std::to_string(CUTLASS_MAJOR) + "." +
        std::to_string(CUTLASS_MINOR) + "." +
        std::to_string(CUTLASS_PATCH);
    info["goal"] = "1B";
    info["wrapper_compiled_architecture"] = "sm_103a";
    info["compiled_architecture"] = "sm_103a";
    info["bf16_grouped_gemm_provider"] =
        "at::cuda::detail::bf16bf16_grouped_mm (PyTorch binary)";
    info["bf16_provider_implementation"] =
        "PyTorch CUDA GroupMM; CUTLASS implementation requires target-host smoke";
    info["grouped_gemm_provider_architecture"] =
        "precompiled PyTorch binary; runtime requires get_arch_list() SM103";
    info["external_cutlass_headers_version"] = external_cutlass_version;
    info["cutlass_version"] = external_cutlass_version;
    info["pytorch_cxx_version"] = TORCH_VERSION;
    info["cuda_runtime_version"] = CUDART_VERSION;
    info["nvfp4_compiled"] = nvfp4_grouped_gemm_available_cuda();
    info["uses_fallback"] = false;
    return info;
  });
#else
  module.def("build_info", []() {
    pybind11::dict info;
    info["goal"] = "dispatch_combine_v0";
    info["compiled_architecture"] = "toolchain_default";
    info["bf16_grouped_gemm_provider"] = pybind11::none();
    info["cutlass_version"] = pybind11::none();
    info["cuda_runtime_version"] = pybind11::none();
    info["nvfp4_compiled"] = false;
    info["uses_fallback"] = false;
    return info;
  });
#endif
}
