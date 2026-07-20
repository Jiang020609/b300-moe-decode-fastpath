# Goal 1B environment audit

Audit date: 2026-07-20 (Asia/Shanghai)

This document separates facts observed on the current development host from the
requirements of the B300 verification host. A configured filename, requested
compiler flag, or fallback implementation is not treated as proof that a B300 or
NVFP4 kernel was built or executed.

## Result

The current host is a CPU-only Windows development environment. It cannot build
or run the Goal 1B CUDA extension:

- no NVIDIA GPU or NVIDIA driver tools are visible;
- the installed PyTorch is a CPU build;
- no CUDA toolkit or `nvcc` is installed;
- no host C++ compiler, CMake, or Ninja is available on `PATH`;
- no CUTLASS checkout or installed CUTLASS package was found.

Consequently, the actual compiled CUDA architecture on this host is **none**,
the actual runtime backend is **`torch`**, and NVFP4 is **not enabled**. B300
build, kernel correctness, and latency results remain unverified until the same
repository is run on the target host described below. There is no silent CUDA
fallback: requesting a compiled backend must fail when its prerequisites are
absent.

## Observed host

| Item | Observed value |
|---|---|
| OS | Windows 11 x86-64 (`Windows-11-10.0.26200-SP0`) |
| Python | 3.12.10, MSC v.1943, `C:\Users\34310\AppData\Local\Programs\Python\Python312\python.exe` |
| pip | 25.0.1 |
| PyTorch | `2.12.0+cpu` |
| PyTorch CUDA version | `None` |
| PyTorch CUDA build | `torch._C._has_cuda == False` |
| CUDA runtime available | `False` |
| CUDA devices | 0 |
| CUDA architecture list | empty |
| `torch.utils.cpp_extension.CUDA_HOME` | `None` |
| GPU visible to the OS audit | Intel Arc 140T plus virtual display adapters; no NVIDIA adapter |
| `nvidia-smi` | not found |
| `nvcc` | not found |
| Host C++ compiler | `cl`, `g++`, and `clang++` not found on `PATH` |
| Build tools | `cmake`, `ninja`, `nmake`, and `msbuild` not found on `PATH` |
| CUTLASS | not found; version unavailable |
| `CUDA_HOME`, `CUDA_PATH` | unset |
| `CUTLASS_PATH`, `CUTLASS_HOME` | unset |
| Requested/actual CUDA architecture | `sm_103a` / none built |
| Actual CUDA backend | unavailable |
| Actual NVFP4 backend | unavailable; not built or executed |

Calling `torch.cuda.get_device_capability()` on this host raises
`AssertionError: Torch not compiled with CUDA enabled`; this is recorded instead
of fabricating a capability value. Forcing the current extension build with
`MOE_BUILD_CUDA=1` fails during setup because `CUDA_HOME` is unavailable.

## CUTLASS and SM103 audit

There is no local CUTLASS source tree, header installation, package, submodule,
or version file to inspect. Therefore none of the following is locally proven:

- the CUTLASS architecture tag selected by a compiled kernel;
- compilation of `-arch=sm_103a` or
  `-gencode=arch=compute_103a,code=sm_103a`;
- availability of the Blackwell MoE grouped-GEMM examples;
- availability of SM103 block-scaled NVFP4 grouped GEMM;
- the activation, weight, or scale datatype used by an executed kernel.

The B300 build flow must validate these items from the selected `CUTLASS_PATH`
and the compiler output. It must not infer NVFP4 support from a Python
`quant_mode` string.

## Required B300 verification host

The reproducible target is a Linux B300/GB300 host with:

- an NVIDIA B300-class GPU reporting compute capability `(10, 3)`;
- a CUDA 13.x development toolkit whose `nvcc` accepts `sm_103a`;
- a CUDA-enabled PyTorch build compatible with that toolkit and driver;
- a CUTLASS checkout with explicit SM103 support and the required grouped GEMM
  examples/headers;
- a supported host C++ compiler, CMake, and Ninja;
- `CUDA_HOME` and `CUTLASS_PATH` pointing at those exact installations.

The build and smoke scripts are responsible for printing the resolved paths,
versions, GPU name/capability, and complete architecture flags before compiling.
Runtime metadata must report the compiled backend and datatype actually used.

## Commands executed on this host

```text
python -c "import sys, torch; from torch.utils.cpp_extension import CUDA_HOME; ..."
python -c "import torch; print(torch.__config__.show())"
python -m pip --version
where.exe python
where.exe nvcc
nvcc --version
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader
cl
cmake --version
ninja --version
python -m pytest -q
```

The initial repository test baseline at audit time was `33 passed, 12 skipped`;
all 12 skips were CUDA-extension tests. After the Goal 1B work, the full suite
reported `86 passed, 109 skipped`; the Goal 1B-focused selection reported
`50 passed, 94 skipped, 51 deselected`. The additional skips are the explicit
B300/CC 10.3 CUDA matrices, so neither result is B300 kernel evidence.

The final preflight also verifies Linux host support, a host C++ compiler,
PyTorch `>=2.12`, a CUDA 13.x PyTorch build, the internal GroupMM header/operator,
an SM103 entry in `torch.cuda.get_arch_list()`, and a throwaway
`nvcc -arch=sm_103a` translation-unit compile. On this host the compile probe
could not start because `nvcc` is absent; preflight returned status `FAILED`
and exit code 2, as required, and did not invoke the extension build.
