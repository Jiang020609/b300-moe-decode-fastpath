"""Build configuration for the optional Local-MoE CUDA extensions.

The existing ``MOE_BUILD_CUDA=1`` mode builds the small V0 dispatch/combine
extension.  ``MOE_GOAL1B_BUILD=1`` is deliberately stricter: it requires the
complete Goal 1B source manifest and a recent CUTLASS checkout, and it emits
code for ``sm_103a`` only.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from pathlib import Path

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

ROOT = Path(__file__).resolve().parent

V0_SOURCES = (
    "csrc/bindings.cpp",
    "csrc/dispatch_cuda.cu",
    "csrc/combine_cuda.cu",
)
GOAL1B_SOURCES = (
    "csrc/bindings.cpp",
    "csrc/moe_fastpath.cu",
    "csrc/routing_kernels.cu",
    "csrc/permutation_kernels.cu",
    "csrc/quantization_kernels.cu",
    "csrc/grouped_gemm_sm103.cu",
    "csrc/combine_kernels.cu",
)
GOAL1B_MIN_CUTLASS = (4, 3, 1)
GOAL1B_MIN_CUDA = (13, 0)
GOAL1B_MIN_TORCH = (2, 12, 0)
GOAL1B_CUTLASS_HEADERS = (
    "include/cutlass/cutlass.h",
    "include/cutlass/version.h",
    "include/cutlass/arch/arch.h",
    "include/cutlass/gemm/device/gemm_universal_adapter.h",
    "include/cute/tensor.hpp",
    "tools/util/include/cutlass/util/packed_stride.hpp",
)
GOAL1B_NVCC_FLAGS = (
    "-O3",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-gencode=arch=compute_103a,code=sm_103a",
)
GOAL1B_DEFINES = (
    ("MOE_GOAL1B_BUILD", "1"),
    ("MOE_TARGET_SM103A", "1"),
    ("CUTLASS_ARCH_MMA_SM103A_ENABLED", "1"),
)


def _enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _cutlass_version(version_header: Path) -> tuple[int, int, int] | None:
    try:
        text = version_header.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    values: list[int] = []
    for component in ("MAJOR", "MINOR", "PATCH"):
        match = re.search(
            rf"^\s*#\s*define\s+CUTLASS_{component}\s+(\d+)\b",
            text,
            flags=re.MULTILINE,
        )
        if match is None:
            return None
        values.append(int(match.group(1)))
    return tuple(values)  # type: ignore[return-value]


def _nvcc_version(nvcc: Path) -> tuple[int, ...] | None:
    try:
        completed = subprocess.run(
            [str(nvcc), "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    match = re.search(r"\brelease\s+(\d+)\.(\d+)(?:\.(\d+))?", completed.stdout)
    if match is None:
        return None
    return tuple(int(part) for part in match.groups() if part is not None)


def _package_version(value: str | None) -> tuple[int, int, int] | None:
    if value is None:
        return None
    match = re.match(r"\s*(\d+)\.(\d+)(?:\.(\d+))?", value)
    if match is None:
        return None
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch or 0)


goal1b_build = _enabled("MOE_GOAL1B_BUILD")
force_cuda = _enabled("MOE_BUILD_CUDA") or goal1b_build
disable_cuda = os.environ.get("MOE_BUILD_CUDA", "").lower() in {"0", "false", "no"}
build_cuda = goal1b_build or force_cuda or (CUDA_HOME is not None and not disable_cuda)

if force_cuda and CUDA_HOME is None and not goal1b_build:
    raise RuntimeError(
        "MOE_BUILD_CUDA=1 was set, but CUDA_HOME could not be found. "
        "Install a CUDA toolkit and set CUDA_HOME."
    )

ext_modules = []
cmdclass = {}
if build_cuda:
    sources = V0_SOURCES
    include_dirs: list[str] = []
    define_macros: list[tuple[str, str]] = []
    nvcc_flags = ["-O3", "--expt-relaxed-constexpr"]

    if goal1b_build:
        errors: list[str] = []
        if platform.system() != "Linux":
            errors.append(
                "Goal 1B's PyTorch ATen grouped-MM adapter is supported only on Linux"
            )
        if not any(shutil.which(name) for name in ("c++", "g++", "clang++")):
            errors.append("no supported host C++ compiler was found on PATH")
        torch_version = _package_version(torch.__version__)
        if torch_version is None or torch_version < GOAL1B_MIN_TORCH:
            errors.append(
                f"PyTorch {torch.__version__} is unsupported; Goal 1B requires >=2.12.0"
            )
        torch_cuda_version = _package_version(torch.version.cuda)
        if torch_cuda_version is None or torch_cuda_version[:2] < GOAL1B_MIN_CUDA:
            errors.append(
                f"PyTorch CUDA {torch.version.cuda} is unsupported; "
                "Goal 1B requires a CUDA 13.x PyTorch build"
            )
        groupmm_header = (
            Path(torch.__file__).resolve().parent
            / "include/ATen/native/cuda/GroupMM.h"
        )
        if (
            not groupmm_header.is_file()
            or "bf16bf16_grouped_mm"
            not in groupmm_header.read_text(encoding="utf-8", errors="replace")
        ):
            errors.append(
                "the active PyTorch install lacks the bf16bf16_grouped_mm C++ header"
            )
        if not torch.cuda.is_available():
            errors.append("torch.cuda.is_available() is false; a B300 GPU is required")
        else:
            if not any(
                torch.cuda.get_device_capability(index) == (10, 3)
                for index in range(torch.cuda.device_count())
            ):
                errors.append("no visible CC 10.3 B300 GPU was found")
            torch_arches = tuple(torch.cuda.get_arch_list())
            if not any(
                arch.startswith(("sm_103", "compute_103"))
                for arch in torch_arches
            ):
                print(
                    "Goal 1B warning: torch.cuda.get_arch_list() does not "
                    f"advertise SM103: {torch_arches}; verifying the PyTorch "
                    "grouped-MM provider with an on-device runtime probe"
                )

            target_indices = [
                index
                for index in range(torch.cuda.device_count())
                if torch.cuda.get_device_capability(index) == (10, 3)
            ]

            if target_indices:
                grouped_mm = getattr(torch, "_grouped_mm", None)

                if not callable(grouped_mm):
                    errors.append(
                        "the active PyTorch build does not expose "
                        "torch._grouped_mm for the on-device provider probe"
                    )
                else:
                    device_index = target_indices[0]
                    device = f"cuda:{device_index}"

                    try:
                        with torch.no_grad():
                            tokens_per_expert = 16
                            expert_count = 2
                            k = 64
                            n = 64
                            total_tokens = (
                                tokens_per_expert * expert_count
                            )

                            torch.manual_seed(0)

                            activations = torch.randn(
                                total_tokens,
                                k,
                                device=device,
                                dtype=torch.bfloat16,
                            )
                            weights = torch.randn(
                                expert_count,
                                k,
                                n,
                                device=device,
                                dtype=torch.bfloat16,
                            )
                            offsets = torch.tensor(
                                [
                                    tokens_per_expert,
                                    total_tokens,
                                ],
                                device=device,
                                dtype=torch.int32,
                            )

                            output = grouped_mm(
                                activations,
                                weights,
                                offs=offsets,
                            )
                            reference = torch.cat(
                                [
                                    activations[
                                        :tokens_per_expert
                                    ] @ weights[0],
                                    activations[
                                        tokens_per_expert:
                                    ] @ weights[1],
                                ],
                                dim=0,
                            )

                            torch.cuda.synchronize(device_index)

                            if tuple(output.shape) != tuple(
                                reference.shape
                            ):
                                errors.append(
                                    "PyTorch grouped-MM provider probe "
                                    f"returned shape {tuple(output.shape)}, "
                                    f"expected {tuple(reference.shape)}"
                                )
                            elif output.dtype != torch.bfloat16:
                                errors.append(
                                    "PyTorch grouped-MM provider probe "
                                    f"returned dtype {output.dtype}, "
                                    "expected torch.bfloat16"
                                )
                            elif not bool(
                                torch.allclose(
                                    output.float(),
                                    reference.float(),
                                    rtol=5e-2,
                                    atol=1e-1,
                                )
                            ):
                                max_abs_diff = (
                                    output.float()
                                    - reference.float()
                                ).abs().max().item()
                                errors.append(
                                    "PyTorch grouped-MM provider probe "
                                    "produced incorrect output; "
                                    f"max_abs_diff={max_abs_diff}"
                                )
                            else:
                                print(
                                    "Goal 1B PyTorch grouped-MM "
                                    "runtime probe: PASS"
                                )

                    except Exception as error:
                        errors.append(
                            "PyTorch grouped-MM provider runtime probe "
                            f"failed on {device}: "
                            f"{type(error).__name__}: {error}"
                        )
        sources = GOAL1B_SOURCES
        missing_sources = [source for source in sources if not (ROOT / source).is_file()]
        if missing_sources:
            errors.append(
                "the Goal 1B source manifest is incomplete; missing: "
                + ", ".join(missing_sources)
            )
        if CUDA_HOME is None:
            errors.append(
                "CUDA_HOME could not be found; Goal 1B requires CUDA Toolkit >=13.0"
            )
        else:
            cuda_root = Path(CUDA_HOME)
            nvcc = next(
                (
                    cuda_root / "bin" / filename
                    for filename in ("nvcc", "nvcc.exe")
                    if (cuda_root / "bin" / filename).is_file()
                ),
                None,
            )
            if nvcc is None:
                errors.append(f"nvcc was not found under {cuda_root / 'bin'}")
            else:
                cuda_version = _nvcc_version(nvcc)
                if cuda_version is None:
                    errors.append(f"could not determine the CUDA version from {nvcc}")
                elif cuda_version[:2] < GOAL1B_MIN_CUDA:
                    rendered = ".".join(str(part) for part in cuda_version)
                    errors.append(
                        f"CUDA {rendered} is too old; Goal 1B sm_103a requires >=13.0"
                    )

        cutlass_value = os.environ.get("CUTLASS_PATH", "").strip()
        cutlass_root = Path(cutlass_value).expanduser() if cutlass_value else None
        if cutlass_root is None:
            errors.append(
                "CUTLASS_PATH is not set; point it to a CUTLASS checkout >=4.3.1"
            )
        else:
            cutlass_root = cutlass_root.resolve()
            missing_headers = [
                header
                for header in GOAL1B_CUTLASS_HEADERS
                if not (cutlass_root / header).is_file()
            ]
            if missing_headers:
                errors.append(
                    f"CUTLASS_PATH={cutlass_root} is missing required files: "
                    + ", ".join(missing_headers)
                )
            version = _cutlass_version(cutlass_root / "include/cutlass/version.h")
            if version is None:
                errors.append("could not parse CUTLASS version from include/cutlass/version.h")
            elif version < GOAL1B_MIN_CUTLASS:
                rendered = ".".join(str(part) for part in version)
                errors.append(f"CUTLASS {rendered} is too old; Goal 1B requires >=4.3.1")
            arch_header = cutlass_root / "include/cutlass/arch/arch.h"
            if arch_header.is_file():
                arch_text = arch_header.read_text(encoding="utf-8", errors="replace")
                if re.search(r"\bSm103\b", arch_text) is None:
                    errors.append(
                        "CUTLASS arch.h does not define Sm103; refusing an architecture fallback"
                    )
            include_dirs = [
                str(cutlass_root / "include"),
                str(cutlass_root / "tools/util/include"),
            ]

        if errors:
            detail = "\n  - ".join(errors)
            raise RuntimeError(f"Goal 1B build prerequisites failed:\n  - {detail}")

        nvcc_flags = list(GOAL1B_NVCC_FLAGS)
        define_macros = list(GOAL1B_DEFINES)
    elif _enabled("MOE_B300_SM103A"):
        # Preserve the original V0 opt-in cross-build behavior.
        nvcc_flags.append("-gencode=arch=compute_103a,code=sm_103a")

    ext_modules.append(
        CUDAExtension(
            name="fastpath._C",
            sources=[str(ROOT / source) for source in sources],
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args={"cxx": ["-O3"], "nvcc": nvcc_flags},
        )
    )
    cmdclass["build_ext"] = BuildExtension

setup(
    name="b300-moe-decode-fastpath",
    version="0.2.0",
    description="Local-MoE dispatch/combine CUDA V0 and PyTorch reference",
    packages=find_packages(exclude=("tests", "tests.*")),
    python_requires=">=3.10",
    install_requires=["torch>=2.1", "PyYAML>=6.0"],
    extras_require={"test": ["pytest>=7.4"]},
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
