"""Deterministic preflight and build entry point for the Goal 1B extension.

This module never selects a lower CUDA architecture.  A successful preflight
means that the current interpreter, toolkit, visible GPU, CUTLASS checkout, and
repository source manifest are all suitable for an ``sm_103a`` build.  NVFP4
evidence is reported separately because its absence must not masquerade as an
enabled NVFP4 kernel.
"""

from __future__ import annotations

import argparse
import importlib
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
TARGET_ARCH = "sm_103a"
TARGET_CAPABILITY = (10, 3)
MIN_CUDA_VERSION = (13, 0)
MIN_CUTLASS_VERSION = (4, 3, 1)
MIN_TORCH_VERSION = (2, 12, 0)
MIN_TORCH_CUDA_VERSION = (13, 0)

GOAL1B_SOURCES = (
    "csrc/bindings.cpp",
    "csrc/moe_fastpath.cu",
    "csrc/routing_kernels.cu",
    "csrc/permutation_kernels.cu",
    "csrc/quantization_kernels.cu",
    "csrc/grouped_gemm_sm103.cu",
    "csrc/combine_kernels.cu",
)
REQUIRED_CUTLASS_FILES = (
    "include/cutlass/cutlass.h",
    "include/cutlass/version.h",
    "include/cutlass/arch/arch.h",
    "include/cutlass/gemm/device/gemm_universal_adapter.h",
    "include/cute/tensor.hpp",
    "tools/util/include/cutlass/util/packed_stride.hpp",
)
CUTLASS_INCLUDE_DIRS = ("include", "tools/util/include")
NVCC_FLAGS = (
    "-O3",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-gencode=arch=compute_103a,code=sm_103a",
)
PREPROCESSOR_DEFINES = (
    "MOE_GOAL1B_BUILD=1",
    "MOE_TARGET_SM103A=1",
    "CUTLASS_ARCH_MMA_SM103A_ENABLED=1",
)


@dataclass(frozen=True)
class GPUInfo:
    """One visible CUDA device discovered through the active PyTorch build."""

    index: int
    name: str
    capability: tuple[int, int]
    total_memory: int | None = None

    @property
    def is_target(self) -> bool:
        return self.capability == TARGET_CAPABILITY


@dataclass
class PreflightReport:
    """Machine-readable Goal 1B preflight result."""

    python_executable: str
    python_version: str
    platform_system: str | None = None
    host_compiler: Path | None = None
    host_compiler_version: str | None = None
    torch_version: str | None = None
    torch_cuda_version: str | None = None
    torch_cuda_available: bool = False
    torch_arch_list: tuple[str, ...] = ()
    groupmm_header: Path | None = None
    grouped_mm_api_available: bool = False
    grouped_mm_runtime_probe: bool = False
    cuda_home: Path | None = None
    cuda_home_source: str | None = None
    nvcc_path: Path | None = None
    nvcc_version: tuple[int, ...] | None = None
    nvcc_sm103a_probe: bool = False
    gpus: tuple[GPUInfo, ...] = ()
    target_gpu: GPUInfo | None = None
    cutlass_path: Path | None = None
    cutlass_version: tuple[int, int, int] | None = None
    cutlass_include_dirs: tuple[Path, ...] = ()
    blackwell_grouped_examples: tuple[Path, ...] = ()
    nvfp4_examples: tuple[Path, ...] = ()
    nvfp4_type_headers: tuple[Path, ...] = ()
    missing_sources: tuple[Path, ...] = ()
    missing_cutlass_files: tuple[Path, ...] = ()
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def nvfp4_evidence_available(self) -> bool:
        return bool(self.nvfp4_examples and self.nvfp4_type_headers)


def _version_text(version: Sequence[int] | None) -> str:
    return "unavailable" if version is None else ".".join(map(str, version))


def _parse_nvcc_version(output: str) -> tuple[int, ...] | None:
    """Parse the CUDA toolkit release printed by ``nvcc --version``."""

    match = re.search(r"\brelease\s+(\d+)\.(\d+)(?:\.(\d+))?", output)
    if match is None:
        match = re.search(r"\bV(\d+)\.(\d+)(?:\.(\d+))?\b", output)
    if match is None:
        return None
    return tuple(int(part) for part in match.groups() if part is not None)


def _parse_package_version(value: str | None) -> tuple[int, int, int] | None:
    """Parse the numeric prefix of PyTorch/CUDA package version strings."""

    if value is None:
        return None
    match = re.match(r"\s*(\d+)\.(\d+)(?:\.(\d+))?", value)
    if match is None:
        return None
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch or 0)


def _parse_cutlass_version(version_header: Path) -> tuple[int, int, int] | None:
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
    return values[0], values[1], values[2]


def _resolved_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _discover_cuda_home(
    environment: Mapping[str, str],
    *,
    allow_torch_discovery: bool,
) -> tuple[Path | None, str | None]:
    for variable in ("CUDA_HOME", "CUDA_PATH"):
        value = environment.get(variable, "").strip()
        if value:
            return _resolved_path(value), variable

    if allow_torch_discovery:
        try:
            cpp_extension = importlib.import_module("torch.utils.cpp_extension")
            discovered = getattr(cpp_extension, "CUDA_HOME", None)
        except (ImportError, OSError):
            discovered = None
        if discovered:
            return _resolved_path(str(discovered)), "torch.utils.cpp_extension"

    nvcc_from_path = shutil.which("nvcc", path=environment.get("PATH"))
    if nvcc_from_path:
        nvcc_path = _resolved_path(nvcc_from_path)
        return nvcc_path.parent.parent, "PATH"
    return None, None


def _nvcc_under(cuda_home: Path | None) -> Path | None:
    if cuda_home is None:
        return None
    for filename in ("nvcc", "nvcc.exe"):
        candidate = cuda_home / "bin" / filename
        if candidate.is_file():
            return candidate.resolve()
    return None


def _readable_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _probe_nvcc_sm103a(report: PreflightReport) -> None:
    """Compile a throwaway CUDA TU to prove that nvcc accepts ``sm_103a``."""

    if report.nvcc_path is None:
        return
    try:
        with tempfile.TemporaryDirectory(prefix="goal1b-nvcc-") as directory:
            root = Path(directory)
            source = root / "sm103a_probe.cu"
            output = root / "sm103a_probe.o"
            source.write_text(
                "extern \"C\" __global__ void goal1b_probe() {}\n",
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    str(report.nvcc_path),
                    "-arch=sm_103a",
                    "-c",
                    str(source),
                    "-o",
                    str(output),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
                check=False,
            )
    except (OSError, subprocess.SubprocessError) as error:
        report.errors.append(f"nvcc sm_103a compile probe failed to run: {error}")
        return
    if completed.returncode != 0:
        detail = " | ".join(
            line.strip() for line in completed.stdout.splitlines()[-5:] if line.strip()
        )
        report.errors.append(
            "nvcc rejected the minimal -arch=sm_103a compile probe"
            + (f": {detail}" if detail else "")
        )
        return
    report.nvcc_sm103a_probe = True


def _find_cutlass_examples(cutlass_root: Path) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    examples_root = cutlass_root / "examples"
    if not examples_root.is_dir():
        return (), ()
    try:
        sources = sorted(examples_root.rglob("*.cu"))
    except OSError:
        return (), ()

    def relative_lower(path: Path) -> str:
        return path.relative_to(cutlass_root).as_posix().lower()

    grouped = tuple(
        path
        for path in sources
        if any(marker in relative_lower(path) for marker in ("blackwell", "sm103", "sm10"))
        and any(marker in relative_lower(path) for marker in ("moe", "grouped"))
    )
    nvfp4 = [
        path
        for path in sources
        if any(marker in relative_lower(path) for marker in ("nvfp4", "nv_fp4", "e2m1"))
    ]
    if not nvfp4:
        for path in grouped:
            text = _readable_text(path).lower()
            if any(marker in text for marker in ("nvfp4", "float_e2m1", "e2m1")):
                nvfp4.append(path)
    return grouped, tuple(nvfp4)


def _find_nvfp4_type_headers(cutlass_root: Path) -> tuple[Path, ...]:
    candidates = (
        cutlass_root / "include/cutlass/numeric_types.h",
        cutlass_root / "include/cutlass/float_subbyte.h",
        cutlass_root / "include/cutlass/detail/layout.hpp",
    )
    hits: list[Path] = []
    for path in candidates:
        if path.is_file():
            text = _readable_text(path).lower()
            if any(marker in text for marker in ("float_e2m1", "nvfp4", "e2m1")):
                hits.append(path.resolve())
    return tuple(hits)


def _probe_host(
    report: PreflightReport, environment: Mapping[str, str]
) -> None:
    """Check the host constraints of the ATen grouped-MM adapter."""

    report.platform_system = platform.system()
    for executable in ("c++", "g++", "clang++"):
        resolved = shutil.which(executable, path=environment.get("PATH"))
        if resolved:
            report.host_compiler = _resolved_path(resolved)
            break
    if report.host_compiler is None:
        report.errors.append("no supported host C++ compiler was found on PATH")
    else:
        try:
            completed = subprocess.run(
                [str(report.host_compiler), "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            report.errors.append(f"failed to query the host compiler: {error}")
        else:
            first_line = next(
                (line.strip() for line in completed.stdout.splitlines() if line.strip()),
                "",
            )
            if completed.returncode != 0 or not first_line:
                report.errors.append("the host C++ compiler --version check failed")
            else:
                report.host_compiler_version = first_line
    if report.platform_system != "Linux":
        report.errors.append(
            "Goal 1B's PyTorch ATen grouped-MM adapter is supported only on Linux; "
            f"detected {report.platform_system or 'unknown'}"
        )



def _probe_grouped_mm_runtime(
    report: PreflightReport,
    torch_module: Any,
) -> None:
    """Execute a small BF16 GroupMM on the audited CC 10.3 GPU."""

    if report.target_gpu is None:
        return

    grouped_mm = getattr(torch_module, "_grouped_mm", None)
    if not callable(grouped_mm):
        report.errors.append(
            "cannot run the grouped-MM provider runtime probe because "
            "torch._grouped_mm is unavailable"
        )
        return

    device_index = report.target_gpu.index
    device = f"cuda:{device_index}"

    try:
        with torch_module.no_grad():
            tokens_per_expert = 16
            expert_count = 2
            k = 64
            n = 64
            total_tokens = tokens_per_expert * expert_count

            torch_module.manual_seed(0)

            activations = torch_module.randn(
                total_tokens,
                k,
                device=device,
                dtype=torch_module.bfloat16,
            )
            weights = torch_module.randn(
                expert_count,
                k,
                n,
                device=device,
                dtype=torch_module.bfloat16,
            )
            offsets = torch_module.tensor(
                [tokens_per_expert, total_tokens],
                device=device,
                dtype=torch_module.int32,
            )

            output = grouped_mm(
                activations,
                weights,
                offs=offsets,
            )

            reference = torch_module.cat(
                [
                    activations[:tokens_per_expert] @ weights[0],
                    activations[tokens_per_expert:] @ weights[1],
                ],
                dim=0,
            )

            torch_module.cuda.synchronize(device_index)

            if tuple(output.shape) != tuple(reference.shape):
                report.errors.append(
                    "PyTorch grouped-MM provider runtime probe returned "
                    f"shape {tuple(output.shape)}, expected "
                    f"{tuple(reference.shape)}"
                )
                return

            if output.dtype != torch_module.bfloat16:
                report.errors.append(
                    "PyTorch grouped-MM provider runtime probe returned "
                    f"dtype {output.dtype}, expected torch.bfloat16"
                )
                return

            output_fp32 = output.float()
            reference_fp32 = reference.float()

            if not bool(
                torch_module.allclose(
                    output_fp32,
                    reference_fp32,
                    rtol=5e-2,
                    atol=1e-1,
                )
            ):
                max_abs_diff = (
                    output_fp32 - reference_fp32
                ).abs().max().item()
                report.errors.append(
                    "PyTorch grouped-MM provider runtime probe produced "
                    f"incorrect output; max_abs_diff={max_abs_diff}"
                )
                return

    except Exception as error:
        report.errors.append(
            "PyTorch grouped-MM provider runtime probe failed on "
            f"{device}: {type(error).__name__}: {error}"
        )
        return

    report.grouped_mm_runtime_probe = True


def _probe_torch(
    report: PreflightReport,
    torch_module: Any | None,
    *,
    inspect_installation: bool,
) -> Any | None:
    if torch_module is None:
        try:
            torch_module = importlib.import_module("torch")
        except (ImportError, OSError) as error:
            report.errors.append(f"PyTorch could not be imported: {error}")
            return None

    report.torch_version = str(getattr(torch_module, "__version__", "unknown"))
    parsed_torch = _parse_package_version(report.torch_version)
    if parsed_torch is None:
        report.errors.append(
            f"could not parse the PyTorch version {report.torch_version!r}"
        )
    elif parsed_torch < MIN_TORCH_VERSION:
        report.errors.append(
            f"PyTorch {report.torch_version} is too old for the internal GroupMM ABI; "
            "Goal 1B requires >=2.12.0"
        )
    torch_version_namespace = getattr(torch_module, "version", None)
    compiled_cuda = getattr(torch_version_namespace, "cuda", None)
    report.torch_cuda_version = None if compiled_cuda is None else str(compiled_cuda)
    if compiled_cuda is None:
        report.errors.append(
            "PyTorch is CPU-only (torch.version.cuda is None); install a CUDA-enabled build"
        )
    else:
        parsed_torch_cuda = _parse_package_version(report.torch_cuda_version)
        if parsed_torch_cuda is None or parsed_torch_cuda[:2] < MIN_TORCH_CUDA_VERSION:
            report.errors.append(
                f"PyTorch was built for CUDA {report.torch_cuda_version}; "
                "the Goal 1B ATen provider requires a CUDA 13.x PyTorch build"
            )

    if inspect_installation:
        module_file = getattr(torch_module, "__file__", None)
        if module_file:
            header = (
                Path(str(module_file)).resolve().parent
                / "include/ATen/native/cuda/GroupMM.h"
            )
            if header.is_file() and "bf16bf16_grouped_mm" in _readable_text(header):
                report.groupmm_header = header
            else:
                report.errors.append(
                    "PyTorch does not expose ATen/native/cuda/GroupMM.h with "
                    "bf16bf16_grouped_mm; the internal Goal 1B adapter cannot compile"
                )
        functional = getattr(getattr(torch_module, "nn", None), "functional", None)
        report.grouped_mm_api_available = callable(
            getattr(functional, "grouped_mm", None)
        ) and callable(getattr(torch_module, "_grouped_mm", None))
        if not report.grouped_mm_api_available:
            report.errors.append(
                "the active PyTorch build does not expose the grouped_mm operator"
            )

    cuda_api = getattr(torch_module, "cuda", None)
    try:
        report.torch_cuda_available = bool(cuda_api is not None and cuda_api.is_available())
    except Exception as error:  # CUDA initialization can fail for driver/runtime reasons.
        report.errors.append(f"torch.cuda.is_available() failed: {error}")
        report.torch_cuda_available = False

    if not report.torch_cuda_available:
        report.errors.append(
            "torch.cuda.is_available() is false; an actual CC 10.3 GPU cannot be verified"
        )
        return torch_module

    try:
        report.torch_arch_list = tuple(str(value) for value in cuda_api.get_arch_list())
    except Exception as error:
        report.errors.append(f"torch.cuda.get_arch_list() failed: {error}")
    if not any(
        arch.startswith(("sm_103", "compute_103"))
        for arch in report.torch_arch_list
    ):
        rendered = ", ".join(report.torch_arch_list) or "empty"
        report.warnings.append(
            "torch.cuda.get_arch_list() does not advertise SM103 "
            f"({rendered}); this list is informational for Goal 1B because "
            "the PyTorch grouped-MM provider is verified by an on-device "
            "runtime probe"
        )

    try:
        device_count = int(cuda_api.device_count())
        devices: list[GPUInfo] = []
        for index in range(device_count):
            name = str(cuda_api.get_device_name(index))
            capability_value = cuda_api.get_device_capability(index)
            capability = (int(capability_value[0]), int(capability_value[1]))
            try:
                total_memory = int(cuda_api.get_device_properties(index).total_memory)
            except Exception:
                total_memory = None
            devices.append(GPUInfo(index, name, capability, total_memory))
        report.gpus = tuple(devices)
        report.target_gpu = next((device for device in devices if device.is_target), None)
    except Exception as error:
        report.errors.append(f"CUDA device enumeration failed: {error}")

    if report.target_gpu is None:
        visible = ", ".join(
            f"GPU {device.index}={device.name} CC {device.capability[0]}.{device.capability[1]}"
            for device in report.gpus
        ) or "no visible CUDA devices"
        report.errors.append(
            f"no visible CC 10.3 GPU for {TARGET_ARCH}; found {visible}; refusing an architecture fallback"
        )

    if (
        inspect_installation
        and report.target_gpu is not None
        and report.grouped_mm_api_available
    ):
        _probe_grouped_mm_runtime(report, torch_module)

    return torch_module


def _probe_cutlass(report: PreflightReport, cutlass_value: str | None) -> None:
    if not cutlass_value:
        report.errors.append(
            "CUTLASS_PATH is not set; point it to a CUTLASS source checkout >=4.3.1"
        )
        return

    cutlass_root = _resolved_path(cutlass_value)
    report.cutlass_path = cutlass_root
    if not cutlass_root.is_dir():
        report.errors.append(f"CUTLASS_PATH is not a directory: {cutlass_root}")
        return

    report.cutlass_include_dirs = tuple(
        (cutlass_root / relative).resolve() for relative in CUTLASS_INCLUDE_DIRS
    )
    report.missing_cutlass_files = tuple(
        cutlass_root / relative
        for relative in REQUIRED_CUTLASS_FILES
        if not (cutlass_root / relative).is_file()
    )
    if report.missing_cutlass_files:
        report.errors.append(
            "CUTLASS checkout is missing required files: "
            + ", ".join(str(path) for path in report.missing_cutlass_files)
        )

    version_header = cutlass_root / "include/cutlass/version.h"
    report.cutlass_version = _parse_cutlass_version(version_header)
    if report.cutlass_version is None:
        report.errors.append(f"could not parse CUTLASS version macros from {version_header}")
    elif report.cutlass_version < MIN_CUTLASS_VERSION:
        report.errors.append(
            f"CUTLASS {_version_text(report.cutlass_version)} is too old; "
            f"{TARGET_ARCH} requires >= {_version_text(MIN_CUTLASS_VERSION)}"
        )

    arch_header = cutlass_root / "include/cutlass/arch/arch.h"
    if arch_header.is_file() and re.search(r"\bSm103\b", _readable_text(arch_header)) is None:
        report.errors.append(
            f"{arch_header} does not define Sm103; refusing an architecture fallback"
        )

    grouped, nvfp4 = _find_cutlass_examples(cutlass_root)
    report.blackwell_grouped_examples = grouped
    report.nvfp4_examples = nvfp4
    report.nvfp4_type_headers = _find_nvfp4_type_headers(cutlass_root)
    if not grouped:
        report.errors.append(
            f"no Blackwell/SM10x grouped-GEMM or MoE example was found under {cutlass_root / 'examples'}"
        )
    if not report.nvfp4_evidence_available:
        report.warnings.append(
            "CUTLASS NVFP4 datatype plus example evidence is incomplete; "
            "NVFP4 must remain disabled/unavailable even if the BF16 build succeeds"
        )


def probe_environment(
    *,
    cutlass_path: str | os.PathLike[str] | None = None,
    environ: Mapping[str, str] | None = None,
    torch_module: Any | None = None,
) -> PreflightReport:
    """Inspect the active build environment without modifying it."""

    environment = os.environ if environ is None else environ
    report = PreflightReport(
        python_executable=str(Path(sys.executable).resolve()),
        python_version=platform.python_version(),
    )
    if sys.version_info < (3, 10):
        report.errors.append("Python >=3.10 is required")

    _probe_host(report, environment)
    caller_supplied_torch = torch_module is not None
    _probe_torch(
        report,
        torch_module,
        inspect_installation=not caller_supplied_torch,
    )

    report.cuda_home, report.cuda_home_source = _discover_cuda_home(
        environment,
        allow_torch_discovery=not caller_supplied_torch,
    )
    if report.cuda_home is None:
        report.errors.append(
            "CUDA_HOME could not be resolved from CUDA_HOME/CUDA_PATH, PyTorch, or PATH"
        )
    elif not report.cuda_home.is_dir():
        report.errors.append(f"CUDA_HOME is not a directory: {report.cuda_home}")

    report.nvcc_path = _nvcc_under(report.cuda_home)
    if report.nvcc_path is None:
        location = report.cuda_home / "bin" if report.cuda_home else "CUDA_HOME/bin"
        report.errors.append(f"nvcc was not found under {location}")
    else:
        try:
            completed = subprocess.run(
                [str(report.nvcc_path), "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            report.errors.append(f"failed to execute {report.nvcc_path}: {error}")
        else:
            if completed.returncode != 0:
                report.errors.append(
                    f"{report.nvcc_path} --version exited with code {completed.returncode}"
                )
            report.nvcc_version = _parse_nvcc_version(completed.stdout)
            if report.nvcc_version is None:
                report.errors.append(
                    f"could not parse a CUDA release from {report.nvcc_path} --version"
                )
            elif report.nvcc_version[:2] < MIN_CUDA_VERSION:
                report.errors.append(
                    f"CUDA {_version_text(report.nvcc_version)} is too old; "
                    f"{TARGET_ARCH} requires >= {_version_text(MIN_CUDA_VERSION)}"
                )
            else:
                _probe_nvcc_sm103a(report)

    report.missing_sources = tuple(
        ROOT / relative for relative in GOAL1B_SOURCES if not (ROOT / relative).is_file()
    )
    if report.missing_sources:
        report.errors.append(
            "Goal 1B source manifest is incomplete: "
            + ", ".join(str(path) for path in report.missing_sources)
        )

    selected_cutlass = (
        os.fspath(cutlass_path)
        if cutlass_path is not None
        else environment.get("CUTLASS_PATH", "").strip()
    )
    _probe_cutlass(report, selected_cutlass or None)
    return report


def format_report(report: PreflightReport) -> str:
    """Render a stable, human-readable preflight report."""

    lines = [
        "Goal 1B B300 build preflight",
        f"Python executable: {report.python_executable}",
        f"Python version: {report.python_version}",
        f"Host OS: {report.platform_system or 'unavailable'} (Linux required)",
        f"Host C++ compiler: {report.host_compiler or 'unavailable'}",
        f"Host C++ compiler version: {report.host_compiler_version or 'unavailable'}",
        f"PyTorch version: {report.torch_version or 'unavailable'}",
        f"PyTorch CUDA: {report.torch_cuda_version or 'unavailable'}",
        f"torch.cuda available: {report.torch_cuda_available}",
        "PyTorch CUDA architectures: "
        + (", ".join(report.torch_arch_list) if report.torch_arch_list else "unavailable"),
        f"ATen GroupMM header: {report.groupmm_header or 'unavailable'}",
        f"PyTorch grouped_mm operator: {report.grouped_mm_api_available}",
        f"PyTorch grouped_mm runtime probe: {report.grouped_mm_runtime_probe}",
        f"CUDA_HOME: {report.cuda_home or 'unavailable'}"
        + (f" (source: {report.cuda_home_source})" if report.cuda_home_source else ""),
        f"nvcc: {report.nvcc_path or 'unavailable'}",
        f"CUDA toolkit: {_version_text(report.nvcc_version)} (required >=13.0)",
        f"nvcc -arch=sm_103a compile probe: {report.nvcc_sm103a_probe}",
    ]
    if report.gpus:
        for device in report.gpus:
            target = f" [{TARGET_ARCH} target]" if device.is_target else ""
            memory = (
                f", {device.total_memory / (1024**3):.1f} GiB"
                if device.total_memory is not None
                else ""
            )
            lines.append(
                f"GPU {device.index}: {device.name}, CC "
                f"{device.capability[0]}.{device.capability[1]}{memory}{target}"
            )
    else:
        lines.append("Visible GPUs: unavailable")
    lines.extend(
        [
            f"Target architecture: {TARGET_ARCH} only (no fallback)",
            f"CUTLASS_PATH: {report.cutlass_path or 'unavailable'}",
            f"CUTLASS version: {_version_text(report.cutlass_version)} "
            "(required >=4.3.1)",
            "CUTLASS include dirs: "
            + (
                ", ".join(str(path) for path in report.cutlass_include_dirs)
                if report.cutlass_include_dirs
                else "unavailable"
            ),
            "Blackwell grouped/MoE examples: "
            + (
                ", ".join(str(path) for path in report.blackwell_grouped_examples)
                if report.blackwell_grouped_examples
                else "unavailable"
            ),
            "CUTLASS NVFP4 evidence: "
            + ("available" if report.nvfp4_evidence_available else "unavailable"),
            "NVCC flags: " + " ".join(NVCC_FLAGS),
            "Defines: " + " ".join(f"-D{define}" for define in PREPROCESSOR_DEFINES),
            "TORCH_CUDA_ARCH_LIST: unset by builder (explicit sm_103a gencode)",
        ]
    )
    if report.warnings:
        lines.append("Warnings:")
        lines.extend(f"  - {warning}" for warning in report.warnings)
    lines.append(f"Status: {'READY' if report.ok else 'FAILED'}")
    if report.errors:
        lines.append("Errors:")
        lines.extend(f"  - {error}" for error in report.errors)
    return "\n".join(lines)


def _display_command(command: Sequence[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(list(command))
    return shlex.join(command)


def run_editable_build(report: PreflightReport) -> int:
    """Invoke the reproducible editable build after a successful preflight."""

    if not report.ok or report.cutlass_path is None:
        raise RuntimeError("refusing to build because the Goal 1B preflight failed")
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--no-build-isolation",
        "--editable",
        str(ROOT),
    ]
    build_environment = os.environ.copy()
    build_environment.update(
        {
            "CUDA_HOME": str(report.cuda_home),
            "CUTLASS_PATH": str(report.cutlass_path),
            "MOE_BUILD_CUDA": "1",
            "MOE_GOAL1B_BUILD": "1",
            "MOE_B300_SM103A": "1",
        }
    )
    build_environment.pop("TORCH_CUDA_ARCH_LIST", None)
    print(f"Build command: {_display_command(command)}", flush=True)
    completed = subprocess.run(command, cwd=ROOT, env=build_environment, check=False)
    return int(completed.returncode)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preflight and build the B300 Goal 1B sm_103a CUDA extension"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="print diagnostics without invoking pip or modifying the environment",
    )
    parser.add_argument(
        "--cutlass-path",
        type=Path,
        help="CUTLASS checkout root (overrides CUTLASS_PATH for this invocation)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _argument_parser().parse_args(argv)
    report = probe_environment(cutlass_path=arguments.cutlass_path)
    print(format_report(report), flush=True)
    if not report.ok:
        return 2
    if arguments.check_only:
        return 0
    return run_editable_build(report)


if __name__ == "__main__":
    raise SystemExit(main())
