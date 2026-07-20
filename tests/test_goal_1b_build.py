"""CPU-safe tests for the Goal 1B build preflight contract."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastpath import build


class _CpuOnlyCuda:
    @staticmethod
    def is_available() -> bool:
        return False


class _CpuOnlyTorch:
    __version__ = "test+cpu"
    version = SimpleNamespace(cuda=None)
    cuda = _CpuOnlyCuda()


class _MissingSm103Cuda:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def get_arch_list() -> list[str]:
        return ["sm_90", "sm_100"]

    @staticmethod
    def device_count() -> int:
        return 1

    @staticmethod
    def get_device_name(_index: int) -> str:
        return "Synthetic B300"

    @staticmethod
    def get_device_capability(_index: int) -> tuple[int, int]:
        return (10, 3)

    @staticmethod
    def get_device_properties(_index: int) -> SimpleNamespace:
        return SimpleNamespace(total_memory=1)


class _MissingSm103Torch:
    __version__ = "2.12.0+cu130"
    version = SimpleNamespace(cuda="13.0")
    cuda = _MissingSm103Cuda()


def _write(path: Path, text: str = "// test fixture\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_cutlass_fixture(root: Path, version: tuple[int, int, int]) -> None:
    major, minor, patch = version
    for relative in build.REQUIRED_CUTLASS_FILES:
        _write(root / relative)
    _write(
        root / "include/cutlass/version.h",
        f"""#define CUTLASS_MAJOR {major}
#define CUTLASS_MINOR {minor}
#define CUTLASS_PATCH {patch}
""",
    )
    _write(root / "include/cutlass/arch/arch.h", "struct Sm103 {};\n")
    _write(
        root / "include/cutlass/numeric_types.h",
        "struct float_e2m1_t {};\n",
    )
    _write(
        root / "examples/92_blackwell_moe_gemm/92_blackwell_moe_nvfp4.cu",
        "// Blackwell grouped MoE NVFP4 float_e2m1 example\n",
    )


def test_cpu_only_probe_is_truthful(tmp_path: Path) -> None:
    report = build.probe_environment(
        environ={"PATH": "", "CUTLASS_PATH": str(tmp_path / "missing-cutlass")},
        torch_module=_CpuOnlyTorch(),
    )

    assert not report.ok
    assert report.torch_version == "test+cpu"
    assert report.torch_cuda_version is None
    assert not report.torch_cuda_available
    assert report.gpus == ()
    assert report.target_gpu is None
    assert any("CPU-only" in error for error in report.errors)
    assert any("actual CC 10.3 GPU cannot be verified" in error for error in report.errors)
    assert any("CUDA_HOME could not be resolved" in error for error in report.errors)

    rendered = build.format_report(report)
    assert "PyTorch CUDA: unavailable" in rendered
    assert "Visible GPUs: unavailable" in rendered
    assert "Target architecture: sm_103a only (no fallback)" in rendered
    assert "Status: FAILED" in rendered
    assert "Status: READY" not in rendered


def test_nvcc_version_parser_uses_reported_release() -> None:
    output = "Cuda compilation tools, release 13.0, V13.0.48"
    assert build._parse_nvcc_version(output) == (13, 0)
    assert build._parse_nvcc_version("not an nvcc version") is None


def test_package_version_parser_accepts_local_build_suffix() -> None:
    assert build._parse_package_version("2.12.0+cu130") == (2, 12, 0)
    assert build._parse_package_version("13.0") == (13, 0, 0)
    assert build._parse_package_version("development") is None


def test_preflight_rejects_pytorch_binary_without_sm103_provider(
    tmp_path: Path,
) -> None:
    report = build.probe_environment(
        environ={"PATH": "", "CUTLASS_PATH": str(tmp_path / "missing-cutlass")},
        torch_module=_MissingSm103Torch(),
    )

    assert report.target_gpu is not None
    assert report.torch_arch_list == ("sm_90", "sm_100")
    assert any("does not advertise an SM103" in error for error in report.errors)


def test_cutlass_probe_reads_header_version_and_capability_evidence(
    tmp_path: Path,
) -> None:
    cutlass_root = tmp_path / "cutlass"
    _make_cutlass_fixture(cutlass_root, (4, 3, 1))
    report = build.PreflightReport("python", "3.12")

    build._probe_cutlass(report, str(cutlass_root))

    assert report.cutlass_path == cutlass_root.resolve()
    assert report.cutlass_version == (4, 3, 1)
    assert not report.missing_cutlass_files
    assert report.blackwell_grouped_examples
    assert report.nvfp4_evidence_available
    assert not report.errors


def test_old_cutlass_is_rejected_from_parsed_header(tmp_path: Path) -> None:
    cutlass_root = tmp_path / "cutlass"
    _make_cutlass_fixture(cutlass_root, (4, 3, 0))
    report = build.PreflightReport("python", "3.12")

    build._probe_cutlass(report, str(cutlass_root))

    assert any("CUTLASS 4.3.0 is too old" in error for error in report.errors)


def test_goal1b_emits_exactly_one_architecture_target() -> None:
    architecture_flags = [flag for flag in build.NVCC_FLAGS if "gencode" in flag]
    assert architecture_flags == ["-gencode=arch=compute_103a,code=sm_103a"]
    assert not any("compute_" in flag and "103a" not in flag for flag in build.NVCC_FLAGS)


def test_check_only_failure_never_invokes_build(monkeypatch, capsys) -> None:
    failed = build.PreflightReport("python", "3.12", errors=["synthetic failure"])
    monkeypatch.setattr(build, "probe_environment", lambda **_kwargs: failed)

    def _unexpected_build(_report: build.PreflightReport) -> int:
        raise AssertionError("check-only invoked the build")

    monkeypatch.setattr(build, "run_editable_build", _unexpected_build)
    assert build.main(["--check-only"]) == 2
    assert "synthetic failure" in capsys.readouterr().out
