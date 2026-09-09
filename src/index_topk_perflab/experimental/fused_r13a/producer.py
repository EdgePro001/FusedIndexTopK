"""Build and load the producer-side DeepGEMM + R13a candidate fusion."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from index_topk_perflab.provenance import path_fingerprint

_ROOT = Path(__file__).resolve().parent
_CSRC = _ROOT / "csrc"
_KERNEL_HEADER = (
    _CSRC
    / "include"
    / "itk_fused_r13a"
    / "sm90_fp8_mqa_candidate_r13a.cuh"
)
_EXTENSION: Any | None = None
_UPSTREAM_DEVICE_SHA256 = (
    "12e2c0d2f3cb89be5a202c053a64b44ee2a7da9425cfd3ce64c95f23c39b7007"
)
_UPSTREAM_JIT_SHA256 = (
    "412601d9ee6420500617b3a619ad6e5c601689f3ad7f6c6a16eed7078a33f46b"
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha(path: Path, expected: str) -> None:
    actual = _file_sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"fused R13a upstream mismatch for {path}: "
            f"expected sha256={expected}, actual={actual}"
        )


def _deepgemm_source_root() -> Path:
    raw = os.environ.get("DEEPGEMM_SOURCE")
    if not raw:
        raise RuntimeError(
            "DEEPGEMM_SOURCE must point at the frozen DeepGEMM checkout; "
            "the fused R13a experiment never modifies that checkout"
        )
    root = Path(raw).resolve()
    required = root / "csrc" / "jit" / "compiler.hpp"
    if not required.is_file():
        raise RuntimeError(f"invalid DEEPGEMM_SOURCE: missing {required}")
    return root


def _cccl_include(cuda_home: Path) -> Path:
    candidates = [
        cuda_home / "include" / "cccl",
        *sorted((cuda_home / "targets").glob("*/include/cccl")),
    ]
    for candidate in candidates:
        if (candidate / "cuda" / "std" / "utility").is_file():
            return candidate
    raise RuntimeError(f"CUDA CCCL headers are missing below {cuda_home}")


def load_candidate_producer(deep_gemm: Any, *, verbose: bool = False) -> Any:
    """Return the host extension; the device kernel is compiled by DeepGEMM JIT."""

    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION

    from torch.utils.cpp_extension import CUDA_HOME, load

    if CUDA_HOME is None:
        raise RuntimeError("CUDA_HOME is required to build the fused R13a producer")

    deepgemm_source = _deepgemm_source_root()
    deepgemm_package = Path(deep_gemm.__file__).resolve().parent
    if not _KERNEL_HEADER.is_file():
        raise RuntimeError(f"missing fused R13a device source: {_KERNEL_HEADER}")

    upstream_device = (
        deepgemm_source
        / "deep_gemm"
        / "include"
        / "deep_gemm"
        / "impls"
        / "sm90_fp8_mqa_logits.cuh"
    )
    upstream_jit = (
        deepgemm_source
        / "csrc"
        / "jit_kernels"
        / "impls"
        / "smxx_fp8_mqa_logits.hpp"
    )
    installed_device = (
        deepgemm_package
        / "include"
        / "deep_gemm"
        / "impls"
        / "sm90_fp8_mqa_logits.cuh"
    )
    _require_sha(upstream_device, _UPSTREAM_DEVICE_SHA256)
    _require_sha(upstream_jit, _UPSTREAM_JIT_SHA256)
    _require_sha(installed_device, _UPSTREAM_DEVICE_SHA256)

    source_identity = path_fingerprint(_CSRC)
    source_sha = str(source_identity["tree_sha256"])
    include_paths = [
        _CSRC,
        _cccl_include(Path(CUDA_HOME)),
        deepgemm_source / "csrc",
        deepgemm_source / "deep_gemm" / "include",
        deepgemm_source / "third-party" / "cutlass" / "include",
        deepgemm_source / "third-party" / "fmt" / "include",
    ]
    missing = [str(path) for path in include_paths if not path.is_dir()]
    if missing:
        raise RuntimeError(
            f"missing fused R13a build include directories: {missing}"
        )

    module_name = f"itk_fused_r13a_producer_host_{source_sha[:12]}"
    _EXTENSION = load(
        name=module_name,
        sources=[str(_CSRC / "producer_bindings.cpp")],
        extra_include_paths=[str(path) for path in include_paths],
        extra_cflags=["-O3", "-std=c++17", "-Wno-deprecated-declarations"],
        extra_ldflags=["-lcuda", "-lcudart", "-lnvrtc"],
        with_cuda=True,
        verbose=verbose,
    )
    _EXTENSION.init_jit(
        str(deepgemm_package),
        str(CUDA_HOME),
        str(_KERNEL_HEADER),
        source_sha,
    )
    return _EXTENSION
