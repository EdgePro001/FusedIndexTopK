"""Build the same-kernel GEMM and exact Top-K producers."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from fused_index_topk.provenance import path_fingerprint

_ROOT = Path(__file__).resolve().parent
_CSRC = _ROOT / "csrc"
_IMPLEMENTATIONS = {
    "long": (
        _CSRC / "long_fused_bindings.cpp",
        _CSRC / "include" / "fused_index_topk" / "long_fused_topk.cuh",
    ),
}
_EXTENSIONS: dict[str, Any] = {}
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
            f"FusedIndexTopK upstream mismatch for {path}: "
            f"expected sha256={expected}, actual={actual}"
        )


def _deepgemm_source_root() -> Path:
    raw = os.environ.get("DEEPGEMM_SOURCE")
    if not raw:
        raise RuntimeError(
            "DEEPGEMM_SOURCE must point at the frozen DeepGEMM checkout; "
            "FusedIndexTopK never modifies that checkout"
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


def _load(kind: str, deep_gemm: Any, *, verbose: bool) -> Any:
    cached = _EXTENSIONS.get(kind)
    if cached is not None:
        return cached

    from torch.utils.cpp_extension import CUDA_HOME, load

    if CUDA_HOME is None:
        raise RuntimeError("CUDA_HOME is required to build FusedIndexTopK")
    try:
        bindings, kernel_header = _IMPLEMENTATIONS[kind]
    except KeyError as error:
        raise ValueError(f"unknown on-chip implementation: {kind}") from error
    if not bindings.is_file() or not kernel_header.is_file():
        raise RuntimeError(f"missing FusedIndexTopK {kind} implementation sources")

    deepgemm_source = _deepgemm_source_root()
    deepgemm_package = Path(deep_gemm.__file__).resolve().parent
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

    cuda_home = Path(CUDA_HOME)
    include_paths = [
        _CSRC,
        _CSRC / "include" / "fused_index_topk",
        _cccl_include(cuda_home),
        deepgemm_source / "csrc",
        deepgemm_source / "deep_gemm" / "include",
        deepgemm_source / "third-party" / "cutlass" / "include",
        deepgemm_source / "third-party" / "fmt" / "include",
    ]
    missing = [str(path) for path in include_paths if not path.is_dir()]
    if missing:
        raise RuntimeError(f"missing FusedIndexTopK build include directories: {missing}")

    identity = path_fingerprint(kernel_header.parent)
    source_sha = str(identity["tree_sha256"])
    extension = load(
        name=f"fused_index_topk_{kind}_{source_sha[:12]}",
        sources=[str(bindings)],
        extra_include_paths=[str(path) for path in include_paths],
        extra_cflags=["-O3", "-std=c++17", "-Wno-deprecated-declarations"],
        extra_ldflags=["-lcuda", "-lcudart", "-lnvrtc"],
        with_cuda=True,
        verbose=verbose,
    )
    extension.init_jit(
        str(deepgemm_package),
        str(cuda_home),
        str(kernel_header),
        source_sha,
    )
    _EXTENSIONS[kind] = extension
    return extension


def load_long_producer(deep_gemm: Any, *, verbose: bool = False) -> Any:
    """Load the unified bounded-overflow implementation for 8K--160K."""

    return _load("long", deep_gemm, verbose=verbose)
