"""Build the variable-length, complete-row repair producer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fused_index_topk.provenance import path_fingerprint

from . import onchip_producer

_BASE_CSRC = Path(__file__).resolve().parent / "csrc" / "repair"
_BINDINGS = _BASE_CSRC / "repair_producer_bindings.cpp"
_KERNEL_HEADER = _BASE_CSRC / "include" / "fused_index_topk" / "repair_producer.cuh"
_EXTENSION: Any | None = None


def load_repair_producer(deep_gemm: Any, *, verbose: bool = False) -> Any:
    """Build and return the masked repair producer."""

    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION

    from torch.utils.cpp_extension import CUDA_HOME, load

    if CUDA_HOME is None:
        raise RuntimeError("CUDA_HOME is required to build FusedIndexTopK repair")

    deepgemm_source = onchip_producer._deepgemm_source_root()
    deepgemm_package = Path(deep_gemm.__file__).resolve().parent
    kernel_header = _KERNEL_HEADER
    if not kernel_header.is_file():
        raise RuntimeError(f"missing FusedIndexTopK repair device source: {kernel_header}")

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
    onchip_producer._require_sha(
        upstream_device, onchip_producer._UPSTREAM_DEVICE_SHA256
    )
    onchip_producer._require_sha(
        upstream_jit, onchip_producer._UPSTREAM_JIT_SHA256
    )
    onchip_producer._require_sha(
        installed_device, onchip_producer._UPSTREAM_DEVICE_SHA256
    )

    cuda_home = Path(CUDA_HOME)
    include_paths = [
        _BASE_CSRC,
        onchip_producer._cccl_include(cuda_home),
        deepgemm_source / "csrc",
        deepgemm_source / "deep_gemm" / "include",
        deepgemm_source / "third-party" / "cutlass" / "include",
        deepgemm_source / "third-party" / "fmt" / "include",
    ]
    missing = [str(path) for path in include_paths if not path.is_dir()]
    if missing:
        raise RuntimeError(f"missing FusedIndexTopK repair include directories: {missing}")

    source_sha = str(path_fingerprint(_BASE_CSRC)["tree_sha256"])
    module_name = f"fused_index_topk_repair_producer_{source_sha[:12]}"
    _EXTENSION = load(
        name=module_name,
        sources=[str(_BINDINGS)],
        extra_include_paths=[str(path) for path in include_paths],
        extra_cflags=["-O3", "-std=c++17", "-Wno-deprecated-declarations"],
        extra_ldflags=["-lcuda", "-lcudart", "-lnvrtc"],
        with_cuda=True,
        verbose=verbose,
    )
    _EXTENSION.init_jit(
        str(deepgemm_package),
        str(cuda_home),
        str(kernel_header),
        source_sha,
    )
    return _EXTENSION
