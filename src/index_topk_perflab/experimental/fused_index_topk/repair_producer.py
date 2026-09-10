"""Load the complete-row repair producer with a runtime-length host guard.

The measured device kernel accepts runtime ``seq_len_kv`` and its 16 repair
segments remain complete for every N <= 16384. The loader relaxes the original
host assertion while leaving the device implementation unchanged.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from . import producer as fast_producer

_BASE_CSRC = Path(__file__).resolve().parent / "csrc" / "repair"
_HOST_HEADER = _BASE_CSRC / "repair_producer_host.hpp"
_BINDINGS = _BASE_CSRC / "repair_producer_bindings.cpp"
_KERNEL_HEADER = _BASE_CSRC / "include" / "fused_index_topk" / "repair_producer.cuh"
_PATCH_FROM = "DG_HOST_ASSERT(seq_len_kv == kRepairCandidateCapacity);"
_PATCH_TO = (
    "DG_HOST_ASSERT(seq_len_kv > 0 and "
    "seq_len_kv <= kRepairCandidateCapacity);"
)
_EXTENSION: Any | None = None


def _patched_host_source() -> str:
    header = _HOST_HEADER.read_text(encoding="utf-8")
    if header.count(_PATCH_FROM) != 1:
        raise RuntimeError("FusedIndexTopK repair host assertion patch no longer applies")
    header = header.replace(_PATCH_FROM, _PATCH_TO)
    bindings = _BINDINGS.read_text(encoding="utf-8")
    include = '#include "repair_producer_host.hpp"\n'
    if bindings.count(include) != 1:
        raise RuntimeError("FusedIndexTopK repair bindings include changed unexpectedly")
    return header + "\n" + bindings.replace(include, "")


def load_candidate_producer(deep_gemm: Any, *, verbose: bool = False) -> Any:
    """Build the host-only probe adapter around the original device source."""

    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION

    from torch.utils.cpp_extension import CUDA_HOME, load_inline

    if CUDA_HOME is None:
        raise RuntimeError("CUDA_HOME is required to build FusedIndexTopK repair")

    deepgemm_source = fast_producer._deepgemm_source_root()
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
    fast_producer._require_sha(
        upstream_device, fast_producer._UPSTREAM_DEVICE_SHA256
    )
    fast_producer._require_sha(upstream_jit, fast_producer._UPSTREAM_JIT_SHA256)
    fast_producer._require_sha(
        installed_device, fast_producer._UPSTREAM_DEVICE_SHA256
    )

    cuda_home = Path(CUDA_HOME)
    include_paths = [
        _BASE_CSRC,
        fast_producer._cccl_include(cuda_home),
        deepgemm_source / "csrc",
        deepgemm_source / "deep_gemm" / "include",
        deepgemm_source / "third-party" / "cutlass" / "include",
        deepgemm_source / "third-party" / "fmt" / "include",
    ]
    missing = [str(path) for path in include_paths if not path.is_dir()]
    if missing:
        raise RuntimeError(f"missing FusedIndexTopK repair include directories: {missing}")

    host_source = _patched_host_source()
    adapter_sha = hashlib.sha256(host_source.encode("utf-8")).hexdigest()
    module_name = f"fused_index_topk_repair_producer_{adapter_sha[:12]}"
    _EXTENSION = load_inline(
        name=module_name,
        cpp_sources=host_source,
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
        adapter_sha,
    )
    return _EXTENSION
