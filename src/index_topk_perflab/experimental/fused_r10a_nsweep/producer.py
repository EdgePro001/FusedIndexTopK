"""Load R10a's byte-identical device producer with a variable-N host guard.

The released R10a/R5i host wrapper rejects every repair input whose logical N
is not exactly 16384.  The device kernel itself accepts runtime ``seq_len_kv``
and its 16 repair segments remain complete for every N <= 16384.  This probe
changes only that host assertion; it passes the original device header to the
DeepGEMM JIT without editing or copying it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from index_topk_perflab.experimental.fused_r5i import producer as r5i_producer

_BASE_CSRC = Path(r5i_producer.__file__).resolve().parent / "csrc"
_HOST_HEADER = _BASE_CSRC / "smxx_fp8_mqa_candidate_r5i.hpp"
_BINDINGS = _BASE_CSRC / "producer_bindings.cpp"
_PATCH_FROM = "DG_HOST_ASSERT(seq_len_kv == kRepairCandidateCapacity);"
_PATCH_TO = (
    "DG_HOST_ASSERT(seq_len_kv > 0 and "
    "seq_len_kv <= kRepairCandidateCapacity);"
)
_EXTENSION: Any | None = None


def _patched_host_source() -> str:
    header = _HOST_HEADER.read_text(encoding="utf-8")
    if header.count(_PATCH_FROM) != 1:
        raise RuntimeError("R10a N-sweep host assertion patch no longer applies cleanly")
    header = header.replace(_PATCH_FROM, _PATCH_TO)
    bindings = _BINDINGS.read_text(encoding="utf-8")
    include = '#include "smxx_fp8_mqa_candidate_r5i.hpp"\n'
    if bindings.count(include) != 1:
        raise RuntimeError("R10a N-sweep bindings include changed unexpectedly")
    return header + "\n" + bindings.replace(include, "")


def load_candidate_producer(deep_gemm: Any, *, verbose: bool = False) -> Any:
    """Build the host-only probe adapter around the original device source."""

    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION

    from torch.utils.cpp_extension import CUDA_HOME, load_inline

    if CUDA_HOME is None:
        raise RuntimeError("CUDA_HOME is required to build the R10a N-sweep adapter")

    deepgemm_source = r5i_producer._deepgemm_source_root()
    deepgemm_package = Path(deep_gemm.__file__).resolve().parent
    kernel_header = r5i_producer._KERNEL_HEADER
    if not kernel_header.is_file():
        raise RuntimeError(f"missing frozen R10a device source: {kernel_header}")

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
    r5i_producer._require_sha(
        upstream_device, r5i_producer._UPSTREAM_DEVICE_SHA256
    )
    r5i_producer._require_sha(upstream_jit, r5i_producer._UPSTREAM_JIT_SHA256)
    r5i_producer._require_sha(
        installed_device, r5i_producer._UPSTREAM_DEVICE_SHA256
    )

    cuda_home = Path(CUDA_HOME)
    include_paths = [
        _BASE_CSRC,
        r5i_producer._cccl_include(cuda_home),
        deepgemm_source / "csrc",
        deepgemm_source / "deep_gemm" / "include",
        deepgemm_source / "third-party" / "cutlass" / "include",
        deepgemm_source / "third-party" / "fmt" / "include",
    ]
    missing = [str(path) for path in include_paths if not path.is_dir()]
    if missing:
        raise RuntimeError(f"missing R10a N-sweep include directories: {missing}")

    host_source = _patched_host_source()
    adapter_sha = hashlib.sha256(host_source.encode("utf-8")).hexdigest()
    module_name = f"itk_fused_r10a_nsweep_host_{adapter_sha[:12]}"
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
