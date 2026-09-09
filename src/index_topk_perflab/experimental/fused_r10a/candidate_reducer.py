"""Build the sparse-direct-atomic third-histogram reducer used by R10a."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from index_topk_perflab.provenance import path_fingerprint

_ROOT = Path(__file__).resolve().parent
_SOURCE = _ROOT / "csrc" / "segmented_candidate_topk_r10a.cu"
_EXTENSION: Any | None = None


def load_segmented_candidate_reducer(*, verbose: bool = False) -> Any:
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION

    from torch.utils.cpp_extension import CUDA_HOME, load

    if CUDA_HOME is None:
        raise RuntimeError("CUDA_HOME is required to build fused R10a")
    identity = path_fingerprint(_SOURCE)
    module_name = f"itk_fused_r10a_segmented_{identity['sha256'][:12]}"
    extension = load(
        name=module_name,
        sources=[str(_SOURCE)],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "-lineinfo",
        ],
        with_cuda=True,
        verbose=verbose,
    )
    extension.configure()
    _EXTENSION = extension
    return extension
