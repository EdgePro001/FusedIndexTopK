"""Build the candidate-only exact radix reducer used by fused R5i."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from index_topk_perflab.provenance import path_fingerprint

_ROOT = Path(__file__).resolve().parent
_SOURCE = _ROOT / "csrc" / "candidate_topk_r5i.cu"
_SEGMENTED_SOURCE = _ROOT / "csrc" / "segmented_candidate_topk_r5i.cu"
_EXTENSION: Any | None = None
_SEGMENTED_EXTENSION: Any | None = None


def load_candidate_reducer(*, verbose: bool = False) -> Any:
    """Compile and return the project-local candidate radix extension."""

    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION

    from torch.utils.cpp_extension import CUDA_HOME, load

    if CUDA_HOME is None:
        raise RuntimeError("CUDA_HOME is required to build fused R5i")
    identity = path_fingerprint(_SOURCE)
    module_name = f"itk_fused_r5i_candidate_{identity['sha256'][:12]}"
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


def load_segmented_candidate_reducer(*, verbose: bool = False) -> Any:
    """Compile the exact radix reducer for 16 fixed 880-entry segments."""

    global _SEGMENTED_EXTENSION
    if _SEGMENTED_EXTENSION is not None:
        return _SEGMENTED_EXTENSION

    from torch.utils.cpp_extension import CUDA_HOME, load

    if CUDA_HOME is None:
        raise RuntimeError("CUDA_HOME is required to build fused R5i")
    identity = path_fingerprint(_SEGMENTED_SOURCE)
    module_name = f"itk_fused_r5i_segmented_{identity['sha256'][:12]}"
    extension = load(
        name=module_name,
        sources=[str(_SEGMENTED_SOURCE)],
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
    _SEGMENTED_EXTENSION = extension
    return extension
