"""Build FusedIndexTopK's fixed-memory hierarchical exact repair."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from index_topk_perflab.provenance import path_fingerprint

_ROOT = Path(__file__).resolve().parent
_SOURCE = _ROOT / "csrc" / "long_context_repair.cu"
_EXTENSION: Any | None = None


def load_long_context_repair(*, verbose: bool = False) -> Any:
    """Compile and return the chunk-local and merge reducers."""

    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION

    from torch.utils.cpp_extension import CUDA_HOME, load

    if CUDA_HOME is None:
        raise RuntimeError("CUDA_HOME is required to build FusedIndexTopK repair")
    identity = path_fingerprint(_SOURCE)
    module_name = f"fused_index_topk_long_repair_{identity['sha256'][:12]}"
    _EXTENSION = load(
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
    _EXTENSION.configure()
    return _EXTENSION
