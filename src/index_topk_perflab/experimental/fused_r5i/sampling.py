"""Random-token sampling utilities for the producer-side R5i fusion."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from index_topk_perflab.provenance import path_fingerprint

_SOURCE = Path(__file__).resolve().parent / "csrc" / "sampling_threshold_r5i.cu"
_EXTENSION: Any | None = None


def sample_elements_for_context(context_tokens: int) -> int:
    """Use R5i's 0.5% schedule, rounded to 128 independent token samples."""

    if context_tokens <= 0:
        raise ValueError("context_tokens must be positive")
    full_tiles = max(1, context_tokens // 128)
    sample_tiles = max(1, math.ceil(full_tiles / 200))
    return min(context_tokens, sample_tiles * 128)


def guarded_sample_rank(
    sample_elements: int,
    context_tokens: int,
    *,
    target_candidates: int = 3072,
    guard_sigmas: float = 2.0,
) -> int:
    """Return the conservative descending sample rank used by R5i."""

    if not 0 < target_candidates <= context_tokens:
        raise ValueError("target_candidates must lie in [1, context_tokens]")
    if not 0 < sample_elements <= context_tokens:
        raise ValueError("sample_elements must lie in [1, context_tokens]")
    if guard_sigmas < 0 or not math.isfinite(guard_sigmas):
        raise ValueError("guard_sigmas must be finite and non-negative")
    probability = target_candidates / context_tokens
    mean = sample_elements * probability
    deviation = math.sqrt(sample_elements * probability * (1.0 - probability))
    return max(
        1,
        min(sample_elements, math.ceil(mean + guard_sigmas * deviation)),
    )


def common_random_sample_ids(
    torch: Any,
    context_tokens: int,
    sample_elements: int,
    *,
    seed: int,
    device: Any,
) -> Any:
    """Choose independent token positions without replacement, then sort them."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    ids = torch.randperm(
        context_tokens, generator=generator, device="cpu", dtype=torch.int64
    )[:sample_elements]
    return torch.sort(ids).values.to(device=device, dtype=torch.int32).contiguous()


def load_sampling_extension(*, verbose: bool = False) -> Any:
    """Compile and return the sampled-KV gather and threshold extension."""

    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION

    from torch.utils.cpp_extension import CUDA_HOME, load

    if CUDA_HOME is None:
        raise RuntimeError("CUDA_HOME is required to build fused R5i sampling")
    identity = path_fingerprint(_SOURCE)
    module_name = f"itk_fused_r5i_sampling_{identity['sha256'][:12]}"
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
    return _EXTENSION
