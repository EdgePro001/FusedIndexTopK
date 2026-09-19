"""DeepGEMM Indexer paired with frozen DeepSelect exact Top-K.

DeepSelect v1.0.0 does not build SM90 in its released setup.py.  This adapter
therefore instantiates the exact upstream FP32 template selected by its host
dispatch for K in (1024, 4096], without changing the upstream kernel source.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from fused_index_topk.api import (
    PrefillCase,
    PrefillInputs,
    PreparedGraph,
    RunMode,
    StageNode,
    StageSpec,
    VariantDescriptor,
)
from fused_index_topk.provenance import path_fingerprint
from fused_index_topk.variants.common import (
    deepgemm_indexer_stage,
    supports_frozen_deepgemm_case,
)

_ROOT = Path(__file__).resolve().parent
_CSRC = _ROOT / "csrc"
_EXPECTED_COMMIT = "0f03b68748b304863fdf0181a11458d04ae533a9"
_EXTENSION: Any | None = None


def _git_output(source: Path, *arguments: str) -> str:
    try:
        process = subprocess.run(
            ["git", "-C", str(source), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"cannot inspect frozen DeepSelect checkout at {source}") from error
    return process.stdout.strip()


def _deepselect_source_root() -> Path:
    raw = os.environ.get("DEEPSELECT_SOURCE")
    if raw is None:
        raise RuntimeError(
            "DEEPSELECT_SOURCE must point at the frozen DeepSelect checkout; "
            "use scripts/run_h20.sh"
        )
    source = Path(raw).resolve()
    required = (
        source / "csrc" / "cuda_kernels" / "v3_fp32" / "topk_select.cuh",
        source / "csrc" / "cuda_kernels" / "common_parts.cuh",
        source / "csrc" / "3rdparty" / "cutlass" / "include" / "cute" / "tensor.hpp",
        source
        / "csrc"
        / "3rdparty"
        / "kerutils"
        / "include"
        / "kerutils"
        / "kerutils.cuh",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(
            "DeepSelect checkout or submodules are incomplete: " + ", ".join(missing)
        )
    commit = _git_output(source, "rev-parse", "HEAD")
    if commit != _EXPECTED_COMMIT:
        raise RuntimeError(
            f"DeepSelect checkout mismatch: expected {_EXPECTED_COMMIT}, got {commit}"
        )
    dirty = _git_output(source, "status", "--short")
    if dirty:
        raise RuntimeError("DeepSelect checkout must be clean")
    return source


def _load_extension(*, verbose: bool) -> Any:
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION

    from torch.utils.cpp_extension import CUDA_HOME, load

    if CUDA_HOME is None:
        raise RuntimeError("CUDA_HOME is required to build the DeepSelect adapter")
    source = _deepselect_source_root()
    cuda_home = Path(CUDA_HOME)
    include_paths = [
        source / "csrc",
        source / "csrc" / "3rdparty" / "cutlass" / "include",
        source / "csrc" / "3rdparty" / "kerutils" / "include",
    ]
    include_paths.extend(
        path
        for path in (
            cuda_home / "targets" / "x86_64-linux" / "include" / "cccl",
            cuda_home / "targets" / "sbsa-linux" / "include" / "cccl",
        )
        if path.is_dir()
    )
    driver_stubs = [
        path
        for path in (
            cuda_home / "targets" / "x86_64-linux" / "lib" / "stubs",
            cuda_home / "targets" / "sbsa-linux" / "lib" / "stubs",
        )
        if path.is_dir()
    ]
    identity = path_fingerprint(_CSRC)
    module_name = f"fused_index_topk_deepselect_{identity['tree_sha256'][:12]}"
    _EXTENSION = load(
        name=module_name,
        sources=[str(_CSRC / "topk_extension.cu")],
        extra_include_paths=[str(path) for path in include_paths],
        extra_cflags=[
            "-O3",
            "-std=c++20",
            "-DNDEBUG",
            "-Wno-deprecated-declarations",
            "-DKERUTILS_IS_BUILD_ON_CUDA",
        ],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++20",
            "-DNDEBUG",
            "-Wno-deprecated-declarations",
            "-DKERUTILS_IS_BUILD_ON_CUDA",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--use_fast_math",
            "--ftz=false",
            "--ptxas-options=-v,--register-usage-level=10,--warn-on-spills",
            "-lineinfo",
        ],
        extra_ldflags=[*(f"-L{path}" for path in driver_stubs), "-lcuda"],
        with_cuda=True,
        verbose=verbose,
    )
    return _EXTENSION


class DeepGemmDeepSelectTopK:
    """Unmodified DeepSelect FP32 selection behind an SM90 build adapter."""

    descriptor = VariantDescriptor(
        plugin_id="deepgemm_deepselect_topk",
        display_name="DeepGEMM + DeepSelect Top-K",
        api_version="1.0",
        implementation_version="deepselect-v1.0.0-sm90-build-adapter-v1",
        mode="unfused",
        description=(
            "Released DeepGEMM FP8 Indexer plus frozen DeepSelect FP32 exact Top-K"
        ),
        implementation="deepgemm+deepselect-fp32",
        exact_topk=True,
        source_revision=f"deepgemm:7c95b14;deepselect:{_EXPECTED_COMMIT}",
        tags=(
            "baseline-candidate",
            "prefill",
            "sm90",
            "deepselect",
            "upstream-kernel-unmodified",
            "sm90-build-adapter",
        ),
    )

    def __init__(self, options: Mapping[str, Any] | None = None) -> None:
        self.options = dict(options or {})
        unknown = set(self.options) - {"verbose_build"}
        if unknown:
            raise ValueError(f"unknown DeepSelect options: {sorted(unknown)}")

    def supports(self, case: PrefillCase) -> bool:
        return (
            supports_frozen_deepgemm_case(case)
            and 1024 < case.top_k <= 4096
            and case.context_tokens < (1 << 23)
        )

    def fingerprint_metadata(self) -> Mapping[str, Any]:
        source = _deepselect_source_root()
        return {
            "algorithm": "deepselect-v3-fp32-max-topk4096",
            "upstream_commit": _EXPECTED_COMMIT,
            "upstream_kernel_modified": False,
            "upstream_fp32_kernel": path_fingerprint(
                source / "csrc" / "cuda_kernels" / "v3_fp32" / "topk_select.cuh"
            ),
            "upstream_common_kernel": path_fingerprint(
                source / "csrc" / "cuda_kernels" / "common_parts.cuh"
            ),
            "upstream_config": path_fingerprint(
                source / "csrc" / "cuda_kernels" / "config.h"
            ),
            "upstream_arguments": path_fingerprint(source / "csrc" / "structs.h"),
            "upstream_setup": path_fingerprint(source / "setup.py"),
            "adapter_source": path_fingerprint(_CSRC),
            "plugin_python": path_fingerprint(Path(__file__)),
            "source_lock": path_fingerprint(_ROOT / "SOURCE_LOCK.json"),
            "sm90_adaptation": "template instantiation and PyTorch ABI binding only",
            "template": {
                "value": "float32",
                "index": "int32",
                "sorted_value": False,
                "sorted_index": False,
                "return_value": False,
                "max_topk": 4096,
                "threads": 256,
                "target_occupancy": 1,
                "elements_per_round": 4096,
                "reconstruct_threshold": 4096,
                "tma_buffer_depth": 3,
                "elements_per_segment": 512,
                "cluster_size": 1,
            },
        }

    def prepare(
        self,
        case: PrefillCase,
        inputs: PrefillInputs,
        *,
        options: Mapping[str, Any],
        mode: RunMode,
    ) -> PreparedGraph:
        del mode
        if not self.supports(case):
            raise ValueError(f"unsupported case for {self.descriptor.plugin_id}: {case}")
        if options and dict(options) != self.options:
            raise ValueError("variant options changed after plugin construction")

        import deep_gemm
        import torch

        extension = _load_extension(verbose=bool(self.options.get("verbose_build", False)))
        output_ids = torch.empty(
            (case.query_tokens, case.top_k),
            device=inputs.q.device,
            dtype=torch.int32,
        )
        output_view = output_ids.unsqueeze(1)

        def run_topk(context: Any, artifacts: dict[str, Any]) -> None:
            del context
            standard: PrefillInputs = artifacts["inputs"]
            extension.topk_out(artifacts["logits"], standard.k_end, output_ids)
            artifacts["topk_ids"] = output_ids
            artifacts["indices"] = output_view

        nodes = (
            deepgemm_indexer_stage(deep_gemm),
            StageNode(
                StageSpec(
                    stage_id="topk",
                    dependencies=("indexer",),
                    consumes=("inputs", "logits"),
                    produces=("topk_ids", "indices"),
                    semantic_ops=("topk", "output"),
                    description=(
                        "Frozen DeepSelect FP32 exact Top-K; INT32 indices only; "
                        "causal end passed directly"
                    ),
                    kernel_regexes=("topk_kernel",),
                ),
                run_topk,
            ),
        )
        return PreparedGraph(
            descriptor=self.descriptor,
            nodes=nodes,
            initial_artifacts={"inputs": inputs},
            terminal_artifact="indices",
            metadata={
                "topk_library": "deepselect",
                "upstream_version": "1.0.0",
                "upstream_commit": _EXPECTED_COMMIT,
                "upstream_kernel_modified": False,
                "sm90_build_adapter": True,
                "score_dtype": "float32",
                "indices_dtype": "int32",
                "return_value": False,
                "sorted": False,
                "causal_end_consumed": True,
            },
        )


def create_variant(
    options: Mapping[str, Any] | None = None,
) -> DeepGemmDeepSelectTopK:
    return DeepGemmDeepSelectTopK(options)
