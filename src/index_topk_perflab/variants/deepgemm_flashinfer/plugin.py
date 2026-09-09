"""DeepGEMM Indexer paired with locked FlashInfer v0.6.17 exact TopK."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from index_topk_perflab.api import (
    PrefillCase,
    PrefillInputs,
    PreparedGraph,
    RunMode,
    VariantDescriptor,
)
from index_topk_perflab.provenance import path_fingerprint
from index_topk_perflab.variants.common import (
    deepgemm_indexer_stage,
    external_topk_stage,
    int32_output_stage,
    supports_frozen_deepgemm_case,
)

_ROOT = Path(__file__).resolve().parent
_CSRC = _ROOT / "csrc"
_EXPECTED_COMMIT = "a0a6b019b9b27d49d209f85d028a1ae5a9b347d7"
_ALGORITHM_IDS = {"auto": 0, "filtered": 1, "multi_cta": 2}
_KERNEL_REGEXES = {
    "auto": ("FilteredTopK", "RadixTopK"),
    "filtered": ("FilteredTopK",),
    "multi_cta": ("RadixTopK",),
}
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
        raise RuntimeError(f"cannot inspect frozen FlashInfer checkout at {source}") from error
    return process.stdout.strip()


def _flashinfer_source_root() -> Path:
    raw = os.environ.get("FLASHINFER_SOURCE")
    if raw is None:
        raise RuntimeError(
            "FLASHINFER_SOURCE must point at the frozen FlashInfer checkout; "
            "use scripts/run_h20.sh"
        )
    source = Path(raw).resolve()
    header = source / "include" / "flashinfer" / "topk.cuh"
    if not header.is_file():
        raise RuntimeError(
            f"missing FlashInfer v0.6.17 header: {header}; run the candidate setup script"
        )
    commit = _git_output(source, "rev-parse", "HEAD")
    if commit != _EXPECTED_COMMIT:
        raise RuntimeError(
            f"FlashInfer checkout mismatch: expected {_EXPECTED_COMMIT}, got {commit}"
        )
    dirty = _git_output(source, "status", "--short")
    if dirty:
        raise RuntimeError("FlashInfer checkout must be clean")
    return source


def _load_extension(*, verbose: bool) -> Any:
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION

    from torch.utils.cpp_extension import CUDA_HOME, load

    if CUDA_HOME is None:
        raise RuntimeError("CUDA_HOME is required to build the FlashInfer TopK adapter")
    source = _flashinfer_source_root()
    identity = path_fingerprint(_CSRC)
    module_name = f"itk_flashinfer_topk_{identity['tree_sha256'][:12]}"
    _EXTENSION = load(
        name=module_name,
        sources=[str(_CSRC / "topk_extension.cu")],
        extra_include_paths=[str(source / "include")],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
        ],
        with_cuda=True,
        verbose=verbose,
    )
    return _EXTENSION


class DeepGemmFlashInferTopK:
    """One of the locked FlashInfer algorithm-dispatch alternatives."""

    def __init__(
        self,
        algorithm: str,
        options: Mapping[str, Any] | None = None,
    ) -> None:
        if algorithm not in _ALGORITHM_IDS:
            raise ValueError(f"unknown FlashInfer TopK algorithm: {algorithm}")
        self.algorithm = algorithm
        self.options = dict(options or {})
        unknown = set(self.options) - {"verbose_build"}
        if unknown:
            raise ValueError(f"unknown FlashInfer TopK options: {sorted(unknown)}")
        self.descriptor = VariantDescriptor(
            plugin_id=f"deepgemm_flashinfer_topk_{algorithm}",
            display_name=f"DeepGEMM + FlashInfer TopK ({algorithm})",
            api_version="1.0",
            implementation_version="flashinfer-v0.6.17-adapter-v1",
            mode="unfused",
            description=(
                "Released DeepGEMM FP8 Indexer, locked FlashInfer exact TopK, "
                "and released INT32 output pack"
            ),
            implementation=f"deepgemm+flashinfer-{algorithm}",
            exact_topk=True,
            source_revision=f"deepgemm:7c95b14;flashinfer:{_EXPECTED_COMMIT}",
            tags=("baseline", "prefill", "sm90", "flashinfer", algorithm),
        )

    def supports(self, case: PrefillCase) -> bool:
        return supports_frozen_deepgemm_case(case) and (
            self.algorithm != "filtered" or case.top_k <= 2048
        )

    def fingerprint_metadata(self) -> Mapping[str, Any]:
        source = _flashinfer_source_root()
        return {
            "algorithm": self.algorithm,
            "adapter_source": path_fingerprint(_CSRC),
            "plugin_python": path_fingerprint(Path(__file__)),
            "shared_adapter_python": path_fingerprint(_ROOT.parent / "common.py"),
            "source_lock": path_fingerprint(_ROOT / "SOURCE_LOCK.json"),
            "flashinfer_commit": _EXPECTED_COMMIT,
            "flashinfer_topk_header": path_fingerprint(
                source / "include" / "flashinfer" / "topk.cuh"
            ),
        }

    def prepare(
        self,
        case: PrefillCase,
        inputs: PrefillInputs,
        *,
        options: Mapping[str, Any],
        mode: RunMode,
    ) -> PreparedGraph:
        if not self.supports(case):
            raise ValueError(f"unsupported case for {self.descriptor.plugin_id}: {case}")
        if options and dict(options) != self.options:
            raise ValueError("variant options changed after plugin construction")

        import deep_gemm
        import torch

        extension = _load_extension(verbose=bool(self.options.get("verbose_build", False)))
        output_values = torch.empty(
            (case.query_tokens, case.top_k), device=inputs.q.device, dtype=torch.float32
        )
        output_ids = torch.empty(
            (case.query_tokens, case.top_k), device=inputs.q.device, dtype=torch.int32
        )
        row_states = torch.zeros(1024 * 1024, device=inputs.q.device, dtype=torch.uint8)
        algorithm_id = _ALGORITHM_IDS[self.algorithm]

        def launch(physical: Any, ids: Any, values: Any) -> None:
            extension.topk_out(physical, ids, values, row_states, algorithm_id)

        nodes = (
            deepgemm_indexer_stage(deep_gemm),
            external_topk_stage(
                torch,
                description=(
                    f"physical-tail -inf adapter + FlashInfer {self.algorithm} exact TopK"
                ),
                launcher=launch,
                output_values=output_values,
                output_ids=output_ids,
                kernel_regexes=_KERNEL_REGEXES[self.algorithm],
            ),
            int32_output_stage(torch),
        )
        return PreparedGraph(
            descriptor=self.descriptor,
            nodes=nodes,
            initial_artifacts={"inputs": inputs},
            terminal_artifact="indices",
            metadata={
                "topk_library": "flashinfer",
                "topk_algorithm": self.algorithm,
                "upstream_version": "v0.6.17",
                "row_state_bytes": 1024 * 1024,
                "physical_stride_adapter_timed": True,
                "sorted": False,
            },
        )


def create_auto_variant(
    options: Mapping[str, Any] | None = None,
) -> DeepGemmFlashInferTopK:
    return DeepGemmFlashInferTopK("auto", options)


def create_filtered_variant(
    options: Mapping[str, Any] | None = None,
) -> DeepGemmFlashInferTopK:
    return DeepGemmFlashInferTopK("filtered", options)


def create_multi_cta_variant(
    options: Mapping[str, Any] | None = None,
) -> DeepGemmFlashInferTopK:
    return DeepGemmFlashInferTopK("multi_cta", options)
