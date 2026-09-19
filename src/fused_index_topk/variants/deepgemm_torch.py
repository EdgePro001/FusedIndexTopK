"""Exact reference: DeepGEMM Indexer + PyTorch exact TopK + output pack."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..api import (
    PrefillCase,
    PrefillInputs,
    PreparedGraph,
    RunMode,
    VariantDescriptor,
)
from ..provenance import path_fingerprint
from .common import deepgemm_indexer_stage, int32_output_stage, torch_exact_topk_stage


class DeepGemmTorchUnfused:
    descriptor = VariantDescriptor(
        plugin_id="deepgemm_torch_unfused",
        display_name="DeepGEMM + torch.topk unfused",
        api_version="1.0",
        implementation_version="2",
        mode="unfused",
        description="Released SM90 FP8 Indexer, PyTorch exact TopK, and INT32 output pack",
        implementation="deepgemm+pytorch",
        exact_topk=True,
        source_revision="deepgemm:7c95b14;torch:2.10.0+cu130",
        tags=("exact-reference", "prefill", "sm90"),
    )

    def __init__(self, options: Mapping[str, Any] | None = None) -> None:
        self.options = dict(options or {})
        unknown = set(self.options) - {"sorted"}
        if unknown:
            raise ValueError(f"unknown deepgemm_torch_unfused options: {sorted(unknown)}")
        self.sorted = bool(self.options.get("sorted", False))

    def supports(self, case: PrefillCase) -> bool:
        return (
            case.batch_size == 1
            and case.causal
            and case.indexer_heads == 64
            and case.head_dim == 128
            and case.context_tokens >= case.query_tokens
        )

    def fingerprint_metadata(self) -> Mapping[str, Any]:
        return {
            "variant_source": path_fingerprint(Path(__file__)),
            "released_stage_source": path_fingerprint(Path(__file__).with_name("common.py")),
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

        import deep_gemm  # Lazy: CPU-only registry/tests must not import CUDA extensions.
        import torch

        nodes = (
            deepgemm_indexer_stage(deep_gemm),
            torch_exact_topk_stage(torch, sorted_output=self.sorted),
            int32_output_stage(torch),
        )
        return PreparedGraph(
            descriptor=self.descriptor,
            nodes=nodes,
            initial_artifacts={"inputs": inputs},
            terminal_artifact="indices",
            # RunMode is deliberately excluded: the same implementation must
            # have one graph fingerprint in correctness, timing, Nsys, and NCU.
            metadata={"sorted": self.sorted},
        )


def create_variant(options: Mapping[str, Any] | None = None) -> DeepGemmTorchUnfused:
    return DeepGemmTorchUnfused(options)
