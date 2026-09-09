"""Stable NVTX names and lightweight annotation helpers.

The labels in this module are part of the profiling artifact contract.  They do
not include a run ID, process ID, timestamp, or iteration number, so Nsight
Compute application replay sees the same names in every process invocation.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterator, Mapping

from .api import PrefillCase, PreparedGraph, StageNode
from .graph import validate_graph

_SAFE_COMPONENT = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_PREFIX = "ITK"
_MAX_LABEL_BYTES = 240


def validate_label_component(value: str, *, field: str) -> str:
    """Reject characters that have special meaning in NCU NVTX filters.

    In particular, ``/``, ``@``, commas, brackets, and whitespace must never
    occur in a component.  Silently replacing them could make two variants map
    to the same label, so invalid values fail instead.
    """

    if not isinstance(value, str) or not _SAFE_COMPONENT.fullmatch(value):
        raise ValueError(
            f"{field} must match {_SAFE_COMPONENT.pattern!r}; got {value!r}"
        )
    return value


def case_id_for_length(target_length: int) -> str:
    if isinstance(target_length, bool) or not isinstance(target_length, int):
        raise TypeError("target_length must be an integer")
    if target_length <= 0:
        raise ValueError("target_length must be positive")
    return f"prefill-l{target_length}"


def _checked_label(*components: str) -> str:
    label = "::".join((_PREFIX, *components))
    if len(label.encode("utf-8")) > _MAX_LABEL_BYTES:
        raise ValueError(f"NVTX label is longer than {_MAX_LABEL_BYTES} bytes: {label!r}")
    return label


def pipeline_label(variant_id: str, case_id: str) -> str:
    variant_id = validate_label_component(variant_id, field="variant_id")
    case_id = validate_label_component(case_id, field="case_id")
    return _checked_label("RUN", variant_id, case_id)


def stage_label(variant_id: str, case_id: str, stage_id: str) -> str:
    variant_id = validate_label_component(variant_id, field="variant_id")
    case_id = validate_label_component(case_id, field="case_id")
    stage_id = validate_label_component(stage_id, field="stage_id")
    return _checked_label("STAGE", variant_id, case_id, stage_id)


def ncu_push_pop_filter(label: str) -> str:
    """Return the NCU filter expression for a push/pop NVTX range."""

    if not label.startswith(f"{_PREFIX}::") or "/" in label or "@" in label:
        raise ValueError(f"unsafe or foreign NVTX label: {label!r}")
    return f"{label}/"


@dataclass(frozen=True)
class GraphNvtxLabels:
    pipeline: str
    stages: Mapping[str, str]

    def as_mapping(self) -> dict[str, Any]:
        return {"pipeline": self.pipeline, "stages": dict(self.stages)}


def graph_labels(graph: PreparedGraph, case: PrefillCase) -> GraphNvtxLabels:
    ordered = validate_graph(graph)
    variant_id = graph.descriptor.plugin_id
    case_id = case.case_id
    labels = {
        node.spec.stage_id: stage_label(variant_id, case_id, node.spec.stage_id)
        for node in ordered
    }
    return GraphNvtxLabels(
        pipeline=pipeline_label(variant_id, case_id),
        stages=MappingProxyType(labels),
    )


@contextmanager
def nvtx_range(torch_module: Any, label: str) -> Iterator[None]:
    """Annotate a range without importing torch at module import time."""

    torch_module.cuda.nvtx.range_push(label)
    try:
        yield
    finally:
        torch_module.cuda.nvtx.range_pop()


def stage_range_factory(
    torch_module: Any,
    labels: GraphNvtxLabels,
):
    """Build the ``execute_graph(stage_context=...)`` callback."""

    def stage_context(node: StageNode):
        try:
            label = labels.stages[node.spec.stage_id]
        except KeyError as error:  # Defensive: the graph changed after validation.
            raise RuntimeError(f"missing NVTX label for stage {node.spec.stage_id!r}") from error
        return nvtx_range(torch_module, label)

    return stage_context
