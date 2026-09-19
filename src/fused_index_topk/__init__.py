"""FusedIndexTopK operator, baselines, and reproducible evaluation tools."""

from .api import (
    ExecutionContext,
    PrefillCase,
    PrefillInputs,
    PreparedGraph,
    RunMode,
    StageNode,
    StageSpec,
    VariantDescriptor,
    VariantPlugin,
)
from .graph import execute_graph, graph_fingerprint, validate_graph
from .registry import load_variant

__all__ = [
    "ExecutionContext",
    "PrefillCase",
    "PrefillInputs",
    "PreparedGraph",
    "RunMode",
    "StageNode",
    "StageSpec",
    "VariantDescriptor",
    "VariantPlugin",
    "execute_graph",
    "graph_fingerprint",
    "load_variant",
    "validate_graph",
]

__version__ = "0.2.0"
