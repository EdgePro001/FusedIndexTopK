"""GPU-agnostic contracts shared by the core runner and external variants."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable


class RunMode(str, Enum):
    CORRECTNESS = "correctness"
    BENCHMARK = "benchmark"
    NSYS = "nsys"
    NCU = "ncu"


@dataclass(frozen=True)
class PrefillCase:
    case_id: str
    query_tokens: int
    context_tokens: int
    top_k: int
    seed: int
    batch_size: int = 1
    indexer_heads: int = 64
    head_dim: int = 128
    causal: bool = True

    @property
    def query_start(self) -> int:
        return self.context_tokens - self.query_tokens


@dataclass
class PrefillInputs:
    """Standard contiguous Prefill inputs; fields intentionally use ``Any``.

    Importing the framework and listing plugins must not initialize PyTorch/CUDA.
    Real runners populate these fields with CUDA tensors.
    """

    case: PrefillCase
    q: Any
    kv: Any
    kv_scales: Any
    weights: Any
    k_start: Any
    k_end: Any
    generation_context_tokens: int | None = None
    fixture_id: str = "A"
    stream_seeds: Mapping[str, int] = field(default_factory=dict)
    recipe_version: str = "contiguous-prefill-fp8-v2"
    recipe_metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VariantDescriptor:
    plugin_id: str
    display_name: str
    api_version: str
    implementation_version: str
    mode: str
    description: str
    implementation: str
    exact_topk: bool
    source_revision: str | None = None
    supported_arches: tuple[str, ...] = ("sm90",)
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class StageSpec:
    """One physical stage in a variant graph.

    ``semantic_ops`` maps physical stages back to the logical Indexer/TopK/output
    contract. A fused node may cover several logical operations.
    """

    stage_id: str
    dependencies: tuple[str, ...]
    consumes: tuple[str, ...]
    produces: tuple[str, ...]
    semantic_ops: tuple[str, ...]
    description: str = ""
    profile: bool = True
    kernel_regexes: tuple[str, ...] = ()


@dataclass
class ExecutionContext:
    case: PrefillCase
    mode: RunMode
    iteration: int
    stream: Any = None


ArtifactStore = dict[str, Any]
StageCallable = Callable[[ExecutionContext, ArtifactStore], None]


@dataclass
class StageNode:
    spec: StageSpec
    run: StageCallable


@dataclass
class PreparedGraph:
    descriptor: VariantDescriptor
    nodes: Sequence[StageNode]
    initial_artifacts: Mapping[str, Any]
    terminal_artifact: str = "indices"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class VariantPlugin(Protocol):
    descriptor: VariantDescriptor

    def supports(self, case: PrefillCase) -> bool:
        ...

    def fingerprint_metadata(self) -> Mapping[str, Any]:
        """Return stable source/binary hashes used to bind all artifacts."""

        ...

    def prepare(
        self,
        case: PrefillCase,
        inputs: PrefillInputs,
        *,
        options: Mapping[str, Any],
        mode: RunMode,
    ) -> PreparedGraph:
        """Perform JIT/autotune/workspace allocation outside the timed path."""
        ...
