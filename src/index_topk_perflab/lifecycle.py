"""The one input/oracle/prepare/check lifecycle used by every GPU path."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

from .api import (
    ArtifactStore,
    ExecutionContext,
    PrefillCase,
    PrefillInputs,
    PreparedGraph,
    RunMode,
    VariantPlugin,
)
from .artifacts import canonical_hash
from .correctness import (
    build_score_threshold_oracle,
    compare_topk_indices,
    validate_topk_indices,
)
from .graph import execute_prevalidated, graph_fingerprint, validate_graph
from .inputs import (
    input_content_fingerprint,
    input_manifest,
    make_prefill_inputs,
    validate_prefill_inputs,
)

FIXTURE_IDS = ("A", "B")
_INPUT_NAMES = ("q", "kv", "kv_scales", "weights", "k_start", "k_end")
InputFactory = Callable[[PrefillCase, str | Any, str], PrefillInputs]


@dataclass
class FixtureState:
    fixture_id: str
    inputs: PrefillInputs
    input_contract_check: Mapping[str, Any]
    manifest: Mapping[str, Any]
    manifest_hash: str
    content_before: Mapping[str, Any]
    versions_before: Mapping[str, int | None]

    @property
    def content_hash(self) -> str:
        return str(self.content_before["sha256"])


@dataclass
class OracleState:
    fixture_id: str
    indices: Any
    output_check: Mapping[str, Any]
    cutoff_tie_check: Mapping[str, Any]
    cutoff_tie_rows: Mapping[int, Mapping[str, Any]]


def _input_versions(inputs: PrefillInputs) -> dict[str, int | None]:
    return {
        name: getattr(getattr(inputs, name), "_version", None)
        for name in _INPUT_NAMES
    }


class CaseLifecycle:
    """Own one case from compact A/B generation through final integrity check.

    A candidate is prepared exactly once with fixture A.  Every graph execution
    receives its active input through the ``inputs`` artifact, allowing the same
    prepared graph to be checked and timed against fixture B.  A plugin that
    hides an A result in ``prepare()`` therefore fails the B exactness gate.
    """

    def __init__(
        self,
        *,
        case: PrefillCase,
        workload: Any,
        reference: VariantPlugin,
        reference_options: Mapping[str, Any],
        mode: RunMode,
        device: str | Any = "cuda",
        input_factory: InputFactory | None = None,
    ) -> None:
        self.case = case
        self.workload = workload
        self.reference = reference
        self.reference_options = dict(reference_options)
        self.mode = mode
        self.device = device
        self.input_factory = input_factory
        self.fixtures: dict[str, FixtureState] = {}
        self.oracles: dict[str, OracleState] = {}
        self.reference_graph_fingerprint: str | None = None
        self.candidate_graph: PreparedGraph | None = None
        self.candidate_nodes: tuple[Any, ...] = ()
        self.integrity_checks: list[dict[str, Any]] = []
        self._create_fixtures()

    def _create_fixtures(self) -> None:
        for fixture_id in FIXTURE_IDS:
            if self.input_factory is None:
                inputs = make_prefill_inputs(
                    self.case,
                    device=self.device,
                    fixture_id=fixture_id,
                )
            else:
                inputs = self.input_factory(self.case, self.device, fixture_id)
                if inputs.case != self.case:
                    raise ValueError("input factory returned inputs for a different case")
                if inputs.fixture_id != fixture_id:
                    raise ValueError(
                        "input factory returned the wrong A/B fixture identifier"
                    )
            check = validate_prefill_inputs(inputs, self.workload)
            manifest = input_manifest(inputs)
            content = input_content_fingerprint(inputs)
            self.fixtures[fixture_id] = FixtureState(
                fixture_id=fixture_id,
                inputs=inputs,
                input_contract_check=check,
                manifest=manifest,
                manifest_hash=canonical_hash(manifest),
                content_before=content,
                versions_before=_input_versions(inputs),
            )
        hashes = {fixture.content_hash for fixture in self.fixtures.values()}
        if len(hashes) != len(FIXTURE_IDS):
            raise RuntimeError("A/B fixtures must have different tensor content")

    @property
    def input_content_hash(self) -> str:
        return canonical_hash(
            {
                fixture_id: self.fixtures[fixture_id].content_hash
                for fixture_id in FIXTURE_IDS
            }
        )

    def fixture_metadata(self) -> dict[str, Any]:
        return {
            fixture_id: {
                "manifest": fixture.manifest,
                "manifest_hash": fixture.manifest_hash,
                "content": fixture.content_before,
                "input_contract_check": fixture.input_contract_check,
            }
            for fixture_id, fixture in self.fixtures.items()
        }

    def verify_input_integrity(self, phase: str) -> dict[str, Any]:
        """Re-hash all logical bytes and reject version/content mutations."""

        fixtures: dict[str, Any] = {}
        for fixture_id, fixture in self.fixtures.items():
            current = input_content_fingerprint(fixture.inputs)
            versions = _input_versions(fixture.inputs)
            if current["sha256"] != fixture.content_hash:
                raise RuntimeError(
                    f"fixture {fixture_id} tensor content changed during {phase}"
                )
            if versions != fixture.versions_before:
                raise RuntimeError(
                    f"fixture {fixture_id} version counters changed during {phase}"
                )
            fixtures[fixture_id] = {
                "sha256_before": fixture.content_hash,
                "sha256_after": current["sha256"],
                "versions_unchanged": True,
            }
        result = {"status": "passed", "phase": phase, "fixtures": fixtures}
        self.integrity_checks.append(result)
        return result

    def _prepare(
        self,
        plugin: VariantPlugin,
        options: Mapping[str, Any],
        *,
        fixture_id: str = "A",
    ) -> tuple[PreparedGraph, tuple[Any, ...]]:
        if not plugin.supports(self.case):
            raise ValueError(
                f"variant {plugin.descriptor.plugin_id!r} does not support {self.case.case_id}"
            )
        graph = plugin.prepare(
            self.case,
            self.fixtures[fixture_id].inputs,
            options=dict(options),
            mode=self.mode,
        )
        if graph.descriptor.plugin_id != plugin.descriptor.plugin_id:
            raise ValueError("prepared graph descriptor does not match the loaded plugin")
        if "inputs" not in graph.initial_artifacts:
            raise ValueError(
                "prepared graph must expose standard inputs as initial_artifacts['inputs']"
            )
        return graph, validate_graph(graph)

    def bound_artifacts(self, graph: PreparedGraph, fixture_id: str) -> ArtifactStore:
        try:
            fixture = self.fixtures[fixture_id]
        except KeyError as error:
            raise ValueError(f"unknown fixture_id {fixture_id!r}") from error
        artifacts: ArtifactStore = dict(graph.initial_artifacts)
        artifacts["inputs"] = fixture.inputs
        return artifacts

    def execute(
        self,
        graph: PreparedGraph,
        nodes: Sequence[Any],
        *,
        fixture_id: str,
        iteration: int,
        stage_context: Any = None,
        validate_produced: bool = False,
    ) -> ArtifactStore:
        import torch

        stream = torch.cuda.current_stream()
        context = ExecutionContext(
            case=self.case,
            mode=self.mode,
            iteration=iteration,
            stream=stream,
        )
        with torch.inference_mode(), torch.cuda.stream(stream):
            return execute_prevalidated(
                nodes,
                context,
                self.bound_artifacts(graph, fixture_id),
                stage_context=stage_context,
                validate_produced=validate_produced,
            )

    def build_oracles(self) -> dict[str, Any]:
        """Compute exact A/B references before candidate preparation."""

        import torch

        graph, nodes = self._prepare(self.reference, self.reference_options)
        self.reference_graph_fingerprint = graph_fingerprint(graph)
        metadata: dict[str, Any] = {}
        for iteration, fixture_id in enumerate(FIXTURE_IDS):
            outputs = self.execute(
                graph,
                nodes,
                fixture_id=fixture_id,
                iteration=iteration,
                validate_produced=True,
            )
            torch.cuda.synchronize()
            if "logits" not in outputs:
                raise RuntimeError(
                    "fusion-v1 exactness oracle must expose realized FP32 reference logits"
                )
            indices = outputs[graph.terminal_artifact].clone()
            output_check = validate_topk_indices(
                indices,
                self.case,
                name=f"fixture {fixture_id} exact reference",
            )
            tie_check, tie_rows = build_score_threshold_oracle(
                outputs["logits"],
                indices,
                self.case,
            )
            self.oracles[fixture_id] = OracleState(
                fixture_id=fixture_id,
                indices=indices,
                output_check=output_check,
                cutoff_tie_check=tie_check,
                cutoff_tie_rows=tie_rows,
            )
            metadata[fixture_id] = {
                "contract_check": output_check,
                "cutoff_tie_check": tie_check,
                "reference_indices_bytes": int(indices.numel())
                * int(indices.element_size()),
            }
            del outputs
        self.verify_input_integrity("before_candidate")
        return {
            "status": "passed",
            "graph_fingerprint": self.reference_graph_fingerprint,
            "fixtures": metadata,
        }

    def prepare_candidate(
        self,
        plugin: VariantPlugin,
        options: Mapping[str, Any],
    ) -> PreparedGraph:
        if not self.oracles:
            raise RuntimeError("build_oracles() must run before candidate preparation")
        graph, nodes = self._prepare(plugin, options)
        import torch

        torch.cuda.synchronize()
        self.verify_input_integrity("candidate_prepare")
        self.candidate_graph = graph
        self.candidate_nodes = nodes
        return graph

    def check_candidate(
        self,
        *,
        phase: str,
        order: Sequence[str] = ("A", "B"),
        start_iteration: int = 0,
    ) -> dict[str, Any]:
        if self.candidate_graph is None:
            raise RuntimeError("prepare_candidate() must run before candidate checks")
        import torch

        checks: dict[str, Any] = {}
        for offset, fixture_id in enumerate(order):
            outputs = self.execute(
                self.candidate_graph,
                self.candidate_nodes,
                fixture_id=fixture_id,
                iteration=start_iteration + offset,
                validate_produced=True,
            )
            torch.cuda.synchronize()
            result = self.validate_outputs(
                outputs,
                fixture_id=fixture_id,
                name=f"{phase} fixture {fixture_id} candidate",
            )
            checks[f"{offset}:{fixture_id}"] = {
                "fixture_id": fixture_id,
                **result,
            }
            del outputs
        integrity = self.verify_input_integrity(phase)
        return {
            "status": "passed",
            "phase": phase,
            "order": list(order),
            "checks": checks,
            "input_integrity": integrity,
        }

    def validate_outputs(
        self,
        outputs: Mapping[str, Any],
        *,
        fixture_id: str,
        name: str,
    ) -> dict[str, Any]:
        """Apply the terminal contract and the fixture-specific exact oracle."""

        if self.candidate_graph is None:
            raise RuntimeError("prepare_candidate() must run before output validation")
        try:
            oracle = self.oracles[fixture_id]
        except KeyError as error:
            raise ValueError(f"unknown fixture_id {fixture_id!r}") from error
        indices = outputs[self.candidate_graph.terminal_artifact]
        return {
            "contract_check": validate_topk_indices(indices, self.case, name=name),
            "exact_reference_check": compare_topk_indices(
                indices,
                oracle.indices,
                self.case,
                cutoff_tie_rows=oracle.cutoff_tie_rows,
            ),
        }

    def warmup_candidate(self, iterations: int) -> None:
        if iterations < 0:
            raise ValueError("warmup iterations must be non-negative")
        if self.candidate_graph is None:
            raise RuntimeError("prepare_candidate() must run before warmup")
        import torch

        for iteration in range(iterations):
            fixture_id = FIXTURE_IDS[iteration % len(FIXTURE_IDS)]
            outputs = self.execute(
                self.candidate_graph,
                self.candidate_nodes,
                fixture_id=fixture_id,
                iteration=iteration,
            )
            if self.candidate_graph.terminal_artifact not in outputs:
                raise RuntimeError("warmup graph did not produce its terminal artifact")
            del outputs
        torch.cuda.synchronize()

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "case": asdict(self.case),
            "fixture_order": list(FIXTURE_IDS),
            "fixtures": self.fixture_metadata(),
            "input_content_hash": self.input_content_hash,
            "reference_graph_fingerprint": self.reference_graph_fingerprint,
            "candidate_graph_fingerprint": (
                graph_fingerprint(self.candidate_graph)
                if self.candidate_graph is not None
                else None
            ),
            "integrity_checks": list(self.integrity_checks),
            "anti_precompute_gate": {
                "candidate_prepare_fixture": "A",
                "required_execution_fixtures": ["A", "B"],
                "same_shape_different_content": True,
            },
        }
