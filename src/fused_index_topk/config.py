"""Strict, GPU-agnostic configuration loading for FusedIndexTopK."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .api import PrefillCase

_HEX_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SAFE_NAME = re.compile(r"^[a-zA-Z][a-zA-Z0-9_.-]*$")


class ConfigError(ValueError):
    """Raised when a configuration violates the public experiment contract."""


@dataclass(frozen=True)
class TargetConfig:
    gpu_name: str
    compute_capability: tuple[int, int]
    multiprocessor_count: int
    total_memory_bytes: int
    require_single_visible_gpu: bool


@dataclass(frozen=True)
class SourceConfig:
    deep_gemm_commit: str
    torch_version: str


@dataclass(frozen=True)
class CaseShapeConfig:
    query_tokens: int
    context_tokens: int


@dataclass(frozen=True)
class WorkloadConfig:
    batch_size: int
    benchmark_cases: tuple[CaseShapeConfig, ...]
    correctness_cases: tuple[CaseShapeConfig, ...]
    top_k: int
    indexer_heads: int
    head_dim: int
    causal: bool
    q_dtype: str
    kv_dtype: str
    kv_scale_dtype: str
    weight_dtype: str
    range_dtype: str
    score_dtype: str
    output_dtype: str
    selection: str
    output_order: str
    padding_index: int
    tie_policy: str
    causal_range: str


@dataclass(frozen=True)
class TimingConfig:
    method: str
    warmup_iterations: int
    event_trials: int
    kineto_trials: int
    l2_flush_bytes: int
    cooldown_seconds: float
    campaign_repetitions: int
    campaign_orders: tuple[str, ...]


@dataclass(frozen=True)
class ProfilingConfig:
    target_cases: tuple[CaseShapeConfig, ...]
    nsys_cases: tuple[CaseShapeConfig, ...]
    ncu_cases: tuple[CaseShapeConfig, ...]
    captured_iterations: int
    ncu_metrics: tuple[str, ...]
    ncu_sections: tuple[str, ...]


@dataclass(frozen=True)
class VariantConfig:
    factory: str
    options: Mapping[str, Any]


@dataclass(frozen=True)
class LabConfig:
    schema_version: int
    name: str
    seed: int
    target: TargetConfig
    sources: SourceConfig
    workload: WorkloadConfig
    timing: TimingConfig
    profiling: ProfilingConfig
    exact_reference_variant: str
    baseline_variant: str
    variants: Mapping[str, VariantConfig]

    def case(self, context_tokens: int, *, query_tokens: int | None = None) -> PrefillCase:
        """Return a profiling/benchmark case for a configured context length.

        Profiling targets are a subset of the benchmark matrix, so this helper
        is used by the profiler wrappers while ``make_case`` remains strict
        about the requested lifecycle purpose.
        """

        profile_matches = [
            shape
            for shape in self.profiling.target_cases
            if shape.context_tokens == context_tokens
            and (query_tokens is None or shape.query_tokens == query_tokens)
        ]
        purpose = "profile" if profile_matches else "benchmark"
        return self.make_case(
            context_tokens,
            query_tokens=query_tokens,
            purpose=purpose,
        )

    def variant_factory(self, plugin_id: str) -> str:
        try:
            return self.variants[plugin_id].factory
        except KeyError as error:
            raise ConfigError(
                f"unknown configured variant {plugin_id!r}; available={sorted(self.variants)}"
            ) from error

    def variant_options(self, plugin_id: str) -> dict[str, Any]:
        try:
            return dict(self.variants[plugin_id].options)
        except KeyError as error:
            raise ConfigError(
                f"unknown configured variant {plugin_id!r}; available={sorted(self.variants)}"
            ) from error

    def make_case(
        self,
        context_tokens: int,
        *,
        query_tokens: int | None = None,
        purpose: str = "benchmark",
    ) -> PrefillCase:
        """Create one canonical Prefill case from the frozen workload contract."""

        if purpose not in {"benchmark", "correctness", "profile"}:
            raise ConfigError("purpose must be benchmark, correctness, or profile")
        allowed = {
            "benchmark": self.workload.benchmark_cases,
            "correctness": self.workload.correctness_cases,
            "profile": self.profiling.target_cases,
        }[purpose]
        matches = [
            shape
            for shape in allowed
            if shape.context_tokens == context_tokens
            and (query_tokens is None or shape.query_tokens == query_tokens)
        ]
        if not matches:
            raise ConfigError(
                f"case Q={query_tokens or '*'}, N={context_tokens} is not configured "
                f"for purpose={purpose!r}"
            )
        if len(matches) != 1:
            raise ConfigError(
                f"context_tokens={context_tokens} is ambiguous for purpose={purpose!r}; "
                "pass query_tokens explicitly"
            )
        shape = matches[0]
        workload = self.workload
        case_id = (
            f"prefill-{purpose}-q{shape.query_tokens}-"
            f"kv{context_tokens}-k{workload.top_k}"
        )
        return PrefillCase(
            case_id=case_id,
            query_tokens=shape.query_tokens,
            context_tokens=context_tokens,
            top_k=workload.top_k,
            seed=self.seed,
            batch_size=workload.batch_size,
            indexer_heads=workload.indexer_heads,
            head_dim=workload.head_dim,
            causal=workload.causal,
        )

    def benchmark_cases(self) -> tuple[PrefillCase, ...]:
        return tuple(
            self.make_case(
                shape.context_tokens,
                query_tokens=shape.query_tokens,
            )
            for shape in self.workload.benchmark_cases
        )

    def correctness_cases(self) -> tuple[PrefillCase, ...]:
        return tuple(
            self.make_case(
                shape.context_tokens,
                query_tokens=shape.query_tokens,
                purpose="correctness",
            )
            for shape in self.workload.correctness_cases
        )

    def profile_cases(self) -> tuple[PrefillCase, ...]:
        return tuple(
            self.make_case(
                shape.context_tokens,
                query_tokens=shape.query_tokens,
                purpose="profile",
            )
            for shape in self.profiling.target_cases
        )

    def profile_shapes(self, mode: str) -> tuple[CaseShapeConfig, ...]:
        if mode == "nsys":
            return self.profiling.nsys_cases
        if mode == "ncu":
            return self.profiling.ncu_cases
        raise ConfigError("profile mode must be nsys or ncu")

    def profile_lengths(self, mode: str) -> tuple[int, ...]:
        return tuple(shape.context_tokens for shape in self.profile_shapes(mode))

    def as_mapping(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot without embedding the source path."""

        return asdict(self)


def _mapping(value: object, path: str, keys: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{path} must be an object")
    unknown = set(value) - keys
    missing = keys - set(value)
    if unknown:
        raise ConfigError(f"{path} contains unknown field(s): {sorted(unknown)}")
    if missing:
        raise ConfigError(f"{path} is missing field(s): {sorted(missing)}")
    return value


def _integer(value: object, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{path} must be an integer >= {minimum}")
    return value


def _number(value: object, path: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path} must be a finite number >= {minimum}")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ConfigError(f"{path} must be a finite number >= {minimum}")
    return result


def _boolean(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{path} must be a boolean")
    return value


def _string(value: object, path: str, *, safe: bool = False) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{path} must be a non-empty string")
    if safe and not _SAFE_NAME.fullmatch(value):
        raise ConfigError(f"{path} contains unsafe characters: {value!r}")
    return value


def _ordered_unique_ints(value: object, path: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{path} must be a non-empty list")
    result = tuple(
        _integer(item, f"{path}[{index}]", minimum=1)
        for index, item in enumerate(value)
    )
    if tuple(sorted(set(result))) != result:
        raise ConfigError(f"{path} must contain strictly increasing, unique integers")
    return result


def _ordered_unique_case_shapes(value: object, path: str) -> tuple[CaseShapeConfig, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{path} must be a non-empty list")
    result: list[CaseShapeConfig] = []
    for index, raw in enumerate(value):
        item = _mapping(
            raw,
            f"{path}[{index}]",
            {"query_tokens", "context_tokens"},
        )
        shape = CaseShapeConfig(
            query_tokens=_integer(
                item["query_tokens"],
                f"{path}[{index}].query_tokens",
                minimum=1,
            ),
            context_tokens=_integer(
                item["context_tokens"],
                f"{path}[{index}].context_tokens",
                minimum=1,
            ),
        )
        if shape.context_tokens < shape.query_tokens:
            raise ConfigError(f"{path}[{index}] requires context_tokens >= query_tokens")
        result.append(shape)
    ordered = tuple(result)
    canonical_order = tuple(
        sorted(set(ordered), key=lambda item: (item.query_tokens, item.context_tokens))
    )
    if canonical_order != ordered:
        raise ConfigError(f"{path} must contain lexicographically ordered, unique Q/N cases")
    return ordered


def _unique_strings(value: object, path: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{path} must be a non-empty list")
    result = tuple(_string(item, f"{path}[{index}]") for index, item in enumerate(value))
    if len(set(result)) != len(result):
        raise ConfigError(f"{path} must not contain duplicates")
    return result


def _load_variants(value: object) -> dict[str, VariantConfig]:
    if not isinstance(value, Mapping) or not value:
        raise ConfigError("variants must be a non-empty object")
    result: dict[str, VariantConfig] = {}
    for plugin_id, raw in value.items():
        _string(plugin_id, "variants key", safe=True)
        item = _mapping(raw, f"variants.{plugin_id}", {"factory", "options"})
        factory = _string(item["factory"], f"variants.{plugin_id}.factory")
        if factory.count(":") != 1 or not all(factory.split(":")):
            raise ConfigError(
                f"variants.{plugin_id}.factory must use non-empty module:object syntax"
            )
        options = item["options"]
        if not isinstance(options, Mapping):
            raise ConfigError(f"variants.{plugin_id}.options must be an object")
        result[plugin_id] = VariantConfig(factory=factory, options=dict(options))
    return result


def parse_config(raw: object) -> LabConfig:
    """Validate an already decoded JSON object and return the typed config."""

    if not isinstance(raw, Mapping):
        raise ConfigError("config must be an object")
    normalized_raw = dict(raw)
    # Historical configs predate the explicit separation between the exact
    # oracle and the performance baseline.  They retain their former behavior.
    if (
        "exact_reference_variant" not in normalized_raw
        and "baseline_variant" in normalized_raw
    ):
        normalized_raw["exact_reference_variant"] = normalized_raw[
            "baseline_variant"
        ]
    root = _mapping(
        normalized_raw,
        "config",
        {
            "schema_version",
            "name",
            "seed",
            "target",
            "sources",
            "workload",
            "timing",
            "profiling",
            "exact_reference_variant",
            "baseline_variant",
            "variants",
        },
    )
    schema_version = _integer(root["schema_version"], "schema_version", minimum=1)
    if schema_version not in {1, 2, 3, 4}:
        raise ConfigError(
            f"unsupported schema_version={schema_version}; expected one of [1, 2, 3, 4]"
        )

    target_raw = _mapping(
        root["target"],
        "target",
        {
            "gpu_name",
            "compute_capability",
            "multiprocessor_count",
            "total_memory_bytes",
            "require_single_visible_gpu",
        },
    )
    capability_raw = target_raw["compute_capability"]
    if not isinstance(capability_raw, list) or len(capability_raw) != 2:
        raise ConfigError("target.compute_capability must be [major, minor]")
    capability = tuple(
        _integer(value, f"target.compute_capability[{index}]")
        for index, value in enumerate(capability_raw)
    )
    target = TargetConfig(
        gpu_name=_string(target_raw["gpu_name"], "target.gpu_name"),
        compute_capability=(capability[0], capability[1]),
        multiprocessor_count=_integer(
            target_raw["multiprocessor_count"],
            "target.multiprocessor_count",
            minimum=1,
        ),
        total_memory_bytes=_integer(
            target_raw["total_memory_bytes"],
            "target.total_memory_bytes",
            minimum=1,
        ),
        require_single_visible_gpu=_boolean(
            target_raw["require_single_visible_gpu"],
            "target.require_single_visible_gpu",
        ),
    )

    sources_raw = _mapping(
        root["sources"],
        "sources",
        {"deep_gemm_commit", "torch_version"},
    )
    commit = _string(sources_raw["deep_gemm_commit"], "sources.deep_gemm_commit")
    if not _HEX_COMMIT.fullmatch(commit):
        raise ConfigError("sources.deep_gemm_commit must be a lowercase 40-character SHA-1")
    sources = SourceConfig(
        deep_gemm_commit=commit,
        torch_version=_string(sources_raw["torch_version"], "sources.torch_version"),
    )

    workload_v1_fields = {
        "batch_size",
        "query_tokens",
        "context_lengths",
        "correctness_lengths",
        "top_k",
        "indexer_heads",
        "head_dim",
        "causal",
        "score_dtype",
        "output_dtype",
    }
    workload_v2_fields = workload_v1_fields | {
        "q_dtype",
        "kv_dtype",
        "kv_scale_dtype",
        "weight_dtype",
        "range_dtype",
        "selection",
        "output_order",
        "padding_index",
        "tie_policy",
        "causal_range",
    }
    workload_v3_fields = (workload_v2_fields - {
        "query_tokens",
        "context_lengths",
        "correctness_lengths",
    }) | {"benchmark_cases", "correctness_cases"}
    workload_raw = _mapping(
        root["workload"],
        "workload",
        workload_v3_fields
        if schema_version >= 3
        else workload_v2_fields
        if schema_version == 2
        else workload_v1_fields,
    )
    if schema_version >= 3:
        benchmark_cases = _ordered_unique_case_shapes(
            workload_raw["benchmark_cases"], "workload.benchmark_cases"
        )
        correctness_cases = _ordered_unique_case_shapes(
            workload_raw["correctness_cases"], "workload.correctness_cases"
        )
    else:
        context_lengths = _ordered_unique_ints(
            workload_raw["context_lengths"], "workload.context_lengths"
        )
        correctness_lengths = _ordered_unique_ints(
            workload_raw["correctness_lengths"], "workload.correctness_lengths"
        )
        query_tokens = _integer(
            workload_raw["query_tokens"], "workload.query_tokens", minimum=1
        )
        if min((*context_lengths, *correctness_lengths)) < query_tokens:
            raise ConfigError(
                "all context and correctness lengths must be >= workload.query_tokens"
            )
        benchmark_cases = tuple(
            CaseShapeConfig(query_tokens, context_tokens)
            for context_tokens in context_lengths
        )
        correctness_cases = tuple(
            CaseShapeConfig(query_tokens, context_tokens)
            for context_tokens in correctness_lengths
        )
    batch_size = _integer(workload_raw["batch_size"], "workload.batch_size", minimum=1)
    if batch_size != 1:
        raise ConfigError("the current Prefill output contract requires workload.batch_size=1")
    causal = _boolean(workload_raw["causal"], "workload.causal")
    if not causal:
        raise ConfigError("the current experiment contract requires causal=true")
    explicit_contract = {
        "q_dtype": "float8_e4m3fn",
        "kv_dtype": "float8_e4m3fn",
        "kv_scale_dtype": "float32",
        "weight_dtype": "float32",
        "range_dtype": "int32",
        "selection": "exact",
        "output_order": "unordered",
        "padding_index": -1,
        "tie_policy": "exact_score_threshold",
        "causal_range": "[0,N-Q+q+1)",
    }
    if schema_version >= 2:
        for field, expected in explicit_contract.items():
            value = workload_raw[field]
            if value != expected:
                raise ConfigError(
                    f"workload.{field} must be {expected!r} for the fusion-v1 contract"
                )
    else:
        # Schema v1 predates explicit tensor/selection fields.  Preserve its
        # historical JSON shape while resolving it to the same frozen contract.
        workload_raw = {**workload_raw, **explicit_contract}
    score_dtype = _string(workload_raw["score_dtype"], "workload.score_dtype")
    output_dtype = _string(workload_raw["output_dtype"], "workload.output_dtype")
    if score_dtype != "float32":
        raise ConfigError("the canonical Indexer score contract requires score_dtype='float32'")
    if output_dtype != "int32":
        raise ConfigError("the canonical TopK output contract requires output_dtype='int32'")
    top_k = _integer(workload_raw["top_k"], "workload.top_k", minimum=1)
    all_case_shapes = (*benchmark_cases, *correctness_cases)
    if top_k > min(shape.context_tokens for shape in all_case_shapes):
        raise ConfigError(
            "workload.top_k must not exceed the smallest configured context length"
        )
    workload = WorkloadConfig(
        batch_size=batch_size,
        benchmark_cases=benchmark_cases,
        correctness_cases=correctness_cases,
        top_k=top_k,
        indexer_heads=_integer(
            workload_raw["indexer_heads"], "workload.indexer_heads", minimum=1
        ),
        head_dim=_integer(workload_raw["head_dim"], "workload.head_dim", minimum=1),
        causal=causal,
        q_dtype=_string(workload_raw["q_dtype"], "workload.q_dtype"),
        kv_dtype=_string(workload_raw["kv_dtype"], "workload.kv_dtype"),
        kv_scale_dtype=_string(
            workload_raw["kv_scale_dtype"], "workload.kv_scale_dtype"
        ),
        weight_dtype=_string(workload_raw["weight_dtype"], "workload.weight_dtype"),
        range_dtype=_string(workload_raw["range_dtype"], "workload.range_dtype"),
        score_dtype=score_dtype,
        output_dtype=output_dtype,
        selection=_string(workload_raw["selection"], "workload.selection"),
        output_order=_string(workload_raw["output_order"], "workload.output_order"),
        padding_index=int(workload_raw["padding_index"]),
        tie_policy=_string(workload_raw["tie_policy"], "workload.tie_policy"),
        causal_range=_string(workload_raw["causal_range"], "workload.causal_range"),
    )

    legacy_timing_fields = {
        "warmup_iterations",
        "clean_trials",
        "attribution_trials",
        "cache_scrub_bytes",
        "cooldown_seconds",
    }
    campaign_timing_fields = legacy_timing_fields | {
        "campaign_repetitions",
        "campaign_orders",
    }
    kineto_timing_fields = {
        "method",
        "warmup_iterations",
        "event_trials",
        "kineto_trials",
        "l2_flush_bytes",
        "cooldown_seconds",
        "campaign_repetitions",
        "campaign_orders",
    }
    raw_timing = root["timing"]
    if not isinstance(raw_timing, Mapping):
        raise ConfigError("timing must be an object")
    if schema_version >= 4:
        timing_raw = _mapping(raw_timing, "timing", kineto_timing_fields)
        method = _string(timing_raw["method"], "timing.method")
        if method != "deepgemm_kineto_cupti_v1":
            raise ConfigError(
                "timing.method must be 'deepgemm_kineto_cupti_v1' for schema v4"
            )
        event_trials = _integer(
            timing_raw["event_trials"], "timing.event_trials", minimum=1
        )
        kineto_trials = _integer(
            timing_raw["kineto_trials"], "timing.kineto_trials", minimum=1
        )
        l2_flush_bytes = _integer(
            timing_raw["l2_flush_bytes"], "timing.l2_flush_bytes", minimum=4
        )
        if l2_flush_bytes % 4:
            raise ConfigError("timing.l2_flush_bytes must be divisible by four")
        campaign_repetitions = _integer(
            timing_raw["campaign_repetitions"],
            "timing.campaign_repetitions",
            minimum=3,
        )
    elif set(raw_timing) == legacy_timing_fields:
        timing_raw = _mapping(raw_timing, "timing", legacy_timing_fields)
        method = "legacy_cuda_event_v1"
        event_trials = _integer(
            timing_raw["clean_trials"], "timing.clean_trials", minimum=1
        )
        kineto_trials = 0
        l2_flush_bytes = _integer(
            timing_raw["cache_scrub_bytes"], "timing.cache_scrub_bytes"
        )
        campaign_repetitions = 1
        campaign_orders = ("ABBA",)
    else:
        timing_raw = _mapping(raw_timing, "timing", campaign_timing_fields)
        method = "legacy_cuda_event_v1"
        event_trials = _integer(
            timing_raw["clean_trials"], "timing.clean_trials", minimum=1
        )
        kineto_trials = 0
        l2_flush_bytes = _integer(
            timing_raw["cache_scrub_bytes"], "timing.cache_scrub_bytes"
        )
        campaign_repetitions = _integer(
            timing_raw["campaign_repetitions"],
            "timing.campaign_repetitions",
            minimum=3,
        )
    if schema_version >= 4 or set(raw_timing) != legacy_timing_fields:
        raw_orders = timing_raw["campaign_orders"]
        if not isinstance(raw_orders, list) or not raw_orders:
            raise ConfigError("timing.campaign_orders must be a non-empty list")
        campaign_orders = tuple(
            _string(order, f"timing.campaign_orders[{index}]")
            for index, order in enumerate(raw_orders)
        )
        if len(campaign_orders) != campaign_repetitions:
            raise ConfigError(
                "timing.campaign_orders length must equal campaign_repetitions"
            )
        if any(order not in {"ABBA", "BAAB"} for order in campaign_orders):
            raise ConfigError("timing.campaign_orders entries must be ABBA or BAAB")
    timing = TimingConfig(
        method=method,
        warmup_iterations=_integer(
            timing_raw["warmup_iterations"], "timing.warmup_iterations"
        ),
        event_trials=event_trials,
        kineto_trials=kineto_trials,
        l2_flush_bytes=l2_flush_bytes,
        cooldown_seconds=_number(
            timing_raw["cooldown_seconds"], "timing.cooldown_seconds"
        ),
        campaign_repetitions=campaign_repetitions,
        campaign_orders=campaign_orders,
    )

    legacy_profiling_fields = {
        "target_lengths",
        "captured_iterations",
        "ncu_metrics",
        "ncu_sections",
    }
    split_profiling_fields = {
        "nsys_target_lengths",
        "ncu_target_lengths",
        "captured_iterations",
        "ncu_metrics",
        "ncu_sections",
    }
    case_profiling_fields = {
        "nsys_cases",
        "ncu_cases",
        "captured_iterations",
        "ncu_metrics",
        "ncu_sections",
    }
    raw_profiling = root["profiling"]
    if not isinstance(raw_profiling, Mapping):
        raise ConfigError("profiling must be an object")
    if schema_version >= 3:
        profiling_raw = _mapping(raw_profiling, "profiling", case_profiling_fields)
        nsys_cases = _ordered_unique_case_shapes(
            profiling_raw["nsys_cases"], "profiling.nsys_cases"
        )
        ncu_cases = _ordered_unique_case_shapes(
            profiling_raw["ncu_cases"], "profiling.ncu_cases"
        )
        target_cases = tuple(
            sorted(
                set(nsys_cases) | set(ncu_cases),
                key=lambda item: (item.query_tokens, item.context_tokens),
            )
        )
    elif set(raw_profiling) == legacy_profiling_fields:
        profiling_raw = _mapping(raw_profiling, "profiling", legacy_profiling_fields)
        target_lengths = _ordered_unique_ints(
            profiling_raw["target_lengths"], "profiling.target_lengths"
        )
        nsys_target_lengths = target_lengths
        ncu_target_lengths = target_lengths
    else:
        profiling_raw = _mapping(raw_profiling, "profiling", split_profiling_fields)
        nsys_target_lengths = _ordered_unique_ints(
            profiling_raw["nsys_target_lengths"], "profiling.nsys_target_lengths"
        )
        ncu_target_lengths = _ordered_unique_ints(
            profiling_raw["ncu_target_lengths"], "profiling.ncu_target_lengths"
        )
        target_lengths = tuple(sorted(set(nsys_target_lengths) | set(ncu_target_lengths)))
    if schema_version < 3:
        legacy_query_tokens = benchmark_cases[0].query_tokens
        nsys_cases = tuple(
            CaseShapeConfig(legacy_query_tokens, length) for length in nsys_target_lengths
        )
        ncu_cases = tuple(
            CaseShapeConfig(legacy_query_tokens, length) for length in ncu_target_lengths
        )
        target_cases = tuple(
            sorted(
                set(nsys_cases) | set(ncu_cases),
                key=lambda item: (item.query_tokens, item.context_tokens),
            )
        )
    if not set(target_cases).issubset(benchmark_cases):
        raise ConfigError("profiling cases must be a subset of workload.benchmark_cases")
    profiling = ProfilingConfig(
        target_cases=target_cases,
        nsys_cases=nsys_cases,
        ncu_cases=ncu_cases,
        captured_iterations=_integer(
            profiling_raw["captured_iterations"],
            "profiling.captured_iterations",
            minimum=1,
        ),
        ncu_metrics=_unique_strings(profiling_raw["ncu_metrics"], "profiling.ncu_metrics"),
        ncu_sections=_unique_strings(profiling_raw["ncu_sections"], "profiling.ncu_sections"),
    )

    variants = _load_variants(root["variants"])
    exact_reference_variant = _string(
        root["exact_reference_variant"],
        "exact_reference_variant",
        safe=True,
    )
    if exact_reference_variant not in variants:
        raise ConfigError("exact_reference_variant must name an entry in variants")
    baseline_variant = _string(root["baseline_variant"], "baseline_variant", safe=True)
    if baseline_variant not in variants:
        raise ConfigError("baseline_variant must name an entry in variants")
    return LabConfig(
        schema_version=schema_version,
        name=_string(root["name"], "name", safe=True),
        seed=_integer(root["seed"], "seed"),
        target=target,
        sources=sources,
        workload=workload,
        timing=timing,
        profiling=profiling,
        exact_reference_variant=exact_reference_variant,
        baseline_variant=baseline_variant,
        variants=variants,
    )


def load_config(path: str | Path) -> LabConfig:
    """Load and strictly validate one JSON configuration file."""

    candidate = Path(path)
    try:
        with candidate.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except json.JSONDecodeError as error:
        raise ConfigError(f"invalid JSON in {candidate}: {error}") from error
    return parse_config(raw)


def build_prefill_cases(
    config: LabConfig,
    *,
    purpose: str = "benchmark",
) -> tuple[PrefillCase, ...]:
    """Functional wrapper used by simple runners and external tooling."""

    dispatch = {
        "benchmark": config.benchmark_cases,
        "correctness": config.correctness_cases,
        "profile": config.profile_cases,
    }
    try:
        factory = dispatch[purpose]
    except KeyError as error:
        raise ConfigError("purpose must be benchmark, correctness, or profile") from error
    return factory()
