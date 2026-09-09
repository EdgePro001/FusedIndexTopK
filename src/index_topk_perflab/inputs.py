"""Deterministic standard inputs for contiguous causal Prefill experiments.

The module deliberately has no module-level torch import.  Listing variants,
reading result files, and running the CPU-only test suite must not initialize
PyTorch or CUDA.  ``make_prefill_inputs`` mirrors the Indexer portion of the
released DSA experiment input generator:

* BF16 random source tensors are scaled by ``1 / sqrt(128)``;
* KV is quantized row-wise to FP8 E4M3 with one FP32 scale per token;
* Q is converted directly to FP8 E4M3;
* causal row ends cover the final query chunk in the context.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict
from typing import Any, Mapping

from .api import PrefillCase, PrefillInputs
from .artifacts import canonical_hash

_INPUT_NAMES = ("q", "kv", "kv_scales", "weights", "k_start", "k_end")
_CONTENT_HASH_CHUNK_BYTES = 16 * 1024 * 1024


def validate_prefill_case(case: PrefillCase) -> None:
    """Validate the shape contract shared by all Prefill variants."""

    if case.batch_size != 1:
        raise ValueError("the v1 Prefill input contract supports batch_size=1")
    if not case.causal:
        raise ValueError("the standard Prefill workload is causal")
    if case.query_tokens <= 0:
        raise ValueError("query_tokens must be positive")
    if case.context_tokens < case.query_tokens:
        raise ValueError("context_tokens must be at least query_tokens")
    if case.top_k <= 0:
        raise ValueError("top_k must be positive")
    if case.indexer_heads <= 0 or case.head_dim <= 0:
        raise ValueError("indexer_heads and head_dim must be positive")


def seed_torch(seed: int) -> None:
    """Legacy helper retained for callers outside the formal v2 input path."""

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def input_stream_seeds(seed: int, fixture_id: str) -> dict[str, int]:
    """Derive stable, independent RNG seeds for one named A/B fixture."""

    if fixture_id not in {"A", "B"}:
        raise ValueError("fixture_id must be 'A' or 'B'")
    result: dict[str, int] = {}
    for stream in ("kv", "q", "weights"):
        material = f"index-topk-prefill-v2:{seed}:{fixture_id}:{stream}".encode()
        # torch.Generator.manual_seed accepts signed-64-bit-safe positive values.
        result[stream] = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & (
            (1 << 63) - 1
        )
    return result


def _torch_generator(torch: Any, device: Any, seed: int) -> Any:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def quantize_contiguous_index_kv(source: Any) -> tuple[Any, Any]:
    """Quantize ``[N, D]`` BF16/FP32 KV into FP8 values and FP32 scales.

    This is the same row-wise quantization rule used by the previous
    ``DSA-KernelExp`` Prefill harness.  The lazy import keeps this module usable
    by CPU-only result tooling.
    """

    import torch

    if source.ndim != 2:
        raise ValueError("contiguous index KV must be rank two")
    scales = source.abs().float().amax(dim=1).clamp_min(1e-4) / 448.0
    values = (source.float() / scales[:, None]).to(torch.float8_e4m3fn)
    return values.contiguous(), scales.contiguous()


def make_prefill_inputs(
    case: PrefillCase,
    device: str | Any = "cuda",
    *,
    generation_context_tokens: int | None = None,
    fixture_id: str = "A",
) -> PrefillInputs:
    """Create one deterministic standard input set for ``case``.

    The allocation and quantization happen before plugin preparation and must
    therefore remain outside every timed or profiled iteration.
    """

    validate_prefill_case(case)

    import torch

    target = torch.device(device)
    generation_tokens = case.context_tokens
    if (
        generation_context_tokens is not None
        and int(generation_context_tokens) != case.context_tokens
    ):
        raise ValueError(
            "v2 inputs require compact per-case storage; generation_context_tokens "
            "must equal case.context_tokens"
        )
    seeds = input_stream_seeds(case.seed, fixture_id)

    # Each logical input owns an independent generator.  Adding a larger N or
    # changing one tensor recipe cannot shift the Q/weight streams of old cases.
    kv_source = torch.randn(
        (generation_tokens, case.head_dim),
        device=target,
        dtype=torch.bfloat16,
        generator=_torch_generator(torch, target, seeds["kv"]),
    ) / math.sqrt(case.head_dim)
    kv, kv_scales = quantize_contiguous_index_kv(kv_source)
    del kv_source

    q = (
        torch.randn(
            (case.query_tokens, case.indexer_heads, case.head_dim),
            device=target,
            dtype=torch.bfloat16,
            generator=_torch_generator(torch, target, seeds["q"]),
        )
        / math.sqrt(case.head_dim)
    ).to(torch.float8_e4m3fn).contiguous()
    weights = (
        torch.randn(
            (case.query_tokens, case.indexer_heads),
            device=target,
            dtype=torch.float32,
            generator=_torch_generator(torch, target, seeds["weights"]),
        )
        / math.sqrt(case.indexer_heads)
    ).contiguous()

    k_start = torch.zeros(case.query_tokens, device=target, dtype=torch.int32)
    k_end = torch.arange(
        case.query_start + 1,
        case.context_tokens + 1,
        device=target,
        dtype=torch.int32,
    )
    if k_end.numel() != case.query_tokens:
        raise RuntimeError("causal row-end construction violated the Prefill contract")

    return PrefillInputs(
        case=case,
        q=q,
        kv=kv,
        kv_scales=kv_scales,
        weights=weights,
        k_start=k_start,
        k_end=k_end,
        generation_context_tokens=generation_tokens,
        fixture_id=fixture_id,
        stream_seeds=seeds,
    )


def validate_prefill_inputs(inputs: PrefillInputs, workload: Any) -> dict[str, Any]:
    """Validate the realized CUDA tensors against the explicit v1 contract.

    This gate runs after generation and outside every timed/profiled region. It
    prevents a config from merely *claiming* a dtype or range convention while
    the actual tensors silently use something else.
    """

    import torch

    case = inputs.case
    dtype_by_name = {
        "float8_e4m3fn": torch.float8_e4m3fn,
        "float32": torch.float32,
        "int32": torch.int32,
    }
    expected = {
        "q": ((case.query_tokens, case.indexer_heads, case.head_dim), workload.q_dtype),
        "kv": ((case.context_tokens, case.head_dim), workload.kv_dtype),
        "kv_scales": ((case.context_tokens,), workload.kv_scale_dtype),
        "weights": ((case.query_tokens, case.indexer_heads), workload.weight_dtype),
        "k_start": ((case.query_tokens,), workload.range_dtype),
        "k_end": ((case.query_tokens,), workload.range_dtype),
    }
    devices = set()
    summary: dict[str, Any] = {}
    for name, (shape, dtype_name) in expected.items():
        tensor = getattr(inputs, name)
        if tuple(tensor.shape) != shape:
            raise ValueError(f"input {name} shape must be {shape}, received {tuple(tensor.shape)}")
        try:
            expected_dtype = dtype_by_name[dtype_name]
        except KeyError as error:
            raise ValueError(f"unsupported contract dtype {dtype_name!r} for {name}") from error
        if tensor.dtype != expected_dtype:
            raise ValueError(
                f"input {name} dtype must be {dtype_name}, received {tensor.dtype}"
            )
        if not tensor.is_contiguous():
            raise ValueError(f"input {name} must be contiguous")
        devices.add(tensor.device)
        description = _tensor_description(tensor)
        if description["storage_offset_elements"] != 0:
            raise ValueError(f"input {name} must have zero storage offset")
        if description["storage_bytes"] != description["logical_bytes"]:
            raise ValueError(
                f"input {name} must use compact storage; logical_bytes="
                f"{description['logical_bytes']} storage_bytes={description['storage_bytes']}"
            )
        summary[name] = {
            "shape": list(shape),
            "dtype": dtype_name,
            "contiguous": True,
            "compact_storage": True,
        }

    if len(devices) != 1:
        raise ValueError(f"all inputs must share one device, received {sorted(map(str, devices))}")
    device = next(iter(devices))
    if device.type != "cuda":
        raise ValueError(f"formal Prefill inputs must be CUDA tensors, received {device}")
    if not torch.equal(inputs.k_start, torch.zeros_like(inputs.k_start)):
        raise ValueError("fusion-v1 supports only contiguous Prefill with k_start == 0")
    expected_end = torch.arange(
        case.query_start + 1,
        case.context_tokens + 1,
        device=device,
        dtype=torch.int32,
    )
    if not torch.equal(inputs.k_end, expected_end):
        raise ValueError("k_end must equal N-Q+q+1 with end-exclusive semantics")
    if not bool(torch.isfinite(inputs.kv_scales).all().item()):
        raise ValueError("kv_scales must be finite")
    if not bool((inputs.kv_scales > 0).all().item()):
        raise ValueError("kv_scales must be strictly positive")
    if not bool(torch.isfinite(inputs.weights).all().item()):
        raise ValueError("weights must be finite")
    if (
        inputs.generation_context_tokens is None
        or inputs.generation_context_tokens != case.context_tokens
    ):
        raise ValueError("generation_context_tokens must equal the realized KV input length")

    return {
        "status": "passed",
        "device": str(device),
        "tensors": summary,
        "causal_range": "[0,N-Q+q+1)",
        "k_start_zero": True,
        "k_end_includes_current_position": True,
    }


def _tensor_description(value: Any) -> dict[str, Any]:
    shape = [int(item) for item in value.shape]
    element_size = int(value.element_size())
    storage_bytes = None
    if hasattr(value, "untyped_storage"):
        try:
            storage_bytes = int(value.untyped_storage().nbytes())
        except (AttributeError, RuntimeError, TypeError):
            storage_bytes = None
    return {
        "shape": shape,
        "stride": [int(item) for item in value.stride()],
        "dtype": str(value.dtype),
        "device": str(value.device),
        "contiguous": bool(value.is_contiguous()),
        "element_size_bytes": element_size,
        "logical_bytes": int(value.numel()) * element_size,
        "storage_bytes": storage_bytes,
        "storage_offset_elements": int(value.storage_offset()),
    }


def input_manifest(inputs: PrefillInputs) -> dict[str, Any]:
    """Return a JSON-serializable manifest without copying tensor payloads."""

    recipe = {
        "version": inputs.recipe_version,
        "seed": inputs.case.seed,
        "fixture_id": inputs.fixture_id,
        "rng_streams": dict(inputs.stream_seeds),
        "generation_context_tokens": inputs.generation_context_tokens,
        "generation_order": "independent_streams",
        "compact_per_case_storage": True,
        "kv_quantization": "row_amax_clamp_1e-4_div_448_fp8_e4m3fn",
    }
    recipe.update(dict(inputs.recipe_metadata))
    return {
        "case": asdict(inputs.case),
        "recipe": recipe,
        "tensors": {
            name: _tensor_description(getattr(inputs, name))
            for name in _INPUT_NAMES
        },
    }


def tensor_content_digest(value: Any, *, chunk_bytes: int = _CONTENT_HASH_CHUNK_BYTES) -> str:
    """Hash the exact logical tensor bytes without retaining a full host copy."""

    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    import torch

    if not isinstance(value, torch.Tensor):
        raise TypeError("content hashing requires a torch.Tensor")
    if not value.is_contiguous():
        raise ValueError("content hashing requires a contiguous tensor")
    raw = value.detach().view(torch.uint8).reshape(-1)
    digest = hashlib.sha256()
    for offset in range(0, raw.numel(), chunk_bytes):
        host = raw[offset : offset + chunk_bytes].cpu()
        digest.update(host.numpy().tobytes())
    return digest.hexdigest()


def input_content_manifest(inputs: PrefillInputs) -> dict[str, Any]:
    """Return exact per-tensor content digests plus logical tensor metadata."""

    tensors: dict[str, Any] = {}
    for name in _INPUT_NAMES:
        value = getattr(inputs, name)
        description = _tensor_description(value)
        tensors[name] = {
            "shape": description["shape"],
            "dtype": description["dtype"],
            "logical_bytes": description["logical_bytes"],
            "sha256": tensor_content_digest(value),
        }
    return {
        "schema_version": 1,
        "algorithm": "sha256-logical-tensor-bytes",
        "tensors": tensors,
    }


def input_content_fingerprint(inputs: PrefillInputs) -> dict[str, Any]:
    """Materialize the auditable content manifest and its canonical digest."""

    manifest = input_content_manifest(inputs)
    return {"sha256": canonical_hash(manifest), "manifest": manifest}


def cases_from_config(
    config: Mapping[str, Any],
    *,
    correctness: bool = False,
) -> tuple[PrefillCase, ...]:
    """Build cases from the checked-in JSON-shaped configuration mapping.

    This small adapter intentionally does not own config file loading; a future
    ``config.load_config(path)`` can pass its mapping here or expose equivalent
    ``config.case(length)`` objects.
    """

    workload = config["workload"]
    cases_key = "correctness_cases" if correctness else "benchmark_cases"
    if cases_key in workload:
        shapes = tuple(
            (int(item["query_tokens"]), int(item["context_tokens"]))
            for item in workload[cases_key]
        )
    else:
        lengths_key = "correctness_lengths" if correctness else "context_lengths"
        shapes = tuple(
            (int(workload["query_tokens"]), int(length))
            for length in workload[lengths_key]
        )
    base_seed = int(config.get("seed", 0))
    return tuple(
        PrefillCase(
            case_id=f"prefill_q{query_tokens}_n{context_tokens}",
            query_tokens=query_tokens,
            context_tokens=context_tokens,
            top_k=int(workload["top_k"]),
            seed=base_seed,
            batch_size=int(workload.get("batch_size", 1)),
            indexer_heads=int(workload.get("indexer_heads", 64)),
            head_dim=int(workload.get("head_dim", 128)),
            causal=bool(workload.get("causal", True)),
        )
        for query_tokens, context_tokens in shapes
    )
