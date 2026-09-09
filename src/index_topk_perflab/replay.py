"""Frozen real-workload replay inputs for the standard Prefill contract.

Replay changes only where inputs come from.  Once loaded, candidates see the
same contiguous tensors and run through the same lifecycle, correctness gates,
DeepGEMM Indexer, TopK implementations, and timing harness as random fixtures.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from .api import PrefillCase, PrefillInputs
from .artifacts import load_json

REPLAY_SCHEMA_VERSION = 1
REPLAY_TENSOR_NAMES = ("q", "kv", "kv_scales", "weights", "k_start", "k_end")


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(value: object) -> Path:
    path = Path(str(value))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"replay item path must be a safe relative path: {path}")
    return path


def validate_replay_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the portable portion of one frozen replay manifest."""

    if int(value.get("schema_version", -1)) != REPLAY_SCHEMA_VERSION:
        raise ValueError("unsupported replay manifest schema_version")
    if not value.get("dataset_id"):
        raise ValueError("replay manifest requires dataset_id")
    if value.get("frozen") is not True:
        raise ValueError("formal replay manifests must be frozen")
    items = value.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("replay manifest requires a non-empty items list")

    keys: set[tuple[str, str, str]] = set()
    item_ids: set[str] = set()
    splits: set[str] = set()
    for item in items:
        if not isinstance(item, Mapping):
            raise ValueError("each replay item must be an object")
        item_id = str(item.get("item_id", ""))
        split = str(item.get("split", ""))
        case_id = str(item.get("case_id", ""))
        fixture_id = str(item.get("fixture_id", ""))
        if not item_id or item_id in item_ids:
            raise ValueError(f"missing or duplicate replay item_id: {item_id!r}")
        if split not in {"tuning", "test_normal", "test_hard"}:
            raise ValueError(f"invalid replay split: {split!r}")
        if fixture_id not in {"A", "B"}:
            raise ValueError(f"invalid fixture_id: {fixture_id!r}")
        if not case_id:
            raise ValueError("replay item requires case_id")
        key = (split, case_id, fixture_id)
        if key in keys:
            raise ValueError(f"duplicate replay fixture key: {key}")
        _safe_relative_path(item.get("path"))
        digest = str(item.get("sha256", ""))
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError(f"invalid sha256 for replay item {item_id!r}")
        case = item.get("case")
        if not isinstance(case, Mapping):
            raise ValueError(f"replay item {item_id!r} requires case metadata")
        for field in ("query_tokens", "context_tokens", "top_k", "indexer_heads", "head_dim"):
            if int(case.get(field, 0)) <= 0:
                raise ValueError(f"invalid {field} for replay item {item_id!r}")
        item_ids.add(item_id)
        keys.add(key)
        splits.add(split)

    missing_splits = {"tuning", "test_normal", "test_hard"} - splits
    if missing_splits:
        raise ValueError(f"replay manifest is missing splits: {sorted(missing_splits)}")
    return {
        "status": "passed",
        "schema_version": REPLAY_SCHEMA_VERSION,
        "dataset_id": str(value["dataset_id"]),
        "items": len(items),
        "splits": sorted(splits),
    }


def save_replay_tensors(
    path: str | os.PathLike[str],
    tensors: Mapping[str, Any],
    *,
    metadata: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Atomically save the six standard tensors and return file provenance."""

    from safetensors.torch import save_file

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    missing = set(REPLAY_TENSOR_NAMES) - set(tensors)
    extra = set(tensors) - set(REPLAY_TENSOR_NAMES)
    if missing or extra:
        raise ValueError(
            f"replay tensors mismatch: missing={sorted(missing)} extra={sorted(extra)}"
        )
    compact: dict[str, Any] = {}
    for name in REPLAY_TENSOR_NAMES:
        value = tensors[name]
        if value.device.type != "cpu":
            raise ValueError(f"replay tensor {name} must be on CPU before serialization")
        compact[name] = value.contiguous()

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        save_file(compact, temporary, metadata=dict(metadata or {}))
        os.chmod(temporary, 0o644)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
    }


class ReplayInputFactory:
    """Load one frozen A/B fixture pair for each case in a selected split."""

    def __init__(
        self,
        manifest_path: str | os.PathLike[str],
        *,
        split: str,
        verify_sha256: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.root = self.manifest_path.parent
        self.manifest = load_json(self.manifest_path)
        validate_replay_manifest(self.manifest)
        if split not in {"tuning", "test_normal", "test_hard"}:
            raise ValueError(f"invalid replay split: {split!r}")
        self.split = split
        self.verify_sha256 = verify_sha256
        self._verified_paths: set[Path] = set()
        self._items: dict[tuple[str, str], Mapping[str, Any]] = {}
        for item in self.manifest["items"]:
            if item["split"] == split:
                self._items[(str(item["case_id"]), str(item["fixture_id"]))] = item

    def _item(self, case: PrefillCase, fixture_id: str) -> Mapping[str, Any]:
        try:
            item = self._items[(case.case_id, fixture_id)]
        except KeyError as error:
            raise KeyError(
                f"replay split {self.split!r} has no {fixture_id} fixture for {case.case_id!r}"
            ) from error
        recorded = item["case"]
        for field in ("query_tokens", "context_tokens", "top_k", "indexer_heads", "head_dim"):
            if int(recorded[field]) != int(getattr(case, field)):
                raise ValueError(
                    f"replay case mismatch for {field}: manifest={recorded[field]} "
                    f"runtime={getattr(case, field)}"
                )
        return item

    def __call__(
        self, case: PrefillCase, device: str | Any, fixture_id: str
    ) -> PrefillInputs:
        import torch
        from safetensors.torch import load_file

        item = self._item(case, fixture_id)
        path = (self.root / _safe_relative_path(item["path"])).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"replay item escapes dataset root: {path}")
        if self.verify_sha256 and path not in self._verified_paths:
            actual = sha256_file(path)
            if actual != item["sha256"]:
                raise RuntimeError(
                    f"replay item checksum mismatch for {path}: "
                    f"expected={item['sha256']} actual={actual}"
                )
            self._verified_paths.add(path)
        payload = load_file(path, device="cpu")
        if set(payload) != set(REPLAY_TENSOR_NAMES):
            raise ValueError(f"replay file has unexpected tensors: {sorted(payload)}")
        target = torch.device(device)
        loaded = {
            name: payload[name].to(device=target, non_blocking=False).contiguous()
            for name in REPLAY_TENSOR_NAMES
        }
        recipe_metadata = {
            "dataset_id": self.manifest["dataset_id"],
            "dataset_manifest": str(self.manifest_path),
            "dataset_manifest_sha256": sha256_file(self.manifest_path),
            "replay_split": self.split,
            "replay_item_id": item["item_id"],
            "replay_file": item["path"],
            "replay_file_sha256": item["sha256"],
            "capture": dict(item.get("capture", {})),
            "score_statistics": dict(item.get("score_statistics", {})),
            "generation_order": "frozen_replay_load",
            "kv_quantization": item.get(
                "kv_quantization", "dsv32-act-quant-group128-fp8-e4m3fn"
            ),
        }
        return PrefillInputs(
            case=case,
            q=loaded["q"],
            kv=loaded["kv"],
            kv_scales=loaded["kv_scales"],
            weights=loaded["weights"],
            k_start=loaded["k_start"],
            k_end=loaded["k_end"],
            generation_context_tokens=case.context_tokens,
            fixture_id=fixture_id,
            stream_seeds={},
            recipe_version=str(self.manifest.get("recipe_version", "dsv32-real-replay-v1")),
            recipe_metadata=recipe_metadata,
        )

    def cases(self, *, seed: int = 0) -> tuple[PrefillCase, ...]:
        """Return the cases with complete A/B fixtures in deterministic order."""

        by_id: dict[str, Mapping[str, Any]] = {}
        fixtures: dict[str, set[str]] = {}
        for (case_id, fixture_id), item in self._items.items():
            by_id[case_id] = item["case"]
            fixtures.setdefault(case_id, set()).add(fixture_id)
        missing = [case_id for case_id, ids in fixtures.items() if ids != {"A", "B"}]
        if missing:
            raise ValueError(f"replay cases without complete A/B fixtures: {sorted(missing)}")
        result = []
        for case_id in sorted(
            by_id, key=lambda value: (int(by_id[value]["context_tokens"]), value)
        ):
            value = by_id[case_id]
            result.append(
                PrefillCase(
                    case_id=case_id,
                    query_tokens=int(value["query_tokens"]),
                    context_tokens=int(value["context_tokens"]),
                    top_k=int(value["top_k"]),
                    seed=seed,
                    batch_size=int(value.get("batch_size", 1)),
                    indexer_heads=int(value["indexer_heads"]),
                    head_dim=int(value["head_dim"]),
                    causal=bool(value.get("causal", True)),
                )
            )
        return tuple(result)


def replay_case_metadata(case: PrefillCase) -> dict[str, Any]:
    """Canonical shape metadata stored beside every item."""

    return asdict(case)
