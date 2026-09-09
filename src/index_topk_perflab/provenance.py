"""Stable framework and variant identities used to bind experiment artifacts."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from .api import VariantPlugin
from .artifacts import canonical_hash


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def path_fingerprint(path: str | Path) -> dict[str, Any]:
    """Fingerprint one plugin source file, binary, or source tree.

    Returned names are relative and host/container stable. CUDA plugins should
    expose these records from ``fingerprint_metadata()`` so old correctness
    artifacts cannot be reused after an implementation change.
    """

    source = Path(path).resolve()
    if source.is_file():
        return {
            "kind": "file",
            "name": source.name,
            "bytes": source.stat().st_size,
            "sha256": _file_sha256(source),
        }
    if source.is_dir():
        files = {
            item.relative_to(source).as_posix(): {
                "bytes": item.stat().st_size,
                "sha256": _file_sha256(item),
            }
            for item in sorted(source.rglob("*"))
            if item.is_file() and "__pycache__" not in item.parts
        }
        if not files:
            raise ValueError(f"plugin source tree is empty: {source}")
        return {
            "kind": "directory",
            "name": source.name,
            "files": files,
            "tree_sha256": canonical_hash(files),
        }
    raise ValueError(f"plugin fingerprint path does not exist: {source}")


def framework_source_manifest(package_root: str | Path | None = None) -> dict[str, str]:
    """Fingerprint only top-level formal core modules.

    Operators live below ``variants/`` or ``experimental/`` and carry their own
    source fingerprint.  Keeping this traversal non-recursive prevents an old,
    unrelated experiment from changing the framework or baseline identity.
    """

    package_root = (
        Path(__file__).resolve().parent
        if package_root is None
        else Path(package_root).resolve()
    )
    return {
        str(path.relative_to(package_root)): _file_sha256(path)
        for path in sorted(package_root.glob("*.py"))
        if path.is_file()
    }


def framework_fingerprint() -> str:
    return canonical_hash(framework_source_manifest())


def _factory_source(factory: str) -> dict[str, Any]:
    module_name = factory.split(":", 1)[0]
    module = sys.modules.get(module_name)
    source_name = getattr(module, "__file__", None)
    if source_name is None:
        return {"module": module_name, "sha256": None}
    source = Path(source_name).resolve()
    return {
        "module": module_name,
        "filename": source.name,
        "sha256": _file_sha256(source) if source.is_file() else None,
    }


def variant_identity(
    config: Any,
    configured_id: str,
    plugin: VariantPlugin,
    *,
    options: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a host/container-stable identity without absolute filesystem paths."""

    factory = config.variant_factory(configured_id)
    extra: Any = None
    provider = getattr(plugin, "fingerprint_metadata", None)
    if not callable(provider):
        raise ValueError(
            f"variant {configured_id!r} must implement fingerprint_metadata()"
        )
    extra = provider()
    # Fail early if a plugin returns machine-specific or non-JSON metadata.
    json.dumps(extra, sort_keys=True, allow_nan=False)
    operator_payload = {
        "schema_version": 1,
        "configured_id": configured_id,
        "descriptor": asdict(plugin.descriptor),
        "factory": factory,
        "factory_source": _factory_source(factory),
        "options": dict(options),
        "frozen_sources": asdict(config.sources),
        "plugin_fingerprint_metadata": extra,
    }
    operator_fingerprint = canonical_hash(operator_payload)
    payload = {
        "schema_version": 2,
        "configured_id": configured_id,
        "core_fingerprint": framework_fingerprint(),
        # Kept as a compatibility alias for existing report readers.
        "framework_fingerprint": framework_fingerprint(),
        "operator_fingerprint": operator_fingerprint,
        "operator": operator_payload,
    }
    return {"fingerprint": canonical_hash(payload), "payload": payload}


def experiment_identity(
    *,
    protocol_hash: str,
    plan_hash: str,
    operator_hash: str,
    runtime_hash: str,
    input_content_hash: str,
) -> dict[str, Any]:
    """Compose the five independently auditable experiment identity components."""

    components = {
        "ProtocolHash": protocol_hash,
        "PlanHash": plan_hash,
        "OperatorHash": operator_hash,
        "RuntimeHash": runtime_hash,
        "InputContentHash": input_content_hash,
    }
    return {"sha256": canonical_hash(components), "components": components}
