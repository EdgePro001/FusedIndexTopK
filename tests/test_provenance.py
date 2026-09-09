from __future__ import annotations

import sys
from pathlib import Path

from index_topk_perflab.artifacts import canonical_hash
from index_topk_perflab.config import load_config
from index_topk_perflab.provenance import (
    experiment_identity,
    framework_fingerprint,
    framework_source_manifest,
    path_fingerprint,
    variant_identity,
)
from index_topk_perflab.registry import load_variant

ROOT = Path(__file__).resolve().parents[1]


def test_variant_identity_is_stable_and_cuda_lazy() -> None:
    sys.modules.pop("deep_gemm", None)
    config = load_config(ROOT / "configs" / "r13a_h20_release.json")
    # Exercise core provenance without requiring the H20 FlashInfer
    # source checkout selected by the formal configuration.
    variant_id = "deepgemm_torch_unfused"
    options = config.variant_options(variant_id)
    plugin = load_variant(config.variant_factory(variant_id), options=options)
    first = variant_identity(config, variant_id, plugin, options=options)
    second = variant_identity(config, variant_id, plugin, options=options)
    assert first == second
    assert first["fingerprint"] == canonical_hash(first["payload"])
    assert first["payload"]["framework_fingerprint"] == framework_fingerprint()
    assert first["payload"]["operator_fingerprint"] == canonical_hash(
        first["payload"]["operator"]
    )
    assert "deep_gemm" not in sys.modules


def test_path_fingerprint_changes_with_source_content(tmp_path) -> None:
    source = tmp_path / "kernel.cu"
    source.write_text("version one", encoding="utf-8")
    first = path_fingerprint(tmp_path)
    source.write_text("version two", encoding="utf-8")
    second = path_fingerprint(tmp_path)
    assert first["tree_sha256"] != second["tree_sha256"]


def test_framework_manifest_excludes_nested_operator_sources(tmp_path) -> None:
    core = tmp_path / "core.py"
    old_experiment = tmp_path / "experimental" / "old.py"
    old_experiment.parent.mkdir()
    core.write_text("CORE = 1\n", encoding="utf-8")
    old_experiment.write_text("OLD = 1\n", encoding="utf-8")
    first = framework_source_manifest(tmp_path)
    old_experiment.write_text("OLD = 2\n", encoding="utf-8")
    second = framework_source_manifest(tmp_path)
    assert first == second
    assert set(first) == {"core.py"}
    core.write_text("CORE = 2\n", encoding="utf-8")
    assert framework_source_manifest(tmp_path) != first


def test_experiment_identity_has_exactly_five_components() -> None:
    identity = experiment_identity(
        protocol_hash="p",
        plan_hash="l",
        operator_hash="o",
        runtime_hash="r",
        input_content_hash="i",
    )
    assert identity["sha256"] == canonical_hash(identity["components"])
    assert set(identity["components"]) == {
        "ProtocolHash",
        "PlanHash",
        "OperatorHash",
        "RuntimeHash",
        "InputContentHash",
    }
