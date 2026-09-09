from __future__ import annotations

import hashlib
from collections import deque

import pytest

from index_topk_perflab.real_corpus import (
    TokenDocument,
    pack_balanced_documents,
    stable_document_split,
    validate_document_disjointness,
)
from index_topk_perflab.replay import validate_replay_manifest


def _item(split: str, fixture: str, suffix: str) -> dict:
    return {
        "item_id": f"{split}-{fixture}-{suffix}",
        "split": split,
        "case_id": "prefill_q4096_n16384",
        "fixture_id": fixture,
        "path": f"captured/{split}-{fixture}-{suffix}.safetensors",
        "sha256": hashlib.sha256(suffix.encode()).hexdigest(),
        "case": {
            "query_tokens": 4096,
            "context_tokens": 16384,
            "top_k": 2048,
            "indexer_heads": 64,
            "head_dim": 128,
        },
    }


def test_replay_manifest_requires_all_frozen_splits() -> None:
    value = {
        "schema_version": 1,
        "dataset_id": "unit-test",
        "frozen": True,
        "items": [
            _item(split, fixture, f"{split}-{fixture}")
            for split in ("tuning", "test_normal", "test_hard")
            for fixture in ("A", "B")
        ],
    }
    assert validate_replay_manifest(value)["status"] == "passed"
    value["frozen"] = False
    with pytest.raises(ValueError, match="frozen"):
        validate_replay_manifest(value)


def test_document_split_is_stable_and_content_independent() -> None:
    first = stable_document_split("revision", "document")
    assert first in {"tuning", "test_normal", "hard_pool"}
    assert stable_document_split("revision", "document") == first


def test_balanced_pack_consumes_documents_once() -> None:
    pools = {
        "a": deque(
            [TokenDocument("a", "s", "r", f"a-{i}", tuple(range(5))) for i in range(3)]
        ),
        "b": deque(
            [TokenDocument("b", "s", "r", f"b-{i}", tuple(range(5))) for i in range(3)]
        ),
    }
    packed = pack_balanced_documents(pools, length=20)
    assert len(packed.token_ids) == 20
    fingerprints = [item["document_fingerprint"] for item in packed.documents]
    assert len(fingerprints) == len(set(fingerprints))
    assert all(count > 0 for count in packed.domain_token_counts.values())


def test_document_disjointness_rejects_cross_split_reuse() -> None:
    document = {"document_fingerprint": "same"}
    with pytest.raises(ValueError, match="both"):
        validate_document_disjointness({"tuning": [document], "test_normal": [document]})
