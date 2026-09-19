"""Deterministic document splitting and balanced token packing for replay data."""

from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Sequence

INTERNAL_SPLITS = ("tuning", "test_normal", "hard_pool")


def document_fingerprint(source_revision: str, document_id: str) -> str:
    material = f"{source_revision}\0{document_id}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def stable_document_split(source_revision: str, document_id: str) -> str:
    """Assign whole documents 20/20/60 without inspecting their contents."""

    digest = document_fingerprint(source_revision, document_id)
    bucket = int(digest[:8], 16) % 10_000
    if bucket < 2_000:
        return "tuning"
    if bucket < 4_000:
        return "test_normal"
    return "hard_pool"


@dataclass(frozen=True)
class TokenDocument:
    domain: str
    source: str
    source_revision: str
    document_id: str
    token_ids: tuple[int, ...]

    @property
    def fingerprint(self) -> str:
        return document_fingerprint(self.source_revision, self.document_id)

    @property
    def split(self) -> str:
        return stable_document_split(self.source_revision, self.document_id)


@dataclass(frozen=True)
class PackedTokens:
    token_ids: tuple[int, ...]
    documents: tuple[Mapping[str, Any], ...]
    domain_token_counts: Mapping[str, int]


def pack_balanced_documents(
    pools: MutableMapping[str, deque[TokenDocument]],
    *,
    length: int,
    domain_offset: int = 0,
    bos_token_id: int = 0,
    eos_token_id: int = 1,
) -> PackedTokens:
    """Consume whole, disjoint documents in deterministic domain round-robin order."""

    if length <= 0:
        raise ValueError("packed sequence length must be positive")
    domains = sorted(pools)
    if not domains:
        raise ValueError("at least one corpus domain is required")
    domains = domains[domain_offset % len(domains) :] + domains[: domain_offset % len(domains)]
    output: list[int] = []
    documents: list[Mapping[str, Any]] = []
    counts = {domain: 0 for domain in domains}
    cursor = 0
    empty_rounds = 0
    while len(output) < length:
        domain = domains[cursor % len(domains)]
        cursor += 1
        pool = pools[domain]
        if not pool:
            empty_rounds += 1
            if empty_rounds >= len(domains):
                raise RuntimeError(
                    f"corpus pools exhausted with {length - len(output)} tokens remaining"
                )
            continue
        empty_rounds = 0
        document = pool.popleft()
        remaining = length - len(output)
        if remaining == 1:
            output.append(eos_token_id)
            counts[domain] += 1
            break
        take = min(len(document.token_ids), remaining - 2)
        contribution = [bos_token_id, *document.token_ids[:take], eos_token_id]
        output.extend(contribution)
        counts[domain] += len(contribution)
        documents.append(
            {
                "domain": document.domain,
                "source": document.source,
                "source_revision": document.source_revision,
                "document_id_sha256": hashlib.sha256(
                    document.document_id.encode("utf-8")
                ).hexdigest(),
                "document_fingerprint": document.fingerprint,
                "source_tokens": len(document.token_ids),
                "tokens_used_with_specials": len(contribution),
                "truncated": take < len(document.token_ids),
            }
        )
    if len(output) != length:
        raise RuntimeError("balanced packing did not realize the requested length")
    return PackedTokens(tuple(output), tuple(documents), counts)


def validate_document_disjointness(
    documents_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Reject any source document crossing tuning/normal/hard-pool boundaries."""

    seen: dict[str, str] = {}
    per_split: dict[str, set[str]] = {}
    for split, documents in documents_by_split.items():
        current = per_split.setdefault(split, set())
        for document in documents:
            fingerprint = str(document["document_fingerprint"])
            other = seen.get(fingerprint)
            if other is not None and other != split:
                raise ValueError(
                    f"document {fingerprint} appears in both {other!r} and {split!r}"
                )
            seen[fingerprint] = split
            current.add(fingerprint)
    return {
        "status": "passed",
        "documents_total": len(seen),
        "documents_per_split": {
            split: len(fingerprints) for split, fingerprints in sorted(per_split.items())
        },
    }
