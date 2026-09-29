from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

from .cache import stable_digest
from .config import DatasetLevel


MEMORY_SCHEMA = "phenoair.reference_reliability_memory"
ALIAS_SCHEMA = "phenoair.reference_reliability_memory_alias"
MEMORY_SCHEMA_VERSION = 1
MEMORY_PRODUCER_VERSION = "reference-memory-v1"


def _level(value: DatasetLevel | str) -> DatasetLevel:
    return value if isinstance(value, DatasetLevel) else DatasetLevel.parse(value)


def _read_document(level: DatasetLevel, resource_root: Path | None = None) -> dict[str, Any]:
    filename = "alias.json" if level is DatasetLevel.NOVEL else "memory.json"
    if resource_root is None:
        resource = files("phenoair").joinpath(
            "resources", "reference_memory", level.value, filename
        )
        return json.loads(resource.read_text(encoding="utf-8"))
    path = resource_root / level.value / filename
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class ReferenceReliabilityMemory:
    requested_level: DatasetLevel
    source_level: DatasetLevel
    reference_panel_fingerprint: str
    asset_fingerprint: str
    sources: dict[str, dict[str, Any]]
    moas: dict[str, dict[str, Any]]
    references: dict[str, dict[str, Any]]
    pair_confusions: dict[tuple[str, str], dict[str, Any]]

    def source(self, name: str) -> dict[str, Any] | None:
        return self.sources.get(name)

    def moa(self, name: str) -> dict[str, Any] | None:
        return self.moas.get(name)

    def reference(self, name: str) -> dict[str, Any] | None:
        return self.references.get(name)

    def pair(self, moa_a: str, moa_b: str) -> dict[str, Any] | None:
        return self.pair_confusions.get(tuple(sorted((moa_a, moa_b))))


def load_reference_memory(
    level: DatasetLevel | str,
    resource_root: Path | None = None,
) -> ReferenceReliabilityMemory:
    requested_level = _level(level)
    document = _read_document(requested_level, resource_root)
    source_level = requested_level

    if document.get("schema") == ALIAS_SCHEMA:
        if document.get("schema_version") != MEMORY_SCHEMA_VERSION:
            raise ValueError("Unsupported reference-memory alias schema version")
        if document.get("producer_version") != MEMORY_PRODUCER_VERSION:
            raise ValueError("Unsupported reference-memory alias producer version")
        source_level = _level(document["target_level"])
        target = _read_document(source_level, resource_root)
        if document.get("target_asset_fingerprint") != target.get("asset_fingerprint"):
            raise ValueError("Reference-memory alias fingerprint does not match its target")
        document = target

    if document.get("schema") != MEMORY_SCHEMA:
        raise ValueError("Invalid reference-memory schema")
    if document.get("schema_version") != MEMORY_SCHEMA_VERSION:
        raise ValueError("Unsupported reference-memory schema version")
    if document.get("producer_version") != MEMORY_PRODUCER_VERSION:
        raise ValueError("Unsupported reference-memory producer version")

    fingerprint_payload = dict(document)
    expected_fingerprint = fingerprint_payload.pop("asset_fingerprint", None)
    if not expected_fingerprint or stable_digest(fingerprint_payload) != expected_fingerprint:
        raise ValueError("Reference-memory asset fingerprint verification failed")

    memory = document.get("memory") or {}
    sources = memory.get("sources") or []
    moas = memory.get("moas") or []
    references = memory.get("references") or []
    pairs = (memory.get("pair_confusions") or []) + (
        memory.get("intra_family_pair_confusions") or []
    )
    actual_counts = {
        "sources": len(sources),
        "moas": len(moas),
        "references": len(references),
        "pair_confusions": len(memory.get("pair_confusions") or []),
        "intra_family_pair_confusions": len(
            memory.get("intra_family_pair_confusions") or []
        ),
    }
    if actual_counts != document.get("counts"):
        raise ValueError("Reference-memory counts do not match the manifest")

    return ReferenceReliabilityMemory(
        requested_level=requested_level,
        source_level=source_level,
        reference_panel_fingerprint=str(document["reference_panel_fingerprint"]),
        asset_fingerprint=expected_fingerprint,
        sources={str(item["source"]): item for item in sources},
        moas={str(item["moa"]): item for item in moas},
        references={str(item["ref_drug"]): item for item in references},
        pair_confusions={
            tuple(sorted((str(item["moa_a"]), str(item["moa_b"])))): item
            for item in pairs
        },
    )
