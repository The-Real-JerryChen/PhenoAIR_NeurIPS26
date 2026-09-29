#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phenoair.cache import stable_digest


SCHEMA_VERSION = 1
PRODUCER_VERSION = "reference-memory-v1"
GROUPS = {
    "source_memory": ("sources", "source_memory_item", "source"),
    "moa_memory": ("moas", "moa_memory_item", "moa"),
    "ref_anchor_memory": ("references", "ref_anchor_memory_item", "ref_drug"),
    "pair_confusion_memory": ("pair_confusions", "pair_confusion_memory_item", "moa_a"),
    "intra_family_pair_memory": (
        "intra_family_pair_confusions",
        "intra_family_pair_confusion_memory_item",
        "moa_a",
    ),
}
LEVELS = {
    "core": "moa_verified",
    "extend": "moa_extended",
}
BLOCKED_KEYS = {"messages", "raw_response", "created_at", "trace", "trace_path"}


def _walk(value: Any, trail: tuple[str, ...] = ()):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _walk(child, trail + (str(key),))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, trail + (str(index),))
    else:
        yield trail, value


def _validate_item(item: dict[str, Any], schema: str, internal_level: str, path: Path) -> None:
    if item.get("schema") != schema or item.get("level") != internal_level:
        raise ValueError(f"Unexpected schema or level in {path}")
    if re.fullmatch(r"[0-9a-f]{16}", str(item.get("evidence_hash", ""))) is None:
        raise ValueError(f"Invalid evidence_hash in {path}")
    for trail, value in _walk(item):
        if any(key in BLOCKED_KEYS for key in trail):
            raise ValueError(f"Blocked trace field in {path}: {'.'.join(trail)}")
        if isinstance(value, str) and (value.startswith("/fs/") or value.startswith("/users/")):
            raise ValueError(f"Absolute local path in {path}: {'.'.join(trail)}")


def _load_group(
    source_root: Path,
    internal_level: str,
    group: str,
    public_level: str,
) -> list[dict[str, Any]]:
    output_name, schema, sort_key = GROUPS[group]
    del output_name
    directory = source_root / internal_level / group
    if not directory.exists():
        return []
    items = []
    for path in sorted(directory.glob("*.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        _validate_item(item, schema, internal_level, path)
        item["level"] = public_level
        items.append(item)
    return sorted(
        items,
        key=lambda item: (
            str(item.get(sort_key, "")),
            str(item.get("moa_b", "")),
        ),
    )


def _reference_panel_fingerprint(data_dir: Path, public_level: str) -> str:
    level_dir = data_dir / public_level
    reference = pd.read_parquet(level_dir / "reference.parquet").sort_values("compound_id")
    index = pd.read_parquet(level_dir / "features" / "cellclip_index.parquet")
    matrix = np.load(level_dir / "features" / "cellclip_features.npy", mmap_mode="r")
    reference_ids = set(reference["compound_id"].astype(str))
    selected = index[index["compound_id"].astype(str).isin(reference_ids)].sort_values("row_index")

    digest = hashlib.sha256(b"phenoair-reference-panel-v1")
    table_columns = [
        column
        for column in ("compound_id", "smiles", "inchikey", "primary_moa", "benchmark_label")
        if column in reference.columns
    ]
    digest.update(
        reference[table_columns].to_json(orient="records", force_ascii=True).encode("utf-8")
    )
    index_columns = [
        column
        for column in ("row_index", "compound_id", "source")
        if column in selected.columns
    ]
    digest.update(selected[index_columns].to_json(orient="records", force_ascii=True).encode("utf-8"))
    rows = selected["row_index"].to_numpy(dtype=np.int64)
    digest.update(np.asarray(matrix[rows], dtype=np.float32).tobytes(order="C"))
    return digest.hexdigest()


def export_level(
    source_root: Path,
    data_dir: Path,
    output_root: Path,
    internal_level: str,
    public_level: str,
) -> dict[str, Any]:
    memory = {}
    for source_name, (output_name, _, _) in GROUPS.items():
        memory[output_name] = _load_group(
            source_root, internal_level, source_name, public_level
        )
    counts = {name: len(items) for name, items in memory.items()}
    document = {
        "schema": "phenoair.reference_reliability_memory",
        "schema_version": SCHEMA_VERSION,
        "producer_version": PRODUCER_VERSION,
        "level": public_level,
        "feature_space": "cellclip_white",
        "reference_panel_fingerprint": _reference_panel_fingerprint(data_dir, public_level),
        "provenance": {
            "scope": "reference_only",
            "generation": "LLM-assisted judgments over deterministic reference diagnostics",
            "raw_model_traces_included": False,
            "test_trajectories_included": False,
            "test_ground_truth_used": False,
        },
        "counts": counts,
        "memory": memory,
    }
    document["asset_fingerprint"] = stable_digest(document)
    output_dir = output_root / public_level
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "memory.json").write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description="Export sanitized PhenoAIR reference memory.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    exported = {}
    for internal_level, public_level in LEVELS.items():
        exported[public_level] = export_level(
            args.source_root.resolve(),
            args.data_dir.resolve(),
            args.output_dir.resolve(),
            internal_level,
            public_level,
        )

    verified = exported["moa_verified"]
    alias = {
        "schema": "phenoair.reference_reliability_memory_alias",
        "schema_version": SCHEMA_VERSION,
        "producer_version": PRODUCER_VERSION,
        "level": "moa_novel",
        "target_level": "moa_verified",
        "target_asset_fingerprint": verified["asset_fingerprint"],
        "reason": "MoA-Novel uses the MoA-Verified known-MoA reference panel.",
    }
    alias_dir = args.output_dir.resolve() / "moa_novel"
    alias_dir.mkdir(parents=True, exist_ok=True)
    (alias_dir / "alias.json").write_text(
        json.dumps(alias, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    for level, document in exported.items():
        print(level, document["counts"], document["asset_fingerprint"])
    print("moa_novel", "alias -> moa_verified", alias["target_asset_fingerprint"])


if __name__ == "__main__":
    main()
