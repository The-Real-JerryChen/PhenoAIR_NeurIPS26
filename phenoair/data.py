from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import DatasetLevel
from .models import MoACase, ReferenceCompound


def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item) for item in value if str(item))
    try:
        if pd.isna(value):
            return ()
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return (text,) if text else ()


@dataclass
class ProfileStore:
    """Compound-level profiles aggregated from released well-level matrices."""

    compound_ids: tuple[str, ...]
    matrix: np.ndarray
    _row_by_compound: dict[str, int]

    @classmethod
    def load(cls, level_dir: Path, feature: str = "cellclip") -> "ProfileStore":
        feature_dir = level_dir / "features"
        matrix = np.load(feature_dir / f"{feature}_features.npy", mmap_mode="r")
        index = pd.read_parquet(feature_dir / f"{feature}_index.parquet")
        compound_ids: list[str] = []
        vectors: list[np.ndarray] = []

        # Equal weighting across sources avoids giving heavily replicated sources
        # disproportionate influence over retrieval.
        for compound_id, compound_rows in index.groupby("compound_id", sort=True):
            source_vectors = []
            for _, source_rows in compound_rows.groupby("source", sort=True):
                row_ids = source_rows["row_index"].to_numpy(dtype=np.int64)
                source_vectors.append(np.asarray(matrix[row_ids], dtype=np.float64).mean(axis=0))
            vectors.append(np.stack(source_vectors).mean(axis=0))
            compound_ids.append(str(compound_id))

        aggregated = np.stack(vectors).astype(np.float32, copy=False)
        norms = np.linalg.norm(aggregated, axis=1, keepdims=True)
        aggregated /= np.where(norms > 0, norms, 1.0)
        ids = tuple(compound_ids)
        return cls(ids, aggregated, {compound_id: idx for idx, compound_id in enumerate(ids)})

    def get(self, compound_id: str) -> np.ndarray:
        try:
            return self.matrix[self._row_by_compound[compound_id]]
        except KeyError as exc:
            raise KeyError(f"No released profile for compound {compound_id}") from exc

    def rows(self, compound_ids: Iterable[str]) -> np.ndarray:
        return np.stack([self.get(compound_id) for compound_id in compound_ids])


@dataclass
class MoADataset:
    level: DatasetLevel
    cases: list[MoACase]
    references: list[ReferenceCompound]
    labels: tuple[str, ...]
    profiles: ProfileStore

    @cached_property
    def fingerprint(self) -> str:
        digest = hashlib.sha256(self.level.value.encode("utf-8"))
        for reference in self.references:
            digest.update(
                "\0".join(
                    (reference.compound_id, reference.smiles, reference.primary_moa)
                ).encode("utf-8")
            )
        digest.update("\0".join(self.profiles.compound_ids).encode("utf-8"))
        digest.update(self.profiles.matrix.tobytes(order="C"))
        return digest.hexdigest()[:24]

    @classmethod
    def load(
        cls,
        data_dir: Path,
        level: DatasetLevel,
        feature: str = "cellclip",
    ) -> "MoADataset":
        level_dir = data_dir / level.value
        reference_df = pd.read_parquet(level_dir / "reference.parquet")
        test_df = pd.read_parquet(level_dir / "test.parquet")
        labels_doc = json.loads((level_dir / "labels.json").read_text(encoding="utf-8"))
        labels = tuple(str(item) for item in labels_doc.get("known_primary_moas", []))

        references = []
        for _, row in reference_df.iterrows():
            references.append(
                ReferenceCompound(
                    compound_id=str(row["compound_id"]),
                    smiles=str(row.get("smiles") or ""),
                    inchikey=str(row.get("inchikey") or ""),
                    primary_moa=str(row.get("benchmark_label") or row.get("primary_moa") or "unknown"),
                    valid_moas=_as_tuple(row.get("valid_moas")),
                )
            )

        cases = []
        for _, row in test_df.iterrows():
            valid = _as_tuple(row.get("benchmark_valid_labels")) or _as_tuple(row.get("valid_moas"))
            primary = str(row.get("benchmark_label") or row.get("primary_moa") or "")
            cases.append(
                MoACase(
                    case_id=str(row["sample_id"]),
                    compound_id=str(row["compound_id"]),
                    smiles=str(row.get("smiles") or ""),
                    inchikey=str(row.get("inchikey") or ""),
                    ground_truth=primary,
                    valid_labels=valid or (primary,),
                    is_novel=bool(row.get("is_novel", False)),
                )
            )
        return cls(level, cases, references, labels, ProfileStore.load(level_dir, feature))

    @property
    def allows_novel(self) -> bool:
        return self.level.allows_novel

    def select(self, identifiers: Iterable[str] = (), limit: int | None = None) -> list[MoACase]:
        identifiers = {str(item) for item in identifiers}
        selected = self.cases
        if identifiers:
            selected = [
                case
                for case in selected
                if case.case_id in identifiers or case.compound_id in identifiers
            ]
        return selected[:limit] if limit is not None else list(selected)
