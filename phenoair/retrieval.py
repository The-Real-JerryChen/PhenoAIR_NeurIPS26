from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import cached_property
import hashlib
from typing import Protocol

import numpy as np

from .data import MoADataset
from .cache import JsonCacheStore
from .models import MoACase, Neighbor, ReferenceCompound


class PhenotypeRetriever(Protocol):
    def neighbors(self, case: MoACase, k: int) -> list[Neighbor]: ...

    def distribution(self, case: MoACase, k: int) -> Counter[str]: ...

    def candidate_support(self, case: MoACase, moa: str, k: int) -> dict: ...


@dataclass
class CellClipRetriever:
    dataset: MoADataset
    cache: JsonCacheStore | None = None

    def __post_init__(self) -> None:
        self._reference_ids = tuple(item.compound_id for item in self.dataset.references)
        self._reference_matrix = self.dataset.profiles.rows(self._reference_ids)

    def neighbors(self, case: MoACase, k: int = 10) -> list[Neighbor]:
        k = min(max(int(k), 1), len(self._reference_ids))
        cache_key = {
            "dataset": self.dataset.fingerprint,
            "query_compound_id": case.compound_id,
            "k": k,
        }
        if self.cache is not None:
            cached = self.cache.get("phenotype_neighbors", "cellclip-cosine-v1", cache_key)
            if cached is not None:
                return [Neighbor(**item) for item in cached]
        query = self.dataset.profiles.get(case.compound_id)
        similarities = self._reference_matrix @ query
        indices = np.argsort(-similarities)[:k]
        result = [
            Neighbor(
                rank=rank,
                compound_id=self.dataset.references[int(idx)].compound_id,
                moa=self.dataset.references[int(idx)].primary_moa,
                similarity=float(similarities[int(idx)]),
            )
            for rank, idx in enumerate(indices, start=1)
        ]
        if self.cache is not None:
            self.cache.set(
                "phenotype_neighbors",
                "cellclip-cosine-v1",
                cache_key,
                [vars(item) for item in result],
            )
        return result

    def distribution(self, case: MoACase, k: int = 50) -> Counter[str]:
        return Counter(item.moa for item in self.neighbors(case, k))

    def candidate_support(self, case: MoACase, moa: str, k: int = 50) -> dict:
        neighbors = self.neighbors(case, k)
        matches = [item for item in neighbors if item.moa == moa]
        best_rank = matches[0].rank if matches else None
        top10 = sum(item.rank <= 10 for item in matches)
        if top10 >= 2 or (best_rank is not None and best_rank <= 3):
            support = "strong"
        elif best_rank is not None and best_rank <= 10:
            support = "moderate"
        elif matches:
            support = "weak"
        else:
            support = "none"
        return {
            "moa": moa,
            "support": support,
            "count": len(matches),
            "best_rank": best_rank,
            "supporting_compounds": [item.compound_id for item in matches[:5]],
        }


@dataclass
class StructureRetriever:
    references: list[ReferenceCompound]
    cache: JsonCacheStore | None = None

    @cached_property
    def _reference_fingerprint(self) -> str:
        digest = hashlib.sha256()
        for reference in self.references:
            digest.update(
                "\0".join(
                    (reference.compound_id, reference.smiles, reference.primary_moa)
                ).encode("utf-8")
            )
        return digest.hexdigest()[:24]

    @cached_property
    def _index(self):
        try:
            from rdkit import Chem
            from rdkit.Chem import rdFingerprintGenerator
        except ImportError as exc:
            raise RuntimeError("RDKit is required for structure retrieval") from exc
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        entries = []
        for reference in self.references:
            molecule = Chem.MolFromSmiles(reference.smiles)
            if molecule is not None:
                entries.append((reference, generator.GetFingerprint(molecule)))
        return generator, entries

    def neighbors(self, case: MoACase, k: int = 10) -> list[Neighbor]:
        from rdkit import Chem, DataStructs

        k = min(max(int(k), 1), len(self.references))
        cache_key = {
            "reference_set": self._reference_fingerprint,
            "query_smiles": case.smiles,
            "k": k,
        }
        if self.cache is not None:
            cached = self.cache.get("structure_neighbors", "morgan-r2-fp2048-v1", cache_key)
            if cached is not None:
                return [Neighbor(**item) for item in cached]
        generator, entries = self._index
        molecule = Chem.MolFromSmiles(case.smiles)
        if molecule is None:
            return []
        query_fp = generator.GetFingerprint(molecule)
        similarities = DataStructs.BulkTanimotoSimilarity(query_fp, [item[1] for item in entries])
        indices = np.argsort(-np.asarray(similarities))[: min(k, len(entries))]
        result = [
            Neighbor(
                rank=rank,
                compound_id=entries[int(idx)][0].compound_id,
                moa=entries[int(idx)][0].primary_moa,
                similarity=float(similarities[int(idx)]),
            )
            for rank, idx in enumerate(indices, start=1)
        ]
        if self.cache is not None:
            self.cache.set(
                "structure_neighbors",
                "morgan-r2-fp2048-v1",
                cache_key,
                [vars(item) for item in result],
            )
        return result

    @staticmethod
    def properties(smiles: str) -> dict[str, float | int | str | None]:
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, rdMolDescriptors
        except ImportError as exc:
            raise RuntimeError("RDKit is required for molecular descriptors") from exc
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return {"smiles": smiles, "valid": False}
        return {
            "smiles": smiles,
            "valid": True,
            "molecular_weight": round(float(Descriptors.MolWt(molecule)), 3),
            "logp": round(float(Descriptors.MolLogP(molecule)), 3),
            "tpsa": round(float(rdMolDescriptors.CalcTPSA(molecule)), 3),
            "h_bond_donors": int(rdMolDescriptors.CalcNumHBD(molecule)),
            "h_bond_acceptors": int(rdMolDescriptors.CalcNumHBA(molecule)),
        }
