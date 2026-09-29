from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .models import MechanismOutput, PhenotypeOutput


_SUPPORT_VALUE = {"none": 0, "weak": 1, "moderate": 2, "strong": 3}


@dataclass
class CandidateMemoryEntry:
    moa: str
    status: str = "active"
    phenotype_score: float = 0.0
    phenotype_support: str = "none"
    phenotype_contradiction: str = "none"
    mechanism_verdict: str = "uncertain"
    mechanism_strength: str = "none"
    sources: set[str] = field(default_factory=set)
    first_seen_round: int = 0
    last_seen_round: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def suppressed(self) -> bool:
        return self.status.startswith("suppressed") or self.status == "archived"

    @property
    def eligible(self) -> bool:
        return not self.suppressed and (
            self.phenotype_score > 0
            or self.phenotype_support in {"weak", "moderate", "strong"}
            or self.mechanism_verdict == "support"
        )

    def score(self) -> float:
        score = self.phenotype_score
        score += 0.12 * _SUPPORT_VALUE.get(self.phenotype_support, 0)
        if self.mechanism_verdict == "support":
            score += 0.25 + 0.08 * _SUPPORT_VALUE.get(self.mechanism_strength, 0)
        elif self.mechanism_verdict == "reject":
            score -= 0.12 + 0.06 * _SUPPORT_VALUE.get(self.mechanism_strength, 0)
        if self.suppressed:
            score -= 1.0
        return score

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["sources"] = sorted(self.sources)
        result["eligible"] = self.eligible
        result["memory_score"] = self.score()
        return result


class CandidateMemory:
    def __init__(self) -> None:
        self._entries: dict[str, CandidateMemoryEntry] = {}

    def ensure(self, moa: str, round_idx: int, source: str) -> CandidateMemoryEntry:
        moa = moa.strip()
        entry = self._entries.get(moa)
        if entry is None:
            entry = CandidateMemoryEntry(moa=moa, first_seen_round=round_idx, last_seen_round=round_idx)
            self._entries[moa] = entry
        entry.sources.add(source)
        entry.last_seen_round = round_idx
        return entry

    def update_phenotype(self, output: PhenotypeOutput, round_idx: int) -> None:
        evidence_by_moa = {item.moa: item for item in output.candidate_evidence}
        for candidate in output.ranked_candidates:
            entry = self.ensure(candidate.moa, round_idx, "P")
            entry.phenotype_score = max(entry.phenotype_score, candidate.score)
            evidence = evidence_by_moa.get(candidate.moa)
            if evidence:
                if _SUPPORT_VALUE.get(evidence.phenotype_support, 0) >= _SUPPORT_VALUE.get(entry.phenotype_support, 0):
                    entry.phenotype_support = evidence.phenotype_support
                entry.phenotype_contradiction = evidence.phenotype_contradiction
                if evidence.evidence:
                    entry.notes.append(evidence.evidence)
            if entry.status == "archived":
                entry.status = "active"

    def update_mechanism(self, output: MechanismOutput, round_idx: int) -> None:
        for assessment in output.candidate_assessment:
            entry = self.ensure(assessment.moa, round_idx, "M")
            entry.mechanism_verdict = assessment.verdict
            entry.mechanism_strength = assessment.verdict_strength
            if assessment.rationale:
                entry.notes.append(assessment.rationale)
            if (
                assessment.verdict == "reject"
                and assessment.verdict_strength == "strong"
                and assessment.suppression_recommendation == "suppress_now"
                and entry.phenotype_contradiction == "strong"
            ):
                entry.status = "suppressed_by_mechanism"
        if output.mechanistic_candidate:
            candidate = output.mechanistic_candidate
            entry = self.ensure(candidate.moa, round_idx, "M_alternative")
            entry.mechanism_verdict = "support"
            entry.mechanism_strength = candidate.strength
            if entry.phenotype_score == 0:
                entry.status = "needs_phenotype_support"
        if output.mechanistic_candidate_group:
            for moa in output.mechanistic_candidate_group.members:
                entry = self.ensure(moa, round_idx, "M_group")
                if entry.phenotype_score == 0:
                    entry.status = "family_only"

    def active(self) -> list[CandidateMemoryEntry]:
        return sorted(
            (entry for entry in self._entries.values() if not entry.suppressed),
            key=lambda entry: entry.score(),
            reverse=True,
        )

    def eligible(self) -> list[CandidateMemoryEntry]:
        return [entry for entry in self.active() if entry.eligible]

    def best(self) -> CandidateMemoryEntry | None:
        candidates = self.eligible() or self.active()
        return candidates[0] if candidates else None

    def get(self, moa: str) -> CandidateMemoryEntry | None:
        return self._entries.get(moa)

    def summary(self, max_candidates: int = 12) -> list[dict[str, Any]]:
        return [entry.to_dict() for entry in self.active()[:max_candidates]]

    def to_dict(self) -> dict[str, Any]:
        return {moa: entry.to_dict() for moa, entry in sorted(self._entries.items())}

    def __len__(self) -> int:
        return len(self._entries)

