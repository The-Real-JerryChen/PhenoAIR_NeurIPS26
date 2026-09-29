from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


Reliability = Literal["high", "medium", "low"]
Strength = Literal["strong", "moderate", "weak", "none"]
Verdict = Literal["support", "reject", "uncertain"]


@dataclass(frozen=True)
class MoACase:
    case_id: str
    compound_id: str
    smiles: str
    inchikey: str
    ground_truth: str
    valid_labels: tuple[str, ...]
    is_novel: bool = False


@dataclass(frozen=True)
class ReferenceCompound:
    compound_id: str
    smiles: str
    inchikey: str
    primary_moa: str
    valid_moas: tuple[str, ...]


@dataclass(frozen=True)
class Neighbor:
    rank: int
    compound_id: str
    moa: str
    similarity: float

    @property
    def distance(self) -> float:
        return 1.0 - self.similarity


@dataclass
class Candidate:
    moa: str
    score: float

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Candidate":
        return cls(moa=str(value.get("moa", "")).strip(), score=float(value.get("score", 0.0)))


@dataclass
class CandidateEvidence:
    moa: str
    phenotype_support: Strength = "none"
    phenotype_contradiction: Literal["strong", "weak", "none"] = "none"
    evidence: str = ""


@dataclass
class PhenotypeOutput:
    ranked_candidates: list[Candidate] = field(default_factory=list)
    retrieval_reliability: Reliability = "low"
    flags: list[str] = field(default_factory=list)
    candidate_evidence: list[CandidateEvidence] = field(default_factory=list)
    phenotype_summary: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "PhenotypeOutput":
        value = value or {}
        ranked = [Candidate.from_dict(item) for item in value.get("ranked_candidates", []) if item.get("moa")]
        ranked.sort(key=lambda item: item.score, reverse=True)
        evidence = []
        for item in value.get("candidate_evidence", []) or []:
            moa = str(item.get("moa", "")).strip()
            if not moa:
                continue
            support = str(item.get("phenotype_support", "none")).lower()
            if support not in {"strong", "moderate", "weak", "none"}:
                support = "none"
            contradiction = str(item.get("phenotype_contradiction", "none")).lower()
            if contradiction not in {"strong", "weak", "none"}:
                contradiction = "none"
            evidence.append(
                CandidateEvidence(
                    moa=moa,
                    phenotype_support=support,
                    phenotype_contradiction=contradiction,
                    evidence=str(item.get("evidence", item.get("summary", ""))),
                )
            )
        reliability = str(value.get("retrieval_reliability", "low")).lower()
        if reliability not in {"high", "medium", "low"}:
            reliability = "low"
        return cls(
            ranked_candidates=ranked,
            retrieval_reliability=reliability,
            flags=[str(item) for item in value.get("flags", [])],
            candidate_evidence=evidence,
            phenotype_summary=str(value.get("phenotype_summary", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateAssessment:
    moa: str
    verdict: Verdict
    verdict_strength: Strength = "weak"
    suppression_recommendation: str = "keep_active"
    rationale: str = ""


@dataclass
class MechanisticCandidate:
    moa: str
    strength: Strength = "weak"
    rationale: str = ""


@dataclass
class MechanisticGroup:
    family_name: str
    members: list[str]
    strength: Strength = "weak"


@dataclass
class MechanismOutput:
    candidate_assessment: list[CandidateAssessment] = field(default_factory=list)
    mechanistic_candidate: MechanisticCandidate | None = None
    mechanistic_candidate_group: MechanisticGroup | None = None
    overall_agreement_with_p: Reliability = "low"
    mechanism_summary: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "MechanismOutput":
        value = value or {}
        assessments = []
        for item in value.get("candidate_assessment", []) or []:
            moa = str(item.get("moa", item.get("candidate", ""))).strip()
            if not moa:
                continue
            verdict = str(item.get("verdict", "uncertain")).lower()
            if verdict not in {"support", "reject", "uncertain"}:
                verdict = "uncertain"
            strength = _normalize_strength(item.get("verdict_strength", "weak"))
            assessments.append(
                CandidateAssessment(
                    moa=moa,
                    verdict=verdict,
                    verdict_strength=strength,
                    suppression_recommendation=str(item.get("suppression_recommendation", "keep_active")),
                    rationale=str(item.get("rationale", item.get("reasoning", ""))),
                )
            )

        candidate_doc = value.get("mechanistic_candidate")
        candidate = None
        if isinstance(candidate_doc, dict) and candidate_doc.get("moa"):
            candidate = MechanisticCandidate(
                moa=str(candidate_doc["moa"]).strip(),
                strength=_normalize_strength(candidate_doc.get("strength", candidate_doc.get("confidence", "weak"))),
                rationale=str(candidate_doc.get("rationale", "")),
            )

        group_doc = value.get("mechanistic_candidate_group")
        group = None
        if isinstance(group_doc, dict):
            members = group_doc.get("members", group_doc.get("labels", [])) or []
            if members:
                group = MechanisticGroup(
                    family_name=str(group_doc.get("family_name", "mechanistic group")),
                    members=[str(item) for item in members],
                    strength=_normalize_strength(group_doc.get("strength", group_doc.get("confidence", "weak"))),
                )

        agreement = str(value.get("overall_agreement_with_P", value.get("overall_agreement_with_p", "low"))).lower()
        if agreement not in {"high", "medium", "low"}:
            agreement = "low"
        return cls(
            candidate_assessment=assessments,
            mechanistic_candidate=candidate,
            mechanistic_candidate_group=group,
            overall_agreement_with_p=agreement,
            mechanism_summary=str(value.get("mechanism_summary", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["overall_agreement_with_P"] = result.pop("overall_agreement_with_p")
        return result


@dataclass
class ArbiterOutput:
    primary_moa: str
    other_possible_moas: list[str] = field(default_factory=list)
    confidence: Reliability = "low"
    reasoning: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "ArbiterOutput":
        value = value or {}
        confidence = str(value.get("confidence", "low")).lower()
        if confidence not in {"high", "medium", "low"}:
            confidence = "low"
        return cls(
            primary_moa=str(value.get("primary_moa", "")).strip(),
            other_possible_moas=[str(item) for item in value.get("other_possible_moas", [])],
            confidence=confidence,
            reasoning=str(value.get("reasoning", value.get("final_reasoning", ""))),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ControllerDecision:
    action: str
    rule_id: str
    reason: str
    parameters: dict[str, Any] = field(default_factory=dict)
    source: str = "hand-crafted"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrajectoryStep:
    round_idx: int
    phenotype: PhenotypeOutput
    mechanism: MechanismOutput
    controller: ControllerDecision

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round_idx,
            "P": self.phenotype.to_dict(),
            "M": self.mechanism.to_dict(),
            "C": self.controller.to_dict(),
        }


def _normalize_strength(value: Any) -> Strength:
    text = str(value or "weak").lower()
    if text == "high":
        return "strong"
    if text == "medium":
        return "moderate"
    if text == "low":
        return "weak"
    if text not in {"strong", "moderate", "weak", "none"}:
        return "weak"
    return text

