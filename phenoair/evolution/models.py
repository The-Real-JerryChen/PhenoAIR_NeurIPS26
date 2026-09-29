from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..agents import ArbiterAgent
from ..controller import RuleRegistry, rule_from_spec


@dataclass
class FailureCluster:
    cluster_id: str
    description: str
    example_query_ids: list[str]
    frequency: int
    distinguishing_signature: dict[str, Any]
    why_current_rules_dont_handle: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FailureCluster":
        return cls(
            cluster_id=str(value["cluster_id"]),
            description=str(value.get("description", "")),
            example_query_ids=[str(item) for item in value.get("example_query_ids", [])],
            frequency=int(value.get("frequency", 0)),
            distinguishing_signature=dict(value.get("distinguishing_signature", {})),
            why_current_rules_dont_handle=str(value.get("why_current_rules_dont_handle", "")),
        )


@dataclass
class EvolutionProposal:
    """Portable evolution output with no prescribed round or selection policy."""

    controller_rules: list[dict[str, Any]] = field(default_factory=list)
    arbiter_heuristics: list[dict[str, Any]] = field(default_factory=list)
    comments: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "EvolutionProposal":
        value = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            controller_rules=list(value.get("controller_rules", [])),
            arbiter_heuristics=list(value.get("arbiter_heuristics", [])),
            comments=str(value.get("comments", "")),
            metadata=dict(value.get("metadata", {})),
        )

    def apply_rules(
        self,
        registry: RuleRegistry,
        selected_rule_ids: set[str] | None = None,
        replace: bool = False,
    ) -> list[str]:
        applied = []
        for spec in self.controller_rules:
            rule_id = str(spec.get("rule_id", ""))
            if not rule_id or (selected_rule_ids is not None and rule_id not in selected_rule_ids):
                continue
            registry.add(rule_from_spec(spec, source="llm"), replace=replace)
            applied.append(rule_id)
        return applied

    def apply_arbiter_heuristic(self, arbiter: ArbiterAgent, candidate_id: str) -> None:
        candidate = next(
            (item for item in self.arbiter_heuristics if item.get("candidate_id") == candidate_id),
            None,
        )
        if candidate is None:
            raise KeyError(f"Unknown Arbiter candidate: {candidate_id}")
        arbiter.decision_heuristics = str(candidate["decision_heuristics"])

