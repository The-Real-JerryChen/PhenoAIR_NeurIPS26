from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

from .memory import CandidateMemory
from .models import ControllerDecision, MechanismOutput, PhenotypeOutput


RuleSource = Literal["hand-crafted", "llm", "user"]
Predicate = Callable[["ControllerContext"], bool]
ActionFactory = Callable[["ControllerContext"], ControllerDecision]


@dataclass
class ControllerState:
    round_idx: int = 0
    max_rounds: int = 4
    already_expanded_k: bool = False
    already_checked_distribution: bool = False
    checked_candidates: set[str] = field(default_factory=set)
    already_checked_group: bool = False
    already_checked_active_candidate_set: bool = False
    already_rescued_candidate_set: bool = False
    already_requested_group_mode: bool = False
    family_exhaustion_count: int = 0
    mechanism_assessment_count: int = 0
    mechanism_rejection_count: int = 0
    benchmark_mode: str = "closed_set"
    p_top_history: list[str] = field(default_factory=list)
    p_reliability_history: list[str] = field(default_factory=list)

    @property
    def oscillation_detected(self) -> bool:
        if len(self.p_top_history) < 4:
            return False
        return sum(
            current != previous
            for previous, current in zip(self.p_top_history, self.p_top_history[1:])
        ) >= 3


@dataclass
class ControllerContext:
    phenotype: PhenotypeOutput
    mechanism: MechanismOutput
    memory: CandidateMemory
    state: ControllerState

    def signals(self) -> dict[str, Any]:
        ranked = self.phenotype.ranked_candidates
        top = ranked[0] if ranked else None
        runner_up = ranked[1] if len(ranked) > 1 else None
        top_assessment = next(
            (
                item
                for item in self.mechanism.candidate_assessment
                if top is not None and item.moa == top.moa
            ),
            None,
        )
        top3_labels = {item.moa for item in ranked[:3]}
        top3_assessments = [
            item for item in self.mechanism.candidate_assessment if item.moa in top3_labels
        ]
        active = self.memory.active()
        memory_entries = list(self.memory.to_dict().values())
        suppressed = [
            entry for entry in memory_entries if str(entry.get("status", "")).startswith("suppressed")
        ]
        family_only = [entry for entry in memory_entries if entry.get("status") == "family_only"]
        if any(entry.phenotype_support == "strong" for entry in active):
            evidence_ceiling = "strong"
        elif any(
            entry.phenotype_support == "moderate" or entry.mechanism_verdict == "support"
            for entry in active
        ):
            evidence_ceiling = "moderate"
        elif active:
            evidence_ceiling = "weak"
        else:
            evidence_ceiling = "none"
        rejection_rate = (
            self.state.mechanism_rejection_count / self.state.mechanism_assessment_count
            if self.state.mechanism_assessment_count
            else 0.0
        )
        return {
            "p": {
                "top1": top.moa if top else None,
                "top1_score": top.score if top else 0.0,
                "top_gap": (top.score - runner_up.score) if top and runner_up else (top.score if top else 0.0),
                "retrieval_reliability": self.phenotype.retrieval_reliability,
                "flags": list(self.phenotype.flags),
                "num_candidates": len(ranked),
            },
            "m": {
                "top_verdict": top_assessment.verdict if top_assessment else "uncertain",
                "top_verdict_strength": top_assessment.verdict_strength if top_assessment else "none",
                "overall_agreement": self.mechanism.overall_agreement_with_p,
                "mechanistic_candidate": (
                    self.mechanism.mechanistic_candidate.moa
                    if self.mechanism.mechanistic_candidate
                    else None
                ),
                "mechanistic_candidate_strength": (
                    self.mechanism.mechanistic_candidate.strength
                    if self.mechanism.mechanistic_candidate
                    else "none"
                ),
                "group_members": (
                    list(self.mechanism.mechanistic_candidate_group.members)
                    if self.mechanism.mechanistic_candidate_group
                    else []
                ),
                "group_family_name": (
                    self.mechanism.mechanistic_candidate_group.family_name
                    if self.mechanism.mechanistic_candidate_group
                    else None
                ),
                "group_strength": (
                    self.mechanism.mechanistic_candidate_group.strength
                    if self.mechanism.mechanistic_candidate_group
                    else "none"
                ),
            },
            "memory": {
                "num_total": len(self.memory),
                "num_active": len(self.memory.active()),
                "num_eligible": len(self.memory.eligible()),
                "num_suppressed": len(suppressed),
                "num_family_only": len(family_only),
                "active_candidates": [entry.moa for entry in self.memory.active()],
                "suppressed_candidates": [str(entry.get("moa")) for entry in suppressed],
                "family_only_candidates": [str(entry.get("moa")) for entry in family_only],
            },
            "state": {
                "round_idx": self.state.round_idx,
                "max_rounds": self.state.max_rounds,
                "already_expanded_k": self.state.already_expanded_k,
                "already_checked_distribution": self.state.already_checked_distribution,
                "already_checked_group": self.state.already_checked_group,
                "already_checked_active_candidate_set": self.state.already_checked_active_candidate_set,
                "already_rescued_candidate_set": self.state.already_rescued_candidate_set,
                "already_requested_group_mode": self.state.already_requested_group_mode,
                "oscillation_detected": self.state.oscillation_detected,
                "checked_candidates": sorted(self.state.checked_candidates),
                "p_top_history": list(self.state.p_top_history),
                "p_reliability_history": list(self.state.p_reliability_history),
            },
            "diagnostics": {
                "benchmark_mode": self.state.benchmark_mode,
                "cumulative_rejection_rate": rejection_rate,
                "family_exhaustion_count": self.state.family_exhaustion_count,
                "evidence_ceiling": evidence_ceiling,
                "top3_all_rejected": (
                    bool(top3_labels)
                    and len(top3_assessments) == len(top3_labels)
                    and all(item.verdict == "reject" for item in top3_assessments)
                ),
                "no_supported_canonical_candidate": not bool(self.memory.eligible()),
            },
        }


@dataclass
class ControllerRule:
    rule_id: str
    priority: int
    predicate: Predicate
    action_factory: ActionFactory
    description: str = ""
    source: RuleSource = "hand-crafted"
    enabled: bool = True
    tags: tuple[str, ...] = ()
    structured_spec: dict[str, Any] | None = None

    def evaluate(self, context: ControllerContext) -> ControllerDecision | None:
        if not self.enabled or not self.predicate(context):
            return None
        decision = self.action_factory(context)
        decision.rule_id = self.rule_id
        decision.source = self.source
        return decision

    def metadata(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "priority": self.priority,
            "description": self.description,
            "source": self.source,
            "enabled": self.enabled,
            "tags": list(self.tags),
            "structured_spec": self.structured_spec,
        }


class RuleRegistry:
    """Mutable rule registry supporting add, replace, remove, inspect, and filtering."""

    def __init__(self, rules: Iterable[ControllerRule] = ()) -> None:
        self._rules: dict[str, ControllerRule] = {}
        for rule in rules:
            self.add(rule)

    def add(self, rule: ControllerRule, replace: bool = False) -> None:
        if rule.rule_id in self._rules and not replace:
            raise KeyError(f"Rule already registered: {rule.rule_id}")
        self._rules[rule.rule_id] = rule

    def remove(self, rule_id: str) -> ControllerRule:
        try:
            return self._rules.pop(rule_id)
        except KeyError as exc:
            raise KeyError(f"Unknown rule: {rule_id}") from exc

    def get(self, rule_id: str) -> ControllerRule:
        return self._rules[rule_id]

    def set_enabled(self, rule_id: str, enabled: bool) -> None:
        self.get(rule_id).enabled = enabled

    def iter_rules(
        self,
        source: RuleSource | None = None,
        enabled_only: bool = True,
    ) -> list[ControllerRule]:
        rules = self._rules.values()
        if source is not None:
            rules = (rule for rule in rules if rule.source == source)
        if enabled_only:
            rules = (rule for rule in rules if rule.enabled)
        return sorted(rules, key=lambda rule: (rule.priority, rule.rule_id))

    def decide(self, context: ControllerContext) -> ControllerDecision:
        for rule in self.iter_rules():
            decision = rule.evaluate(context)
            if decision is not None:
                return decision
        raise RuntimeError("No controller rule matched; register a default STOP rule")

    def snapshot(self, enabled_only: bool = False) -> list[dict[str, Any]]:
        return [rule.metadata() for rule in self.iter_rules(enabled_only=enabled_only)]

    def load_specs(self, path: Path, source: RuleSource = "user", replace: bool = False) -> None:
        document = json.loads(path.read_text(encoding="utf-8"))
        specs = document.get("rules", document) if isinstance(document, dict) else document
        if not isinstance(specs, list):
            raise ValueError("Rule file must be a JSON list or an object containing `rules`")
        for spec in specs:
            self.add(rule_from_spec(spec, source=source), replace=replace)


def _get_path(value: dict[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _condition_matches(actual: Any, operator: str, expected: Any) -> bool:
    if operator == "eq":
        return actual == expected
    if operator == "ne":
        return actual != expected
    if operator == "in":
        return actual in expected
    if operator == "not_in":
        return actual not in expected
    if operator == "contains":
        return expected in (actual or [])
    if operator == "gt":
        return actual is not None and actual > expected
    if operator == "gte":
        return actual is not None and actual >= expected
    if operator == "lt":
        return actual is not None and actual < expected
    if operator == "lte":
        return actual is not None and actual <= expected
    if operator == "truthy":
        return bool(actual)
    if operator == "falsy":
        return not bool(actual)
    raise ValueError(f"Unsupported rule operator: {operator}")


def _resolve_dynamic(value: Any, signals: dict[str, Any]) -> Any:
    if isinstance(value, str) and value.startswith("$"):
        return _get_path(signals, value[1:])
    if isinstance(value, list):
        return [_resolve_dynamic(item, signals) for item in value]
    if isinstance(value, dict):
        return {key: _resolve_dynamic(item, signals) for key, item in value.items()}
    return value


def rule_from_spec(spec: dict[str, Any], source: RuleSource = "llm") -> ControllerRule:
    required = {"rule_id", "priority", "conditions", "action"}
    missing = required - set(spec)
    if missing:
        raise ValueError(f"Rule is missing required fields: {sorted(missing)}")
    conditions = list(spec["conditions"])
    action = dict(spec["action"])

    def predicate(context: ControllerContext) -> bool:
        signals = context.signals()
        return all(
            _condition_matches(
                _get_path(signals, str(condition["path"])),
                str(condition.get("op", "eq")),
                condition.get("value"),
            )
            for condition in conditions
        )

    def action_factory(context: ControllerContext) -> ControllerDecision:
        signals = context.signals()
        return ControllerDecision(
            action=str(action.get("type", action.get("decision", "STOP"))),
            rule_id=str(spec["rule_id"]),
            reason=str(action.get("reason", spec.get("description", "structured rule matched"))),
            parameters=_resolve_dynamic(dict(action.get("parameters", action.get("params", {}))), signals),
            source=source,
        )

    return ControllerRule(
        rule_id=str(spec["rule_id"]),
        priority=int(spec["priority"]),
        description=str(spec.get("description", "")),
        source=source,
        enabled=bool(spec.get("enabled", True)),
        tags=tuple(str(item) for item in spec.get("tags", [])),
        predicate=predicate,
        action_factory=action_factory,
        structured_spec=spec,
    )


def _decision(action: str, reason: str, **parameters: Any) -> ActionFactory:
    return lambda context: ControllerDecision(action, "", reason, parameters)


def build_default_registry() -> RuleRegistry:
    """Return the hand-crafted baseline. Callers may add/remove/replace rules."""
    rules = [
        ControllerRule(
            "stop_budget_exhausted",
            10,
            lambda ctx: ctx.state.round_idx >= ctx.state.max_rounds,
            _decision("STOP", "refinement budget exhausted"),
            "Terminate once the configured refinement budget is exhausted.",
            tags=("terminal",),
        ),
        ControllerRule(
            "expand_no_candidates",
            100,
            lambda ctx: not ctx.phenotype.ranked_candidates and not ctx.state.already_expanded_k,
            _decision("EXPAND_K", "phenotype agent returned no candidates", next_k=20),
            tags=("phenotype", "recovery"),
        ),
        ControllerRule(
            "validate_mechanistic_alternative",
            120,
            lambda ctx: bool(
                ctx.mechanism.mechanistic_candidate
                and ctx.mechanism.mechanistic_candidate.moa not in ctx.state.checked_candidates
                and ctx.mechanism.mechanistic_candidate.strength in {"moderate", "strong"}
            ),
            lambda ctx: ControllerDecision(
                "CHECK_CANDIDATE_IN_NEIGHBORHOOD",
                "",
                "validate M's mechanistic alternative in phenotype space",
                {"target_candidate": ctx.mechanism.mechanistic_candidate.moa},
            ),
            tags=("mechanism", "validation"),
        ),
        ControllerRule(
            "validate_mechanistic_group",
            140,
            lambda ctx: bool(
                ctx.mechanism.mechanistic_candidate_group and not ctx.state.already_checked_group
            ),
            lambda ctx: ControllerDecision(
                "CHECK_GROUP_IN_NEIGHBORHOOD",
                "",
                "discriminate M's candidate group using phenotype evidence",
                {
                    "target_group": list(ctx.mechanism.mechanistic_candidate_group.members),
                    "family_name": ctx.mechanism.mechanistic_candidate_group.family_name,
                },
            ),
            tags=("mechanism", "group"),
        ),
        ControllerRule(
            "expand_low_reliability",
            180,
            lambda ctx: (
                ctx.phenotype.retrieval_reliability == "low"
                and not ctx.state.already_expanded_k
            ),
            _decision("EXPAND_K", "low phenotype retrieval reliability", next_k=20),
            tags=("phenotype",),
        ),
        ControllerRule(
            "broaden_rejected_top",
            220,
            lambda ctx: (
                ctx.signals()["m"]["top_verdict"] == "reject"
                and not ctx.state.already_checked_distribution
            ),
            _decision("GET_BROADER_SUMMARY", "M rejects P top-1; inspect broader phenotype distribution", max_k=50),
            tags=("phenotype", "conflict"),
        ),
        ControllerRule(
            "stop_high_agreement",
            500,
            lambda ctx: (
                ctx.phenotype.retrieval_reliability == "high"
                and ctx.mechanism.overall_agreement_with_p == "high"
                and ctx.signals()["m"]["top_verdict"] == "support"
            ),
            _decision("STOP", "high-confidence P/M agreement"),
            tags=("terminal",),
        ),
        ControllerRule(
            "stop_medium_after_refinement",
            700,
            lambda ctx: (
                ctx.state.round_idx >= 1
                and ctx.phenotype.retrieval_reliability in {"high", "medium"}
                and ctx.signals()["m"]["top_verdict"] != "reject"
            ),
            _decision("STOP", "evidence stabilized after refinement"),
            tags=("terminal",),
        ),
        ControllerRule(
            "inspect_distribution_once",
            900,
            lambda ctx: not ctx.state.already_checked_distribution,
            _decision("GET_BROADER_SUMMARY", "inspect broader phenotype evidence before stopping", max_k=50),
            tags=("phenotype",),
        ),
        ControllerRule(
            "stop_default",
            10_000,
            lambda ctx: True,
            _decision("STOP", "no higher-priority controller rule matched"),
            tags=("terminal", "default"),
        ),
    ]
    return RuleRegistry(rules)
