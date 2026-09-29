from __future__ import annotations

import json
from typing import Any

from .models import FailureCluster


ARCHITECTURE = """P proposes candidate MOAs from Cell Painting retrieval.
M evaluates candidates using structure and mechanism evidence.
The registry-driven Controller chooses a refinement action.
A produces the final answer from candidate memory and the trajectory.

Controller actions: EXPAND_K, GET_BROADER_SUMMARY,
CHECK_CANDIDATE_IN_NEIGHBORHOOD, CHECK_GROUP_IN_NEIGHBORHOOD,
CHECK_ACTIVE_CANDIDATE_SET, RESCUE_CANDIDATE_SET, REQUEST_GROUP_MODE_FROM_M,
and STOP."""


SIGNAL_CATALOG = {
    "p": ["top1", "top1_score", "top_gap", "retrieval_reliability", "flags", "num_candidates"],
    "m": [
        "top_verdict",
        "top_verdict_strength",
        "overall_agreement",
        "mechanistic_candidate",
        "mechanistic_candidate_strength",
        "group_members",
        "group_family_name",
        "group_strength",
    ],
    "memory": [
        "num_total",
        "num_active",
        "num_eligible",
        "num_suppressed",
        "num_family_only",
        "active_candidates",
        "suppressed_candidates",
        "family_only_candidates",
    ],
    "state": [
        "round_idx",
        "max_rounds",
        "already_expanded_k",
        "already_checked_distribution",
        "already_checked_group",
        "already_checked_active_candidate_set",
        "already_rescued_candidate_set",
        "already_requested_group_mode",
        "oscillation_detected",
        "checked_candidates",
        "p_top_history",
        "p_reliability_history",
    ],
    "diagnostics": [
        "benchmark_mode",
        "cumulative_rejection_rate",
        "family_exhaustion_count",
        "evidence_ceiling",
        "top3_all_rejected",
        "no_supported_canonical_candidate",
    ],
}


def build_clustering_prompt(
    rule_snapshot: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    successes: list[dict[str, Any]],
    prior_changes: list[dict[str, Any]] | None = None,
    result_change_summary: dict[str, Any] | None = None,
) -> str:
    return f"""You are analyzing systematic failure patterns in a multi-agent system for drug MoA prediction.
Cluster failures by root cause. Do not suggest fixes yet.

SYSTEM ARCHITECTURE
{ARCHITECTURE}

CURRENT CONTROLLER RULES
{json.dumps(rule_snapshot, indent=2)}

PRIOR CHANGES (optional context, not a prescribed iteration scheme)
{json.dumps(prior_changes or [], indent=2)}

RESULT CHANGE SUMMARY
{json.dumps(result_change_summary or {}, indent=2)}

FAILURE TRAJECTORIES
{json.dumps(failures, indent=2)}

SUCCESS TRAJECTORIES FOR CONTRAST
{json.dumps(successes, indent=2)}

Assign each failure to at most one cluster. Treat `moa_novel` failures separately
from closed-set failures even when surface trajectories look similar. Clusters with
frequency below 3 should be merged or marked as unsystematic one-offs. Cite specific
trajectory observations.

Return valid JSON:
{{
  "clusters": [{{
    "cluster_id": "...",
    "description": "...",
    "example_query_ids": ["..."],
    "frequency": 3,
    "distinguishing_signature": {{
      "p_pattern": "...", "m_pattern": "...",
      "controller_pattern": "...", "memory_pattern": "..."
    }},
    "why_current_rules_dont_handle": "..."
  }}],
  "unsystematic_one_offs": [],
  "overall_failure_summary": "..."
}}"""


def build_rule_generation_prompt(
    cluster: FailureCluster,
    rule_snapshot: list[dict[str, Any]],
    positive_examples: list[dict[str, Any]],
    negative_examples: list[dict[str, Any]],
) -> str:
    return f"""Generate controller-rule candidates for one systematic failure cluster.

SYSTEM ARCHITECTURE
{ARCHITECTURE}

EXISTING RULES
{json.dumps(rule_snapshot, indent=2)}

AVAILABLE SIGNALS
{json.dumps(SIGNAL_CATALOG, indent=2)}

TARGET CLUSTER
{json.dumps(cluster.__dict__, indent=2)}

POSITIVE EXAMPLES
{json.dumps(positive_examples, indent=2)}

NEGATIVE SUCCESS EXAMPLES
{json.dumps(negative_examples, indent=2)}

Propose 1-3 rules. Conditions may only use paths from the signal catalog and
operators eq, ne, in, not_in, contains, gt, gte, lt, lte, truthy, or falsy.
Actions must come from the architecture action list. Dynamic action parameters may
reference a signal with `$`, for example `$m.mechanistic_candidate`. Lower numeric
priority fires first. Explicitly distinguish positive from negative examples.

Return valid JSON:
{{
  "controller_rules": [{{
    "rule_id": "...", "priority": 100, "description": "...",
    "conditions": [{{"path": "p.retrieval_reliability", "op": "eq", "value": "low"}}],
    "action": {{"type": "EXPAND_K", "parameters": {{"next_k": 20}}, "reason": "..."}},
    "tags": ["..."], "enabled": true,
    "rationale": "...", "trigger_precision_argument": "...", "priority_argument": "..."
  }}],
  "comments": "..."
}}"""


def build_arbiter_generation_prompt(
    cluster: FailureCluster,
    base_heuristics: str,
    positive_examples: list[dict[str, Any]],
    negative_examples: list[dict[str, Any]],
) -> str:
    return f"""Propose a replacement for the mutable decision-heuristics section of
the Arbiter prompt. The role definition, input contract, and output schema are fixed.

CURRENT DECISION HEURISTICS
{base_heuristics}

TARGET CLUSTER
{json.dumps(cluster.__dict__, indent=2)}

POSITIVE EXAMPLES
{json.dumps(positive_examples, indent=2)}

NEGATIVE SUCCESS EXAMPLES
{json.dumps(negative_examples, indent=2)}

Return valid JSON:
{{
  "arbiter_heuristics": [{{
    "candidate_id": "...",
    "decision_heuristics": "...",
    "rationale": "...",
    "safety_notes": "..."
  }}],
  "comments": "..."
}}"""
