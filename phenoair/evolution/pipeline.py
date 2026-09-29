from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..controller import RuleRegistry
from ..llm_client import LLMClient
from ..prompts import DEFAULT_ARBITER_HEURISTICS
from .models import EvolutionProposal, FailureCluster
from .prompts import (
    build_arbiter_generation_prompt,
    build_clustering_prompt,
    build_rule_generation_prompt,
)
from .trajectories import compress_failure, compress_success


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(text)


@dataclass
class EvolutionEngine:
    """Two-step evolution helper without an imposed round or retention policy."""

    client: LLMClient
    registry: RuleRegistry

    def analyze_failures(
        self,
        failures: list[dict[str, Any]],
        successes: list[dict[str, Any]],
        prior_changes: list[dict[str, Any]] | None = None,
        result_change_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        prompt = build_clustering_prompt(
            self.registry.snapshot(enabled_only=False),
            [compress_failure(item) for item in failures],
            [compress_success(item) for item in successes],
            prior_changes=prior_changes,
            result_change_summary=result_change_summary,
        )
        response = self.client.chat(
            [{"role": "user", "content": prompt}],
            json_mode=True,
            max_tokens=8000,
        )
        return _parse_json(response)

    def propose_controller_rules(
        self,
        cluster: FailureCluster,
        positive_examples: list[dict[str, Any]],
        negative_examples: list[dict[str, Any]],
    ) -> EvolutionProposal:
        prompt = build_rule_generation_prompt(
            cluster,
            self.registry.snapshot(enabled_only=False),
            [compress_failure(item) for item in positive_examples],
            [compress_success(item) for item in negative_examples],
        )
        response = self.client.chat(
            [{"role": "user", "content": prompt}],
            json_mode=True,
            max_tokens=8000,
        )
        value = _parse_json(response)
        return EvolutionProposal(
            controller_rules=list(value.get("controller_rules", value.get("proposed_rules", []))),
            comments=str(value.get("comments", "")),
            metadata={"cluster_id": cluster.cluster_id, "proposal_type": "controller"},
        )

    def propose_arbiter_heuristics(
        self,
        cluster: FailureCluster,
        positive_examples: list[dict[str, Any]],
        negative_examples: list[dict[str, Any]],
        base_heuristics: str = DEFAULT_ARBITER_HEURISTICS,
    ) -> EvolutionProposal:
        prompt = build_arbiter_generation_prompt(
            cluster,
            base_heuristics,
            [compress_failure(item) for item in positive_examples],
            [compress_success(item) for item in negative_examples],
        )
        response = self.client.chat(
            [{"role": "user", "content": prompt}],
            json_mode=True,
            max_tokens=8000,
        )
        value = _parse_json(response)
        return EvolutionProposal(
            arbiter_heuristics=list(value.get("arbiter_heuristics", [])),
            comments=str(value.get("comments", "")),
            metadata={"cluster_id": cluster.cluster_id, "proposal_type": "arbiter"},
        )

