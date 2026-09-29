from __future__ import annotations

from typing import Any


def _prediction_label(record: dict[str, Any]) -> str:
    return str((record.get("prediction") or {}).get("primary_moa", ""))


def _top_assessments(mechanism: dict[str, Any], labels: list[str]) -> list[dict[str, Any]]:
    by_label = {
        str(item.get("moa", item.get("candidate", ""))): item
        for item in mechanism.get("candidate_assessment", []) or []
    }
    return [by_label[label] for label in labels if label in by_label]


def compress_failure(record: dict[str, Any], max_rounds: int = 4) -> dict[str, Any]:
    """Produce the information-dense failure skeleton used by evolution prompts."""
    trajectory = record.get("trajectory", []) or []
    if len(trajectory) > max_rounds:
        selected = [trajectory[0], *trajectory[-(max_rounds - 1) :]]
    else:
        selected = trajectory
    rounds = []
    for step in selected:
        phenotype = step.get("P", {}) or {}
        mechanism = step.get("M", {}) or {}
        top3 = (phenotype.get("ranked_candidates", []) or [])[:3]
        labels = [str(item.get("moa", "")) for item in top3]
        rounds.append(
            {
                "round": step.get("round"),
                "p_top3": top3,
                "p_reliability": phenotype.get("retrieval_reliability"),
                "p_flags": phenotype.get("flags", []),
                "m_assessments_on_p_top3": _top_assessments(mechanism, labels),
                "m_mechanistic_candidate": mechanism.get("mechanistic_candidate"),
                "m_mechanistic_group": mechanism.get("mechanistic_candidate_group"),
                "controller": step.get("C", {}),
            }
        )
    memory = record.get("candidate_memory", {}) or {}
    return {
        "query_id": record.get("id"),
        "level": record.get("level"),
        "query_smiles": record.get("query_smiles"),
        "ground_truth": (record.get("ground_truth") or {}).get("primary_moa"),
        "predicted": _prediction_label(record),
        "rounds": rounds,
        "final_memory": {
            "active": [item for item in memory.values() if item.get("status") == "active"],
            "suppressed": [item for item in memory.values() if str(item.get("status", "")).startswith("suppressed")],
            "archived": [item for item in memory.values() if item.get("status") == "archived"],
            "ground_truth_status": memory.get(
                (record.get("ground_truth") or {}).get("primary_moa"),
                "never proposed",
            ),
        },
        "arbiter_output": record.get("prediction", {}),
    }


def compress_success(record: dict[str, Any]) -> dict[str, Any]:
    trajectory = record.get("trajectory", []) or []
    final_step = trajectory[-1] if trajectory else {}
    phenotype = final_step.get("P", {}) or {}
    mechanism = final_step.get("M", {}) or {}
    top1 = ((phenotype.get("ranked_candidates", []) or [{}])[0]).get("moa")
    top_assessment = _top_assessments(mechanism, [str(top1)])
    return {
        "query_id": record.get("id"),
        "level": record.get("level"),
        "ground_truth": (record.get("ground_truth") or {}).get("primary_moa"),
        "predicted": _prediction_label(record),
        "p_top1": top1,
        "p_reliability": phenotype.get("retrieval_reliability"),
        "m_top1_assessment": top_assessment[0] if top_assessment else None,
        "num_refinement_rounds": max(len(trajectory) - 1, 0),
        "final_controller_rule": (final_step.get("C") or {}).get("rule_id"),
    }

