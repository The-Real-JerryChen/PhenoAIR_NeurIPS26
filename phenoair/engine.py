from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

from .agents import ArbiterAgent, MechanismAgent, PhenotypeAgent, ToolCallingAgent
from .cache import JsonCacheStore
from .config import RunConfig
from .controller import ControllerContext, ControllerState, RuleRegistry, build_default_registry
from .data import MoADataset
from .memory import CandidateMemory
from .models import ArbiterOutput, MechanismOutput, MoACase, PhenotypeOutput, TrajectoryStep
from .providers import create_llm_client
from .retrieval import CellClipRetriever, StructureRetriever


@dataclass
class PhenoAIR:
    """Composable PhenoAIR orchestrator.

    Agents, retrievers, and the registry are constructor arguments so downstream
    users can replace any component without modifying the orchestration loop.
    """

    dataset: MoADataset
    phenotype_agent: PhenotypeAgent
    mechanism_agent: MechanismAgent
    arbiter_agent: ArbiterAgent
    registry: RuleRegistry
    cache_store: JsonCacheStore | None = None
    max_refine_rounds: int = 4

    @classmethod
    def from_config(
        cls,
        config: RunConfig,
        registry: RuleRegistry | None = None,
    ) -> "PhenoAIR":
        config.validate()
        dataset = MoADataset.load(config.data_dir, config.level)
        client = create_llm_client(config.model)
        cache_store = JsonCacheStore(config.cache_dir, config.cache_policy)
        registry = registry or build_default_registry()
        if config.rules_file:
            registry.load_specs(config.rules_file, source="user", replace=True)
        heuristics = (
            config.arbiter_heuristics_file.read_text(encoding="utf-8")
            if config.arbiter_heuristics_file
            else None
        )
        return cls(
            dataset=dataset,
            phenotype_agent=PhenotypeAgent(
                ToolCallingAgent(client, max_turns=config.max_agent_turns),
                CellClipRetriever(dataset, cache=cache_store),
                initial_k=config.initial_k,
                broader_k=config.broader_k,
            ),
            mechanism_agent=MechanismAgent(
                ToolCallingAgent(client, max_turns=config.max_agent_turns),
                StructureRetriever(dataset.references, cache=cache_store),
            ),
            arbiter_agent=ArbiterAgent(
                ToolCallingAgent(client, max_turns=min(config.max_agent_turns, 3)),
                decision_heuristics=heuristics,
            ),
            registry=registry,
            cache_store=cache_store,
            max_refine_rounds=config.max_refine_rounds,
        )

    @property
    def label_vocabulary(self) -> tuple[str, ...]:
        if self.dataset.allows_novel:
            return self.dataset.labels
        return tuple(sorted({reference.primary_moa for reference in self.dataset.references}))

    def predict(self, case: MoACase) -> dict[str, Any]:
        memory = CandidateMemory()
        state = ControllerState(
            max_rounds=self.max_refine_rounds,
            benchmark_mode="open_set" if self.dataset.allows_novel else "closed_set",
        )
        trajectory: list[TrajectoryStep] = []
        tool_log: list[dict[str, Any]] = []
        previous_decision = None

        while True:
            phenotype, p_log = self.phenotype_agent.run(
                case,
                memory,
                state.round_idx,
                allowed_labels=self.label_vocabulary if self.dataset.allows_novel else (),
                decision=previous_decision,
            )
            phenotype = self._sanitize_phenotype(phenotype)
            memory.update_phenotype(phenotype, state.round_idx)
            tool_log.extend({"agent": "P", "round": state.round_idx, **item} for item in p_log)

            mechanism, m_log = self.mechanism_agent.run(
                case,
                phenotype,
                memory,
                state.round_idx,
                label_vocabulary=self.label_vocabulary,
                decision=previous_decision,
            )
            mechanism = self._sanitize_mechanism(mechanism)
            memory.update_mechanism(mechanism, state.round_idx)
            state.mechanism_assessment_count += len(mechanism.candidate_assessment)
            state.mechanism_rejection_count += sum(
                item.verdict == "reject" for item in mechanism.candidate_assessment
            )
            tool_log.extend({"agent": "M", "round": state.round_idx, **item} for item in m_log)

            if phenotype.ranked_candidates:
                state.p_top_history.append(phenotype.ranked_candidates[0].moa)
            state.p_reliability_history.append(phenotype.retrieval_reliability)
            decision = self.registry.decide(ControllerContext(phenotype, mechanism, memory, state))
            trajectory.append(TrajectoryStep(state.round_idx, phenotype, mechanism, decision))

            if decision.action == "STOP":
                break
            self._update_state(state, decision)
            previous_decision = decision
            state.round_idx += 1

        prediction, a_log = self.arbiter_agent.run(
            case,
            trajectory,
            memory,
            allowed_labels=self.label_vocabulary,
            novel_allowed=self.dataset.allows_novel,
        )
        tool_log.extend({"agent": "A", "round": state.round_idx, **item} for item in a_log)
        prediction, validation = self._validate_prediction(prediction, memory)

        return {
            "id": case.case_id,
            "compound_id": case.compound_id,
            "query_smiles": case.smiles,
            "level": self.dataset.level.value,
            "is_novel": case.is_novel,
            "ground_truth": {
                "primary_moa": case.ground_truth,
                "all_valid_moas": list(case.valid_labels),
            },
            "prediction": prediction.to_dict(),
            "validation": validation,
            "trajectory": [step.to_dict() for step in trajectory],
            "candidate_memory": memory.to_dict(),
            "controller_rules": [step.controller.rule_id for step in trajectory],
            "tool_call_log": tool_log,
        }

    def _sanitize_phenotype(self, output: PhenotypeOutput) -> PhenotypeOutput:
        vocabulary = set(self.label_vocabulary)
        output.ranked_candidates = [item for item in output.ranked_candidates if item.moa in vocabulary]
        output.candidate_evidence = [item for item in output.candidate_evidence if item.moa in vocabulary]
        return output

    def _sanitize_mechanism(self, output: MechanismOutput) -> MechanismOutput:
        vocabulary = set(self.label_vocabulary)
        output.candidate_assessment = [item for item in output.candidate_assessment if item.moa in vocabulary]
        if output.mechanistic_candidate and output.mechanistic_candidate.moa not in vocabulary:
            output.mechanistic_candidate = None
        if output.mechanistic_candidate_group:
            output.mechanistic_candidate_group.members = [
                item for item in output.mechanistic_candidate_group.members if item in vocabulary
            ]
            if not output.mechanistic_candidate_group.members:
                output.mechanistic_candidate_group = None
        return output

    @staticmethod
    def _update_state(state: ControllerState, decision) -> None:
        if decision.action == "EXPAND_K":
            state.already_expanded_k = True
        elif decision.action == "GET_BROADER_SUMMARY":
            state.already_checked_distribution = True
        elif decision.action == "CHECK_CANDIDATE_IN_NEIGHBORHOOD":
            target = decision.parameters.get("target_candidate")
            if target:
                state.checked_candidates.add(str(target))
        elif decision.action == "CHECK_GROUP_IN_NEIGHBORHOOD":
            state.already_checked_group = True
            state.family_exhaustion_count += 1
        elif decision.action == "CHECK_ACTIVE_CANDIDATE_SET":
            state.already_checked_active_candidate_set = True
        elif decision.action == "RESCUE_CANDIDATE_SET":
            state.already_rescued_candidate_set = True
        elif decision.action == "REQUEST_GROUP_MODE_FROM_M":
            state.already_requested_group_mode = True

    def _validate_prediction(
        self,
        prediction: ArbiterOutput,
        memory: CandidateMemory,
    ) -> tuple[ArbiterOutput, dict[str, Any]]:
        vocabulary = set(self.label_vocabulary)
        primary = prediction.primary_moa
        best = memory.best()
        adequately_supported = any(
            entry.phenotype_support in {"moderate", "strong"}
            or (entry.phenotype_score > 0 and entry.mechanism_verdict == "support")
            for entry in memory.eligible()
        )
        fallback_reason = None

        if primary == "novel moa":
            if not self.dataset.allows_novel:
                fallback_reason = "novel_not_allowed"
            elif adequately_supported:
                fallback_reason = "supported_canonical_candidate_available"
            else:
                return prediction, {"fallback_triggered": False, "novel_gate_passed": True}
        elif primary not in vocabulary:
            fallback_reason = "label_outside_vocabulary"
        else:
            entry = memory.get(primary)
            if entry is None or entry.suppressed:
                fallback_reason = "candidate_not_visible_or_suppressed"

        if fallback_reason is None:
            prediction.other_possible_moas = [
                label for label in prediction.other_possible_moas if label in vocabulary and label != primary
            ]
            return prediction, {"fallback_triggered": False, "novel_gate_passed": None}

        if best is not None:
            prediction.primary_moa = best.moa
            prediction.other_possible_moas = [
                entry.moa for entry in memory.eligible()[1:4] if entry.moa != best.moa
            ]
            prediction.confidence = "low"
            prediction.reasoning = (
                prediction.reasoning
                + f" System fallback selected the strongest visible candidate ({fallback_reason})."
            ).strip()
        elif self.dataset.allows_novel:
            prediction.primary_moa = "novel moa"
            prediction.other_possible_moas = []
            prediction.confidence = "low"
        return prediction, {
            "fallback_triggered": True,
            "fallback_reason": fallback_reason,
            "novel_gate_passed": prediction.primary_moa == "novel moa",
        }

    def run_cases(
        self,
        cases: Iterable[MoACase],
        output_dir: Path,
        max_workers: int = 4,
        force: bool = False,
    ) -> list[dict[str, Any]]:
        cases = list(cases)
        output_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = output_dir / "predictions.jsonl"
        json_path = output_dir / "predictions.json"
        all_existing = self._load_jsonl(jsonl_path)
        requested_ids = {case.case_id for case in cases}
        existing = {} if force else {
            case_id: record
            for case_id, record in all_existing.items()
            if case_id in requested_ids
        }
        # Failed records are checkpoints, not completed predictions. Retrying them
        # by default makes interrupted/rate-limited runs resumable without --force.
        pending = [
            case
            for case in cases
            if case.case_id not in existing or "error" in existing[case.case_id]
        ]
        results_by_id = dict(existing)
        write_lock = threading.Lock()

        def predict_and_write(case: MoACase) -> dict[str, Any]:
            result = self.predict(case)
            with write_lock, jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            return result

        if pending:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(predict_and_write, case): case for case in pending}
                for future in tqdm(as_completed(futures), total=len(futures), desc="phenoair"):
                    case = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {
                            "id": case.case_id,
                            "compound_id": case.compound_id,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                        with write_lock, jsonl_path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    results_by_id[case.case_id] = result
        results = list(results_by_id.values())
        results.sort(key=lambda item: item["id"])
        # The append-only writes above are crash-tolerant. Once the run completes,
        # compact them atomically so consumers see exactly one latest row per ID.
        latest_records = self._load_jsonl(jsonl_path)
        latest_records.update(results_by_id)
        temporary_jsonl = jsonl_path.with_suffix(".jsonl.tmp")
        with temporary_jsonl.open("w", encoding="utf-8") as handle:
            for item in sorted(latest_records.values(), key=lambda value: value["id"]):
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        temporary_jsonl.replace(jsonl_path)
        json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        return results

    @staticmethod
    def _load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
        if not path.exists():
            return {}
        result = {}
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    item = json.loads(line)
                    result[item["id"]] = item
        return result
