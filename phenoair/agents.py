from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable

from .llm_client import LLMClient
from .memory import CandidateMemory
from .models import ArbiterOutput, ControllerDecision, MechanismOutput, MoACase, PhenotypeOutput, TrajectoryStep
from .prompts import (
    MECHANISM_SYSTEM_PROMPT,
    PHENOTYPE_SYSTEM_PROMPT,
    build_arbiter_system_prompt,
)
from .retrieval import CellClipRetriever, StructureRetriever


ToolDispatcher = Callable[[str, dict[str, Any]], str]


def _function_tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


class ToolCallingAgent:
    """Provider-neutral tool loop shared by P, M, and A."""

    def __init__(self, client: LLMClient, max_turns: int = 8) -> None:
        self.client = client
        self.max_turns = max_turns

    @property
    def is_claude(self) -> bool:
        return self.client.model.startswith("anthropic/")

    def run(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        submit_tool: str,
        dispatcher: ToolDispatcher,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        log: list[dict[str, Any]] = []
        submit_only = [tool for tool in tools if tool["function"]["name"] == submit_tool]
        for turn in range(self.max_turns):
            response = self._call(messages, tools, "auto", turn)
            message = response.choices[0].message
            tool_calls = message.tool_calls or []
            assistant = {"role": "assistant", "content": message.content}
            if tool_calls:
                assistant["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in tool_calls
                ]
            messages.append(assistant)
            for call in tool_calls:
                try:
                    arguments = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                if call.function.name == submit_tool:
                    log.append({"tool": submit_tool, "args": arguments, "result": "submitted"})
                    return arguments, log
                result = dispatcher(call.function.name, arguments)
                log.append({"tool": call.function.name, "args": arguments, "result": result[:500]})
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
            if not tool_calls:
                break

        messages.append({"role": "user", "content": f"Submit now using `{submit_tool}`."})
        choices = ["auto"] if self.is_claude else ["required", "auto"]
        for choice in choices:
            response = self._call(messages, submit_only, choice, self.max_turns)
            calls = response.choices[0].message.tool_calls or []
            call = next((item for item in calls if item.function.name == submit_tool), None)
            if call is None:
                continue
            try:
                result = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                result = {}
            log.append({"tool": submit_tool, "args": result, "result": f"forced submit ({choice})"})
            return result, log
        return {}, log

    def _call(self, messages: list[dict], tools: list[dict], tool_choice: str, attempt: int):
        try:
            return self.client.chat_raw(messages, tools=tools, tool_choice=tool_choice)
        except Exception:
            if attempt >= 2:
                raise
            time.sleep(2 ** attempt)
            return self.client.chat_raw(messages, tools=tools, tool_choice=tool_choice)


_SUBMIT_P = _function_tool(
    "submit_phenotype_output",
    "Submit the phenotype analysis.",
    {
        "ranked_candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"moa": {"type": "string"}, "score": {"type": "number"}},
                "required": ["moa", "score"],
            },
        },
        "retrieval_reliability": {"type": "string", "enum": ["high", "medium", "low"]},
        "flags": {"type": "array", "items": {"type": "string"}},
        "candidate_evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "moa": {"type": "string"},
                    "phenotype_support": {"type": "string", "enum": ["strong", "moderate", "weak", "none"]},
                    "phenotype_contradiction": {"type": "string", "enum": ["strong", "weak", "none"]},
                    "evidence": {"type": "string"},
                },
                "required": ["moa", "phenotype_support", "phenotype_contradiction", "evidence"],
            },
        },
        "phenotype_summary": {"type": "string"},
    },
    ["ranked_candidates", "retrieval_reliability", "flags", "candidate_evidence", "phenotype_summary"],
)


_SUBMIT_M = _function_tool(
    "submit_mechanism_output",
    "Submit the mechanism analysis.",
    {
        "candidate_assessment": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "moa": {"type": "string"},
                    "verdict": {"type": "string", "enum": ["support", "reject", "uncertain"]},
                    "verdict_strength": {"type": "string", "enum": ["strong", "moderate", "weak"]},
                    "suppression_recommendation": {
                        "type": "string",
                        "enum": ["keep_active", "suppress_if_P_uncertain", "suppress_now", "needs_more_phenotype_check"],
                    },
                    "rationale": {"type": "string"},
                },
                "required": ["moa", "verdict", "verdict_strength", "suppression_recommendation", "rationale"],
            },
        },
        "mechanistic_candidate": {
            "anyOf": [
                {"type": "null"},
                {
                    "type": "object",
                    "properties": {
                        "moa": {"type": "string"},
                        "strength": {"type": "string", "enum": ["strong", "moderate", "weak"]},
                        "rationale": {"type": "string"},
                    },
                    "required": ["moa", "strength", "rationale"],
                },
            ]
        },
        "mechanistic_candidate_group": {
            "anyOf": [
                {"type": "null"},
                {
                    "type": "object",
                    "properties": {
                        "family_name": {"type": "string"},
                        "members": {"type": "array", "items": {"type": "string"}},
                        "strength": {"type": "string", "enum": ["strong", "moderate", "weak"]},
                    },
                    "required": ["family_name", "members", "strength"],
                },
            ]
        },
        "overall_agreement_with_P": {"type": "string", "enum": ["high", "medium", "low"]},
        "mechanism_summary": {"type": "string"},
    },
    ["candidate_assessment", "overall_agreement_with_P", "mechanism_summary"],
)


_SUBMIT_A = _function_tool(
    "submit_final_prediction",
    "Submit the final MOA prediction.",
    {
        "primary_moa": {"type": "string"},
        "other_possible_moas": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "reasoning": {"type": "string"},
    },
    ["primary_moa", "other_possible_moas", "confidence", "reasoning"],
)


@dataclass
class PhenotypeAgent:
    loop: ToolCallingAgent
    retriever: CellClipRetriever
    system_prompt: str = PHENOTYPE_SYSTEM_PROMPT
    initial_k: int = 10
    broader_k: int = 50

    def run(
        self,
        case: MoACase,
        memory: CandidateMemory,
        round_idx: int,
        allowed_labels: tuple[str, ...] = (),
        decision: ControllerDecision | None = None,
    ) -> tuple[PhenotypeOutput, list[dict[str, Any]]]:
        initial_neighbors = self.retriever.neighbors(case, self.initial_k)
        prompt = {
            "query_smiles": case.smiles,
            "round": round_idx,
            "controller_request": decision.to_dict() if decision else None,
            "allowed_labels": list(allowed_labels),
            "candidate_memory": memory.summary(),
            "initial_neighbors": [item.__dict__ | {"distance": item.distance} for item in initial_neighbors],
        }

        def dispatch(name: str, arguments: dict[str, Any]) -> str:
            if name == "get_knn_neighbors":
                items = self.retriever.neighbors(case, int(arguments.get("k", self.initial_k)))
                return json.dumps([item.__dict__ | {"distance": item.distance} for item in items], indent=2)
            if name == "get_moa_distribution":
                counts = self.retriever.distribution(case, int(arguments.get("max_k", self.broader_k)))
                return json.dumps(dict(counts.most_common()), indent=2)
            if name == "check_candidate_in_neighborhood":
                return json.dumps(
                    self.retriever.candidate_support(
                        case,
                        str(arguments.get("moa", "")),
                        int(arguments.get("max_k", self.broader_k)),
                    ),
                    indent=2,
                )
            if name == "check_group_in_neighborhood":
                labels = [str(item) for item in arguments.get("labels", [])]
                return json.dumps(
                    [self.retriever.candidate_support(case, label, self.broader_k) for label in labels],
                    indent=2,
                )
            return f"Unknown phenotype tool: {name}"

        tools = [
            _function_tool(
                "get_knn_neighbors",
                "Retrieve Cell Painting nearest neighbors.",
                {"k": {"type": "integer", "enum": [5, 10, 20, 50]}},
                ["k"],
            ),
            _function_tool(
                "get_moa_distribution",
                "Summarize MOA frequencies in a broader neighborhood.",
                {"max_k": {"type": "integer", "enum": [20, 50, 100]}},
                ["max_k"],
            ),
            _function_tool(
                "check_candidate_in_neighborhood",
                "Measure phenotype support for one candidate.",
                {"moa": {"type": "string"}, "max_k": {"type": "integer", "enum": [20, 50, 100]}},
                ["moa"],
            ),
            _function_tool(
                "check_group_in_neighborhood",
                "Compare a constrained group of candidate labels.",
                {"labels": {"type": "array", "items": {"type": "string"}}},
                ["labels"],
            ),
            _SUBMIT_P,
        ]
        result, log = self.loop.run(
            [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": json.dumps(prompt, indent=2)}],
            tools,
            "submit_phenotype_output",
            dispatch,
        )
        return PhenotypeOutput.from_dict(result), log


@dataclass
class MechanismAgent:
    loop: ToolCallingAgent
    retriever: StructureRetriever
    system_prompt: str = MECHANISM_SYSTEM_PROMPT

    def run(
        self,
        case: MoACase,
        phenotype: PhenotypeOutput,
        memory: CandidateMemory,
        round_idx: int,
        label_vocabulary: tuple[str, ...] = (),
        decision: ControllerDecision | None = None,
    ) -> tuple[MechanismOutput, list[dict[str, Any]]]:
        prompt = {
            "query_smiles": case.smiles,
            "round": round_idx,
            "phenotype_output": phenotype.to_dict(),
            "candidate_memory": memory.summary(),
            "canonical_label_vocabulary": list(label_vocabulary),
            "controller_request": decision.to_dict() if decision else None,
        }

        def dispatch(name: str, arguments: dict[str, Any]) -> str:
            if name == "get_structure_neighbors":
                items = self.retriever.neighbors(case, int(arguments.get("k", 10)))
                return json.dumps([item.__dict__ for item in items], indent=2)
            if name == "get_compound_properties":
                return json.dumps(self.retriever.properties(case.smiles), indent=2)
            return f"Unknown mechanism tool: {name}"

        tools = [
            _function_tool(
                "get_structure_neighbors",
                "Retrieve Morgan-fingerprint structural nearest neighbors.",
                {"k": {"type": "integer", "enum": [5, 10, 20]}},
                ["k"],
            ),
            _function_tool("get_compound_properties", "Calculate RDKit molecular descriptors.", {}, []),
            _SUBMIT_M,
        ]
        result, log = self.loop.run(
            [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": json.dumps(prompt, indent=2)}],
            tools,
            "submit_mechanism_output",
            dispatch,
        )
        return MechanismOutput.from_dict(result), log


@dataclass
class ArbiterAgent:
    loop: ToolCallingAgent
    decision_heuristics: str | None = None

    @property
    def system_prompt(self) -> str:
        return build_arbiter_system_prompt(self.decision_heuristics)

    def run(
        self,
        case: MoACase,
        trajectory: list[TrajectoryStep],
        memory: CandidateMemory,
        allowed_labels: tuple[str, ...],
        novel_allowed: bool,
    ) -> tuple[ArbiterOutput, list[dict[str, Any]]]:
        prompt = {
            "query_smiles": case.smiles,
            "novel_allowed": novel_allowed,
            "allowed_labels": list(allowed_labels),
            "trajectory": [step.to_dict() for step in trajectory],
            "final_candidate_memory": memory.to_dict(),
        }
        result, log = self.loop.run(
            [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": json.dumps(prompt, indent=2)}],
            [_SUBMIT_A],
            "submit_final_prediction",
            lambda name, arguments: f"Unknown Arbiter tool: {name}",
        )
        return ArbiterOutput.from_dict(result), log
