import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from phenoair.cache import CachePolicy, JsonCacheStore
from phenoair.agents import ToolCallingAgent
from phenoair.cli import build_parser
from phenoair.config import DatasetLevel, ModelBackbone, ModelConfig, RunConfig
from phenoair.controller import ControllerContext, ControllerState, build_default_registry, rule_from_spec
from phenoair.data import MoADataset, ProfileStore
from phenoair.engine import PhenoAIR
from phenoair.evolution.models import EvolutionProposal
from phenoair.memory import CandidateMemory
from phenoair.models import (
    ArbiterOutput,
    Candidate,
    CandidateAssessment,
    CandidateEvidence,
    MechanismOutput,
    MoACase,
    PhenotypeOutput,
    ReferenceCompound,
)
from phenoair.reference_memory import load_reference_memory


class ReleaseConfigTests(unittest.TestCase):
    def test_cli_paths_are_relative_to_the_run_root(self) -> None:
        args = build_parser().parse_args(["--level", "novel", "--model", "gpt"])
        self.assertEqual(args.data_dir, Path("data"))
        self.assertEqual(args.output_dir, Path("outputs"))
        self.assertEqual(args.cache_dir, Path(".cache/phenoair"))
        self.assertEqual(args.cache_policy, "read-write")

    def test_level_aliases(self) -> None:
        self.assertIs(DatasetLevel.parse("verified"), DatasetLevel.VERIFIED)
        self.assertIs(DatasetLevel.parse("extend"), DatasetLevel.EXTENDED)
        self.assertIs(DatasetLevel.parse("novel"), DatasetLevel.NOVEL)

    def test_config_is_provider_and_level_parametric(self) -> None:
        config = RunConfig(
            level=DatasetLevel.NOVEL,
            model=ModelConfig(ModelBackbone.GPT, azure_endpoint="https://example.test"),
            data_dir=Path("data"),
            output_dir=Path("outputs"),
        )
        self.assertTrue(config.level.allows_novel)
        self.assertEqual(config.model.backbone, ModelBackbone.GPT)


class CacheTests(unittest.TestCase):
    def test_json_cache_is_content_addressed_and_versioned(self) -> None:
        with TemporaryDirectory() as directory:
            store = JsonCacheStore(Path(directory), CachePolicy.READ_WRITE)
            key = {"dataset": "abc", "query": "C1", "k": 10}
            value = [{"rank": 1, "compound_id": "R1", "similarity": 0.8}]
            store.set("neighbors", "tool-v1", key, value)
            self.assertEqual(store.get("neighbors", "tool-v1", key), value)
            self.assertIsNone(store.get("neighbors", "tool-v2", key))
            self.assertEqual(store.stats()["writes"], 1)
            self.assertEqual(store.stats()["hits"], 1)

    def test_read_only_cache_does_not_write(self) -> None:
        with TemporaryDirectory() as directory:
            store = JsonCacheStore(Path(directory), CachePolicy.READ_ONLY)
            store.set("neighbors", "tool-v1", {"query": "C1"}, [1])
            self.assertFalse(any(Path(directory).rglob("*.json")))

    def test_cache_rejects_path_segments_that_escape_the_root(self) -> None:
        with TemporaryDirectory() as directory:
            store = JsonCacheStore(Path(directory), CachePolicy.READ_WRITE)
            with self.assertRaises(ValueError):
                store.set("../outside", "tool-v1", {"query": "C1"}, [1])
            with self.assertRaises(ValueError):
                store.get("neighbors", "../../tool-v1", {"query": "C1"})


class ToolCallingTests(unittest.TestCase):
    def test_claude_forced_submission_keeps_tool_choice_auto(self) -> None:
        class ClaudeStub:
            model = "anthropic/claude-sonnet-4"

            def __init__(self) -> None:
                self.tool_choices = []

            def chat_raw(self, messages, **kwargs):
                self.tool_choices.append(kwargs["tool_choice"])
                if len(self.tool_choices) == 1:
                    message = SimpleNamespace(content="", tool_calls=[])
                else:
                    call = SimpleNamespace(
                        id="submit-1",
                        function=SimpleNamespace(
                            name="submit_result",
                            arguments=json.dumps({"value": "ok"}),
                        ),
                    )
                    message = SimpleNamespace(content="", tool_calls=[call])
                return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        client = ClaudeStub()
        tool = {
            "type": "function",
            "function": {
                "name": "submit_result",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        result, _ = ToolCallingAgent(client, max_turns=1).run(
            [{"role": "user", "content": "test"}],
            [tool],
            "submit_result",
            lambda name, arguments: "unused",
        )
        self.assertEqual(result, {"value": "ok"})
        self.assertEqual(client.tool_choices, ["auto", "auto"])


class ReferenceMemoryTests(unittest.TestCase):
    def test_released_memory_counts_and_alias(self) -> None:
        verified = load_reference_memory(DatasetLevel.VERIFIED)
        extended = load_reference_memory(DatasetLevel.EXTENDED)
        novel = load_reference_memory(DatasetLevel.NOVEL)

        self.assertEqual(len(verified.sources), 5)
        self.assertEqual(len(verified.moas), 26)
        self.assertEqual(len(verified.references), 63)
        self.assertEqual(len(verified.pair_confusions), 28)
        self.assertEqual(len(extended.sources), 5)
        self.assertEqual(len(extended.moas), 119)
        self.assertEqual(len(extended.references), 533)
        self.assertEqual(len(extended.pair_confusions), 302)
        self.assertIs(novel.source_level, DatasetLevel.VERIFIED)
        self.assertEqual(novel.asset_fingerprint, verified.asset_fingerprint)

    def test_memory_lookup(self) -> None:
        memory = load_reference_memory("verified")
        self.assertEqual(memory.moa("RAF inhibitor")["risk_level"], "high")
        self.assertIsNotNone(memory.reference("GNF-5"))
        self.assertIsNotNone(
            memory.pair("sodium channel blocker", "potassium channel blocker")
        )


class RegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = ControllerContext(
            phenotype=PhenotypeOutput(
                ranked_candidates=[Candidate("example inhibitor", 0.7)],
                retrieval_reliability="low",
            ),
            mechanism=MechanismOutput(),
            memory=CandidateMemory(),
            state=ControllerState(max_rounds=4),
        )

    def test_registry_supports_remove_and_filter(self) -> None:
        registry = build_default_registry()
        removed = registry.remove("stop_default")
        self.assertEqual(removed.rule_id, "stop_default")
        self.assertNotIn("stop_default", [item["rule_id"] for item in registry.snapshot()])
        self.assertTrue(all(rule.source == "hand-crafted" for rule in registry.iter_rules(source="hand-crafted")))

    def test_structured_rule_can_be_applied_without_round_layout(self) -> None:
        spec = {
            "rule_id": "custom_low_reliability_stop",
            "priority": 1,
            "conditions": [
                {"path": "p.retrieval_reliability", "op": "eq", "value": "low"}
            ],
            "action": {"type": "STOP", "reason": "custom test", "parameters": {}},
        }
        registry = build_default_registry()
        registry.add(rule_from_spec(spec, source="user"))
        decision = registry.decide(self.context)
        self.assertEqual(decision.rule_id, "custom_low_reliability_stop")
        self.assertEqual(decision.source, "user")

    def test_evolution_proposal_requires_explicit_apply(self) -> None:
        registry = build_default_registry()
        before = len(registry.snapshot())
        proposal = EvolutionProposal(
            controller_rules=[
                {
                    "rule_id": "proposed_rule",
                    "priority": 2,
                    "conditions": [{"path": "state.round_idx", "op": "gte", "value": 1}],
                    "action": {"type": "STOP", "parameters": {}, "reason": "proposal"},
                }
            ]
        )
        self.assertEqual(len(registry.snapshot()), before)
        proposal.apply_rules(registry)
        self.assertEqual(len(registry.snapshot()), before + 1)


class EngineCompositionTests(unittest.TestCase):
    def test_engine_accepts_replaceable_agents(self) -> None:
        class PhenotypeStub:
            def run(self, *args, **kwargs):
                return PhenotypeOutput(
                    [Candidate("CDK inhibitor", 0.9)],
                    "high",
                    [],
                    [CandidateEvidence("CDK inhibitor", "strong", "none", "coherent")],
                    "coherent phenotype",
                ), []

        class MechanismStub:
            def run(self, *args, **kwargs):
                return MechanismOutput(
                    [CandidateAssessment("CDK inhibitor", "support", "strong")],
                    overall_agreement_with_p="high",
                ), []

        class ArbiterStub:
            def run(self, *args, **kwargs):
                return ArbiterOutput("CDK inhibitor", [], "high", "P/M agreement"), []

        case = MoACase("T1", "C1", "CC", "IK", "CDK inhibitor", ("CDK inhibitor",))
        reference = ReferenceCompound("R1", "CC", "IR", "CDK inhibitor", ("CDK inhibitor",))
        profiles = ProfileStore(
            ("C1", "R1"),
            np.ones((2, 2), dtype=np.float32),
            {"C1": 0, "R1": 1},
        )
        dataset = MoADataset(
            DatasetLevel.VERIFIED,
            [case],
            [reference],
            ("CDK inhibitor",),
            profiles,
        )
        system = PhenoAIR(
            dataset,
            PhenotypeStub(),
            MechanismStub(),
            ArbiterStub(),
            build_default_registry(),
        )
        result = system.predict(case)
        self.assertEqual(result["prediction"]["primary_moa"], "CDK inhibitor")
        self.assertEqual(result["controller_rules"], ["stop_high_agreement"])


class ResumeTests(unittest.TestCase):
    def test_resume_retries_errors_and_returns_only_requested_cases(self) -> None:
        class RunnerStub:
            _load_jsonl = staticmethod(PhenoAIR._load_jsonl)

            def __init__(self) -> None:
                self.calls = 0

            def predict(self, case):
                self.calls += 1
                return {"id": case.case_id, "prediction": {"primary_moa": "CDK inhibitor"}}

        case = MoACase("T1", "C1", "CC", "IK", "CDK inhibitor", ("CDK inhibitor",))
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            jsonl_path = output_dir / "predictions.jsonl"
            jsonl_path.write_text(
                "\n".join(
                    [
                        json.dumps({"id": "T1", "compound_id": "C1", "error": "temporary"}),
                        json.dumps({"id": "T2", "prediction": {"primary_moa": "other"}}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            runner = RunnerStub()
            first = PhenoAIR.run_cases(runner, [case], output_dir, max_workers=1)
            self.assertEqual(runner.calls, 1)
            self.assertEqual([item["id"] for item in first], ["T1"])
            self.assertNotIn("error", first[0])
            compacted = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
            self.assertEqual([item["id"] for item in compacted], ["T1", "T2"])
            self.assertNotIn("error", compacted[0])

            second = PhenoAIR.run_cases(runner, [case], output_dir, max_workers=1)
            self.assertEqual(runner.calls, 1)
            self.assertEqual(second, first)


if __name__ == "__main__":
    unittest.main()
