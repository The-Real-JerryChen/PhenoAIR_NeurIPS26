"""Minimal example: generate a proposal without imposing evolution rounds."""

import json
import os
from pathlib import Path

from phenoair.config import ModelBackbone, ModelConfig
from phenoair.controller import build_default_registry
from phenoair.evolution import EvolutionEngine, FailureCluster
from phenoair.providers import create_llm_client


records = [json.loads(line) for line in Path("predictions.jsonl").read_text().splitlines() if line]
failures = [
    item
    for item in records
    if item.get("prediction", {}).get("primary_moa")
    not in item.get("ground_truth", {}).get("all_valid_moas", [])
]
successes = [item for item in records if item not in failures]

client = create_llm_client(
    ModelConfig(
        backbone=ModelBackbone.GPT,
        azure_endpoint=os.environ["AZURE_ENDPOINT"],
        azure_deployment=os.environ["AZURE_DEPLOYMENT"],
    )
)
registry = build_default_registry()
evolver = EvolutionEngine(client, registry)

analysis = evolver.analyze_failures(failures, successes)
cluster = FailureCluster.from_dict(analysis["clusters"][0])
positive_ids = set(cluster.example_query_ids)
positives = [item for item in failures if item["id"] in positive_ids]
proposal = evolver.propose_controller_rules(cluster, positives, successes)
proposal.save(Path("proposal.json"))

# Deliberately explicit: inspect proposal.json before selecting and applying rules.

