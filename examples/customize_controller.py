"""Minimal example: modify rules before constructing the shared engine."""

import os
from pathlib import Path

from phenoair.config import DatasetLevel, ModelBackbone, ModelConfig, RunConfig
from phenoair.controller import build_default_registry, rule_from_spec
from phenoair.engine import PhenoAIR


registry = build_default_registry()
registry.remove("inspect_distribution_once")
registry.add(
    rule_from_spec(
        {
            "rule_id": "custom_medium_reliability_check",
            "priority": 160,
            "conditions": [
                {"path": "p.retrieval_reliability", "op": "eq", "value": "medium"},
                {"path": "state.already_checked_distribution", "op": "eq", "value": False},
            ],
            "action": {
                "type": "GET_BROADER_SUMMARY",
                "parameters": {"max_k": 50},
                "reason": "custom medium-reliability check",
            },
        },
        source="user",
    )
)

config = RunConfig(
    level=DatasetLevel.NOVEL,
    model=ModelConfig(
        backbone=ModelBackbone.GPT,
        azure_endpoint=os.environ["AZURE_ENDPOINT"],
        azure_deployment=os.environ["AZURE_DEPLOYMENT"],
    ),
    data_dir=Path("data"),
    output_dir=Path("outputs/custom"),
    max_examples=1,
)
system = PhenoAIR.from_config(config, registry=registry)
system.run_cases(system.dataset.select(limit=1), config.output_dir, max_workers=1)
