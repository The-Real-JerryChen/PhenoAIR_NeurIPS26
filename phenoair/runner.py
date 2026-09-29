from __future__ import annotations

import json

from .config import RunConfig
from .engine import PhenoAIR


def run(config: RunConfig) -> list[dict]:
    """Load configured components, run selected cases, and save run metadata."""
    system = PhenoAIR.from_config(config)
    cases = system.dataset.select(config.selected_ids, config.max_examples)
    results = system.run_cases(
        cases,
        output_dir=config.output_dir,
        max_workers=config.max_workers,
        force=config.force,
    )
    metadata = {
        "level": config.level.value,
        "model": config.model.backbone.value,
        "num_requested_cases": len(cases),
        "num_records": len(results),
        "max_refine_rounds": config.max_refine_rounds,
        "tool_cache": system.cache_store.stats() if system.cache_store else None,
        "controller_rules": system.registry.snapshot(enabled_only=False),
        "custom_rules_file": str(config.rules_file) if config.rules_file else None,
        "custom_arbiter_heuristics_file": (
            str(config.arbiter_heuristics_file) if config.arbiter_heuristics_file else None
        ),
    }
    (config.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return results
