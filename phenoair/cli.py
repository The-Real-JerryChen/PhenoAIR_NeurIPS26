from __future__ import annotations

import argparse
import os
from pathlib import Path

from .cache import CachePolicy
from .config import DatasetLevel, ModelBackbone, ModelConfig, RunConfig
from .runner import run


# Relative defaults intentionally resolve from the user's working directory.
# This keeps the cloned code repository, rather than its parent, as the run root
# and continues to work when phenoair is installed as a wheel.
DEFAULT_DATA_DIR = Path("data")
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_CACHE_DIR = Path(".cache/phenoair")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PhenoAIR MoA inference.")
    parser.add_argument(
        "--level",
        required=True,
        choices=["moa_verified", "moa_extended", "moa_novel", "verified", "extended", "novel"],
    )
    parser.add_argument("--model", required=True, choices=["gpt", "claude"])
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--cache-policy",
        choices=[policy.value for policy in CachePolicy],
        default=CachePolicy.READ_WRITE.value,
        help="Local deterministic-tool cache policy (default: read-write).",
    )
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--selected-id", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-agent-turns", type=int, default=8)
    parser.add_argument("--max-refine-rounds", type=int, default=4)
    parser.add_argument("--initial-k", type=int, default=10)
    parser.add_argument("--broader-k", type=int, default=50)
    parser.add_argument("--rules-file", type=Path, help="Optional structured controller rules JSON.")
    parser.add_argument(
        "--arbiter-heuristics-file",
        type=Path,
        help="Optional replacement for the mutable Arbiter heuristic section.",
    )
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--max-tokens", type=int, default=8000)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--azure-endpoint", default=os.getenv("AZURE_ENDPOINT"))
    parser.add_argument("--azure-deployment", default=os.getenv("AZURE_DEPLOYMENT", "gpt-5.1-chat"))
    parser.add_argument("--azure-api-version", default=os.getenv("AZURE_API_VERSION", "2024-12-01-preview"))
    parser.add_argument("--claude-model", default=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"))
    parser.add_argument("--no-claude-thinking", action="store_true")
    parser.add_argument("--claude-thinking-budget", type=int, default=2048)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    level = DatasetLevel.parse(args.level)
    backbone = ModelBackbone(args.model)
    output_dir = (args.output_dir / level.value / backbone.value).resolve()
    model = ModelConfig(
        backbone=backbone,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout_seconds,
        reasoning_effort=args.reasoning_effort,
        azure_endpoint=args.azure_endpoint,
        azure_deployment=args.azure_deployment,
        azure_api_version=args.azure_api_version,
        claude_model=args.claude_model,
        claude_thinking=not args.no_claude_thinking,
        claude_thinking_budget=args.claude_thinking_budget,
    )
    config = RunConfig(
        level=level,
        model=model,
        data_dir=args.data_dir.resolve(),
        output_dir=output_dir,
        cache_dir=args.cache_dir.resolve(),
        cache_policy=CachePolicy(args.cache_policy),
        max_workers=args.max_workers,
        max_examples=args.max_examples,
        selected_ids=tuple(args.selected_id),
        force=args.force,
        max_agent_turns=args.max_agent_turns,
        max_refine_rounds=args.max_refine_rounds,
        initial_k=args.initial_k,
        broader_k=args.broader_k,
        rules_file=args.rules_file.resolve() if args.rules_file else None,
        arbiter_heuristics_file=(
            args.arbiter_heuristics_file.resolve() if args.arbiter_heuristics_file else None
        ),
    )
    results = run(config)
    print(f"Completed {len(results)} records. Output: {output_dir}")
