from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .cache import CachePolicy


class DatasetLevel(str, Enum):
    VERIFIED = "moa_verified"
    EXTENDED = "moa_extended"
    NOVEL = "moa_novel"

    @classmethod
    def parse(cls, value: str) -> "DatasetLevel":
        aliases = {
            "verified": cls.VERIFIED,
            "core": cls.VERIFIED,
            "extended": cls.EXTENDED,
            "extend": cls.EXTENDED,
            "novel": cls.NOVEL,
        }
        return aliases[value] if value in aliases else cls(value)

    @property
    def allows_novel(self) -> bool:
        return self is self.NOVEL


class ModelBackbone(str, Enum):
    GPT = "gpt"
    CLAUDE = "claude"


@dataclass(frozen=True)
class ModelConfig:
    backbone: ModelBackbone
    max_tokens: int = 8000
    timeout_seconds: int = 300
    reasoning_effort: str = "medium"
    azure_endpoint: str | None = None
    azure_deployment: str = "gpt-5.1-chat"
    azure_api_version: str = "2024-12-01-preview"
    claude_model: str = "claude-sonnet-4-6"
    claude_thinking: bool = True
    claude_thinking_budget: int = 2048

    def validate(self) -> None:
        if self.backbone is ModelBackbone.GPT:
            if not os.getenv("AZURE_API_KEY"):
                raise RuntimeError("AZURE_API_KEY is required for --model gpt")
            if not self.azure_endpoint:
                raise RuntimeError("Set AZURE_ENDPOINT or pass --azure-endpoint for GPT.")
        elif not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is required for --model claude")


@dataclass(frozen=True)
class RunConfig:
    level: DatasetLevel
    model: ModelConfig
    data_dir: Path
    output_dir: Path
    cache_dir: Path = Path(".cache/phenoair")
    cache_policy: CachePolicy = CachePolicy.READ_WRITE
    max_workers: int = 4
    max_examples: int | None = None
    selected_ids: tuple[str, ...] = ()
    force: bool = False
    max_agent_turns: int = 8
    max_refine_rounds: int = 4
    initial_k: int = 10
    broader_k: int = 50
    rules_file: Path | None = None
    arbiter_heuristics_file: Path | None = None

    def validate(self, require_credentials: bool = True) -> None:
        level_dir = self.data_dir / self.level.value
        required = (
            level_dir / "reference.parquet",
            level_dir / "test.parquet",
            level_dir / "labels.json",
            level_dir / "features" / "cellclip_features.npy",
            level_dir / "features" / "cellclip_index.parquet",
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing release data files:\n- " + "\n- ".join(missing))
        if self.max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        if self.max_examples is not None and self.max_examples < 1:
            raise ValueError("max_examples must be positive")
        if self.max_agent_turns < 1 or self.max_refine_rounds < 0:
            raise ValueError("agent turns and refinement rounds must be non-negative")
        if require_credentials:
            self.model.validate()
