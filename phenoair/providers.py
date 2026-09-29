from __future__ import annotations

import os

from .config import ModelBackbone, ModelConfig
from .llm_client import LLMClient


def create_llm_client(config: ModelConfig) -> LLMClient:
    """Create the shared provider client. Credentials are read only from the environment."""
    config.validate()
    if config.backbone is ModelBackbone.GPT:
        return LLMClient.azure(
            deployment=config.azure_deployment,
            endpoint=str(config.azure_endpoint),
            api_key=os.environ["AZURE_API_KEY"],
            api_version=config.azure_api_version,
            reasoning_effort=config.reasoning_effort,
            default_max_tokens=config.max_tokens,
            timeout=config.timeout_seconds,
        )

    kwargs = {
        "default_max_tokens": config.max_tokens,
        "timeout": config.timeout_seconds,
    }
    if config.claude_thinking:
        kwargs["thinking"] = {
            "type": "enabled",
            "budget_tokens": config.claude_thinking_budget,
        }
    return LLMClient.claude(
        model=config.claude_model,
        api_key=os.environ["ANTHROPIC_API_KEY"],
        **kwargs,
    )

