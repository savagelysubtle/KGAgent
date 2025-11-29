"""Shared LLM configuration for all agents.

This module provides a centralized LLM configuration that all agents
(both the original KGAgent and multi-agent specialists) should use.

Usage:
    from kg_agent.agent.llm import create_lm_studio_model

    agent = Agent(
        model=create_lm_studio_model(),
        system_prompt="...",
    )
"""

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from ..core.config import settings
from ..core.logging import logger


def create_lm_studio_model() -> OpenAIChatModel:
    """
    Create an OpenAI-compatible model configured for LM Studio.

    This is the canonical way to create an LLM for all agents in the system.
    It uses settings from the config:
    - LLM_BASE_URL: The LM Studio API URL (default: http://localhost:1234/v1)
    - LLM_API_KEY: The API key (default: "lm-studio")
    - LLM_MODEL_NAME: The model to use (default: "local-model")

    Returns:
        OpenAIChatModel configured for LM Studio
    """
    logger.debug(
        f"Creating LM Studio model: {settings.LLM_MODEL_NAME} at {settings.LLM_BASE_URL}"
    )

    provider = OpenAIProvider(
        base_url=settings.LLM_BASE_URL,
        api_key=settings.LLM_API_KEY,
    )

    return OpenAIChatModel(
        settings.LLM_MODEL_NAME,
        provider=provider,
    )


# Alias for backwards compatibility
create_llm = create_lm_studio_model

