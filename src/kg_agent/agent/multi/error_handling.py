"""Error handling utilities for the multi-agent system."""

from functools import wraps
from typing import Any, Callable, Dict

from langchain_core.runnables import RunnableConfig

from ...core.logging import logger
from .state import MultiAgentState, create_thinking_step

try:
    from copilotkit.langgraph import copilotkit_emit_state  # type: ignore[assignment]
except ImportError:

    async def copilotkit_emit_state(config, state) -> None:
        """Fallback when CopilotKit is not installed."""


def with_error_handling(agent_name: str):
    """
    Decorator to add error handling to node functions.

    Catches exceptions, logs them, and returns graceful error state.

    Args:
        agent_name: Name of the agent for error reporting
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(
            state: MultiAgentState, config: RunnableConfig
        ) -> Dict[str, Any]:
            try:
                return await func(state, config)
            except Exception as e:
                logger.error(f"{agent_name} node failed: {e}", exc_info=True)

                # Create error thinking step
                thinking_steps = list(state.get("thinking_steps", []))
                thinking_steps.append(
                    create_thinking_step(
                        agent=agent_name,
                        thought=f"Error: {str(e)[:100]}",
                        status="error",
                    )
                )

                # Emit error state
                await copilotkit_emit_state(
                    config,
                    {
                        "thinking_steps": thinking_steps,
                        "current_agent": agent_name,
                        "last_error": str(e),
                    },
                )

                # Return state that allows graph to continue
                result_key = f"{agent_name}_result"
                delegation_queue = state.get("delegation_queue", [])

                return {
                    "thinking_steps": thinking_steps,
                    result_key: f"❌ {agent_name} encountered an error: {str(e)}",
                    "last_error": str(e),
                    "delegation_queue": delegation_queue[1:]
                    if delegation_queue
                    else [],
                    "current_delegation": None,
                }

        return wrapper

    return decorator


class MultiAgentError(Exception):
    """Base exception for multi-agent system."""

    pass


class DelegationError(MultiAgentError):
    """Error during delegation routing."""

    pass


class SpecialistError(MultiAgentError):
    """Error in specialist execution."""

    def __init__(self, agent_name: str, message: str):
        self.agent_name = agent_name
        super().__init__(f"{agent_name}: {message}")


class ConfigurationError(MultiAgentError):
    """Error in system configuration."""

    pass


class StateValidationError(MultiAgentError):
    """Error when state validation fails."""

    pass
