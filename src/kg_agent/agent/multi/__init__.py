"""Multi-agent system for KGAgent using LangGraph + Pydantic AI.

This module provides a hierarchical multi-agent system where:
- Manager agent analyzes intent and delegates to specialists
- Four specialist agents handle domain-specific tasks
- Results are synthesized into a final response
- Real-time reasoning is streamed to the UI via CopilotKit

Usage:
    from kg_agent.agent.multi import get_multi_agent, invoke_multi_agent

    # Simple invocation
    result = await invoke_multi_agent("Search for Python tutorials")
    print(result["final_response"])

    # With session persistence
    result = await invoke_multi_agent(
        "Remember my name is Steve",
        session_id="session-123"
    )
"""

from .document_lead import documents_node
from .error_handling import (
    ConfigurationError,
    DelegationError,
    MultiAgentError,
    SpecialistError,
    StateValidationError,
    with_error_handling,
)
from .graph import (
    create_multi_agent_graph,
    get_multi_agent,
    invoke_multi_agent,
    stream_multi_agent,
)
from .knowledge_lead import knowledge_node
from .memory_lead import memory_node
from .prompts import (
    DOCUMENT_LEAD_PROMPT,
    KNOWLEDGE_LEAD_PROMPT,
    MANAGER_SYSTEM_PROMPT,
    MEMORY_LEAD_PROMPT,
    RESEARCH_LEAD_PROMPT,
    get_prompt_for_agent,
)
from .research_lead import research_node
from .session import (
    SessionManager,
    get_session_manager,
)
from .optimization import (
    QueryCache,
    get_query_cache,
    get_stats_cache,
    optimize_thinking_steps,
    should_emit_state,
)
from .state import (
    DelegationRequest,
    MultiAgentState,
    ThinkingStep,
    create_delegation_request,
    create_initial_state,
    create_thinking_step,
)

__all__ = [
    # State
    "MultiAgentState",
    "ThinkingStep",
    "DelegationRequest",
    "create_initial_state",
    "create_thinking_step",
    "create_delegation_request",
    # Graph
    "create_multi_agent_graph",
    "get_multi_agent",
    "invoke_multi_agent",
    "stream_multi_agent",
    # Session
    "SessionManager",
    "get_session_manager",
    # Optimization
    "QueryCache",
    "get_query_cache",
    "get_stats_cache",
    "optimize_thinking_steps",
    "should_emit_state",
    # Error handling
    "with_error_handling",
    "MultiAgentError",
    "DelegationError",
    "SpecialistError",
    "ConfigurationError",
    "StateValidationError",
    # Prompts
    "get_prompt_for_agent",
    "MANAGER_SYSTEM_PROMPT",
    "RESEARCH_LEAD_PROMPT",
    "MEMORY_LEAD_PROMPT",
    "KNOWLEDGE_LEAD_PROMPT",
    "DOCUMENT_LEAD_PROMPT",
    # Specialist nodes
    "research_node",
    "memory_node",
    "knowledge_node",
    "documents_node",
]
