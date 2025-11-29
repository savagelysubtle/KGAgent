"""Pydantic AI Agent module for Knowledge Graph RAG.

The multi-agent system provides hierarchical orchestration with:
- Manager agent for intent analysis and delegation
- Specialist agents for research, memory, knowledge, and documents
"""

from .tools import RAGTools
from .llm import create_lm_studio_model

# Re-export multi-agent components for convenience
from .multi import (
    MultiAgentState,
    ThinkingStep,
    DelegationRequest,
    create_initial_state,
    create_multi_agent_graph,
    get_multi_agent,
    invoke_multi_agent,
    stream_multi_agent,
    SessionManager,
    get_session_manager,
)

__all__ = [
    # Tools
    "RAGTools",
    "create_lm_studio_model",
    # Multi-agent
    "MultiAgentState",
    "ThinkingStep",
    "DelegationRequest",
    "create_initial_state",
    "create_multi_agent_graph",
    "get_multi_agent",
    "invoke_multi_agent",
    "stream_multi_agent",
    "SessionManager",
    "get_session_manager",
]
