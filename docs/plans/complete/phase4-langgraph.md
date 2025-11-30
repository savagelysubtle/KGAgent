# Phase 4: LangGraph Wiring - Complete Orchestration

> **Parent Plan:** [multi_agent.plan.md](./multi_agent.plan.md) > **Status:**
> Not Started **Estimated Effort:** 3-4 hours **Dependencies:** Phases 1-3
> complete **Last Updated:** November 29, 2025

---

## 🎯 Objectives

1. Complete the LangGraph `StateGraph` with all nodes and edges
2. Implement intelligent routing based on Manager decisions
3. Add LangGraph checkpointing for session persistence
4. Enable multi-delegation (sequential specialist execution)
5. Handle errors and edge cases gracefully

---

## 📋 Prerequisites

- [ ] Phase 1: State definitions working
- [ ] Phase 2: Manager node implemented
- [ ] Phase 3: All specialist nodes implemented

---

## 🔬 Technical Research Summary

### Key Findings from CopilotKit Docs

1. **`copilotkit_emit_state`** - Emits intermediate state to UI during
   long-running nodes
2. **`copilotkit_emit_message`** - Manually emit messages (must also return in
   node output)
3. **`copilotkit_customize_config`** - Control what gets streamed (messages,
   tool calls)
4. **`copilotkit_exit`** - Signal graph completion to CopilotKit
5. **Predictive State Updates** - Emit state updates mid-node for real-time UI
   feedback

### LangGraph 0.2+ API Changes

1. **Command Pattern** - Modern routing uses `Command(goto=..., update={...})`
   from `langgraph.types`
2. **Checkpointer Changes**:
   - `MemorySaver` - In-memory (dev/testing)
   - `AsyncSqliteSaver` - File-based async (requires
     `langgraph.checkpoint.sqlite.aio`)
   - `AsyncPostgresSaver` - Production database (requires
     `langgraph.checkpoint.postgres.aio`)
3. **State as TypedDict** - Current implementation is correct (extends
   `CopilotKitState`)
4. **Conditional Edges** - `add_conditional_edges` with path_map is correct
   pattern

### CopilotKit Integration Pattern

From the `with-langgraph-python` example:

```python
# Modern pattern uses Command for routing
from langgraph.types import Command

async def node(state, config):
    # ... do work ...
    return Command(
        goto="next_node",  # or END
        update={"key": "value"}
    )
```

---

## 🔀 Routing Logic Design

### State Machine Diagram

```
                                        ┌─────────────────────────────────────────────┐
                                        │              MULTI-AGENT GRAPH              │
                                        └─────────────────────────────────────────────┘

                                                        ┌───────────┐
                                                        │   START   │
                                                        └─────┬─────┘
                                                              │
                                                              ▼
        ┌─────────────────────────────────────────────────────────────────────────────────────┐
        │                                    MANAGER                                           │
        │  • Analyzes user intent                                                              │
        │  • Creates delegation_queue                                                          │
        │  • Routes: needs_clarification? → END                                                │
        │           delegation_queue empty? → synthesize                                       │
        │           else → first delegation target                                             │
        └───────────┬────────────────────────┬────────────────────────┬────────────────────────┤
                    │                        │                        │                        │
                    │ delegate_to_research   │ delegate_to_memory     │ delegate_to_knowledge  │ delegate_to_documents
                    ▼                        ▼                        ▼                        ▼
        ┌───────────────────┐    ┌───────────────────┐    ┌───────────────────┐    ┌───────────────────┐
        │     RESEARCH      │    │      MEMORY       │    │     KNOWLEDGE     │    │     DOCUMENTS     │
        │                   │    │                   │    │                   │    │                   │
        │ • Executes task   │    │ • Executes task   │    │ • Executes task   │    │ • Executes task   │
        │ • Sets result     │    │ • Sets result     │    │ • Sets result     │    │ • Sets result     │
        │ • Pops queue      │    │ • Pops queue      │    │ • Pops queue      │    │ • Pops queue      │
        └─────────┬─────────┘    └─────────┬─────────┘    └─────────┬─────────┘    └─────────┬─────────┘
                  │                        │                        │                        │
                  └────────────────────────┴────────────────────────┴────────────────────────┘
                                                              │
                                                              ▼
                                            ┌─────────────────────────────────┐
                                            │        ROUTE AFTER SPECIALIST    │
                                            │                                  │
                                            │  delegation_queue not empty?     │
                                            │    YES → next specialist         │
                                            │    NO  → synthesize              │
                                            └───────────────┬──────────────────┘
                                                            │
                                    ┌───────────────────────┼───────────────────────┐
                                    │ more delegations      │ no more delegations   │
                                    ▼                       ▼                       │
                          (loop to next specialist)    ┌───────────────┐            │
                                                       │   SYNTHESIZE  │            │
                                                       │               │            │
                                                       │ • Collect all │            │
                                                       │   results     │            │
                                                       │ • Format resp │            │
                                                       └───────┬───────┘            │
                                                               │                    │
                                                               ▼                    │
                                                           ┌───────┐               │
                                                           │  END  │◀──────────────┘
                                                           └───────┘
```

---

## 📁 Task 1: Update `graph.py` with Full Implementation

### File: `src/kg_agent/agent/multi/graph.py`

**Key Changes from Current Implementation:**

1. Add async checkpointer support with proper context management
2. Add `invoke_multi_agent` and `stream_multi_agent` utility functions
3. Keep current routing logic (it's correct for our pattern)

```python
"""LangGraph StateGraph - Full Implementation.

This module defines the complete multi-agent orchestration graph with:
- Manager node for intent analysis and delegation
- Four specialist nodes (research, memory, knowledge, documents)
- Synthesize node for combining results
- Conditional routing based on delegation queue
- Checkpointing for session persistence
"""

from typing import Literal, Optional, Union
from contextlib import asynccontextmanager
import asyncio

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from ...core.config import settings
from ...core.logging import logger

from .state import MultiAgentState, create_initial_state
from .manager import manager_node, synthesize_node
from .research_lead import research_node
from .memory_lead import memory_node
from .knowledge_lead import knowledge_node
from .document_lead import documents_node


# Type for node names
NodeName = Literal["manager", "research", "memory", "knowledge", "documents", "synthesize"]


# === Routing Functions ===

def route_from_manager(state: MultiAgentState) -> str:
    """
    Route from Manager based on delegation decisions.

    Routes to:
    - END: If should_end is True (clarification needed or no delegations)
    - synthesize: If delegation_queue is empty (nothing to do)
    - First specialist: If delegations are queued
    """
    logger.debug(
        f"Routing from manager. should_end={state.get('should_end')}, "
        f"queue={len(state.get('delegation_queue') or [])}"
    )

    # End conditions
    if state.get("should_end"):
        logger.debug("Routing to END (should_end=True)")
        return END

    if state.get("final_response"):
        logger.debug("Routing to END (final_response set)")
        return END

    # Check delegation queue
    delegation_queue = state.get("delegation_queue", [])
    if not delegation_queue:
        logger.debug("Routing to synthesize (empty queue)")
        return "synthesize"

    # Route to first delegation target
    first_target = delegation_queue[0]["target"]
    logger.debug(f"Routing to specialist: {first_target}")
    return first_target


def route_after_specialist(state: MultiAgentState) -> str:
    """
    Route after a specialist completes.

    Routes to:
    - Next specialist: If more delegations in queue
    - synthesize: If all delegations processed
    """
    remaining = state.get("delegation_queue", [])

    logger.debug(f"Routing after specialist. Remaining delegations: {len(remaining)}")

    # Check if more delegations to process
    if remaining:
        next_target = remaining[0]["target"]
        logger.debug(f"Routing to next specialist: {next_target}")
        return next_target

    # All done - synthesize
    logger.debug("Routing to synthesize (queue empty)")
    return "synthesize"


# === Graph Builder ===

def create_multi_agent_graph(
    checkpointer: Optional[MemorySaver] = None
) -> StateGraph:
    """
    Create and compile the multi-agent StateGraph.

    Graph Structure:

    START → manager → [research|memory|knowledge|documents]* → synthesize → END

    The manager can delegate to one or more specialists sequentially.
    After all specialists complete, results are synthesized.

    Args:
        checkpointer: Optional checkpointer for session persistence.
                     Use MemorySaver for in-memory (dev/test).

    Returns:
        Compiled StateGraph ready for invocation.
    """
    logger.info("Creating multi-agent graph")

    # Create graph with state type
    graph = StateGraph(MultiAgentState)

    # === Add Nodes ===

    graph.add_node("manager", manager_node)
    graph.add_node("research", research_node)
    graph.add_node("memory", memory_node)
    graph.add_node("knowledge", knowledge_node)
    graph.add_node("documents", documents_node)
    graph.add_node("synthesize", synthesize_node)

    # === Set Entry Point ===

    graph.set_entry_point("manager")

    # === Add Conditional Edges from Manager ===

    graph.add_conditional_edges(
        source="manager",
        path=route_from_manager,
        path_map={
            "research": "research",
            "memory": "memory",
            "knowledge": "knowledge",
            "documents": "documents",
            "synthesize": "synthesize",
            END: END,
        }
    )

    # === Add Conditional Edges from Specialists ===

    # After each specialist, check if more delegations or go to synthesize
    for specialist in ["research", "memory", "knowledge", "documents"]:
        graph.add_conditional_edges(
            source=specialist,
            path=route_after_specialist,
            path_map={
                "research": "research",
                "memory": "memory",
                "knowledge": "knowledge",
                "documents": "documents",
                "synthesize": "synthesize",
            }
        )

    # === Synthesize → END ===

    graph.add_edge("synthesize", END)

    # === Compile ===

    if checkpointer:
        logger.info(f"Compiling graph with checkpointer: {type(checkpointer).__name__}")
        compiled = graph.compile(checkpointer=checkpointer)
    else:
        logger.info("Compiling graph without checkpointer")
        compiled = graph.compile()

    return compiled


# === Singleton Instance ===

_multi_agent_graph: Optional[StateGraph] = None
_graph_checkpointer: Optional[MemorySaver] = None


def get_multi_agent(
    use_checkpointer: bool = True,
    force_recreate: bool = False,
) -> StateGraph:
    """
    Get or create the singleton multi-agent graph.

    Args:
        use_checkpointer: Enable session persistence (MemorySaver)
        force_recreate: Force recreation of the graph (useful for testing)

    Returns:
        Compiled StateGraph instance
    """
    global _multi_agent_graph, _graph_checkpointer

    if _multi_agent_graph is None or force_recreate:
        if use_checkpointer:
            _graph_checkpointer = MemorySaver()
            _multi_agent_graph = create_multi_agent_graph(_graph_checkpointer)
        else:
            _multi_agent_graph = create_multi_agent_graph(None)

    return _multi_agent_graph


# === Utility Functions ===

async def invoke_multi_agent(
    user_message: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    config: Optional[dict] = None,
) -> dict:
    """
    Convenience function to invoke the multi-agent graph.

    Args:
        user_message: The user's input
        user_id: Optional user identifier
        session_id: Optional session ID for checkpointing
        config: Optional LangGraph config (for CopilotKit integration)

    Returns:
        Final state dict with results
    """
    graph = get_multi_agent()
    initial_state = create_initial_state(
        user_message=user_message,
        user_id=user_id,
        session_id=session_id,
    )

    # Build config with thread_id for checkpointing
    run_config = config or {}
    if session_id:
        run_config["configurable"] = run_config.get("configurable", {})
        run_config["configurable"]["thread_id"] = session_id

    result = await graph.ainvoke(initial_state, config=run_config)
    return result


async def stream_multi_agent(
    user_message: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    config: Optional[dict] = None,
):
    """
    Stream execution of the multi-agent graph.

    Yields state updates as the graph executes.

    Args:
        user_message: The user's input
        user_id: Optional user identifier
        session_id: Optional session ID
        config: Optional LangGraph config

    Yields:
        Tuples of (node_name, state_update)
    """
    graph = get_multi_agent()
    initial_state = create_initial_state(
        user_message=user_message,
        user_id=user_id,
        session_id=session_id,
    )

    run_config = config or {}
    if session_id:
        run_config["configurable"] = run_config.get("configurable", {})
        run_config["configurable"]["thread_id"] = session_id

    async for event in graph.astream(initial_state, config=run_config):
        for node_name, state_update in event.items():
            yield (node_name, state_update)
```

---

## 📁 Task 2: Add Error Handling Wrapper

### File: `src/kg_agent/agent/multi/error_handling.py`

```python
"""Error handling utilities for the multi-agent system."""

from functools import wraps
from typing import Any, Callable, Dict

from langchain_core.runnables import RunnableConfig

from ...core.logging import logger
from .state import MultiAgentState, ThinkingStep, create_thinking_step

try:
    from copilotkit.langgraph import copilotkit_emit_state
except ImportError:
    async def copilotkit_emit_state(config, state):
        pass


def with_error_handling(agent_name: str):
    """
    Decorator to add error handling to node functions.

    Catches exceptions, logs them, and returns graceful error state.

    Args:
        agent_name: Name of the agent for error reporting
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(state: MultiAgentState, config: RunnableConfig) -> Dict[str, Any]:
            try:
                return await func(state, config)
            except Exception as e:
                logger.error(f"{agent_name} node failed: {e}", exc_info=True)

                # Create error thinking step
                thinking_steps = list(state.get("thinking_steps", []))
                thinking_steps.append(create_thinking_step(
                    agent=agent_name,
                    thought=f"Error: {str(e)[:100]}",
                    status="error",
                ))

                # Emit error state
                await copilotkit_emit_state(config, {
                    "thinking_steps": thinking_steps,
                    "current_agent": agent_name,
                    "last_error": str(e),
                })

                # Return state that allows graph to continue
                result_key = f"{agent_name}_result"
                delegation_queue = state.get("delegation_queue", [])

                return {
                    "thinking_steps": thinking_steps,
                    result_key: f"❌ {agent_name} encountered an error: {str(e)}",
                    "last_error": str(e),
                    "delegation_queue": delegation_queue[1:] if delegation_queue else [],
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
```

---

## 📁 Task 3: Add Session Management

### File: `src/kg_agent/agent/multi/session.py`

```python
"""Session management for multi-agent conversations."""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from ...core.logging import logger


class SessionManager:
    """
    Manages conversation sessions for the multi-agent system.

    Provides:
    - Session ID generation
    - Session metadata storage
    - History retrieval
    """

    def __init__(self):
        self._sessions: Dict[str, Dict[str, Any]] = {}

    def create_session(
        self,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a new session.

        Args:
            user_id: Optional user identifier
            metadata: Optional session metadata

        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())

        self._sessions[session_id] = {
            "id": session_id,
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "last_active": datetime.now().isoformat(),
            "message_count": 0,
            "metadata": metadata or {},
        }

        logger.info(f"Created session: {session_id}")
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session info."""
        return self._sessions.get(session_id)

    def update_session(self, session_id: str, **updates):
        """Update session metadata."""
        if session_id in self._sessions:
            self._sessions[session_id]["last_active"] = datetime.now().isoformat()
            self._sessions[session_id].update(updates)

    def increment_message_count(self, session_id: str):
        """Increment the message count for a session."""
        if session_id in self._sessions:
            self._sessions[session_id]["message_count"] += 1
            self._sessions[session_id]["last_active"] = datetime.now().isoformat()

    def list_sessions(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List sessions, optionally filtered by user."""
        sessions = list(self._sessions.values())
        if user_id:
            sessions = [s for s in sessions if s.get("user_id") == user_id]
        return sorted(sessions, key=lambda s: s["last_active"], reverse=True)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"Deleted session: {session_id}")
            return True
        return False


# Singleton instance
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get the singleton session manager."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
```

---

## 📁 Task 4: Update `__init__.py` Exports

### File: `src/kg_agent/agent/multi/__init__.py`

```python
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

from .state import (
    MultiAgentState,
    ThinkingStep,
    DelegationRequest,
    create_initial_state,
    create_thinking_step,
    create_delegation_request,
)

from .graph import (
    create_multi_agent_graph,
    get_multi_agent,
    invoke_multi_agent,
    stream_multi_agent,
)

from .session import (
    SessionManager,
    get_session_manager,
)

from .prompts import (
    get_prompt_for_agent,
    MANAGER_SYSTEM_PROMPT,
    RESEARCH_LEAD_PROMPT,
    MEMORY_LEAD_PROMPT,
    KNOWLEDGE_LEAD_PROMPT,
    DOCUMENT_LEAD_PROMPT,
)

# Specialist nodes (for direct testing)
from .research_lead import research_node
from .memory_lead import memory_node
from .knowledge_lead import knowledge_node
from .document_lead import documents_node


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
```

---

## 📁 Task 5: Integration Tests

### File: `tests/test_multi_agent_integration.py`

```python
"""Integration tests for the multi-agent system."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from langchain_core.messages import HumanMessage

from kg_agent.agent.multi import (
    get_multi_agent,
    invoke_multi_agent,
    create_initial_state,
    MultiAgentState,
)
from kg_agent.agent.multi.manager import DelegationDecision


@pytest.fixture
def mock_llm_response():
    """Create a mock LLM response factory."""
    def create_response(delegations):
        return type("Result", (), {
            "output": DelegationDecision(
                reasoning="Test delegation",
                delegations=delegations,
            )
        })()
    return create_response


class TestMultiAgentFlow:
    """Test complete multi-agent flows."""

    @pytest.mark.asyncio
    async def test_single_delegation_flow(self, mock_llm_response):
        """Test flow with single specialist delegation."""
        with patch("kg_agent.agent.multi.manager.manager_agent") as mock_manager:
            mock_manager.run = AsyncMock(return_value=mock_llm_response([
                {"target": "research", "task": "Search for Python"}
            ]))

            # Also mock the research agent
            with patch("kg_agent.agent.multi.research_lead.research_agent") as mock_research:
                mock_research.run = AsyncMock(return_value=type("Result", (), {
                    "output": "Found Python tutorials"
                })())

                result = await invoke_multi_agent("Search for Python tutorials")

                assert result.get("should_end") == True
                assert result.get("final_response") is not None
                assert "manager" in result.get("execution_path", [])

    @pytest.mark.asyncio
    async def test_multi_delegation_flow(self, mock_llm_response):
        """Test flow with multiple specialist delegations."""
        with patch("kg_agent.agent.multi.manager.manager_agent") as mock_manager:
            mock_manager.run = AsyncMock(return_value=mock_llm_response([
                {"target": "memory", "task": "Remember name is Steve"},
                {"target": "research", "task": "Search for Steve's projects"},
            ]))

            with patch("kg_agent.agent.multi.memory_lead.memory_agent") as mock_memory:
                mock_memory.run = AsyncMock(return_value=type("Result", (), {
                    "output": "Remembered: name is Steve"
                })())

                with patch("kg_agent.agent.multi.research_lead.research_agent") as mock_research:
                    mock_research.run = AsyncMock(return_value=type("Result", (), {
                        "output": "Found Steve's projects"
                    })())

                    result = await invoke_multi_agent(
                        "Remember my name is Steve and find my projects"
                    )

                    assert result.get("should_end") == True
                    execution_path = result.get("execution_path", [])
                    assert "memory" in execution_path
                    assert "research" in execution_path

    @pytest.mark.asyncio
    async def test_session_persistence(self):
        """Test that sessions are persisted correctly."""
        from kg_agent.agent.multi.session import get_session_manager

        sm = get_session_manager()
        session_id = sm.create_session(user_id="test-user")

        assert sm.get_session(session_id) is not None
        assert sm.get_session(session_id)["user_id"] == "test-user"

        sm.increment_message_count(session_id)
        assert sm.get_session(session_id)["message_count"] == 1


class TestRoutingLogic:
    """Test routing decision logic."""

    def test_route_to_specialist(self):
        """Test routing to specialist when delegations exist."""
        from kg_agent.agent.multi.graph import route_from_manager
        from kg_agent.agent.multi.state import create_delegation_request

        state: MultiAgentState = {
            "messages": [],
            "delegation_queue": [create_delegation_request(target="research", task="test")],
            "should_end": False,
        }

        result = route_from_manager(state)
        assert result == "research"

    def test_route_to_synthesize(self):
        """Test routing to synthesize when queue empty."""
        from kg_agent.agent.multi.graph import route_from_manager

        state: MultiAgentState = {
            "messages": [],
            "delegation_queue": [],
            "should_end": False,
        }

        result = route_from_manager(state)
        assert result == "synthesize"

    def test_route_to_end(self):
        """Test routing to END when should_end is True."""
        from kg_agent.agent.multi.graph import route_from_manager
        from langgraph.graph import END

        state: MultiAgentState = {
            "messages": [],
            "should_end": True,
        }

        result = route_from_manager(state)
        assert result == END


class TestErrorHandling:
    """Test error handling in the multi-agent system."""

    @pytest.mark.asyncio
    async def test_specialist_error_recovery(self, mock_llm_response):
        """Test that system recovers from specialist errors."""
        with patch("kg_agent.agent.multi.manager.manager_agent") as mock_manager:
            mock_manager.run = AsyncMock(return_value=mock_llm_response([
                {"target": "research", "task": "Search"}
            ]))

            with patch("kg_agent.agent.multi.research_lead.research_agent") as mock_research:
                # Research agent raises an error
                mock_research.run = AsyncMock(side_effect=Exception("Test error"))

                result = await invoke_multi_agent("Search for something")

                # Should still complete with error message
                assert result.get("should_end") == True
                # Either has error in result or last_error field
                has_error = (
                    "error" in result.get("research_result", "").lower()
                    or result.get("last_error") is not None
                )
                assert has_error
```

---

## 📚 CopilotKit SDK Reference

### Key Functions (from `copilotkit.langgraph`)

| Function                                        | Purpose                       | Usage                    |
| ----------------------------------------------- | ----------------------------- | ------------------------ |
| `copilotkit_emit_state(config, state)`          | Emit intermediate state to UI | Long-running nodes       |
| `copilotkit_emit_message(config, msg)`          | Emit message mid-node         | Status updates           |
| `copilotkit_emit_tool_call(config, name, args)` | Emit tool execution           | Before tool runs         |
| `copilotkit_customize_config(config, **opts)`   | Control streaming behavior    | Disable/filter emissions |
| `copilotkit_exit(config)`                       | Signal graph completion       | End of workflow          |

### State Emission Best Practice

```python
from copilotkit.langgraph import copilotkit_emit_state

async def my_node(state: MultiAgentState, config: RunnableConfig):
    # Update thinking steps
    thinking_steps = list(state.get("thinking_steps", []))
    thinking_steps.append(create_thinking_step(
        agent="my_agent",
        thought="Processing...",
        status="thinking",
    ))

    # Emit intermediate state (UI updates immediately)
    await copilotkit_emit_state(config, {
        "thinking_steps": thinking_steps,
        "current_agent": "my_agent",
    })

    # Do work...

    # Return final state (becomes ground truth)
    return {
        "thinking_steps": thinking_steps,
        "result": "...",
    }
```

---

## ✅ Phase 4 Definition of Done

- [ ] `graph.py` fully implemented:

  - [ ] All nodes registered
  - [ ] Conditional routing working
  - [ ] Checkpointing integration (MemorySaver)
  - [ ] Utility functions (`invoke_multi_agent`, `stream_multi_agent`)

- [ ] Routing logic verified:

  - [ ] Manager → specialists based on delegation queue
  - [ ] Specialists → next specialist or synthesize
  - [ ] Synthesize → END

- [ ] Error handling:

  - [ ] `error_handling.py` created
  - [ ] Graceful degradation on specialist failure

- [ ] Session management:

  - [ ] `session.py` created
  - [ ] Session create/get/update working

- [ ] Tests pass:
  - [ ] Single delegation flow
  - [ ] Multi-delegation flow
  - [ ] Routing logic tests
  - [ ] Error recovery tests

---

## 🔗 Next Phase

→ [Phase 5: API Integration](./phase5-api.md) - Create FastAPI endpoints with
CopilotKit

---

## 📝 Implementation Notes

### What's Already Implemented (from Phases 1-3)

1. **`state.py`** - MultiAgentState TypedDict extending CopilotKitState ✅
2. **`graph.py`** - Basic graph structure with routing ✅
3. **`manager.py`** - Manager node with delegation logic ✅
4. **`research_lead.py`** - Research specialist with RAG tools ✅
5. **`prompts.py`** - All agent prompts ✅

### What Phase 4 Adds

1. **`error_handling.py`** - Decorator and exception classes (NEW)
2. **`session.py`** - Session management (NEW)
3. **`graph.py`** updates - `invoke_multi_agent`, `stream_multi_agent` utilities
4. **`__init__.py`** updates - Export new modules
5. **Tests** - Integration tests for full flows

### Current Routing Pattern (Correct)

The existing `route_from_manager` and `route_after_specialist` functions use the
correct pattern for LangGraph 0.2+. The `add_conditional_edges` with `path_map`
is the standard approach.

### Checkpointer Decision

Using `MemorySaver` for now (in-memory, suitable for dev/test). For production
with persistence:

- Use `AsyncSqliteSaver` from `langgraph.checkpoint.sqlite.aio`
- Or `AsyncPostgresSaver` from `langgraph.checkpoint.postgres.aio`

---

_Created: November 29, 2025_ _Updated: November 29, 2025 - Added technical
research findings_
