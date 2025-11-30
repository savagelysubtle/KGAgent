# Phase 1: Foundation - Core Infrastructure

> **Parent Plan:** [multi_agent.plan.md](./multi_agent.plan.md) > **Status:**
> Not Started **Estimated Effort:** 2-3 hours **Dependencies:** None (this is
> the foundation)

---

## 🎯 Objectives

1. Create the multi-agent directory structure
2. Define shared state models (`MultiAgentState`, `ThinkingStep`)
3. Create all system prompts in a centralized file
4. Set up the basic LangGraph skeleton (no logic yet)

---

## 📋 Prerequisites

- [x] Python dependencies already installed (`langgraph`, `copilotkit>=0.1.40`)
- [x] Existing `RAGTools` class understood (928 lines of tool implementations)
- [x] Current agent tools mapped to specialist roles

---

## 📁 Task 1: Create Directory Structure

### Files to Create

```
src/kg_agent/agent/multi/
├── __init__.py          # Public exports
├── state.py             # State definitions
├── prompts.py           # All system prompts
├── graph.py             # LangGraph skeleton
├── manager.py           # (empty placeholder)
├── research_lead.py     # (empty placeholder)
├── memory_lead.py       # (empty placeholder)
├── knowledge_lead.py    # (empty placeholder)
└── document_lead.py     # (empty placeholder)
```

### `__init__.py` Template

```python
"""Multi-agent system for KGAgent using LangGraph + CopilotKit."""

from .state import (
    MultiAgentState,
    ThinkingStep,
    DelegationRequest,
    create_thinking_step,
    create_delegation_request,
    create_initial_state,
)
from .graph import create_multi_agent_graph, get_multi_agent

__all__ = [
    # State types (TypedDicts)
    "MultiAgentState",
    "ThinkingStep",
    "DelegationRequest",
    # Factory functions
    "create_thinking_step",
    "create_delegation_request",
    "create_initial_state",
    # Graph
    "create_multi_agent_graph",
    "get_multi_agent",
]
```

### Acceptance Criteria

- [ ] All files created
- [ ] `__init__.py` exports the correct symbols
- [ ] `from kg_agent.agent.multi import MultiAgentState` works

---

## 📁 Task 2: Define State Models (`state.py`)

> ⚠️ **CRITICAL NOTE (from research):** `CopilotKitState` is a **TypedDict**,
> NOT a regular class!
>
> - Cannot use `@dataclass` decorator with TypedDict inheritance
> - Cannot use `field(default_factory=...)` - use plain type annotations instead
> - LangGraph reducers go in `Annotated[type, reducer]`
> - CopilotKitState already provides `messages` with `add_messages` reducer
>   built-in

### CopilotKitState Built-in Fields (do NOT redeclare)

```python
# These fields are ALREADY in CopilotKitState - don't redeclare them:
class CopilotKitState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], add_messages]  # Built-in!
    steps: list[ThinkingStep]  # For UI progress display
```

### Full Implementation

```python
"""Shared state definitions for the multi-agent system.

IMPORTANT: CopilotKitState is a TypedDict, not a class!
- Use TypedDict inheritance, not @dataclass
- Use Annotated[] for reducers, not field()
- total=False makes all fields optional with None default
"""

from datetime import datetime
from typing import Annotated, Literal, NotRequired, Sequence, TypedDict

from copilotkit.langgraph import CopilotKitState
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ThinkingStep(TypedDict, total=False):
    """
    A single reasoning step from an agent.

    Displayed in the UI via CopilotKit's state rendering.
    Uses TypedDict for JSON serialization compatibility.
    """
    agent: str
    """Which agent produced this step: 'manager', 'research', 'memory', 'knowledge', 'documents'"""

    thought: str
    """What the agent is thinking or doing"""

    action: str  # Optional in total=False
    """Tool name if executing a tool"""

    result: str  # Optional in total=False
    """Result of the action (truncated for display)"""

    status: Literal["thinking", "delegating", "executing", "complete", "error"]
    """Current status of this step (defaults to 'thinking' in factory)"""

    timestamp: str
    """When this step occurred (ISO format)"""

    duration_ms: int  # Optional in total=False
    """How long this step took (set on completion)"""


class DelegationRequest(TypedDict, total=False):
    """A request from Manager to a specialist agent."""
    target: Literal["research", "memory", "knowledge", "documents"]  # Required
    task: str  # Required
    context: str  # Optional
    priority: int  # 1 = normal, 2 = high, defaults to 1


class MultiAgentState(CopilotKitState, total=False):
    """
    Shared state across all agents in the multi-agent hierarchy.

    Extends CopilotKitState (a TypedDict) to enable:
    - Real-time state streaming to UI via emit_state
    - Message history management (inherited from CopilotKitState)
    - Checkpoint persistence

    INHERITED FROM CopilotKitState (do NOT redeclare):
    - messages: Annotated[Sequence[BaseMessage], add_messages]
    - steps: list[ThinkingStep] (for UI progress)

    State Flow:
    1. User message → messages updated
    2. Manager thinks → thinking_steps updated, emit_state
    3. Manager delegates → delegation_queue updated
    4. Specialist executes → specialist_result updated
    5. Manager synthesizes → final_response set
    """

    # NOTE: `messages` is inherited from CopilotKitState with add_messages reducer!
    # NOTE: `steps` is inherited from CopilotKitState for UI progress!

    # === Reasoning Trail (additional to inherited steps) ===
    thinking_steps: list[ThinkingStep]
    """All reasoning steps from all agents - streamed to UI"""

    # === Current Execution Context ===
    current_agent: str
    """Which agent is currently active (default: 'manager')"""

    delegation_queue: list[DelegationRequest]
    """Pending delegations from Manager to specialists"""

    current_delegation: DelegationRequest
    """The delegation currently being processed"""

    # === Results from Specialists ===
    research_result: str
    """Result from Research Lead"""

    memory_result: str
    """Result from Memory Lead"""

    knowledge_result: str
    """Result from Knowledge Lead"""

    document_result: str
    """Result from Document Lead"""

    # === Final Output ===
    final_response: str
    """The synthesized final response to the user"""

    should_end: bool
    """Flag to signal graph completion"""

    # === Error Handling ===
    last_error: str
    """Last error message (if any)"""

    # === Metadata & Telemetry ===
    total_llm_calls: int
    """Total LLM API calls made in this request"""

    execution_path: list[str]
    """Ordered list of nodes visited: ['manager', 'research', 'manager', ...]"""

    start_time: str
    """When this request started processing (ISO format)"""

    # === User Context (passed through from frontend) ===
    user_id: str
    """Current user identifier"""

    session_id: str
    """Current session identifier for checkpointing"""


def create_thinking_step(
    agent: str,
    thought: str,
    status: Literal["thinking", "delegating", "executing", "complete", "error"] = "thinking",
    action: str | None = None,
    result: str | None = None,
) -> ThinkingStep:
    """
    Factory function to create a ThinkingStep with defaults.

    Since TypedDict doesn't support default values, use this factory.
    """
    step: ThinkingStep = {
        "agent": agent,
        "thought": thought,
        "status": status,
        "timestamp": datetime.now().isoformat(),
    }
    if action:
        step["action"] = action
    if result:
        step["result"] = result
    return step


def create_delegation_request(
    target: Literal["research", "memory", "knowledge", "documents"],
    task: str,
    context: str | None = None,
    priority: int = 1,
) -> DelegationRequest:
    """Factory function to create a DelegationRequest with defaults."""
    request: DelegationRequest = {
        "target": target,
        "task": task,
        "priority": priority,
    }
    if context:
        request["context"] = context
    return request


def create_initial_state(
    user_message: str,
    user_id: str | None = None,
    session_id: str | None = None,
) -> dict:
    """
    Create initial state dict for a new multi-agent request.

    Args:
        user_message: The user's input message
        user_id: Optional user identifier
        session_id: Optional session ID for persistence

    Returns:
        Dictionary suitable for graph.invoke()

    Note:
        Returns a dict, not MultiAgentState, because LangGraph
        expects a dict for invoke() and handles the typing internally.
    """
    from langchain_core.messages import HumanMessage

    state: dict = {
        "messages": [HumanMessage(content=user_message)],
        "thinking_steps": [],
        "current_agent": "manager",
        "delegation_queue": [],
        "execution_path": [],
        "total_llm_calls": 0,
        "start_time": datetime.now().isoformat(),
        "should_end": False,
    }
    if user_id:
        state["user_id"] = user_id
    if session_id:
        state["session_id"] = session_id
    return state
```

### Key Design Decisions

| Decision                       | Rationale                                                                                         |
| ------------------------------ | ------------------------------------------------------------------------------------------------- |
| **TypedDict, not @dataclass**  | `CopilotKitState` is a TypedDict; Python doesn't allow mixing TypedDict with dataclass            |
| **`total=False`**              | Makes all fields optional (default `None`), avoiding need for default values                      |
| **Factory functions**          | `create_thinking_step()` and `create_delegation_request()` provide defaults since TypedDict can't |
| **Don't redeclare `messages`** | Already in CopilotKitState with `add_messages` reducer                                            |
| **`thinking_steps` list**      | Ordered timeline for UI display                                                                   |
| **`delegation_queue`**         | Manager can queue multiple specialists                                                            |
| **`execution_path`**           | Debugging: track node visitation order                                                            |
| **`should_end` bool**          | Clean termination signal for graph routing                                                        |

### TypedDict vs Dataclass Comparison

```python
# ❌ WRONG - This would cause runtime errors
@dataclass
class MultiAgentState(CopilotKitState):
    messages: Annotated[...] = field(default_factory=list)  # FAILS!

# ✅ CORRECT - TypedDict inheritance
class MultiAgentState(CopilotKitState, total=False):
    thinking_steps: list[ThinkingStep]  # Optional due to total=False
```

### Acceptance Criteria

- [ ] `ThinkingStep` is a TypedDict with all status types
- [ ] `DelegationRequest` is a TypedDict
- [ ] `MultiAgentState` extends `CopilotKitState` (TypedDict)
- [ ] Factory functions work: `create_thinking_step()`,
      `create_delegation_request()`
- [ ] `create_initial_state()` returns proper dict for graph.invoke()
- [ ] Type hints pass `ty` type checker
- [ ] No `@dataclass` or `field()` usage with TypedDict

---

## 📁 Task 3: Create System Prompts (`prompts.py`)

### Full Implementation

```python
"""Centralized system prompts for all agents in the multi-agent system."""

# Token budget: ~200 tokens for Manager, ~250 for specialists
# Total: ~1200 tokens for all prompts combined

MANAGER_SYSTEM_PROMPT = """You are the Manager of a knowledge management system. Your role is to:

1. **Understand** what the user needs
2. **Delegate** to the right specialist(s)
3. **Synthesize** results into a helpful response

## Your Team

- **🔍 Research Lead**: Searches the knowledge base (vector/graph/hybrid search)
- **🧠 Memory Lead**: Manages user context, preferences, and past conversations
- **📊 Knowledge Lead**: Creates entities and relationships in the knowledge graph
- **📁 Document Lead**: Manages document lifecycle (list, delete, stats)

## Rules

1. ALWAYS delegate to specialists - you don't have direct tools
2. You CAN delegate to multiple specialists for complex requests
3. Wait for specialist results before synthesizing
4. If a specialist fails, acknowledge it and continue with what you have
5. Be concise but helpful in your final response

## Delegation Format

Think step-by-step about which specialist(s) to use, then delegate.
"""

RESEARCH_LEAD_PROMPT = """You are the Research Lead. You specialize in finding information in the knowledge base.

## Your Tools

- `search_knowledge_base(query, search_type)` - Search with 'vector', 'graph', or 'hybrid' mode
- `search_by_source(source_pattern, limit)` - Find docs from specific sources
- `get_database_statistics()` - Get stats about the databases

## Guidelines

1. Choose the right search type:
   - **vector**: For semantic/conceptual queries ("explain X", "what is Y")
   - **graph**: For structured queries ("who works at X", "what relates to Y")
   - **hybrid**: When unsure or for comprehensive results

2. Always cite sources in your response
3. Summarize findings concisely - don't dump raw results
4. If nothing found, say so clearly

When given a task, determine the best search strategy, execute it, and return a summary of findings.
"""

MEMORY_LEAD_PROMPT = """You are the Memory Lead. You manage the user's personal context and conversation history.

## Your Tools

- `recall_past_conversations(query, limit)` - Search past discussions
- `remember_about_user(fact, category)` - Store a fact about the user
- `add_user_info(property_name, property_value)` - Add user property
- `get_user_profile_summary()` - Get user profile overview

## Guidelines

1. **Recall**: Search past conversations when asked about previous discussions
2. **Remember**: Store new facts proactively when user shares information
3. **Profile**: Maintain a coherent user profile across sessions
4. Categories: general, work, interests, skills, preferences

Be personal and helpful. Context matters - use what you know about the user.
"""

KNOWLEDGE_LEAD_PROMPT = """You are the Knowledge Lead. You build and maintain the knowledge graph.

## Your Tools

- `create_entity(name, entity_type, description)` - Create a new entity
- `create_relationship(source_entity, target_entity, relationship_type)` - Link entities
- `search_graph(query, limit)` - Search existing graph structure
- `get_graph_stats()` - Get graph statistics

## Entity Types (use exactly)

Person, Organization, Technology, Concept, Location, Event, Product

## Relationship Types (examples)

KNOWS, WORKS_AT, CREATED, USES, LOCATED_IN, PART_OF, RELATED_TO, DEPENDS_ON

## Guidelines

1. Normalize entity names (e.g., "Python" not "python" or "PYTHON")
2. Use specific relationship types (CREATED_BY better than RELATED_TO)
3. Check if entities exist before creating duplicates
4. Report what you created/found clearly

Be structured and precise in graph operations.
"""

DOCUMENT_LEAD_PROMPT = """You are the Document Lead. You manage the document lifecycle.

## Your Tools

- `list_documents(status, source_type, search, limit)` - List/search documents
- `get_document_statistics()` - Get document stats
- `delete_document(doc_id, delete_vectors, delete_graph_nodes)` - Delete single doc
- `delete_documents_by_source(source_pattern)` - Delete by source
- `clear_all_data(confirm)` - Delete everything (requires confirm=True)

## Status Values

pending, processing, completed, failed, deleted

## Source Types

web_crawl, file_upload, api

## Guidelines

1. **Listing**: Default to limit=10, increase if user needs more
2. **Deletion**: Confirm destructive actions before executing
3. **Stats**: Include both status breakdown and source type breakdown
4. For `clear_all_data`, ALWAYS warn the user first

Be careful with deletions. Double-check before removing data.
"""


def get_prompt_for_agent(agent_name: str) -> str:
    """Get the system prompt for a specific agent."""
    prompts = {
        "manager": MANAGER_SYSTEM_PROMPT,
        "research": RESEARCH_LEAD_PROMPT,
        "memory": MEMORY_LEAD_PROMPT,
        "knowledge": KNOWLEDGE_LEAD_PROMPT,
        "documents": DOCUMENT_LEAD_PROMPT,
    }
    return prompts.get(agent_name, "")


# Prompt metadata for debugging/logging
PROMPT_METADATA = {
    "manager": {"tokens": 200, "tools": 4, "version": "1.0"},
    "research": {"tokens": 250, "tools": 3, "version": "1.0"},
    "memory": {"tokens": 250, "tools": 4, "version": "1.0"},
    "knowledge": {"tokens": 250, "tools": 4, "version": "1.0"},
    "documents": {"tokens": 250, "tools": 5, "version": "1.0"},
}
```

### Acceptance Criteria

- [ ] All 5 prompts defined
- [ ] Each prompt is under 300 tokens
- [ ] `get_prompt_for_agent()` helper works
- [ ] Prompts use consistent formatting

---

## 📁 Task 4: Create LangGraph Skeleton (`graph.py`)

> ⚠️ **NOTE:** Since `MultiAgentState` is a TypedDict, access state fields using
> **dict-style access** (`state["field"]`) or `.get("field")`, not attribute
> access.

### Full Implementation

```python
"""LangGraph StateGraph definition for the multi-agent system.

IMPORTANT: MultiAgentState is a TypedDict, so:
- Use state["field"] or state.get("field"), not state.field
- Return dict from nodes (partial state updates)
"""

from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from .state import MultiAgentState

# Node type hints
NodeName = Literal["manager", "research", "memory", "knowledge", "documents", "synthesize"]


def manager_node(state: MultiAgentState) -> dict:
    """
    Manager node - analyzes request and delegates to specialists.

    Implementation in Phase 2.
    """
    # Placeholder - returns immediately for skeleton testing
    return {"current_agent": "manager", "should_end": True}


def research_node(state: MultiAgentState) -> dict:
    """Research Lead node - searches knowledge base."""
    # Placeholder for Phase 3
    return {"research_result": "[Research placeholder]"}


def memory_node(state: MultiAgentState) -> dict:
    """Memory Lead node - handles user context."""
    # Placeholder for Phase 3
    return {"memory_result": "[Memory placeholder]"}


def knowledge_node(state: MultiAgentState) -> dict:
    """Knowledge Lead node - manages entities/relationships."""
    # Placeholder for Phase 3
    return {"knowledge_result": "[Knowledge placeholder]"}


def documents_node(state: MultiAgentState) -> dict:
    """Document Lead node - manages documents."""
    # Placeholder for Phase 3
    return {"document_result": "[Documents placeholder]"}


def synthesize_node(state: MultiAgentState) -> dict:
    """
    Synthesize node - combines specialist results into final response.

    Implementation in Phase 4.

    Note: Uses .get() for safe access since TypedDict fields may be None.
    """
    # Placeholder - combine any available results
    parts = []

    # Use .get() for safe TypedDict access
    if state.get("research_result"):
        parts.append(f"Research: {state['research_result']}")
    if state.get("memory_result"):
        parts.append(f"Memory: {state['memory_result']}")
    if state.get("knowledge_result"):
        parts.append(f"Knowledge: {state['knowledge_result']}")
    if state.get("document_result"):
        parts.append(f"Documents: {state['document_result']}")

    final = "\n\n".join(parts) if parts else "No specialist results available."
    return {"final_response": final, "should_end": True}


def route_from_manager(state: MultiAgentState) -> str:
    """
    Routing function - determines next node based on manager's delegation.

    Full implementation in Phase 4.

    Note: Uses .get() for safe TypedDict access.
    """
    # Placeholder - go straight to end for skeleton
    if state.get("should_end"):
        return END

    # Check if there are pending delegations
    delegation_queue = state.get("delegation_queue", [])
    if delegation_queue:
        delegation = delegation_queue[0]
        return delegation["target"]  # TypedDict uses dict access

    return "synthesize"


def route_after_specialist(state: MultiAgentState) -> str:
    """
    Routing after a specialist completes - back to manager or synthesize.

    Full implementation in Phase 4.
    """
    # Check if more delegations pending
    delegation_queue = state.get("delegation_queue", [])
    if len(delegation_queue) > 1:
        return "manager"  # Manager will process next delegation

    return "synthesize"


def create_multi_agent_graph(checkpointer: Optional[MemorySaver] = None) -> StateGraph:
    """
    Create the multi-agent StateGraph.

    Graph Structure:

        ┌─────────────────────────────────────────────────────┐
        │                                                     │
        │    ┌─────────┐                                      │
        │    │ START   │                                      │
        │    └────┬────┘                                      │
        │         │                                           │
        │         ▼                                           │
        │    ┌─────────┐      ┌──────────┐                    │
        │    │ manager │──────│ research │───┐                │
        │    └────┬────┘      └──────────┘   │                │
        │         │           ┌──────────┐   │                │
        │         ├───────────│  memory  │───┤                │
        │         │           └──────────┘   │                │
        │         │           ┌──────────┐   │   ┌───────────┐│
        │         ├───────────│knowledge │───┼───│synthesize ││
        │         │           └──────────┘   │   └─────┬─────┘│
        │         │           ┌──────────┐   │         │      │
        │         └───────────│documents │───┘         │      │
        │                     └──────────┘             │      │
        │                                              ▼      │
        │                                           ┌─────┐   │
        │                                           │ END │   │
        │                                           └─────┘   │
        └─────────────────────────────────────────────────────┘

    Args:
        checkpointer: Optional MemorySaver for persistence

    Returns:
        Compiled StateGraph ready for invocation
    """
    # Create the graph with our state type
    graph = StateGraph(MultiAgentState)

    # Add all nodes
    graph.add_node("manager", manager_node)
    graph.add_node("research", research_node)
    graph.add_node("memory", memory_node)
    graph.add_node("knowledge", knowledge_node)
    graph.add_node("documents", documents_node)
    graph.add_node("synthesize", synthesize_node)

    # Set entry point
    graph.set_entry_point("manager")

    # Add conditional edges from manager
    graph.add_conditional_edges(
        "manager",
        route_from_manager,
        {
            "research": "research",
            "memory": "memory",
            "knowledge": "knowledge",
            "documents": "documents",
            "synthesize": "synthesize",
            END: END,
        }
    )

    # Specialists route back (to manager for more delegations, or synthesize)
    for specialist in ["research", "memory", "knowledge", "documents"]:
        graph.add_conditional_edges(
            specialist,
            route_after_specialist,
            {
                "manager": "manager",
                "synthesize": "synthesize",
            }
        )

    # Synthesize goes to END
    graph.add_edge("synthesize", END)

    # Compile with optional checkpointer
    if checkpointer:
        return graph.compile(checkpointer=checkpointer)
    return graph.compile()


# Singleton instance
_multi_agent_graph = None


def get_multi_agent(use_checkpointer: bool = True) -> StateGraph:
    """
    Get or create the singleton multi-agent graph.

    Args:
        use_checkpointer: Whether to enable session persistence

    Returns:
        Compiled StateGraph instance
    """
    global _multi_agent_graph

    if _multi_agent_graph is None:
        checkpointer = MemorySaver() if use_checkpointer else None
        _multi_agent_graph = create_multi_agent_graph(checkpointer)

    return _multi_agent_graph
```

### Acceptance Criteria

- [ ] Graph compiles without errors
- [ ] All 6 nodes defined (manager + 4 specialists + synthesize)
- [ ] Routing functions return valid node names
- [ ] Graph can be invoked with test state
- [ ] Checkpointer integration works

---

## 📁 Task 5: Create Empty Placeholder Files

### `manager.py`

```python
"""Manager Agent - Orchestrator for the multi-agent system.

Implementation in Phase 2.
"""

# TODO: Implement manager logic
# - Parse user intent
# - Decide which specialists to delegate to
# - Emit thinking steps to UI
# - Synthesize specialist results
```

### `research_lead.py`

```python
"""Research Lead Agent - Specialist for knowledge base search.

Implementation in Phase 3.

Tools:
- search_knowledge_base(query, search_type)
- search_by_source(source_pattern, limit)
- get_database_statistics()
"""

# TODO: Implement research specialist
# - Port tools from kg_agent.py
# - Add thinking step emissions
# - Handle errors gracefully
```

### `memory_lead.py`

```python
"""Memory Lead Agent - Specialist for user context and conversations.

Implementation in Phase 3.

Tools:
- recall_past_conversations(query, limit)
- remember_about_user(fact, category)
- add_user_info(property_name, property_value)
- get_user_profile_summary()
"""

# TODO: Implement memory specialist
# - Port tools from kg_agent.py
# - Add thinking step emissions
# - Integrate with conversation_memory service
```

### `knowledge_lead.py`

```python
"""Knowledge Lead Agent - Specialist for knowledge graph operations.

Implementation in Phase 3.

Tools:
- create_entity(name, entity_type, description)
- create_relationship(source_entity, target_entity, relationship_type)
- search_graph(query, limit)
- get_graph_stats()
"""

# TODO: Implement knowledge specialist
# - Port tools from kg_agent.py
# - Add thinking step emissions
# - Integrate with graphiti_service
```

### `document_lead.py`

```python
"""Document Lead Agent - Specialist for document management.

Implementation in Phase 3.

Tools:
- list_documents(status, source_type, search, limit)
- get_document_statistics()
- delete_document(doc_id, delete_vectors, delete_graph_nodes)
- delete_documents_by_source(source_pattern)
- clear_all_data(confirm)
"""

# TODO: Implement document specialist
# - Port tools from kg_agent.py
# - Add thinking step emissions
# - Add confirmation prompts for destructive operations
```

---

## ✅ Phase 1 Definition of Done

- [ ] All 9 files created in `src/kg_agent/agent/multi/`
- [ ] `state.py` compiles and types check
      (`ty check src/kg_agent/agent/multi/state.py`)
- [ ] `prompts.py` has all 5 agent prompts
- [ ] `graph.py` compiles and can be imported
- [ ] No `@dataclass` decorators used with TypedDict classes
- [ ] Factory functions exist for ThinkingStep and DelegationRequest
- [ ] Basic smoke test passes:

```python
# Test script: scripts/test_phase1.py
"""Smoke test for Phase 1 multi-agent foundation."""

from kg_agent.agent.multi import MultiAgentState, get_multi_agent
from kg_agent.agent.multi.state import (
    create_initial_state,
    create_thinking_step,
    create_delegation_request,
)

# Test 1: Factory functions work
step = create_thinking_step(
    agent="manager",
    thought="Analyzing user request",
    status="thinking"
)
assert step["agent"] == "manager"
assert "timestamp" in step
print("✅ ThinkingStep factory works")

delegation = create_delegation_request(
    target="research",
    task="Search for Python tutorials"
)
assert delegation["target"] == "research"
assert delegation["priority"] == 1  # Default
print("✅ DelegationRequest factory works")

# Test 2: Graph compiles without error
graph = get_multi_agent(use_checkpointer=False)
print("✅ Graph compiled")

# Test 3: Graph can be invoked
initial = create_initial_state("Hello, what can you do?")
assert "messages" in initial
assert initial["current_agent"] == "manager"
print("✅ Initial state created")

result = graph.invoke(initial)
assert result["should_end"] == True
print("✅ Graph invocation succeeded")

print("\n🎉 Phase 1 complete!")
```

---

## 📚 Research Notes

> These notes document findings from deep research on CopilotKit, LangGraph, and
> TypedDict patterns. **Last updated:** November 29, 2025

### CopilotKitState Analysis

**Source:** `copilotkit.langgraph.CopilotKitState` (v0.1.40+)

```python
# Actual implementation from CopilotKit SDK:
class CopilotKitState(TypedDict, total=False):
    """
    Base state class for LangGraph agents.

    Attributes:
        messages: List of messages in the conversation (with add_messages reducer)
        steps: List of thinking steps for progress display
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    steps: list[ThinkingStep]  # For UI progress
```

**Key findings:**

1. `CopilotKitState` is a **TypedDict**, not a regular class
2. It already includes `messages` with the `add_messages` reducer
3. It includes `steps` for progress tracking
4. Uses `total=False` so all fields are optional

### Why @dataclass Fails with TypedDict

```python
# ❌ This causes a TypeError at class definition time:
from dataclasses import dataclass, field

@dataclass
class MyState(CopilotKitState):
    my_field: str = field(default_factory=str)

# TypeError: cannot inherit non-TypedDict class
```

**The fix:** Use TypedDict inheritance pattern with factory functions for
defaults.

### LangGraph State Access Pattern

When state is a TypedDict:

- Use `state["field"]` or `state.get("field")` for access
- Do NOT use `state.field` (attribute access)
- Return partial dicts from nodes (LangGraph merges them)

### add_messages Reducer

The `add_messages` reducer in LangGraph:

- Automatically deduplicates messages by ID
- Handles message updates/replacements
- Works with `Annotated[Sequence[BaseMessage], add_messages]`

**Do not redeclare** this in subclasses - it's already in CopilotKitState!

---

## 🔗 Next Phase

→ [Phase 2: Manager Agent](./phase2-manager.md) - Implement delegation logic and
reasoning emission

---

_Created: November 29, 2025_ _Updated: November 29, 2025 - Fixed TypedDict
patterns based on research_
