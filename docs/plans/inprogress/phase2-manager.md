# Phase 2: Manager Agent - The Orchestrator

> **Parent Plan:** [multi_agent.plan.md](./multi_agent.plan.md)
> **Status:** Not Started
> **Estimated Effort:** 3-4 hours
> **Dependencies:** Phase 1 complete
> **Last Technical Review:** November 29, 2025

---

## 🎯 Objectives

1. Implement the Manager agent with delegation logic
2. Create delegation tools (`delegate_to_*`)
3. Add real-time reasoning emission via CopilotKit
4. Test manager routing with mock specialist responses

---

## 📋 Prerequisites

- [ ] Phase 1 complete (state, prompts, graph skeleton)
- [ ] `MultiAgentState` and `ThinkingStep` defined
- [ ] LangGraph skeleton compiles

---

## ⚠️ Technical Notes (Research-Validated)

### Pydantic AI API (v0.1.x - Current)
- **Use `output_type`** not `result_type` for Agent constructor
- **Access result via `result.output`** (not `.data`)
- `OpenAIChatModel` + `OpenAIProvider` pattern is correct

### CopilotKit State Emission
- `copilotkit_emit_state(config, state_dict)` expects a **JSON-serializable dict**
- Don't pass `state.__dict__` directly for dataclass-based state
- Use explicit dict conversion for nested dataclasses

### LangGraph State Management
- Phase 1 correctly uses `CopilotKitState` base class (dataclass)
- `ThinkingStep` and `DelegationRequest` are dataclasses (good - avoids Pydantic serialization issues)
- Nodes must return **dict updates**, not full state objects

---

## 🧠 Manager Design Overview

The Manager is the **only** agent that decides routing. It:

1. Receives the user message
2. Analyzes intent to determine which specialist(s) to invoke
3. Emits thinking steps for UI transparency
4. Queues delegations
5. After all specialists complete, synthesizes the final response

### Manager Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         MANAGER NODE                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. Receive State                                                        │
│     └── Check messages[-1] for user input                               │
│                                                                          │
│  2. Emit "Thinking" Step                                                 │
│     └── copilotkit_emit_state() → UI shows "🎯 Manager is analyzing..." │
│                                                                          │
│  3. LLM Inference (Intent Classification)                               │
│     └── Determine: Research? Memory? Knowledge? Documents? Multiple?    │
│                                                                          │
│  4. Emit "Delegating" Steps (one per specialist)                        │
│     └── For each target: emit_state with delegation info                │
│                                                                          │
│  5. Update State                                                         │
│     └── delegation_queue = [DelegationRequest(...), ...]                │
│     └── execution_path.append("manager")                                │
│     └── total_llm_calls += 1                                            │
│                                                                          │
│  6. Return Updated State                                                 │
│     └── Router picks next node based on delegation_queue                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Task 1: Create Manager Agent Implementation

### File: `src/kg_agent/agent/multi/manager.py`

```python
"""Manager Agent - Orchestrator for the multi-agent system.

The Manager:
1. Analyzes user intent
2. Delegates to specialist agents
3. Synthesizes final responses
4. Emits reasoning steps to the UI
"""

import json
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from ...core.config import settings
from ...core.logging import logger

try:
    from copilotkit.langgraph import copilotkit_emit_state
except ImportError:
    # Fallback for local testing without CopilotKit
    async def copilotkit_emit_state(config, state):
        logger.debug(f"copilotkit_emit_state fallback: {state.get('current_agent', 'unknown')}")

from .prompts import MANAGER_SYSTEM_PROMPT
from .state import DelegationRequest, MultiAgentState, ThinkingStep


# === Delegation Decision Schema ===

class DelegationDecision(BaseModel):
    """Structured output from Manager's intent analysis.

    This Pydantic model constrains the LLM output to a reliable format
    that can be parsed and acted upon without string parsing.
    """

    reasoning: str = Field(
        description="Brief explanation of why these specialists are needed"
    )

    delegations: List[Dict[str, str]] = Field(
        default_factory=list,
        description="List of delegations: [{'target': 'research|memory|knowledge|documents', 'task': 'what to do'}]"
    )

    needs_clarification: bool = Field(
        default=False,
        description="True if the request is too vague to delegate"
    )

    clarification_question: Optional[str] = Field(
        default=None,
        description="Question to ask user if needs_clarification is True"
    )


# === LLM Setup ===

def create_manager_llm() -> OpenAIChatModel:
    """Create the LLM for the Manager agent.

    Uses OpenAIProvider for OpenAI-compatible APIs (LM Studio, Ollama, etc.)
    """
    provider = OpenAIProvider(
        base_url=settings.LLM_BASE_URL,
        api_key=settings.LLM_API_KEY or "not-needed",  # Some local LLMs don't require keys
    )
    return OpenAIChatModel(
        settings.LLM_MODEL_NAME,
        provider=provider,
    )


# === Manager Pydantic AI Agent ===
# NOTE: Use `output_type` (not `result_type`) - Pydantic AI v0.1.x API

manager_agent = Agent(
    model=create_manager_llm(),
    system_prompt=MANAGER_SYSTEM_PROMPT,
    output_type=DelegationDecision,  # CORRECT: output_type, not result_type
    retries=2,
)


# === Intent Classification Prompt ===

DELEGATION_PROMPT = """Analyze the user's request and decide which specialist(s) to delegate to.

User Request: {user_message}

Previous Context (if any):
{context}

Available Specialists:
- research: For searching/finding information in the knowledge base
- memory: For user preferences, past conversations, personal context
- knowledge: For creating entities/relationships in the knowledge graph
- documents: For listing, deleting, or managing documents

Respond with JSON:
{{
  "reasoning": "Brief explanation",
  "delegations": [
    {{"target": "specialist_name", "task": "specific task description"}}
  ],
  "needs_clarification": false,
  "clarification_question": null
}}

Rules:
- You can delegate to MULTIPLE specialists if needed
- Be specific in task descriptions
- If the request is unclear, set needs_clarification=true
"""


# === Helper Functions ===

def state_to_emittable_dict(state: MultiAgentState, **overrides) -> Dict[str, Any]:
    """Convert MultiAgentState to a JSON-serializable dict for CopilotKit emission.

    CopilotKit's copilotkit_emit_state() requires a plain dict that can be
    JSON-serialized. This helper properly converts dataclass instances.

    Args:
        state: The current MultiAgentState
        **overrides: Values to override in the emitted state

    Returns:
        JSON-serializable dict suitable for copilotkit_emit_state()
    """
    # Convert thinking_steps (list of dataclasses) to list of dicts
    thinking_steps = overrides.get("thinking_steps", state.thinking_steps)
    serialized_steps = [
        asdict(step) if hasattr(step, '__dataclass_fields__') else step
        for step in thinking_steps
    ]

    # Convert delegation_queue
    delegation_queue = overrides.get("delegation_queue", state.delegation_queue)
    serialized_delegations = [
        asdict(d) if hasattr(d, '__dataclass_fields__') else d
        for d in delegation_queue
    ]

    # Build the emittable dict
    return {
        "current_agent": overrides.get("current_agent", state.current_agent),
        "thinking_steps": serialized_steps,
        "delegation_queue": serialized_delegations,
        "research_result": overrides.get("research_result", state.research_result),
        "memory_result": overrides.get("memory_result", state.memory_result),
        "knowledge_result": overrides.get("knowledge_result", state.knowledge_result),
        "document_result": overrides.get("document_result", state.document_result),
        "final_response": overrides.get("final_response", state.final_response),
        "should_end": overrides.get("should_end", state.should_end),
    }


def add_thinking_step(
    state: MultiAgentState,
    thought: str,
    status: str = "thinking",
    action: Optional[str] = None,
    result: Optional[str] = None,
) -> ThinkingStep:
    """Create and add a thinking step to state."""
    step = ThinkingStep(
        agent="manager",
        thought=thought,
        status=status,
        action=action,
        result=result,
    )
    state.thinking_steps.append(step)
    return step


def get_conversation_context(state: MultiAgentState, max_turns: int = 3) -> str:
    """Get recent conversation context for the LLM."""
    messages = state.messages[-max_turns * 2:] if state.messages else []

    context_parts = []
    for msg in messages:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
        context_parts.append(f"{role}: {content}")

    return "\n".join(context_parts) if context_parts else "No previous context"


# === Main Manager Node ===

async def manager_node(
    state: MultiAgentState,
    config: RunnableConfig
) -> Dict[str, Any]:
    """
    Manager node implementation.

    Analyzes user intent and creates delegation queue.
    Emits thinking steps for UI display.

    Args:
        state: Current multi-agent state
        config: LangGraph runtime config (contains CopilotKit context)

    Returns:
        State updates including delegation_queue
    """
    logger.info("Manager node executing")

    # Track execution
    execution_path = list(state.execution_path)
    execution_path.append("manager")

    # Get the latest user message
    user_message = ""
    if state.messages:
        last_msg = state.messages[-1]
        if isinstance(last_msg, HumanMessage):
            user_message = last_msg.content

    if not user_message:
        logger.warning("No user message found in state")
        return {
            "final_response": "I didn't receive a message. How can I help you?",
            "should_end": True,
            "execution_path": execution_path,
        }

    # Step 1: Emit "analyzing" thinking step
    thinking_steps = list(state.thinking_steps)
    thinking_steps.append(ThinkingStep(
        agent="manager",
        thought=f"Analyzing request: '{user_message[:100]}{'...' if len(user_message) > 100 else ''}'",
        status="thinking",
    ))

    # Emit state to UI - use helper for proper serialization
    await copilotkit_emit_state(
        config,
        state_to_emittable_dict(
            state,
            thinking_steps=thinking_steps,
            current_agent="manager",
        )
    )

    # Step 2: Get conversation context
    context = get_conversation_context(state)

    # Step 3: Run LLM to decide delegations
    prompt = DELEGATION_PROMPT.format(
        user_message=user_message,
        context=context,
    )

    try:
        result = await manager_agent.run(prompt)
        decision: DelegationDecision = result.output

        logger.info(f"Manager decision: {decision.reasoning}")
        logger.info(f"Delegations: {decision.delegations}")

    except Exception as e:
        logger.error(f"Manager LLM failed: {e}")
        # Fallback: try research for any query
        decision = DelegationDecision(
            reasoning="Defaulting to research due to analysis error",
            delegations=[{"target": "research", "task": user_message}],
            needs_clarification=False,
        )

    # Step 4: Handle clarification needed
    if decision.needs_clarification:
        thinking_steps.append(ThinkingStep(
            agent="manager",
            thought=f"Need clarification: {decision.clarification_question}",
            status="complete",
        ))

        return {
            "thinking_steps": thinking_steps,
            "final_response": decision.clarification_question,
            "should_end": True,
            "execution_path": execution_path,
            "total_llm_calls": state.total_llm_calls + 1,
        }

    # Step 5: Create delegation queue
    delegation_queue = []
    for d in decision.delegations:
        target = d.get("target", "").lower()
        task = d.get("task", user_message)

        if target in ["research", "memory", "knowledge", "documents"]:
            delegation_queue.append(DelegationRequest(
                target=target,
                task=task,
                context=context,
            ))

            # Emit delegation step
            thinking_steps.append(ThinkingStep(
                agent="manager",
                thought=f"Delegating to {target}: {task[:80]}{'...' if len(task) > 80 else ''}",
                status="delegating",
                action=f"delegate_to_{target}",
            ))

    # Emit updated state with delegations - use helper for serialization
    await copilotkit_emit_state(
        config,
        state_to_emittable_dict(
            state,
            thinking_steps=thinking_steps,
            delegation_queue=delegation_queue,
            current_agent="manager",
        )
    )

    # No delegations? Provide direct response
    if not delegation_queue:
        thinking_steps.append(ThinkingStep(
            agent="manager",
            thought="No specialist needed - responding directly",
            status="complete",
        ))

        return {
            "thinking_steps": thinking_steps,
            "final_response": f"I understand you're asking about: {user_message}. However, I'm not sure which specialist can help with this. Could you rephrase or be more specific?",
            "should_end": True,
            "execution_path": execution_path,
            "total_llm_calls": state.total_llm_calls + 1,
        }

    # Return state updates
    return {
        "thinking_steps": thinking_steps,
        "delegation_queue": delegation_queue,
        "current_delegation": delegation_queue[0] if delegation_queue else None,
        "execution_path": execution_path,
        "total_llm_calls": state.total_llm_calls + 1,
        "current_agent": delegation_queue[0].target if delegation_queue else "synthesize",
    }


# === Synthesize Node ===

async def synthesize_node(
    state: MultiAgentState,
    config: RunnableConfig,
) -> Dict[str, Any]:
    """
    Synthesize specialist results into final response.

    Called after all delegations complete.
    """
    logger.info("Synthesize node executing")

    thinking_steps = list(state.thinking_steps)
    execution_path = list(state.execution_path)
    execution_path.append("synthesize")

    # Collect all results
    results = []

    if state.research_result:
        results.append(("Research", state.research_result))
    if state.memory_result:
        results.append(("Memory", state.memory_result))
    if state.knowledge_result:
        results.append(("Knowledge", state.knowledge_result))
    if state.document_result:
        results.append(("Documents", state.document_result))

    # Emit synthesizing step
    thinking_steps.append(ThinkingStep(
        agent="manager",
        thought=f"Synthesizing {len(results)} specialist result(s)...",
        status="thinking",
    ))
    await copilotkit_emit_state(
        config,
        state_to_emittable_dict(state, thinking_steps=thinking_steps)
    )

    # Simple synthesis (Phase 4 will add LLM-based synthesis)
    if not results:
        final_response = "I wasn't able to gather relevant information. Could you try rephrasing your question?"
    elif len(results) == 1:
        # Single result - use directly
        final_response = results[0][1]
    else:
        # Multiple results - combine with headers
        parts = []
        for name, content in results:
            parts.append(f"**{name}:**\n{content}")
        final_response = "\n\n".join(parts)

    # Mark complete
    thinking_steps.append(ThinkingStep(
        agent="manager",
        thought="Response ready",
        status="complete",
    ))

    await copilotkit_emit_state(
        config,
        state_to_emittable_dict(
            state,
            thinking_steps=thinking_steps,
            final_response=final_response,
            should_end=True,
        )
    )

    return {
        "thinking_steps": thinking_steps,
        "final_response": final_response,
        "should_end": True,
        "execution_path": execution_path,
        "current_agent": "manager",
    }
```

### Key Design Patterns

| Pattern | Implementation | Why |
|---------|---------------|-----|
| Structured Output | `DelegationDecision` Pydantic model with `output_type` | Reliable parsing of LLM decisions |
| State Emission | `state_to_emittable_dict()` + `copilotkit_emit_state()` | Proper JSON serialization for UI |
| Graceful Fallback | Default to research on error | System stays functional |
| Context Window | Last 3 turns of conversation | Keep prompt size manageable |
| Dataclass Serialization | `asdict()` for nested dataclasses | Avoid pickle/JSON issues |

---

## 📁 Task 2: Update Graph to Use New Manager

### File: `src/kg_agent/agent/multi/graph.py` (updates)

```python
# Replace the placeholder imports and nodes with:

from .manager import manager_node, synthesize_node

# In create_multi_agent_graph(), update the node registrations:
graph.add_node("manager", manager_node)
graph.add_node("synthesize", synthesize_node)
```

---

## 📁 Task 3: Create Delegation Tools (Alternative Approach)

If using Pydantic AI tool decoration instead of structured output:

```python
"""Alternative: Manager delegation tools (if not using structured output)."""

from pydantic_ai import RunContext

@manager_agent.tool
async def delegate_to_research(
    ctx: RunContext,
    task: str
) -> str:
    """
    Delegate a task to the Research Lead.

    Use when you need to:
    - Search the knowledge base
    - Find information about a topic
    - Get database statistics

    Args:
        task: What the Research Lead should do

    Returns:
        Confirmation that the delegation was queued
    """
    # This would update shared state
    # Implementation depends on how state is passed
    return f"Delegated to Research: {task}"


@manager_agent.tool
async def delegate_to_memory(
    ctx: RunContext,
    task: str
) -> str:
    """
    Delegate a task to the Memory Lead.

    Use when you need to:
    - Recall past conversations
    - Store user preferences
    - Get user profile information

    Args:
        task: What the Memory Lead should do
    """
    return f"Delegated to Memory: {task}"


@manager_agent.tool
async def delegate_to_knowledge(
    ctx: RunContext,
    task: str
) -> str:
    """
    Delegate a task to the Knowledge Lead.

    Use when you need to:
    - Create entities in the knowledge graph
    - Create relationships between entities
    - Search the graph structure

    Args:
        task: What the Knowledge Lead should do
    """
    return f"Delegated to Knowledge: {task}"


@manager_agent.tool
async def delegate_to_documents(
    ctx: RunContext,
    task: str
) -> str:
    """
    Delegate a task to the Document Lead.

    Use when you need to:
    - List documents in the system
    - Delete documents
    - Get document statistics

    Args:
        task: What the Document Lead should do
    """
    return f"Delegated to Documents: {task}"
```

**Note:** The structured output approach (Task 1) is preferred because:
- More reliable than tool-based delegation
- Single LLM call instead of multiple tool invocations
- Easier to debug and test

---

## 📁 Task 4: Test Manager in Isolation

### Test File: `tests/test_manager.py`

```python
"""Unit tests for the Manager agent."""

import pytest
from unittest.mock import AsyncMock, patch
from langchain_core.messages import HumanMessage

from kg_agent.agent.multi.state import MultiAgentState, create_initial_state
from kg_agent.agent.multi.manager import (
    manager_node,
    synthesize_node,
    DelegationDecision,
)


@pytest.fixture
def mock_config():
    """Create a mock RunnableConfig."""
    return {}


@pytest.fixture
def sample_state():
    """Create a sample initial state."""
    return MultiAgentState(
        messages=[HumanMessage(content="Search for Python tutorials")],
        thinking_steps=[],
        current_agent="manager",
        delegation_queue=[],
        execution_path=[],
        total_llm_calls=0,
    )


class TestManagerIntentClassification:
    """Test Manager's intent classification."""

    @pytest.mark.asyncio
    async def test_research_intent(self, sample_state, mock_config):
        """Manager should delegate search queries to Research."""
        # Mock the LLM response
        with patch("kg_agent.agent.multi.manager.manager_agent") as mock_agent:
            mock_agent.run = AsyncMock(return_value=type("Result", (), {
                "output": DelegationDecision(
                    reasoning="User wants to search",
                    delegations=[{"target": "research", "task": "Search for Python tutorials"}],
                )
            })())

            result = await manager_node(sample_state, mock_config)

            assert len(result["delegation_queue"]) == 1
            assert result["delegation_queue"][0].target == "research"

    @pytest.mark.asyncio
    async def test_multiple_delegations(self, mock_config):
        """Manager should queue multiple specialists when needed."""
        state = MultiAgentState(
            messages=[HumanMessage(content="Remember my name is Steve and search for my projects")],
            thinking_steps=[],
            current_agent="manager",
            delegation_queue=[],
            execution_path=[],
            total_llm_calls=0,
        )

        with patch("kg_agent.agent.multi.manager.manager_agent") as mock_agent:
            mock_agent.run = AsyncMock(return_value=type("Result", (), {
                "output": DelegationDecision(
                    reasoning="User wants to store info and search",
                    delegations=[
                        {"target": "memory", "task": "Remember user's name is Steve"},
                        {"target": "research", "task": "Search for Steve's projects"},
                    ],
                )
            })())

            result = await manager_node(state, mock_config)

            assert len(result["delegation_queue"]) == 2
            assert result["delegation_queue"][0].target == "memory"
            assert result["delegation_queue"][1].target == "research"


class TestSynthesizeNode:
    """Test the synthesis node."""

    @pytest.mark.asyncio
    async def test_single_result_synthesis(self, mock_config):
        """Single specialist result should be returned directly."""
        state = MultiAgentState(
            messages=[],
            thinking_steps=[],
            research_result="Found 5 Python tutorials",
            execution_path=["manager", "research"],
        )

        result = await synthesize_node(state, mock_config)

        assert result["final_response"] == "Found 5 Python tutorials"
        assert result["should_end"] == True

    @pytest.mark.asyncio
    async def test_multiple_results_synthesis(self, mock_config):
        """Multiple results should be combined with headers."""
        state = MultiAgentState(
            messages=[],
            thinking_steps=[],
            research_result="Found tutorials",
            memory_result="User prefers video content",
            execution_path=["manager", "research", "memory"],
        )

        result = await synthesize_node(state, mock_config)

        assert "**Research:**" in result["final_response"]
        assert "**Memory:**" in result["final_response"]
```

---

## 📁 Task 5: Integration Test with Mock Specialists

```python
"""Integration test for Manager + mock specialists."""

import pytest
from kg_agent.agent.multi import get_multi_agent
from kg_agent.agent.multi.state import create_initial_state


@pytest.mark.asyncio
async def test_full_manager_flow():
    """Test complete manager flow with mock specialists."""
    graph = get_multi_agent(use_checkpointer=False)

    initial = create_initial_state("Search for LangGraph documentation")

    # Run the graph
    result = graph.invoke(initial)

    # Should complete
    assert result["should_end"] == True

    # Should have execution path
    assert "manager" in result["execution_path"]

    # Should have thinking steps
    assert len(result["thinking_steps"]) > 0

    # Should have final response
    assert result["final_response"] is not None
```

---

## ✅ Phase 2 Definition of Done

- [ ] `manager.py` implemented with:
  - [ ] Intent classification via structured output (`output_type=DelegationDecision`)
  - [ ] Delegation queue management
  - [ ] Thinking step emissions via `state_to_emittable_dict()`
  - [ ] Error handling / fallbacks

- [ ] `synthesize_node` implemented:
  - [ ] Combines specialist results
  - [ ] Emits completion thinking steps

- [ ] Tests pass:
  - [ ] Unit tests for intent classification
  - [ ] Unit tests for synthesis
  - [ ] Integration test with mock specialists
  - [ ] Verify state serialization works (no pickle errors)

- [ ] Manual verification:
  - [ ] Run `python -c "from kg_agent.agent.multi.manager import manager_node; print('OK')"`
  - [ ] Graph invocation completes without errors
  - [ ] CopilotKit receives state updates (check browser console/network)

---

## 🔗 Next Phase

→ [Phase 3: Specialist Agents](./phase3-specialists.md) - Port existing tools to specialist agents

---

## 📚 Research Notes & Technical References

### Key Library Versions (as of Nov 2025)
- **Pydantic AI**: v0.1.x - Uses `output_type` parameter
- **LangGraph**: v0.2.x - Supports both TypedDict and Pydantic BaseModel for state
- **CopilotKit Python SDK**: `copilotkit` package on PyPI

### Verified API Patterns

#### Pydantic AI Structured Output
```python
from pydantic_ai import Agent
from pydantic import BaseModel

class MyOutput(BaseModel):
    field: str

agent = Agent(
    model=my_model,
    output_type=MyOutput,  # NOT result_type
)
result = await agent.run(prompt)
print(result.output)  # Access via .output, NOT .data
```

#### CopilotKit State Emission
```python
from copilotkit.langgraph import copilotkit_emit_state

# copilotkit_emit_state expects JSON-serializable dict
await copilotkit_emit_state(config, {
    "key": "value",
    "nested": {"also": "works"},
    "lists": ["of", "primitives"]
})

# For dataclass state, convert explicitly:
from dataclasses import asdict
await copilotkit_emit_state(config, asdict(my_dataclass))
```

#### LangGraph Node Return Pattern
```python
# Nodes MUST return dict updates, not full state
async def my_node(state: MyState, config: RunnableConfig) -> Dict[str, Any]:
    # Do work...
    return {
        "field_to_update": new_value,
        # Only include fields that changed
    }
```

### Known Pitfalls

1. **Pydantic BaseModel in LangGraph State**: Can cause pickle serialization issues with nested models. Phase 1 correctly uses dataclasses for `ThinkingStep` and `DelegationRequest`.

2. **State Emission with `__dict__`**: Don't use `state.__dict__` directly - nested dataclasses won't serialize correctly to JSON.

3. **OpenAI Provider API Key**: Some local LLMs (LM Studio, Ollama) don't require API keys. Use `api_key="not-needed"` as a placeholder.

### Source Links
- [Pydantic AI Docs - Output](https://ai.pydantic.dev/output/)
- [CopilotKit LangGraph SDK](https://docs.copilotkit.ai/reference/sdk/python/LangGraph)
- [LangGraph State Model Discussion](https://github.com/langchain-ai/langgraph/discussions/1306)

---

*Created: November 29, 2025*
*Last Technical Review: November 29, 2025*

