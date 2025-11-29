# Multi-Agent System Design Plan

> **Branch:** `feature/multi-agent` > **Date:** November 29, 2025 **Status:**

> Planning

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Design Decisions](#design-decisions)
3. [Architecture](#architecture)
4. [Agent Hierarchy](#agent-hierarchy)
5. [Tool Distribution](#tool-distribution)
6. [State Management](#state-management)
7. [Show Reasoning (CopilotKit Integration)](#show-reasoning-copilotkit-integration)
8. [File Structure](#file-structure)
9. [Implementation Plan](#implementation-plan)
10. [Dependencies](#dependencies)

---

## Overview

Transform the current monolithic KGAgent into a **hierarchical multi-agent

system** using:

- **LangGraph** for orchestration and state management
- **Pydantic AI** for individual specialist agents
- **CopilotKit** for real-time reasoning display in the UI

### Goals

- Split large system prompt into smaller, focused prompts per agent
- Improve response quality through specialization
- Enable transparent reasoning visible to users
- Maintain sequential execution (local LLM constraint)
- Keep all routing through a central Manager for traceability

---

## Design Decisions

| Decision              | Choice                  | Rationale                                                 |

| --------------------- | ----------------------- | --------------------------------------------------------- |

| **Execution Model**   | Sequential              | Single local LLM endpoint (LM Studio) - can't parallelize |

| **Routing Pattern**   | Manager-only routing    | All requests go through Manager for clear logic flow      |

| **State Persistence** | LangGraph checkpointing | Built-in, enables session continuity                      |

| **Show Reasoning**    | CopilotKit `emit_state` | Real-time UI updates as agents think                      |

| **Agent Framework**   | Pydantic AI + LangGraph | Type-safe, already in use, familiar patterns              |

---

## Architecture

### High-Level Flow

```
                          ┌─────────────────────────────────┐
                          │         🎯 MANAGER              │
                          │      (Orchestrator Agent)       │
                          │                                 │
                          │  System Prompt: ~200 tokens     │
                          │  Tools: delegate_to_*           │
                          │  Role: Route & Synthesize       │
                          └───────────────┬─────────────────┘
                                          │
            ┌─────────────────────────────┼─────────────────────────────┐
            │                             │                             │
            ▼                             ▼                             ▼
┌───────────────────────┐   ┌───────────────────────┐   ┌───────────────────────┐
│  🔍 RESEARCH LEAD     │   │  🧠 MEMORY LEAD       │   │  📊 KNOWLEDGE LEAD    │
│                       │   │                       │   │                       │
│  Prompt: ~300 tokens  │   │  Prompt: ~300 tokens  │   │  Prompt: ~300 tokens  │
│                       │   │                       │   │                       │
│  Tools:               │   │  Tools:               │   │  Tools:               │
│  • search_knowledge   │   │  • recall_convos      │   │  • create_entity      │
│  • search_by_source   │   │  • remember_user      │   │  • create_relationship│
│  • get_db_stats       │   │  • add_user_info      │   │  • search_graph       │
│                       │   │  • get_user_profile   │   │  • get_graph_stats    │
└───────────────────────┘   └───────────────────────┘   └───────────────────────┘
            │
            │
            ▼
┌───────────────────────┐
│  📁 DOCUMENT LEAD     │
│                       │
│  Prompt: ~300 tokens  │
│                       │
│  Tools:               │
│  • list_documents     │
│  • get_doc_stats      │
│  • delete_document    │
│  • delete_by_source   │
│  • clear_all_data     │
└───────────────────────┘
```

### Request Flow Example

```
User: "Research LangGraph, remember I'm interested in it, and create entities"

┌──────────────────────────────────────────────────────────────┐
│ MANAGER receives request                                      │
│                                                               │
│ Thinks: "This needs Research, Memory, AND Knowledge"         │
│                                                               │
│ Plan:                                                         │
│   1. delegate_to_research("Find info about LangGraph")       │
│   2. delegate_to_memory("Remember user is interested")       │
│   3. delegate_to_knowledge("Create entities for concepts")   │
└──────────────────────────────────────────────────────────────┘
                              │
           ┌──────────────────┼──────────────────┐
           ▼                  ▼                  ▼
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │  RESEARCH   │    │   MEMORY    │    │  KNOWLEDGE  │
    │             │    │             │    │             │
    │ search_kb   │    │ remember    │    │ create_ent  │
    │ ("LangGraph │    │ ("User is   │    │ ("LangGraph"│
    │  framework")│    │  interested │    │  Technology)│
    │             │    │  in Lang-   │    │             │
    │ Returns:    │    │  Graph")    │    │ create_rel  │
    │ "LangGraph  │    │             │    │ (LangGraph  │
    │  is..."     │    │ Returns:    │    │  PART_OF    │
    │             │    │ "Stored!"   │    │  LangChain) │
    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
           │                  │                  │
           └──────────────────┼──────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────┐
│ MANAGER synthesizes results                                   │
│                                                               │
│ "LangGraph is a framework for building stateful AI agents... │
│  I've noted your interest in it and created entities for     │
│  the key concepts: LangGraph (Technology), StateGraph..."    │
└──────────────────────────────────────────────────────────────┘
```

---

## Agent Hierarchy

### Manager Agent

**Role:** Orchestrator - routes requests and synthesizes responses

**System Prompt (~200 tokens):**

```
You are the Manager of a knowledge management system. Your job is to:
1. Understand what the user needs
2. Delegate to the right specialist(s)
3. Combine results into a helpful response

You have 4 team leads:
- Research Lead: Finding and searching information
- Memory Lead: User preferences, past conversations, personal context
- Knowledge Lead: Creating entities and relationships in the knowledge graph
- Document Lead: Managing documents (list, delete, stats)

Analyze the user's request and delegate to ONE OR MORE leads as needed.
Always synthesize the results into a coherent response.
```

**Tools:**

- `delegate_to_research(task: str)`
- `delegate_to_memory(task: str)`
- `delegate_to_knowledge(task: str)`
- `delegate_to_documents(task: str)`

---

### Research Lead

**Role:** Finding and searching information

**System Prompt (~250 tokens):**

```
You are the Research Lead. You specialize in finding information.

Your capabilities:
- Search the knowledge base (vector, graph, or hybrid search)
- Search by source pattern (find docs from specific sites)
- Get database statistics

When given a research task:
1. Determine the best search strategy
2. Execute the search
3. Return relevant findings with sources

Be thorough but concise. Always cite your sources.
```

**Tools:**

- `search_knowledge_base(query, search_type)`
- `search_by_source(source_pattern, limit)`
- `get_database_statistics()`

---

### Memory Lead

**Role:** User context and conversation history

**System Prompt (~250 tokens):**

```
You are the Memory Lead. You are the guardian of the user's personal context.

Your capabilities:
- Recall past conversations and discussions
- Remember facts about the user
- Store user preferences and information
- Retrieve the user's profile summary

When given a memory task:
1. Search for relevant past context
2. Store new information appropriately
3. Return what you know about the topic

Be personal and helpful. Remember that context matters.
```

**Tools:**

- `recall_past_conversations(query, limit)`
- `remember_about_user(fact, category)`
- `add_user_info(property_name, property_value)`
- `get_user_profile_summary()`

---

### Knowledge Lead

**Role:** Building and maintaining the knowledge graph

**System Prompt (~250 tokens):**

```
You are the Knowledge Lead. You build and maintain the knowledge graph.

Your capabilities:
- Create entities (Person, Organization, Technology, Concept, etc.)
- Create relationships between entities
- Search the graph structure
- Get graph statistics

When given a knowledge task:
1. Identify entities and relationships to create
2. Use proper entity types and relationship names
3. Report what was created/found

Be structured and precise. Use clear entity types and relationship names.
```

**Tools:**

- `create_entity(name, entity_type, description)`
- `create_relationship(source_entity, target_entity, relationship_type)`
- `search_graph(query, limit)` *(new)*
- `get_graph_stats()` *(new)*

---

### Document Lead

**Role:** Document lifecycle management

**System Prompt (~250 tokens):**

```
You are the Document Lead. You manage the document lifecycle.

Your capabilities:
- List documents with filters (status, type, search)
- Get document statistics
- Delete individual documents
- Delete documents by source
- Clear all data (requires confirmation)

When given a document task:
1. Understand what documents are involved
2. Execute the appropriate action
3. Report the results clearly

Be careful with deletions. Always confirm destructive actions.
```

**Tools:**

- `list_documents(status, source_type, search, limit)`
- `get_document_statistics()`
- `delete_document(doc_id, delete_vectors, delete_graph_nodes)`
- `delete_documents_by_source(source_pattern)`
- `clear_all_data(confirm)`

---

## Tool Distribution

| Agent                 | Tools                                                                                                          | Count |

| --------------------- | -------------------------------------------------------------------------------------------------------------- | ----- |

| **🎯 Manager**        | `delegate_to_research`, `delegate_to_memory`, `delegate_to_knowledge`, `delegate_to_documents`                 | 4     |

| **🔍 Research Lead**  | `search_knowledge_base`, `search_by_source`, `get_database_statistics`                                         | 3     |

| **🧠 Memory Lead**    | `recall_past_conversations`, `remember_about_user`, `add_user_info`, `get_user_profile_summary`                | 4     |

| **📊 Knowledge Lead** | `create_entity`, `create_relationship`, `search_graph`, `get_graph_stats`                                      | 4     |

| **📁 Document Lead**  | `list_documents`, `get_document_statistics`, `delete_document`, `delete_documents_by_source`, `clear_all_data` | 5     |

**Total:** 20 tools across 5 agents (vs 14 tools in 1 agent before)

---

## State Management

### Shared State Definition

```python
from copilotkit import CopilotKitState
from typing import Literal, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class ThinkingStep:
    """A single reasoning step from an agent."""
    agent: str                    # "manager", "research", "memory", etc.
    thought: str                  # What the agent is thinking
    action: Optional[str] = None  # What action it's taking (tool name)
    status: Literal["thinking", "delegating", "executing", "complete"] = "thinking"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

class MultiAgentState(CopilotKitState):
    """Shared state across all agents in the hierarchy."""

    # === Reasoning Trail ===
    thinking_steps: List[ThinkingStep] = field(default_factory=list)

    # === Current Execution Context ===
    current_agent: str = "manager"
    delegation_target: Optional[str] = None
    delegation_task: Optional[str] = None

    # === Results from Specialists ===
    research_result: Optional[str] = None
    memory_result: Optional[str] = None
    knowledge_result: Optional[str] = None
    document_result: Optional[str] = None

    # === Final Output ===
    final_response: Optional[str] = None

    # === Metadata ===
    total_llm_calls: int = 0
    execution_path: List[str] = field(default_factory=list)  # ["manager", "research", "manager"]
```

### State Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        STATE UPDATES                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. User sends message                                          │
│     state.messages.append(user_message)                         │
│                                                                  │
│  2. Manager thinks                                              │
│     state.thinking_steps.append({"agent": "manager", ...})      │
│     state.current_agent = "manager"                             │
│     emit_state(config, state)  ← UI updates!                    │
│                                                                  │
│  3. Manager delegates                                           │
│     state.delegation_target = "research"                        │
│     state.thinking_steps.append({"status": "delegating"})       │
│     emit_state(config, state)  ← UI shows delegation            │
│                                                                  │
│  4. Specialist executes                                         │
│     state.current_agent = "research"                            │
│     state.thinking_steps.append({"status": "executing"})        │
│     emit_state(config, state)  ← UI shows tool execution        │
│                                                                  │
│  5. Specialist completes                                        │
│     state.research_result = "..."                               │
│     state.thinking_steps.append({"status": "complete"})         │
│     emit_state(config, state)  ← UI shows completion            │
│                                                                  │
│  6. Manager synthesizes                                         │
│     state.final_response = "..."                                │
│     emit_state(config, state)  ← UI shows final response        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Show Reasoning (CopilotKit Integration)

### Python Backend Functions

```python
from copilotkit.langgraph import (
    copilotkit_emit_state,      # Emit state updates mid-execution
    copilotkit_emit_message,    # Emit messages mid-execution
    copilotkit_emit_tool_call,  # Emit tool calls
    copilotkit_customize_config # Configure what gets streamed
)
from copilotkit import CopilotKitState  # Base state class
```

| Function                                        | Purpose                  | When to Use                       |

| ----------------------------------------------- | ------------------------ | --------------------------------- |

| `copilotkit_emit_state(config, state)`          | Stream state to frontend | After each thinking step          |

| `copilotkit_emit_message(config, msg)`          | Send message to UI       | Status updates like "Thinking..." |

| `copilotkit_emit_tool_call(config, name, args)` | Show tool execution      | Before executing a tool           |

### Example: Manager Node with Reasoning

```python
async def manager_node(state: MultiAgentState, config: RunnableConfig):
    """Manager analyzes request and delegates to specialists."""

    # Step 1: Show we're thinking
    state["thinking_steps"].append(ThinkingStep(
        agent="manager",
        thought="Analyzing user request to determine which specialist(s) to delegate to...",
        status="thinking"
    ))
    await copilotkit_emit_state(config, state)

    # Step 2: LLM decides delegation (actual inference)
    response = await manager_llm.ainvoke(...)
    delegation_target = parse_delegation(response)

    # Step 3: Show delegation decision
    state["thinking_steps"].append(ThinkingStep(
        agent="manager",
        thought=f"This request requires {delegation_target} expertise",
        action=f"delegate_to_{delegation_target}",
        status="delegating"
    ))
    state["delegation_target"] = delegation_target
    await copilotkit_emit_state(config, state)

    return {"delegation_target": delegation_target, ...}
```

### Frontend: Rendering in Chat

```tsx
// components/agent-reasoning.tsx
import { useCoAgentStateRender } from '@copilotkit/react-core';

type ThinkingStep = {
  agent: string;
  thought: string;
  action?: string;
  status: 'thinking' | 'delegating' | 'executing' | 'complete';
};

type AgentState = {
  thinking_steps: ThinkingStep[];
  current_agent: string;
};

const agentIcons: Record<string, string> = {
  manager: '🎯',
  research: '🔍',
  memory: '🧠',
  knowledge: '📊',
  documents: '📁',
};

const statusColors: Record<string, string> = {
  thinking: 'text-yellow-500',
  delegating: 'text-blue-500',
  executing: 'text-purple-500',
  complete: 'text-green-500',
};

export function AgentReasoningRenderer() {
  useCoAgentStateRender<AgentState>({
    name: 'kg_multi_agent',
    render: ({ state }) => {
      if (!state.thinking_steps?.length) return null;

      return (
        <div className="bg-gray-900 rounded-lg p-3 my-2 font-mono text-sm">
          <div className="text-gray-400 mb-2">🧠 Agent Reasoning:</div>
          {state.thinking_steps.map((step, i) => (
            <div
              key={i}
              className="flex items-start gap-2 py-1 border-l-2 border-gray-700 pl-2 ml-2"
            >
              <span>{agentIcons[step.agent] || '🤖'}</span>
              <div>
                <span className="text-gray-300 font-semibold">
                  {step.agent}
                </span>
                <span className={`ml-2 text-xs ${statusColors[step.status]}`}>
                  [{step.status}]
                </span>
                <p className="text-gray-400">{step.thought}</p>
                {step.action && (
                  <p className="text-blue-400 text-xs">→ {step.action}</p>
                )}
              </div>
            </div>
          ))}
        </div>
      );
    },
  });

  return null;
}
```

### Frontend: Status Panel (Outside Chat)

```tsx
// components/agent-status-panel.tsx
import { useCoAgent } from '@copilotkit/react-core';

export function AgentStatusPanel() {
  const { state } = useCoAgent<AgentState>({ name: 'kg_multi_agent' });

  return (
    <div className="fixed bottom-4 right-4 bg-gray-800 p-4 rounded-lg w-80">
      <h3 className="text-white font-bold mb-2">🤖 Agent Status</h3>

      <div className="text-sm space-y-1">
        <p className="text-gray-400">
          Current: <span className="text-white">{state.current_agent}</span>
        </p>

        {state.delegation_target && (
          <p className="text-blue-400">
            → Delegating to: {state.delegation_target}
          </p>
        )}

        <p className="text-gray-400">
          Steps: {state.thinking_steps?.length || 0}
        </p>

        <p className="text-gray-400">LLM Calls: {state.total_llm_calls || 0}</p>
      </div>
    </div>
  );
}
```

---

## File Structure

```
src/kg_agent/
├── agent/
│   ├── __init__.py
│   ├── kg_agent.py              # Original monolithic agent (backward compat)
│   ├── tools.py                 # Original tools (shared utilities)
│   ├── README.md
│   │
│   └── multi/                   # NEW: Multi-agent system
│       ├── __init__.py          # Exports: get_multi_agent, MultiAgentState
│       ├── state.py             # MultiAgentState, ThinkingStep definitions
│       ├── graph.py             # LangGraph StateGraph definition
│       ├── manager.py           # Manager agent (orchestrator)
│       ├── research_lead.py     # Research specialist agent
│       ├── memory_lead.py       # Memory specialist agent
│       ├── knowledge_lead.py    # Knowledge graph specialist agent
│       ├── document_lead.py     # Document management specialist agent
│       └── prompts.py           # All system prompts in one place
│
├── api/routes/
│   ├── agent.py                 # Original agent routes
│   └── multi_agent.py           # NEW: Multi-agent API endpoints
│
└── services/
    └── (existing services)

dashboard/
├── components/
│   ├── agent-reasoning.tsx      # NEW: Reasoning renderer for chat
│   ├── agent-status-panel.tsx   # NEW: Status panel component
│   └── (existing components)
```

---

## Implementation Plan

> **📁 Detailed phase plans are available in separate files:**

### Phase 1: Foundation (Core Infrastructure)
> **[📄 Detailed Plan: phase1-foundation.md](./phase1-foundation.md)** | Est: 2-3 hours

- [ ] Create `src/kg_agent/agent/multi/` directory structure
- [ ] Define `state.py` with `MultiAgentState` and `ThinkingStep`
- [ ] Create `prompts.py` with all agent system prompts
- [ ] Set up basic LangGraph `graph.py` skeleton

### Phase 2: Manager Agent
> **[📄 Detailed Plan: phase2-manager.md](./phase2-manager.md)** | Est: 3-4 hours

- [ ] Implement `manager.py` with delegation logic
- [ ] Add reasoning emission (`copilotkit_emit_state`)
- [ ] Create delegation tools (`delegate_to_*`)
- [ ] Test Manager routing with mock specialists

### Phase 3: Specialist Agents
> **[📄 Detailed Plan: phase3-specialists.md](./phase3-specialists.md)** | Est: 4-5 hours

- [ ] Implement `research_lead.py` (port existing search tools)
- [ ] Implement `memory_lead.py` (port existing memory tools)
- [ ] Implement `knowledge_lead.py` (port existing entity tools)
- [ ] Implement `document_lead.py` (port existing document tools)

### Phase 4: LangGraph Wiring
> **[📄 Detailed Plan: phase4-langgraph.md](./phase4-langgraph.md)** | Est: 3-4 hours

- [ ] Complete `graph.py` with all nodes and edges
- [ ] Add conditional routing based on Manager decisions
- [ ] Implement synthesize node for final response
- [ ] Add LangGraph checkpointing for persistence

### Phase 5: API Integration
> **[📄 Detailed Plan: phase5-api.md](./phase5-api.md)** | Est: 2-3 hours

- [ ] Create `api/routes/multi_agent.py` endpoints
- [ ] Integrate with CopilotKit remote endpoint
- [ ] Test full request/response cycle

### Phase 6: Frontend Integration
> **[📄 Detailed Plan: phase6-frontend.md](./phase6-frontend.md)** | Est: 3-4 hours

- [ ] Create `agent-reasoning.tsx` component
- [ ] Add `useCoAgentStateRender` for in-chat reasoning
- [ ] Create `agent-status-panel.tsx` for outside-chat status
- [ ] Update `copilot-provider.tsx` to register multi-agent

### Phase 7: Testing & Polish
> **[📄 Detailed Plan: phase7-testing.md](./phase7-testing.md)** | Est: 3-4 hours

- [ ] Write integration tests
- [ ] Test with various query types
- [ ] Performance optimization
- [ ] Documentation updates

---

### 📊 Total Estimated Effort: 21-27 hours

---

## Dependencies

### Python (Already Installed)

```toml
# pyproject.toml
dependencies = [
    # ... existing ...
    "langgraph>=0.2.0",
    "langchain-core>=0.3.0",
    "langchain-openai>=0.2.0",
    "langsmith>=0.1.0",
    "copilotkit>=0.1.72",
]
```

### Dashboard (Check if needed)

```json
// package.json - verify these are present
{
  "dependencies": {
    "@copilotkit/react-core": "latest",
    "@copilotkit/react-ui": "latest"
  }
}
```

---

## Open Questions

1. **Fallback behavior:** What happens if a specialist fails? Should Manager

retry or return partial results?

2. **Maximum delegation depth:** Should we limit how many specialists can be

called per request?

3. **Caching:** Should specialist results be cached for similar queries?

4. **User interruption:** Should users be able to cancel mid-execution?

5. **Streaming vs final:** Should final response stream or wait for full

synthesis?

---

## References

- [CopilotKit LangGraph SDK](https://docs.copilotkit.ai/reference/sdk/python/LangGraph)
- [CopilotKit Generative UI](https://docs.copilotkit.ai/coagents/generative-ui)
- [CopilotKit Predictive State Updates](https://docs.copilotkit.ai/coagents/shared-state/predictive-state-updates)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Pydantic AI Documentation](https://ai.pydantic.dev/)

---

*Last updated: November 29, 2025*