# Multi-Agent System Architecture

> KGAgent's hierarchical multi-agent system for intelligent task orchestration.

## Overview

The KGAgent multi-agent system uses a **hierarchical architecture** where a Manager agent orchestrates specialized agents, each handling a specific domain:

```
                    ┌─────────────────┐
                    │     User        │
                    │   Message       │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │    Manager      │
                    │     Agent       │
                    │  (Orchestrator) │
                    └────────┬────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
    ┌──────▼──────┐  ┌───────▼───────┐  ┌──────▼──────┐
    │  Research   │  │    Memory     │  │  Knowledge  │
    │    Lead     │  │     Lead      │  │    Lead     │
    └─────────────┘  └───────────────┘  └─────────────┘
           │                 │                 │
           └─────────────────┼─────────────────┘
                             │
                    ┌────────▼────────┐
                    │   Synthesize    │
                    │    Response     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Final Response │
                    └─────────────────┘
```

## Agents

### Manager Agent
**Role:** Orchestration and intent analysis

- Analyzes user intent from messages
- Decides which specialist(s) to delegate to
- Can delegate to multiple specialists in sequence
- Synthesizes results from all specialists
- Handles clarification when intent is unclear

### Research Lead
**Role:** Knowledge base search and statistics

**Tools:**
- `search_knowledge_base` - Hybrid vector + graph search
- `search_vectors_only` - Pure vector similarity search
- `search_graph_only` - Graph traversal search
- `get_database_statistics` - Vector and graph stats

### Memory Lead
**Role:** User context and conversation history

**Tools:**
- `recall_conversation_context` - Get relevant past conversations
- `remember_about_user` - Store user facts/preferences
- `get_user_profile` - Retrieve user profile
- `summarize_recent_interactions` - Conversation summaries

### Knowledge Lead
**Role:** Knowledge graph entity management

**Tools:**
- `create_knowledge_entity` - Create new entities
- `find_related_entities` - Discover relationships
- `get_entity_details` - Entity information
- `list_entity_types` - Available entity types

### Document Lead
**Role:** Document lifecycle management

**Tools:**
- `list_documents` - List with filters
- `get_document_details` - Document metadata
- `get_document_statistics` - Document stats
- `delete_document` - Remove documents

## Quick Start

### Simple Query

```python
from kg_agent.agent.multi import invoke_multi_agent

# Single query
result = await invoke_multi_agent("Search for Python tutorials")
print(result["final_response"])

# View thinking steps
for step in result["thinking_steps"]:
    print(f"{step.agent}: {step.thought}")
```

### With Session Persistence

```python
from kg_agent.agent.multi import invoke_multi_agent

# First message - creates session
result = await invoke_multi_agent(
    "My name is Steve",
    session_id="user-session-123"
)

# Later message - same session for context
result = await invoke_multi_agent(
    "What's my name?",
    session_id="user-session-123"
)
```

### Streaming Results

```python
from kg_agent.agent.multi import stream_multi_agent

async for node_name, state_update in stream_multi_agent("Search for AI papers"):
    print(f"[{node_name}] {state_update.get('current_agent')}")

    # Access thinking steps as they happen
    if state_update.get("thinking_steps"):
        latest = state_update["thinking_steps"][-1]
        print(f"  → {latest.thought}")
```

## API Endpoints

### Chat Endpoint

```http
POST /api/v1/multi-agent/chat
Content-Type: application/json

{
  "message": "Search for Python tutorials",
  "session_id": "optional-session-id",
  "user_id": "optional-user-id"
}
```

**Response:**
```json
{
  "response": "I found several Python tutorials...",
  "session_id": "session-123",
  "thinking_steps": [
    {"agent": "manager", "thought": "Analyzing request", "status": "thinking"},
    {"agent": "research", "thought": "Searching knowledge base", "status": "executing"}
  ],
  "execution_path": ["manager", "research", "synthesize"],
  "total_llm_calls": 3
}
```

### Streaming Chat

```http
POST /api/v1/multi-agent/chat/stream
Content-Type: application/json

{
  "message": "Search for Python tutorials"
}
```

Returns Server-Sent Events (SSE):
```
data: {"node": "manager", "current_agent": "manager", "thinking_steps": [...]}

data: {"node": "research", "current_agent": "research", "thinking_steps": [...]}

data: {"node": "synthesize", "final_response": "..."}

data: {"done": true}
```

### Session Management

```http
# Create session
POST /api/v1/multi-agent/session
{"user_id": "user-123", "metadata": {"source": "web"}}

# Get session
GET /api/v1/multi-agent/session/{session_id}

# List sessions
GET /api/v1/multi-agent/sessions

# Delete session
DELETE /api/v1/multi-agent/session/{session_id}
```

### System Status

```http
GET /api/v1/multi-agent/status
```

```json
{
  "status": "healthy",
  "graph_compiled": true,
  "checkpointer_type": "MemorySaver",
  "active_sessions": 5,
  "specialists": ["research", "memory", "knowledge", "documents"]
}
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_BASE_URL` | LM Studio/OpenAI API URL | `http://localhost:1234/v1` |
| `LLM_MODEL_NAME` | Model to use | `local-model` |
| `LLM_API_KEY` | API key (optional for local) | `not-needed` |

### Logging

The multi-agent system uses structured logging:

```python
from kg_agent.core.logging import logger

# Debug routing decisions
logger.debug(f"Routing to specialist: {target}")

# Info for major events
logger.info(f"Multi-agent invocation complete")

# Errors with context
logger.error(f"Specialist {agent} failed: {error}", exc_info=True)
```

## State Model

The `MultiAgentState` TypedDict tracks all agent state:

```python
class MultiAgentState(TypedDict, total=False):
    # Core
    messages: list[BaseMessage]
    user_id: str | None
    session_id: str | None

    # Orchestration
    current_agent: str  # 'manager', 'research', etc.
    delegation_queue: list[DelegationRequest]

    # Results
    research_result: str | None
    memory_result: str | None
    knowledge_result: str | None
    document_result: str | None

    # Thinking (streamed to UI)
    thinking_steps: list[ThinkingStep]
    total_llm_calls: int
    execution_path: list[str]

    # Control
    should_end: bool
    final_response: str | None
    last_error: str | None
```

## Frontend Integration

### CopilotKit Setup

```tsx
import { CopilotKit } from '@copilotkit/react-core';
import { AgentReasoningRenderer } from './components/agent-reasoning';
import { AgentStatusPanel } from './components/agent-status-panel';

function App() {
  return (
    <CopilotKit runtimeUrl="/api/copilotkit">
      <AgentReasoningRenderer />
      <AgentStatusPanel position="bottom-right" />
      {/* Your app content */}
    </CopilotKit>
  );
}
```

### Hooks

```tsx
// Access state anywhere
const { state, running } = useCoAgent<AgentState>({
  name: 'kg_multi_agent'
});

// Render in chat stream
useCoAgentStateRender<AgentState>({
  name: 'kg_multi_agent',
  render: ({ state, status }) => <ReasoningDisplay state={state} />
});
```

## Troubleshooting

### Common Issues

**1. "Checkpointer requires thread_id"**
```python
# Always provide session_id when using persistence
result = await invoke_multi_agent(
    "Hello",
    session_id="some-session-id"  # Required!
)
```

**2. "No response from specialist"**
- Check LM Studio is running
- Verify `LLM_BASE_URL` is correct
- Check logs for detailed error

**3. "State not updating in UI"**
- Verify `copilotkit_emit_state` is called in nodes
- Check agent name matches: `kg_multi_agent`
- Ensure CopilotKit provider wraps your app

### Debug Mode

Enable debug logging:
```python
import logging
logging.getLogger("kg_agent.agent.multi").setLevel(logging.DEBUG)
```

Or via environment:
```bash
export LOG_LEVEL=DEBUG
```

## Architecture Decisions

### Why Hierarchical?

1. **Single entry point** - Manager handles all routing
2. **Specialist focus** - Each agent has clear responsibilities
3. **Composable** - Easy to add new specialists
4. **Observable** - Clear execution path

### Why LangGraph?

1. **State management** - TypedDict state with clear contracts
2. **Checkpointing** - Session persistence out of the box
3. **Conditional routing** - Dynamic graph based on decisions
4. **Streaming** - Native async streaming support

### Why Pydantic AI?

1. **Structured output** - DelegationDecision is type-safe
2. **Tool integration** - Easy tool registration with type hints
3. **Provider agnostic** - Works with LM Studio, OpenAI, etc.
4. **Retry logic** - Built-in retry with backoff

---

*Created: November 29, 2025*

