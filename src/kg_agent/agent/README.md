# Knowledge Graph Multi-Agent System

This module provides a **hierarchical multi-agent system** using LangGraph and
Pydantic AI that connects to LM Studio for local LLM inference with tools for
RAG (Retrieval-Augmented Generation).

## Architecture

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
                    └─────────────────┘
```

## Features

- **LM Studio Integration**: Uses OpenAI-compatible API for local LLM inference
- **Hierarchical Multi-Agent**: Manager orchestrates specialist agents
- **Real-time Reasoning**: Thinking steps streamed to UI via CopilotKit
- **Vector Search (ChromaDB)**: Semantic similarity search
- **Graph Search (FalkorDB)**: Structured knowledge graph queries
- **Session Persistence**: Checkpointing for conversation continuity
- **CopilotKit Integration**: AG-UI protocol for frontend integration

## Agents

| Agent              | Role                           | Tools                                              |
| ------------------ | ------------------------------ | -------------------------------------------------- |
| **Manager**        | Orchestration, intent analysis | Delegation decisions                               |
| **Research Lead**  | Knowledge base search          | `search_knowledge_base`, `get_database_statistics` |
| **Memory Lead**    | User context, history          | `recall_context`, `remember_about_user`            |
| **Knowledge Lead** | Entity management              | `create_entity`, `find_related_entities`           |
| **Document Lead**  | Document lifecycle             | `list_documents`, `delete_document`                |

## Usage

### Python API

```python
from kg_agent.agent import invoke_multi_agent, stream_multi_agent

# Simple invocation
result = await invoke_multi_agent("Search for Python tutorials")
print(result["final_response"])

# With session persistence
result = await invoke_multi_agent(
    "Remember my name is Steve",
    session_id="session-123"
)

# Streaming
async for node, state in stream_multi_agent("Search for AI papers"):
    print(f"[{node}] {state.get('current_agent')}")
```

### RAG Tools Direct Access

```python
from kg_agent.agent.tools import get_rag_tools

tools = get_rag_tools()
await tools.initialize()

# Vector search
results = await tools.search_vectors("machine learning", n_results=5)

# Graph search
results = await tools.search_graph("Python", limit=10)

# Hybrid search
results = await tools.hybrid_search("AI applications")

# Statistics
vector_stats = await tools.get_vector_stats()
graph_stats = await tools.get_graph_stats()
```

### REST API Endpoints

Start the server:

```bash
uv run python main.py
```

**Chat with Multi-Agent:**

```bash
curl -X POST http://localhost:8000/api/v1/multi-agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Search for Python tutorials"}'
```

**Stream Chat Response:**

```bash
curl -X POST http://localhost:8000/api/v1/multi-agent/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "What documents do we have?"}'
```

**System Status:**

```bash
curl http://localhost:8000/api/v1/multi-agent/status
```

## Configuration

Set these environment variables in `.env`:

```env
# LM Studio / Local LLM
LLM_BASE_URL=http://localhost:1234/v1
LLM_API_KEY=lm-studio
LLM_MODEL_NAME=local-model

# ChromaDB
CHROMA_PERSIST_DIR=./data/chroma_db
CHROMA_COLLECTION_NAME=document_chunks

# FalkorDB (via Graphiti)
GRAPH_DRIVER=falkordb
FALKORDB_HOST=localhost
FALKORDB_PORT=6380

# Embeddings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

## Module Structure

```
agent/
├── __init__.py          # Module exports
├── llm.py               # Shared LLM configuration
├── tools.py             # RAG tools (vector/graph search, CRUD)
├── README.md            # This file
└── multi/               # Multi-agent system
    ├── __init__.py      # Public exports
    ├── state.py         # Shared state TypedDict
    ├── prompts.py       # System prompts
    ├── graph.py         # LangGraph StateGraph
    ├── manager.py       # Manager agent (orchestrator)
    ├── research_lead.py # Research specialist
    ├── memory_lead.py   # Memory specialist
    ├── knowledge_lead.py# Knowledge specialist
    ├── document_lead.py # Document specialist
    ├── session.py       # Session management
    ├── error_handling.py# Error handling utilities
    └── optimization.py  # Performance utilities
```

## Testing

Run the multi-agent tests:

```bash
uv run pytest tests/test_multi_agent*.py -v
```

## Troubleshooting

### LM Studio Connection Error

- Ensure LM Studio is running with a model loaded
- Check the server is started (Developer → Local Server)
- Verify the port matches `LLM_BASE_URL`

### FalkorDB Connection Error

- Start FalkorDB: `docker compose -f docker-compose.dev.yml up -d`
- Check that port 6380 is accessible
- Wait for FalkorDB to fully start

### ChromaDB Issues

- Delete `./data/chroma_db` to reset
- Check disk space and permissions

### No Reasoning in UI

- Verify `copilotkit_emit_state` is called in agent nodes
- Check agent name matches: `kg_multi_agent`
- Ensure CopilotKit provider wraps the app

## See Also

- [Multi-Agent Architecture](../../../docs/multi-agent-architecture.md) - Full
  documentation
