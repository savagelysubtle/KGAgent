# Pydantic AI Knowledge Graph Agent

This module provides a Pydantic AI agent that connects to LM Studio for local LLM inference and has tools for RAG (Retrieval-Augmented Generation) using ChromaDB and FalkorDB via Graphiti.

## Features

- **LM Studio Integration**: Uses OpenAI-compatible API to connect to any model running in LM Studio
- **Vector Search (ChromaDB)**: Semantic similarity search across document chunks
- **Graph Search (FalkorDB)**: Structured search in the knowledge graph via Graphiti
- **Hybrid Search**: Combines both vector and graph search for comprehensive results
- **Entity Management**: Create, search, and delete entities and relationships
- **Document Management**: Track, search, and delete documents
- **FastAPI Endpoints**: REST API for chat and tool access
- **CopilotKit Integration**: AI assistant in the dashboard UI

## Prerequisites

1. **LM Studio**: Download from [lmstudio.ai](https://lmstudio.ai/)
   - Load a model (e.g., Qwen 2.5, Llama 3.2, Mistral)
   - Start the local server (Developer → Local Server → Start)
   - Default endpoint: `http://localhost:1234/v1`

2. **FalkorDB**: Running instance (Docker recommended)
   ```bash
   docker compose -f docker-compose.dev.yml up -d
   ```

3. **ChromaDB**: Automatically initialized in `./data/chroma_db`

4. **Sentence Transformers**: Local embeddings (auto-downloaded)

## Configuration

Set these environment variables in `.env`:

```env
# LM Studio / Local LLM
LLM_BASE_URL=http://localhost:1234/v1
LLM_API_KEY=lm-studio
LLM_MODEL_NAME=local-model  # Any name, LM Studio uses the loaded model

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

## Usage

### Python API

```python
from kg_agent.agent import get_kg_agent

# Get the agent instance
agent = get_kg_agent()

# Initialize (connects to databases)
await agent.initialize()

# Chat with the agent
response = await agent.chat("What documents do we have about Python?")
print(response)

# Stream responses
async for chunk in agent.chat_stream("Tell me about machine learning"):
    print(chunk, end="")
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

# Hybrid search (combines vector + graph)
results = await tools.hybrid_search("AI applications")

# Get statistics
vector_stats = await tools.get_vector_stats()
graph_stats = await tools.get_graph_stats()

# Create entity
result = await tools.create_entity(
    name="OpenAI",
    entity_type="Organization",
    description="AI research company"
)

# Create relationship
result = await tools.create_relationship(
    source_entity="OpenAI",
    target_entity="GPT-4",
    relationship_type="CREATED"
)

# Delete document (with vectors and graph nodes)
result = await tools.delete_document(
    doc_id="abc123",
    delete_vectors=True,
    delete_graph_nodes=True
)
```

### REST API Endpoints

Start the server:
```bash
uv run python main.py
```

**Chat with Agent:**
```bash
curl -X POST http://localhost:8000/api/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What can you help me with?"}'
```

**Stream Chat Response:**
```bash
curl -X POST http://localhost:8000/api/v1/agent/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain knowledge graphs"}'
```

**Search Knowledge Base:**
```bash
# Hybrid search (default)
curl "http://localhost:8000/api/v1/agent/tools/search?query=Python&search_type=hybrid"

# Vector only
curl "http://localhost:8000/api/v1/agent/tools/search?query=Python&search_type=vector"

# Graph only
curl "http://localhost:8000/api/v1/agent/tools/search?query=Python&search_type=graph"
```

**Get Statistics:**
```bash
curl http://localhost:8000/api/v1/agent/tools/stats
```

**Health Check:**
```bash
curl http://localhost:8000/api/v1/agent/health
```

## Agent Tools

The agent has access to these tools:

| Tool | Description |
|------|-------------|
| `search_knowledge_base` | Search using vector, graph, or hybrid search |
| `get_database_statistics` | Get stats about ChromaDB and FalkorDB |
| `search_by_source` | Find documents from a specific source |
| `list_documents` | List all tracked documents |
| `create_entity` | Create a new entity in the knowledge graph |
| `create_relationship` | Create a relationship between entities |
| `delete_document` | Delete a document and its data |
| `delete_by_source` | Delete all data from a specific source |
| `clear_all_data` | Clear all data from the system |

## Dashboard Integration

The agent is integrated with the CopilotKit dashboard. Available actions:

| Action | Description |
|--------|-------------|
| `searchKnowledgeBase` | Search the knowledge base |
| `askAgent` | Ask the AI agent a question |
| `getDatabaseStats` | Get database statistics |
| `checkAgentHealth` | Check agent health status |
| `startCrawl` | Start a web crawl |
| `uploadDocument` | Upload a document |
| `deleteDocument` | Delete a document |
| `createEntity` | Create a new entity |
| `createRelationship` | Create a relationship |

## Architecture

```
agent/
├── __init__.py          # Module exports
├── kg_agent.py          # Main Pydantic AI agent
├── tools.py             # RAG tools (vector/graph search, CRUD)
└── README.md            # This file
```

### Data Flow

```
User Query
    │
    ▼
┌─────────────────┐
│  KG Agent       │ ← Pydantic AI with LM Studio
│  (kg_agent.py)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  RAG Tools      │
│  (tools.py)     │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌───────────┐
│ChromaDB│ │ FalkorDB  │
│(Vector)│ │ (Graph)   │
└───────┘ └───────────┘
```

## Testing

Run the test script:
```bash
uv run python scripts/test_agent.py
```

Or test specific functionality:
```python
import asyncio
from kg_agent.agent import get_kg_agent

async def test():
    agent = get_kg_agent()
    await agent.initialize()

    # Test chat
    response = await agent.chat("Hello, what can you do?")
    print(response)

    # Test search
    results = await agent.tools.hybrid_search("Python")
    print(f"Found {len(results)} results")

asyncio.run(test())
```

## Troubleshooting

### LM Studio Connection Error
- Ensure LM Studio is running with a model loaded
- Check the server is started (Developer → Local Server)
- Verify the port matches `LLM_BASE_URL`
- Try a smaller model if responses are slow

### FalkorDB Connection Error
- Start FalkorDB: `docker compose -f docker-compose.dev.yml up -d`
- Check that port 6380 is accessible
- Wait for FalkorDB to fully start (check logs)
- Verify `GRAPH_DRIVER=falkordb` in `.env`

### ChromaDB Issues
- Delete `./data/chroma_db` to reset
- Check disk space
- Ensure write permissions
- Check for SQLite lock issues

### Slow Responses
- Use a smaller/faster model in LM Studio
- Reduce search result limits
- Check if embedding model is using GPU

### Entity Extraction Failures
- Check LM Studio logs for errors
- Ensure model can output valid JSON
- Try a more capable model (Qwen 2.5 recommended)
