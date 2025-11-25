# Pydantic AI Knowledge Graph Agent

This module provides a Pydantic AI agent that connects to LM Studio for local LLM inference and has tools for RAG (Retrieval-Augmented Generation) using ChromaDB and Neo4j.

## Features

- **LM Studio Integration**: Uses OpenAI-compatible API to connect to any model running in LM Studio
- **Vector Search (ChromaDB)**: Semantic similarity search across document chunks
- **Graph Search (Neo4j)**: Structured search in the knowledge graph
- **Hybrid Search**: Combines both vector and graph search for comprehensive results
- **FastAPI Endpoints**: REST API for chat and tool access

## Prerequisites

1. **LM Studio**: Download from [lmstudio.ai](https://lmstudio.ai/)
   - Load a model (e.g., Llama 3, Mistral, etc.)
   - Start the local server (default: `http://localhost:1234/v1`)

2. **Neo4j**: Running instance (Docker recommended)
   ```bash
   docker compose -f docker-compose.dev.yml up -d
   ```

3. **ChromaDB**: Automatically initialized in `./data/chroma_db`

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

# Neo4j
NEO4J_URI=neo4j://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
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

# Hybrid search
results = await tools.hybrid_search("AI applications")

# Get statistics
vector_stats = await tools.get_vector_stats()
graph_stats = await tools.get_graph_stats()
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

**Search Knowledge Base:**
```bash
curl -X POST "http://localhost:8000/api/v1/agent/tools/search?query=Python&search_type=hybrid"
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

1. **search_knowledge_base**: Search for information using vector, graph, or hybrid search
2. **get_database_statistics**: Get current stats about ChromaDB and Neo4j
3. **search_by_source**: Find documents from a specific source

## Dashboard Integration

The agent is integrated with the CopilotKit dashboard. Available actions:

- `searchKnowledgeBase`: Search the knowledge base
- `askAgent`: Ask the AI agent a question
- `getDatabaseStats`: Get database statistics
- `checkAgentHealth`: Check agent health status
- `startCrawl`: Start a web crawl

## Testing

Run the test script:
```bash
uv run python scripts/test_agent.py
```

## Architecture

```
agent/
├── __init__.py          # Module exports
├── kg_agent.py          # Main Pydantic AI agent
├── tools.py             # RAG tools (vector/graph search)
└── README.md            # This file
```

## Troubleshooting

### LM Studio Connection Error
- Ensure LM Studio is running with a model loaded
- Check the server is started (Developer → Local Server)
- Verify the port matches `LLM_BASE_URL`

### Neo4j Connection Error
- Start Neo4j: `docker compose -f docker-compose.dev.yml up -d`
- Check credentials in `.env`
- Wait for Neo4j to fully start (check logs)

### ChromaDB Issues
- Delete `./data/chroma_db` to reset
- Check disk space
- Ensure write permissions

