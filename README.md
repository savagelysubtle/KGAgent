# KG Agent - Knowledge Graph ETL Pipeline with AI Agent

[![Support this project](https://img.shields.io/badge/Support-PayPal-blue.svg)](https://paypal.me/safeappealnavigator)

A comprehensive knowledge graph system that combines web crawling, document
processing, entity extraction, and an AI-powered agent for intelligent querying.
Built with FastAPI, Graphiti, FalkorDB, ChromaDB, and local LLM inference via LM
Studio.

> 💖 **If you find this project useful, please consider
> [supporting development](https://paypal.me/safeappealnavigator)!**

## Features

### Core Pipeline

- **Web Crawling**: High-performance async crawling using Crawl4AI and
  Playwright
- **Document Processing**: Upload and process PDFs, HTML, Markdown, and text
  files
- **Entity Extraction**: LLM-powered extraction of entities and relationships
- **Resumable Processing**: Pause/resume long-running extraction jobs

### Knowledge Graph

- **FalkorDB Integration**: Graph database with vector search via Graphiti
- **Temporal Knowledge**: Track entity changes over time with Graphiti's
  temporal model
- **Automatic Deduplication**: Smart entity resolution and relationship merging
- **Hybrid Search**: Combined vector + graph search for comprehensive retrieval

### AI Agent

- **Local LLM**: Run entirely locally with LM Studio (Qwen, Llama, Mistral,
  etc.)
- **RAG Tools**: Vector search, graph search, and hybrid retrieval
- **CopilotKit Integration**: AI assistant in the dashboard UI
- **Tool Calling**: Agent can search, create entities, and manage data

### Dashboard

- **Modern React UI**: Next.js 16 with Tailwind CSS and shadcn/ui
- **Real-time Updates**: Live pipeline status and entity counts
- **Document Management**: Upload, view, and reprocess documents
- **Graph Explorer**: Visualize knowledge graph entities and relationships

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Dashboard (Next.js)                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────────┐ │
│  │ Crawler  │ │ Upload   │ │ Graph    │ │ AI Agent (CopilotKit)│ │
│  │ Control  │ │ Manager  │ │ Explorer │ │                      │ │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │ REST API
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────────┐ │
│  │ Crawler  │ │ Document │ │ Reprocess│ │ Agent API            │ │
│  │ Routes   │ │ Routes   │ │ Routes   │ │ (Chat, Tools, Stats) │ │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                        Services Layer                            │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────────┐  │
│  │ Graphiti     │ │ Vector Store │ │ Resumable Pipeline       │  │
│  │ Service      │ │ (ChromaDB)   │ │ (Entity Extraction)      │  │
│  └──────────────┘ └──────────────┘ └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                      External Services                           │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────────┐  │
│  │ FalkorDB    │ │ LM Studio    │ │ Sentence Transformers    │  │
│  │ (Graph DB)   │ │ (Local LLM)  │ │ (Embeddings)             │  │
│  └──────────────┘ └──────────────┘ └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+ (or Bun)
- Docker (for FalkorDB)
- [LM Studio](https://lmstudio.ai/) with a loaded model

### 1. Clone and Install

```bash
git clone <repository-url>
cd kg-agent

# Install Python dependencies
uv sync --all-groups

# Install dashboard dependencies
cd dashboard && bun install && cd ..

# Install Playwright browsers
playwright install chromium
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```env
# LM Studio (Local LLM)
LLM_BASE_URL=http://localhost:1234/v1
LLM_API_KEY=lm-studio
LLM_MODEL_NAME=local-model

# FalkorDB
GRAPH_DRIVER=falkordb
FALKORDB_HOST=localhost
FALKORDB_PORT=6380

# ChromaDB
CHROMA_PERSIST_DIR=./data/chroma_db
CHROMA_COLLECTION_NAME=document_chunks

# Embeddings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### 3. Start Services

**Option A: VS Code Tasks (Recommended)**

Press `Ctrl+Shift+B` to run "Start All Services" which starts:

- FalkorDB (Docker)
- Backend API (FastAPI)
- Dashboard (Next.js)

**Option B: Manual Start**

```bash
# Terminal 1: Start FalkorDB
docker compose -f docker-compose.dev.yml up

# Terminal 2: Start Backend
uv run python main.py

# Terminal 3: Start Dashboard
cd dashboard && bun run dev
```

### 4. Load a Model in LM Studio

1. Open LM Studio
2. Download/load a model (recommended: Qwen 2.5, Llama 3.2, or Mistral)
3. Start the local server (Developer → Local Server → Start)
4. Ensure it's running on `http://localhost:1234`

### 5. Access the Application

- **Dashboard**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## Usage

### Dashboard Features

| Page          | Description                                         |
| ------------- | --------------------------------------------------- |
| **Home**      | Overview with stats, pipeline activity, and AI chat |
| **Crawler**   | Start web crawls with configurable depth            |
| **Upload**    | Upload documents (PDF, HTML, MD, TXT)               |
| **Documents** | View and manage processed documents                 |
| **Reprocess** | Run entity extraction on documents                  |
| **Graph**     | Explore entities and relationships                  |
| **Pipeline**  | Monitor processing status                           |

### AI Agent

The AI agent (powered by CopilotKit) can:

- **Search knowledge base**: "What do we know about Python?"
- **Get statistics**: "How many entities are in the graph?"
- **Start crawls**: "Crawl https://example.com"
- **Create entities**: "Create an entity for OpenAI"
- **Delete data**: "Delete document xyz"

### API Examples

**Chat with Agent:**

```bash
curl -X POST http://localhost:8000/api/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What documents do we have?"}'
```

**Search Knowledge Base:**

```bash
curl "http://localhost:8000/api/v1/agent/tools/search?query=machine+learning&search_type=hybrid"
```

**Upload Document:**

```bash
curl -X POST http://localhost:8000/api/v1/upload/file \
  -F "file=@document.pdf"
```

**Start Entity Extraction:**

```bash
curl -X POST http://localhost:8000/api/v1/reprocess/start \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "your-doc-id"}'
```

**Get Graph Stats:**

```bash
curl http://localhost:8000/api/v1/reprocess/graph/stats
```

## Configuration

### Environment Variables

| Variable                 | Default                                | Description                              |
| ------------------------ | -------------------------------------- | ---------------------------------------- |
| **LLM Settings**         |                                        |                                          |
| `LLM_BASE_URL`           | http://localhost:1234/v1               | LM Studio API endpoint                   |
| `LLM_API_KEY`            | lm-studio                              | API key (any value for LM Studio)        |
| `LLM_MODEL_NAME`         | local-model                            | Model name (LM Studio uses loaded model) |
| **Graph Database**       |                                        |                                          |
| `GRAPH_DRIVER`           | falkordb                               | Graph driver: `falkordb` or `neo4j`      |
| `FALKORDB_HOST`          | localhost                              | FalkorDB host                            |
| `FALKORDB_PORT`          | 6380                                   | FalkorDB port                            |
| **Vector Store**         |                                        |                                          |
| `CHROMA_PERSIST_DIR`     | ./data/chroma_db                       | ChromaDB storage path                    |
| `CHROMA_COLLECTION_NAME` | document_chunks                        | Collection name                          |
| **Embeddings**           |                                        |                                          |
| `EMBEDDING_MODEL`        | sentence-transformers/all-MiniLM-L6-v2 | Embedding model                          |
| **Crawler**              |                                        |                                          |
| `CRAWLER_MAX_CONCURRENT` | 5                                      | Max concurrent crawls                    |
| `CRAWLER_TIMEOUT`        | 30                                     | Crawl timeout (seconds)                  |

### VS Code Tasks

Available tasks in `.vscode/tasks.json`:

| Task                         | Description                                   |
| ---------------------------- | --------------------------------------------- |
| `Start All Services`         | Start Docker, Backend, and Dashboard together |
| `Start Dev (without Docker)` | Start Backend and Dashboard only              |
| `Start Docker Services`      | Start FalkorDB and Redis                      |
| `Start Backend API`          | Start FastAPI server                          |
| `Start Dashboard`            | Start Next.js dashboard                       |
| `Stop Docker Services`       | Stop all Docker containers                    |

## Project Structure

```
kg-agent/
├── src/kg_agent/
│   ├── agent/           # AI Agent with RAG tools
│   │   ├── kg_agent.py  # Pydantic AI agent
│   │   └── tools.py     # Search & management tools
│   ├── api/routes/      # FastAPI endpoints
│   │   ├── agent.py     # Agent chat & tools
│   │   ├── documents.py # Document management
│   │   ├── reprocess.py # Entity extraction
│   │   └── upload.py    # File uploads
│   ├── services/
│   │   ├── graphiti_service.py    # FalkorDB via Graphiti
│   │   ├── vector_store.py        # ChromaDB operations
│   │   ├── resumable_pipeline.py  # Entity extraction
│   │   └── document_tracker.py    # Document metadata
│   ├── crawler/         # Web crawling service
│   └── core/            # Config, logging
├── dashboard/           # Next.js frontend
│   ├── app/             # Pages (App Router)
│   ├── components/      # React components
│   └── lib/             # API client, utilities
├── docker-compose.dev.yml  # Development services
├── main.py              # FastAPI entry point
└── .env.example         # Environment template
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/kg_agent

# Run specific test
pytest tests/test_agent.py -v
```

### Code Quality

```bash
# Format code
ruff format src/

# Lint code
ruff check src/

# Type checking
ty check src/
```

### Useful Scripts

```bash
# Test Graphiti connection
uv run python scripts/check_graphiti.py

# Recover entities from failed job
uv run python scripts/recover_entities.py
```

## Troubleshooting

### LM Studio Connection Error

- Ensure LM Studio is running with a model loaded
- Check the server is started (Developer → Local Server)
- Verify port matches `LLM_BASE_URL` (default: 1234)

### FalkorDB Connection Error

- Start FalkorDB: `docker compose -f docker-compose.dev.yml up -d`
- Check port 6380 is accessible
- Wait for FalkorDB to fully initialize

### Entity Extraction Not Saving

- Check backend logs for errors
- Ensure FalkorDB is running and connected
- Verify LM Studio is responding (check for timeout errors)

### Dashboard Not Updating

- Check browser console for API errors
- Verify backend is running on port 8000
- Try refreshing with Ctrl+F5

## Tech Stack

- **Backend**: FastAPI, Pydantic, asyncio
- **Knowledge Graph**: FalkorDB, Graphiti
- **Vector Store**: ChromaDB, sentence-transformers
- **LLM**: LM Studio (local), OpenAI-compatible API
- **Agent**: Pydantic AI, CopilotKit
- **Frontend**: Next.js 16, React, Tailwind CSS, shadcn/ui
- **Crawler**: Crawl4AI, Playwright
- **Package Manager**: uv (Python), bun (Node.js)

---

> 🙏 **Enjoying KG Agent?** >
> [Support the project on PayPal](https://paypal.me/safeappealnavigator) to help
> with ongoing development!

## Support the Project

If you find KG Agent useful, please consider supporting its development:

[![Support on PayPal](https://img.shields.io/badge/Donate-PayPal-blue.svg?style=for-the-badge&logo=paypal)](https://paypal.me/safeappealnavigator)

**[☕ Buy me a coffee via PayPal](https://paypal.me/safeappealnavigator)**

Your support helps maintain and improve this project!

## License

This project is licensed under the MIT License - see the LICENSE file for
details.

## Help & Questions

For questions and support, please open an issue on GitHub.

---

<p align="center">
  Made with ❤️ | <a href="https://paypal.me/safeappealnavigator">Support this project</a>
</p>
