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

---

## 🐳 Docker Deployment (Standalone App)

Run KG Agent as a fully containerized standalone application—no Python or Node.js
installation required on your machine!

### Prerequisites

- **Docker Desktop** with at least 8GB RAM allocated
- **LM Studio** running on your host machine (or another OpenAI-compatible LLM)

### Quick Start with Docker

**Windows (PowerShell):**

```powershell
# Clone the repository
git clone <repository-url>
cd kg-agent

# Start everything with one command
.\scripts\start-local.ps1 -Build -Detached
```

**Linux/macOS:**

```bash
# Clone the repository
git clone <repository-url>
cd kg-agent

# Make script executable and run
chmod +x scripts/start-local.sh
./scripts/start-local.sh --build --detached
```

**Or use Docker Compose directly:**

```bash
# Copy the Docker environment template
cp .env.docker .env

# Build and start all services
docker compose -f docker-compose.local.yml up --build -d
```

### What Gets Started

| Service       | Port | Description                          |
| ------------- | ---- | ------------------------------------ |
| **Dashboard** | 3000 | Next.js web interface                |
| **API**       | 8000 | FastAPI backend with Swagger docs    |
| **FalkorDB**  | 6380 | Graph database with vector search    |
| **Redis**     | 6379 | Message broker for background tasks  |
| **Worker**    | -    | Celery worker for async processing   |

### Managing the Docker Stack

**Windows (PowerShell):**

```powershell
# View logs
.\scripts\start-local.ps1 -Logs

# Check status
.\scripts\start-local.ps1 -Status

# Stop everything
.\scripts\start-local.ps1 -Down

# Rebuild after code changes
.\scripts\start-local.ps1 -Build -Detached
```

**Linux/macOS or direct Docker commands:**

```bash
# View logs
docker compose -f docker-compose.local.yml logs -f

# Check status
docker compose -f docker-compose.local.yml ps

# Stop everything
docker compose -f docker-compose.local.yml down

# Rebuild
docker compose -f docker-compose.local.yml up --build -d
```

### LLM Configuration for Docker

By default, the Docker containers connect to LM Studio running on your host machine
via `host.docker.internal:1234`.

**To use a different LLM provider**, edit `.env`:

```env
# OpenAI
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-your-api-key
LLM_MODEL_NAME=gpt-4o-mini

# Ollama (running on host)
LLM_BASE_URL=http://host.docker.internal:11434/v1
LLM_API_KEY=ollama
LLM_MODEL_NAME=llama3.2
```

### Data Persistence

All data is persisted in local directories that are mounted into the containers:

| Directory     | Purpose                              |
| ------------- | ------------------------------------ |
| `./storage`   | SQLite databases, uploaded files     |
| `./data`      | ChromaDB vectors, processed chunks   |
| `./models`    | Downloaded embedding models          |
| `./cache`     | Crawler cache                        |
| `./logs`      | Application logs                     |

Your data survives container restarts and rebuilds!

### Docker Resource Requirements

| Resource | Minimum | Recommended |
| -------- | ------- | ----------- |
| RAM      | 4GB     | 8GB+        |
| CPU      | 2 cores | 4+ cores    |
| Disk     | 5GB     | 20GB+       |

The first build downloads embedding models (~100MB) which are cached for future runs.

---

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
- **Remember conversations**: All chats are saved to the knowledge graph
- **Recall past discussions**: "What have we talked about before?"
- **Learn about you**: "Remember that I prefer Python"

### Conversation Memory

The agent **continuously learns** from your conversations:

- **Auto-save**: Every conversation is automatically saved to FalkorDB
- **User profile**: The agent builds a profile of your preferences over time
- **Context recall**: Past discussions are retrieved to provide better answers
- **Persistent memory**: Knowledge persists across sessions

**Teach the agent about yourself:**

```bash
# Store a preference
curl -X POST http://localhost:8000/api/v1/agent/memory/learn/preference \
  -H "Content-Type: application/json" \
  -d '{"key": "favorite_language", "value": "Python"}'

# Store a fact
curl -X POST http://localhost:8000/api/v1/agent/memory/learn/fact \
  -H "Content-Type: application/json" \
  -d '{"fact": "User is a software developer", "category": "work"}'

# View your profile
curl http://localhost:8000/api/v1/agent/memory/profile

# Search past conversations
curl "http://localhost:8000/api/v1/agent/memory/recall?query=Python+programming"
```

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
