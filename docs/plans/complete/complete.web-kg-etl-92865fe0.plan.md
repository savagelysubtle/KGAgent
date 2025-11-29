<!-- 92865fe0-d718-47d6-803b-6a6a93eaebc9 5f481ba9-9b86-4872-8c6c-153777ecced5 -->
# Web-to-Knowledge-Graph ETL Pipeline System Architecture

## Executive Summary

This plan details a production-grade automated ETL pipeline that transforms web content into a queryable Knowledge Graph. The system integrates **Crawl4AI** (web crawling), **Docling** (document parsing/OCR), **ChromaDB** (vector storage), **HuggingFace models** (LLM inference), **LLM Studio** (local inference orchestration), **Graphiti** (knowledge graph), and **FastAPI** (API orchestration) into a modular, scalable architecture.

## 1. System Architecture Overview

### Pipeline Flow

```
Web Sources → Crawl4AI → Raw HTML/PDFs
  ↓
Docling → Structured Documents (OCR + Normalization)
  ↓
Chunker → Document Chunks
  ↓
HuggingFace Embedder → Vector Embeddings
  ↓
ChromaDB → Vector Storage (with metadata)
  ↓
LLM Studio → Entity/Relationship Extraction
  ↓
Graphiti → Knowledge Graph Construction
  ↓
FastAPI → Query Interface (Hybrid Vector + KG)
```

### Technology Stack Integration Points

| Component | Technology | Purpose | Integration |

|-----------|-----------|---------|-------------|

| Web Crawling | Crawl4AI | Async web scraping | Playwright-based async crawling |

| Document Parsing | Docling | HTML/PDF/OCR normalization | Parses to structured DoclingDocument format |

| Chunking | Custom + LangChain | Text segmentation | Semantic chunking with overlap |

| Vector DB | ChromaDB | Embedding storage | Persistent client with collections |

| Embeddings | HuggingFace | Text-to-vector | Local models via transformers |

| LLM Inference | LLM Studio | Local LLM orchestration | OpenAI-compatible API (localhost:1234) |

| Entity Extraction | HuggingFace + LLM Studio | NER + relationship extraction | Prompt-based extraction pipeline |

| Knowledge Graph | Graphiti | Graph schema + storage | Neo4j-backed graph construction |

| API Layer | FastAPI | Async orchestration | REST endpoints for all operations |

## 2. Complete Project Structure

```
kgagent/
├── app/
│   ├── __init__.py
│   ├── main.py                      # FastAPI application entry
│   ├── api/
│   │   ├── __init__.py
│   │   ├── v1/
│   │   │   ├── __init__.py
│   │   │   ├── endpoints/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── crawl.py         # POST /crawl - initiate crawl jobs
│   │   │   │   ├── parse.py         # POST /parse - parse documents
│   │   │   │   ├── chunk.py         # POST /chunk - chunk documents
│   │   │   │   ├── embed.py         # POST /embed - generate embeddings
│   │   │   │   ├── extract.py       # POST /extract - extract entities
│   │   │   │   ├── graph.py         # POST /graph/build - build KG
│   │   │   │   ├── query.py         # GET /query - hybrid search
│   │   │   │   └── pipeline.py      # POST /pipeline/run - full ETL
│   │   │   └── dependencies.py      # Shared API dependencies
│   │   └── middleware.py            # CORS, logging, auth middleware
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py                # Settings management (Pydantic)
│   │   ├── security.py              # API key validation
│   │   ├── logging.py               # Structured logging setup
│   │   └── exceptions.py            # Custom exception handlers
│   ├── models/
│   │   ├── __init__.py
│   │   ├── document.py              # Document data models
│   │   ├── chunk.py                 # Chunk data models
│   │   ├── entity.py                # Entity/Relationship models
│   │   ├── graph.py                 # Graph schema models
│   │   └── pipeline.py              # Pipeline job models
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── requests.py              # API request schemas
│   │   ├── responses.py             # API response schemas
│   │   └── internal.py              # Internal data transfer objects
│   ├── services/
│   │   ├── __init__.py
│   │   ├── crawler.py               # Crawl4AI service wrapper
│   │   ├── parser.py                # Docling integration
│   │   ├── chunker.py               # Document chunking logic
│   │   ├── embedder.py              # HuggingFace embedding service
│   │   ├── vector_store.py          # ChromaDB operations
│   │   ├── llm_extractor.py         # LLM Studio integration
│   │   ├── graph_builder.py         # Graphiti KG construction
│   │   ├── query_engine.py          # Hybrid query orchestration
│   │   └── pipeline_orchestrator.py # End-to-end pipeline coordinator
│   ├── workers/
│   │   ├── __init__.py
│   │   ├── celery_app.py            # Celery configuration
│   │   ├── tasks/
│   │   │   ├── __init__.py
│   │   │   ├── crawl_tasks.py       # Async crawling tasks
│   │   │   ├── parse_tasks.py       # Document parsing tasks
│   │   │   ├── embed_tasks.py       # Embedding generation tasks
│   │   │   └── extract_tasks.py     # Entity extraction tasks
│   │   └── monitoring.py            # Task monitoring utilities
│   └── utils/
│       ├── __init__.py
│       ├── retry.py                 # Retry logic decorators
│       ├── validation.py            # Data validation helpers
│       └── text_processing.py       # Text cleaning utilities
├── scripts/
│   ├── install_models.sh            # HuggingFace model download script
│   ├── setup_lm_studio.sh           # LM Studio setup guide
│   ├── init_chromadb.py             # ChromaDB initialization
│   ├── init_graphiti.py             # Graphiti schema setup
│   └── run_pipeline.py              # CLI pipeline runner
├── tests/
│   ├── __init__.py
│   ├── conftest.py                  # Pytest fixtures
│   ├── unit/
│   │   ├── test_crawler.py
│   │   ├── test_parser.py
│   │   ├── test_chunker.py
│   │   ├── test_embedder.py
│   │   ├── test_extractor.py
│   │   └── test_graph_builder.py
│   ├── integration/
│   │   ├── test_pipeline_flow.py
│   │   ├── test_api_endpoints.py
│   │   └── test_hybrid_query.py
│   └── e2e/
│       └── test_full_pipeline.py
├── docker/
│   ├── Dockerfile.api               # FastAPI service
│   ├── Dockerfile.worker            # Celery workers
│   ├── Dockerfile.lm_studio         # LM Studio container
│   └── nginx.conf                   # Nginx reverse proxy config
├── deployment/
│   ├── docker-compose.yml           # Multi-container orchestration
│   ├── kubernetes/
│   │   ├── api-deployment.yaml
│   │   ├── worker-deployment.yaml
│   │   ├── chromadb-statefulset.yaml
│   │   ├── neo4j-statefulset.yaml
│   │   └── ingress.yaml
│   └── terraform/
│       └── main.tf                  # Infrastructure as code
├── config/
│   ├── logging.yaml                 # Logging configuration
│   ├── celery_config.py             # Celery settings
│   └── model_config.yaml            # HuggingFace model specifications
├── docs/
│   ├── architecture.md
│   ├── api_documentation.md
│   ├── deployment_guide.md
│   └── troubleshooting.md
├── .env.example                     # Environment variables template
├── .env                             # Local environment config (gitignored)
├── .gitignore
├── pyproject.toml                   # Dependencies and project metadata
├── Makefile                         # Common commands
└── README.md
```

## 3. Detailed Module Responsibilities

### 3.1 Crawling Module (`services/crawler.py`)

**Responsibility**: Async web crawling using Crawl4AI

**Implementation**:

```python
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

class CrawlerService:
    def __init__(self):
        self.browser_config = BrowserConfig(
            headless=True,
            viewport_width=1920,
            viewport_height=1080
        )
    
    async def crawl_url(self, url: str) -> dict:
        async with AsyncWebCrawler(config=self.browser_config) as crawler:
            result = await crawler.arun(
                url=url,
                config=CrawlerRunConfig(
                    word_count_threshold=10,
                    wait_for="networkidle"
                )
            )
            return {
                "url": url,
                "html": result.html,
                "markdown": result.markdown,
                "metadata": result.metadata
            }
    
    async def crawl_batch(self, urls: list[str]) -> list[dict]:
        return await asyncio.gather(*[self.crawl_url(url) for url in urls])
```

**Key Features**:

- Playwright-based async crawling
- JavaScript rendering support
- Configurable wait strategies
- Batch processing with concurrency limits

### 3.2 Parsing Module (`services/parser.py`)

**Responsibility**: Document parsing with Docling (OCR, PDF, HTML)

**Implementation**:

```python
from docling import DocumentProcessor, DoclingDocument

class ParserService:
    def __init__(self):
        self.processor = DocumentProcessor()
    
    def parse_html(self, html_content: str) -> DoclingDocument:
        return self.processor.process(
            content=html_content,
            source_type="html"
        )
    
    def parse_pdf(self, pdf_path: str) -> DoclingDocument:
        return self.processor.process(
            content=pdf_path,
            source_type="pdf",
            enable_ocr=True
        )
    
    def to_structured_format(self, doc: DoclingDocument) -> dict:
        return {
            "title": doc.title,
            "paragraphs": doc.paragraphs,
            "tables": doc.tables,
            "images": doc.images,
            "metadata": doc.metadata
        }
```

**Key Features**:

- Multi-format support (HTML, PDF, DOCX, images)
- OCR integration (EasyOCR, Tesseract, RapidOCR)
- Table extraction
- Structured output (JSON, Markdown)

### 3.3 Chunking Module (`services/chunker.py`)

**Responsibility**: Semantic text chunking for optimal embedding

**Implementation**:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

class ChunkerService:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def chunk_document(self, parsed_doc: dict) -> list[dict]:
        text = "\n\n".join(parsed_doc["paragraphs"])
        chunks = []
        
        for idx, chunk_text in enumerate(self.splitter.split_text(text)):
            chunks.append({
                "chunk_id": f"{parsed_doc['doc_id']}_chunk_{idx}",
                "text": chunk_text,
                "doc_id": parsed_doc["doc_id"],
                "chunk_index": idx,
                "metadata": {
                    "source_url": parsed_doc["metadata"]["source_url"],
                    "title": parsed_doc["title"],
                    "chunk_size": len(chunk_text)
                }
            })
        
        return chunks
```

**Key Features**:

- Semantic boundary detection
- Configurable chunk size/overlap
- Metadata preservation
- Hierarchical splitting

### 3.4 Embedding Module (`services/embedder.py`)

**Responsibility**: Generate embeddings using HuggingFace models

**Implementation**:

```python
from transformers import AutoTokenizer, AutoModel
import torch

class EmbedderService:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
    
    def embed_text(self, text: str) -> list[float]:
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt", 
                                   truncation=True, max_length=512)
            outputs = self.model(**inputs)
            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
            return embeddings[0].tolist()
    
    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = [self.embed_text(text) for text in batch]
            embeddings.extend(batch_embeddings)
        return embeddings
```

**Key Features**:

- Local HuggingFace model inference
- Batch processing
- GPU acceleration support
- Multiple embedding model options

### 3.5 Vector Store Module (`services/vector_store.py`)

**Responsibility**: ChromaDB operations for vector storage

**Implementation**:

```python
import chromadb
from chromadb.config import Settings

class VectorStoreService:
    def __init__(self, persist_directory: str = "./chroma_data"):
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name="document_chunks",
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_chunks(self, chunks: list[dict], embeddings: list[list[float]]):
        self.collection.add(
            ids=[chunk["chunk_id"] for chunk in chunks],
            embeddings=embeddings,
            documents=[chunk["text"] for chunk in chunks],
            metadatas=[chunk["metadata"] for chunk in chunks]
        )
    
    def query(self, query_embedding: list[float], n_results: int = 10) -> dict:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        return results
```

**Key Features**:

- Persistent storage
- HNSW indexing for fast retrieval
- Metadata filtering
- Similarity search (cosine, L2, IP)

### 3.6 LLM Extraction Module (`services/llm_extractor.py`)

**Responsibility**: Entity/relationship extraction using LLM Studio

**Implementation**:

```python
import httpx
from pydantic import BaseModel

class Entity(BaseModel):
    name: str
    type: str
    properties: dict

class Relationship(BaseModel):
    source: str
    target: str
    type: str
    properties: dict

class LLMExtractorService:
    def __init__(self, base_url: str = "http://localhost:1234/v1"):
        self.base_url = base_url
        self.client = httpx.AsyncClient()
    
    async def extract_entities_relations(self, text: str) -> tuple[list[Entity], list[Relationship]]:
        prompt = f"""Extract entities and relationships from the following text.
        Return JSON with two arrays: 'entities' and 'relationships'.
        
        Entities should have: name, type, properties
        Relationships should have: source, target, type, properties
        
        Text: {text}
        
        JSON:"""
        
        response = await self.client.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": "mistral-7b-instruct",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 2000
            }
        )
        
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        
        # Parse JSON response
        data = json.loads(content)
        entities = [Entity(**e) for e in data["entities"]]
        relationships = [Relationship(**r) for r in data["relationships"]]
        
        return entities, relationships
```

**Key Features**:

- OpenAI-compatible API integration
- Prompt-based extraction
- Structured output parsing
- Async HTTP requests

### 3.7 Graph Builder Module (`services/graph_builder.py`)

**Responsibility**: Graphiti KG construction

**Implementation**:

```python
from graphiti import Graphiti, GraphConfig
from neo4j import GraphDatabase

class GraphBuilderService:
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.graphiti = Graphiti(config=GraphConfig(
            driver=self.driver,
            embedding_dim=384
        ))
    
    async def add_entities(self, entities: list[Entity]):
        with self.driver.session() as session:
            for entity in entities:
                session.run(
                    "MERGE (e:Entity {name: $name, type: $type}) "
                    "SET e += $properties",
                    name=entity.name,
                    type=entity.type,
                    properties=entity.properties
                )
    
    async def add_relationships(self, relationships: list[Relationship]):
        with self.driver.session() as session:
            for rel in relationships:
                session.run(
                    "MATCH (a:Entity {name: $source}) "
                    "MATCH (b:Entity {name: $target}) "
                    "MERGE (a)-[r:RELATION {type: $type}]->(b) "
                    "SET r += $properties",
                    source=rel.source,
                    target=rel.target,
                    type=rel.type,
                    properties=rel.properties
                )
    
    async def build_kg_from_extraction(self, entities: list[Entity], 
                                       relationships: list[Relationship]):
        await self.add_entities(entities)
        await self.add_relationships(relationships)
```

**Key Features**:

- Neo4j-backed storage
- Graph schema validation
- Entity deduplication
- Relationship inference

### 3.8 Query Engine Module (`services/query_engine.py`)

**Responsibility**: Hybrid vector + KG search

**Implementation**:

```python
class QueryEngineService:
    def __init__(self, vector_store: VectorStoreService, 
                 graph_builder: GraphBuilderService,
                 embedder: EmbedderService):
        self.vector_store = vector_store
        self.graph_builder = graph_builder
        self.embedder = embedder
    
    async def hybrid_search(self, query: str, n_results: int = 10) -> dict:
        # Step 1: Vector search
        query_embedding = self.embedder.embed_text(query)
        vector_results = self.vector_store.query(query_embedding, n_results)
        
        # Step 2: Extract entities from query
        entities, _ = await self.llm_extractor.extract_entities_relations(query)
        
        # Step 3: Graph traversal
        graph_results = []
        for entity in entities:
            neighbors = await self.graph_builder.get_neighbors(entity.name, depth=2)
            graph_results.extend(neighbors)
        
        # Step 4: Merge and rank results
        combined_results = self._merge_results(vector_results, graph_results)
        
        return combined_results
```

**Key Features**:

- Hybrid retrieval (vector + graph)
- Result reranking
- Query entity extraction
- Multi-hop graph traversal

## 4. FastAPI Service Architecture

### 4.1 Main Application (`app/main.py`)

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.endpoints import (
    crawl, parse, chunk, embed, extract, graph, query, pipeline
)
from app.core.config import settings
from app.core.logging import setup_logging

app = FastAPI(
    title="Web-to-KG ETL Pipeline API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Setup logging
setup_logging()

# Include routers
app.include_router(crawl.router, prefix="/api/v1/crawl", tags=["crawl"])
app.include_router(parse.router, prefix="/api/v1/parse", tags=["parse"])
app.include_router(chunk.router, prefix="/api/v1/chunk", tags=["chunk"])
app.include_router(embed.router, prefix="/api/v1/embed", tags=["embed"])
app.include_router(extract.router, prefix="/api/v1/extract", tags=["extract"])
app.include_router(graph.router, prefix="/api/v1/graph", tags=["graph"])
app.include_router(query.router, prefix="/api/v1/query", tags=["query"])
app.include_router(pipeline.router, prefix="/api/v1/pipeline", tags=["pipeline"])

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
```

### 4.2 API Endpoints

| Endpoint | Method | Description | Input | Output |

|----------|--------|-------------|-------|--------|

| `/api/v1/crawl` | POST | Initiate web crawl | URLs list | Job ID |

| `/api/v1/parse` | POST | Parse documents | Raw HTML/PDF | Structured doc |

| `/api/v1/chunk` | POST | Chunk documents | Parsed doc | Chunks list |

| `/api/v1/embed` | POST | Generate embeddings | Chunks | Embeddings |

| `/api/v1/extract` | POST | Extract entities | Chunks | Entities/Relations |

| `/api/v1/graph/build` | POST | Build KG | Entities/Relations | KG status |

| `/api/v1/query` | GET | Hybrid search | Query string | Results |

| `/api/v1/pipeline/run` | POST | Full ETL pipeline | URLs + config | Pipeline status |

## 5. Configuration Strategy

### 5.1 Environment Variables (`.env`)

```bash
# Application
APP_ENV=development
APP_PORT=8000
LOG_LEVEL=INFO

# Neo4j (Graphiti)
NEO4J_URL=neo4j://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# ChromaDB
CHROMA_PERSIST_DIR=./chroma_data
CHROMA_HOST=localhost
CHROMA_PORT=8001

# LLM Studio
LLM_STUDIO_BASE_URL=http://localhost:1234/v1
LLM_STUDIO_MODEL=mistral-7b-instruct

# HuggingFace
HF_HOME=./hf_models
HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
HF_EXTRACTION_MODEL=meta-llama/Llama-2-7b-chat-hf

# Celery
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/1

# API Keys
API_KEY_SECRET=your_secret_key

# Crawl4AI
CRAWL4AI_MAX_CONCURRENT=5
CRAWL4AI_TIMEOUT=30

# Pipeline
CHUNK_SIZE=512
CHUNK_OVERLAP=50
EMBEDDING_BATCH_SIZE=32
```

### 5.2 Settings Management (`app/core/config.py`)

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_ENV: str
    APP_PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    
    NEO4J_URL: str
    NEO4J_USERNAME: str
    NEO4J_PASSWORD: str
    
    CHROMA_PERSIST_DIR: str = "./chroma_data"
    
    LLM_STUDIO_BASE_URL: str = "http://localhost:1234/v1"
    LLM_STUDIO_MODEL: str = "mistral-7b-instruct"
    
    HF_HOME: str = "./hf_models"
    HF_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    class Config:
        env_file = ".env"

settings = Settings()
```

## 6. HuggingFace Model Installer

### 6.1 Model Installation Script (`scripts/install_models.sh`)

```bash
#!/bin/bash

echo "=== HuggingFace Model Installer ==="

# Create models directory
mkdir -p ./hf_models
export HF_HOME=./hf_models

# Install required packages
pip install transformers torch sentence-transformers

# Download embedding models
echo "Downloading embedding models..."
python -c "
from transformers import AutoTokenizer, AutoModel
models = [
    'sentence-transformers/all-MiniLM-L6-v2',
    'sentence-transformers/all-mpnet-base-v2',
    'BAAI/bge-small-en-v1.5'
]
for model in models:
    print(f'Downloading {model}...')
    AutoTokenizer.from_pretrained(model)
    AutoModel.from_pretrained(model)
"

# Download extraction models (optional - for local inference)
echo "Downloading extraction models..."
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
models = [
    'mistralai/Mistral-7B-Instruct-v0.2',
    'meta-llama/Llama-2-7b-chat-hf'
]
for model in models:
    print(f'Downloading {model}...')
    AutoTokenizer.from_pretrained(model)
    # Note: Requires HF token for gated models
"

echo "✓ Model installation complete!"
```

## 7. LLM Studio Setup

### 7.1 Setup Guide (`scripts/setup_lm_studio.sh`)

```bash
#!/bin/bash

echo "=== LM Studio Setup Guide ==="

echo "1. Download LM Studio from: https://lmstudio.ai"
echo "2. Install and launch LM Studio"
echo "3. Download recommended models:"
echo "   - mistralai/Mistral-7B-Instruct-v0.2-GGUF"
echo "   - TheBloke/Llama-2-7B-Chat-GGUF"
echo "4. Start local inference server:"
echo "   - Open LM Studio → Developer → Local Server"
echo "   - Select model: Mistral-7B-Instruct"
echo "   - Port: 1234 (default)"
echo "   - Click 'Start Server'"
echo "5. Test connection:"
curl http://localhost:1234/v1/models
echo ""
echo "✓ LM Studio should now be running on http://localhost:1234/v1"
```

## 8. Docker Configuration

### 8.1 Docker Compose (`deployment/docker-compose.yml`)

```yaml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: docker/Dockerfile.api
    ports:
      - "8000:8000"
    environment:
      - NEO4J_URL=neo4j://neo4j:7687
      - CHROMA_HOST=chromadb
      - LLM_STUDIO_BASE_URL=http://lm_studio:1234/v1
      - CELERY_BROKER_URL=redis://redis:6379/0
    depends_on:
      - neo4j
      - chromadb
      - redis
    volumes:
      - ./hf_models:/app/hf_models
    networks:
      - kgagent-network

  worker:
    build:
      context: .
      dockerfile: docker/Dockerfile.worker
    environment:
      - NEO4J_URL=neo4j://neo4j:7687
      - CHROMA_HOST=chromadb
      - CELERY_BROKER_URL=redis://redis:6379/0
    depends_on:
      - redis
      - neo4j
      - chromadb
    volumes:
      - ./hf_models:/app/hf_models
    networks:
      - kgagent-network

  neo4j:
    image: neo4j:5.14
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/your_password
    volumes:
      - neo4j_data:/data
    networks:
      - kgagent-network

  chromadb:
    image: chromadb/chroma:latest
    ports:
      - "8001:8000"
    volumes:
      - chroma_data:/chroma/chroma
    networks:
      - kgagent-network

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    networks:
      - kgagent-network

  lm_studio:
    build:
      context: .
      dockerfile: docker/Dockerfile.lm_studio
    ports:
      - "1234:1234"
    volumes:
      - ./lm_studio_models:/models
    networks:
      - kgagent-network

volumes:
  neo4j_data:
  chroma_data:

networks:
  kgagent-network:
    driver: bridge
```

## 9. Implementation Sequence

### Phase 1: Foundation (Week 1-2)

- Setup project structure
- Install and configure all dependencies
- Implement core configuration management
- Setup logging and monitoring
- Download HuggingFace models
- Initialize Neo4j and ChromaDB

### Phase 2: Individual Services (Week 3-5)

- Implement Crawl4AI service
- Implement Docling parser service
- Implement chunking service
- Implement embedding service with HuggingFace
- Implement ChromaDB vector store service
- Test each service independently

### Phase 3: LLM Integration (Week 6-7)

- Setup LM Studio
- Implement LLM extractor service
- Create entity/relationship extraction prompts
- Implement Graphiti graph builder
- Test extraction pipeline

### Phase 4: API Layer (Week 8-9)

- Implement FastAPI endpoints
- Setup async task workers (Celery)
- Implement pipeline orchestrator
- Setup API authentication/authorization
- Create API documentation

### Phase 5: Query Engine (Week 10)

- Implement hybrid search
- Implement query optimizer
- Setup result reranking
- Test query performance

### Phase 6: Testing & Optimization (Week 11-12)

- Write unit tests
- Write integration tests
- Perform load testing
- Optimize pipeline performance
- Setup monitoring and alerts

### Phase 7: Deployment (Week 13-14)

- Create Docker images
- Setup Docker Compose orchestration
- Create Kubernetes manifests (optional)
- Setup CI/CD pipeline
- Deploy to production environment
- Create user documentation

### To-dos

- [ ] Setup development environment and install base dependencies
- [ ] Create complete project folder structure with all modules
- [ ] Implement configuration management with Pydantic Settings
- [ ] Download and configure HuggingFace embedding models
- [ ] Implement Crawl4AI async web crawling service
- [ ] Implement Docling document parsing service
- [ ] Implement semantic text chunking service
- [ ] Implement HuggingFace embedding service
- [ ] Implement ChromaDB vector storage service
- [ ] Setup LM Studio for local LLM inference
- [ ] Implement LLM entity/relationship extraction service
- [ ] Initialize Graphiti with Neo4j backend
- [ ] Implement Graphiti KG construction service
- [ ] Implement hybrid vector + KG query engine
- [ ] Create FastAPI endpoints for all pipeline stages
- [ ] Implement end-to-end pipeline orchestration service
- [ ] Setup Celery for async task processing
- [ ] Create Docker Compose configuration for all services
- [ ] Write unit and integration tests for all services
- [ ] Create comprehensive API and deployment documentation