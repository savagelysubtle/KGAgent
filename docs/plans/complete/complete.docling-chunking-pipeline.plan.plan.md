# Docling Pipeline & Chunking Strategy

## Executive Summary

This plan outlines the implementation of the document processing pipeline, bridging the gap between the existing Web Crawler and the future Database Ingestion. The pipeline focuses on persisting raw crawl data, parsing it with Docling, and semantically chunking the content for downstream knowledge graph construction.

## 1. Architecture Overview

The pipeline follows a linear data flow:

1.  **Crawler**: Fetches content (HTML/PDF) -> **Raw Storage** (File System).
2.  **Parser (Docling)**: Reads Raw Storage -> Normalizes to Structured Document -> **Parsed Storage** (Intermediate JSON).
3.  **Chunker**: Reads Parsed Storage -> Splits into Semantic Chunks -> **Staging Area** (Ready for DB).

### Data Flow Directories (Proposed)

-   `data/raw/`: Raw HTML and PDF files from the crawler.
-   `data/parsed/`: Docling output (JSON/Markdown).
-   `data/chunks/`: Chunked data ready for embedding/graphing.

## 2. Module Implementation Details

### 2.1 Raw Data Storage (Crawler Update)

**Goal**: Modify or extend the crawler to save artifacts to `data/raw` in addition to the internal cache.

-   **Location**: `src/kg_agent/crawler/storage.py` (New Module)
-   **Responsibility**: Save `CrawlResult.html` or `CrawlResult.pdf_path` to `data/raw/{job_id}/{safe_url_hash}.{ext}`.
-   **Interface**:
    ```python
    class StorageService:
        async def save_raw_content(self, crawl_result: CrawlResult, job_id: str) -> str:
            # Returns file path
            pass
    ```


### 2.2 Docling Parser Service

**Goal**: Use Docling to convert raw HTML/PDF into a unified structured format.

-   **Location**: `src/kg_agent/parser/`
-   **Dependencies**: `docling`
-   **Components**:
    -   `service.py`: Core logic wrapping `DocumentConverter`.
    -   `models.py`: Pydantic models for parsed output (if Docling's native export isn't sufficient).

-   **Key Logic**:
    -   Initialize `DocumentConverter`.
    -   Process files from `data/raw`.
    -   Extract text, tables, and metadata.
    -   Save structured output to `data/parsed`.

### 2.3 Chunking Service

**Goal**: Split structured documents into semantic chunks suitable for vector embedding and relation extraction.

-   **Location**: `src/kg_agent/chunker/`
-   **Dependencies**: `langchain-text-splitters` (or custom logic).
-   **Components**:
    -   `service.py`: Chunking logic.
    -   `strategies.py`: Different chunking strategies (Recursive, Semantic, Hierarchy-aware).

-   **Key Logic**:
    -   Load parsed documents.
    -   Apply `RecursiveCharacterTextSplitter` or Docling's own chunking capabilities (if available/preferred).
    -   Preserve metadata (Source URL, Title, Section Header) in each chunk.
    -   Save chunks to `data/chunks`.

### 2.4 Orchestration

**Goal**: Connect the steps.

-   **Location**: `src/kg_agent/pipeline/` (New Package)
-   **Components**:
    -   `manager.py`: Coordinator to run the flow.
-   **Flow**:

    1.  `crawl_job = crawler.run(urls)`
    2.  `raw_paths = storage.save_batch(crawl_job)`
    3.  `parsed_docs = parser.process_batch(raw_paths)`
    4.  `chunks = chunker.process_batch(parsed_docs)`
    5.  `staging.save(chunks)`

## 3. Implementation Steps

### Step 1: Environment Setup

-   Add `docling` and `langchain-text-splitters` to `pyproject.toml`.
-   Create directory structure: `src/kg_agent/parser`, `src/kg_agent/chunker`, `src/kg_agent/pipeline`.
-   Create data directories: `data/raw`, `data/parsed`, `data/chunks`.

### Step 2: Storage Service

-   Implement `src/kg_agent/crawler/storage.py`.
-   Ensure it handles file naming, directories, and collision avoidance.

### Step 3: Parser Service (Docling)

-   Implement `src/kg_agent/parser/service.py`.
-   Create `parse_file(file_path)` method.
-   Handle both HTML and PDF inputs.

### Step 4: Chunker Service

-   Implement `src/kg_agent/chunker/service.py`.
-   Implement `chunk_document(parsed_doc)` method.
-   Define `Chunk` data model in `src/kg_agent/models/chunk.py`.

### Step 5: Integration/Orchestration

-   Create a simple script or API endpoint to trigger the full flow for testing.
-   Verify data flows from URL -> Raw File -> Parsed JSON -> Chunk JSON.

## 4. Data Models (Drafts)

### Chunk Model (`src/kg_agent/models/chunk.py`)

```python
class Chunk(BaseModel):
    id: str
    doc_id: str
    text: str
    index: int
    metadata: Dict[str, Any]  # source, title, section context
    embedding: Optional[List[float]] = None # To be filled later
```

## 5. Testing Plan

-   **Unit Tests**: Test Parser and Chunker with sample files.
-   **Integration Test**: Run a crawl on a single page, verify it ends up as chunks in `data/chunks`.