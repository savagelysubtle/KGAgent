# Docling to ChromaDB Pipeline Integration Plan

## Executive Summary

This plan outlines the integration of ChromaDB vector storage into the existing Docling-based ETL pipeline. This addition allows the system to not only chunk documents but also generate semantic embeddings and store them for efficient retrieval, laying the groundwork for the Knowledge Graph construction and Hybrid Search.

## 1. Goals

1.  **Persist Embeddings**: Store semantic vectors for each document chunk in ChromaDB.
2.  **Local Processing**: Use `sentence-transformers` for local embedding generation (removing external API dependencies for this step).
3.  **Pipeline Integration**: Seamlessly extend the current `crawl -> parse -> chunk` flow to include `-> embed -> store`.
4.  **Metadata Preservation**: Ensure all source metadata (URL, titles, etc.) is indexed alongside vectors.

## 2. Architecture Updates

### Data Flow
```
... -> [Chunker Service] -> (Chunks) -> [Embedder Service] -> (Embeddings) -> [Vector Store Service] -> (ChromaDB)
```

### New Components

1.  **Embedder Service** (`src/kg_agent/services/embedder.py`)
    *   Wraps `sentence-transformers`.
    *   Generates dense vector embeddings for chunk text.
    *   Supports batch processing for performance.

2.  **Vector Store Service** (`src/kg_agent/services/vector_store.py`)
    *   Wraps `chromadb.PersistentClient`.
    *   Manages collections (e.g., `document_chunks`).
    *   Handles upsert operations (add/update).
    *   Supports querying (for verification/search).

## 3. Implementation Plan

### Phase 1: Dependencies & Configuration

1.  **Update `pyproject.toml`**:
    *   Add `chromadb`
    *   Add `sentence-transformers`
    *   Add `torch` (if not pulled in by sentence-transformers, though usually is)

2.  **Update `src/kg_agent/core/config.py`**:
    *   `CHROMA_PERSIST_DIR`: Path to local vector store (default: `./data/chroma_db`).
    *   `CHROMA_COLLECTION_NAME`: Default collection name (default: `document_chunks`).
    *   `HF_EMBEDDING_MODEL`: Model name (default: `sentence-transformers/all-MiniLM-L6-v2`).

### Phase 2: Core Services Implementation

1.  **Create `src/kg_agent/services/embedder.py`**:
    ```python
    class EmbedderService:
        def __init__(self, model_name: str = settings.HF_EMBEDDING_MODEL): ...
        def embed_text(self, text: str) -> List[float]: ...
        def embed_batch(self, texts: List[str]) -> List[List[float]]: ...
    ```

2.  **Create `src/kg_agent/services/vector_store.py`**:
    ```python
    class VectorStoreService:
        def __init__(self, persist_path: str = settings.CHROMA_PERSIST_DIR): ...
        def add_chunks(self, chunks: List[Chunk]): ...
            # Internally calls embedder if not already embedded, or expects embedded chunks
            # But better design: Pipeline passes embeddings separately or Chunk model has embedding field.
            # Decision: Update Chunk model to include embedding field (already done in previous plan).
    ```

### Phase 3: Pipeline Integration

1.  **Update `src/kg_agent/pipeline/manager.py`**:
    *   Initialize `EmbedderService` and `VectorStoreService` in `__init__`.
    *   In `run_pipeline` and `run_file_pipeline`:
        *   After chunking (`self.chunker.process_batch`), load the generated chunk objects.
        *   Pass chunks to `EmbedderService` to populate `chunk.embedding`.
        *   Pass embedded chunks to `VectorStoreService` for persistence.

### Phase 4: Verification

1.  **Create Test Script**:
    *   Run a simple crawl or file upload.
    *   Verify data exists in `data/chroma_db` (files created).
    *   Query ChromaDB to ensure vectors and metadata are retrievable.

## 4. Action Items

- [ ] Install dependencies (`uv add chromadb sentence-transformers`).
- [ ] Update `Settings` in `config.py`.
- [ ] Implement `EmbedderService`.
- [ ] Implement `VectorStoreService`.
- [ ] Update `PipelineManager` to orchestrate the new steps.
- [ ] Run `test_pipeline.py` (updated) to verify end-to-end flow.

