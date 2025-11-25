"""Pytest fixtures for KG Agent tests."""

import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Generator, AsyncGenerator

import pytest

from src.kg_agent.services.graph_builder import GraphBuilderService
from src.kg_agent.services.vector_store import VectorStoreService
from src.kg_agent.services.embedder import EmbedderService
from src.kg_agent.pipeline.manager import PipelineManager


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp(prefix="kg_agent_test_"))
    yield temp_path
    # Cleanup - use ignore_errors on Windows due to file locks
    if temp_path.exists():
        try:
            shutil.rmtree(temp_path, ignore_errors=True)
        except Exception:
            pass  # Ignore cleanup errors on Windows


@pytest.fixture(scope="function")
def test_html_content() -> str:
    """Sample HTML content for testing."""
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Test Document - Knowledge Graph Pipeline</title></head>
    <body>
        <h1>Introduction to Knowledge Graphs</h1>
        <p>A knowledge graph is a structured representation of real-world entities
        and their relationships. It provides a way to store and query interconnected
        data in a meaningful way.</p>

        <h2>Key Components</h2>
        <p>Knowledge graphs consist of nodes (entities) and edges (relationships).
        Each node represents a concept like a person, place, or thing. Each edge
        represents how two nodes are connected.</p>

        <h3>Entity Types</h3>
        <ul>
            <li>Person - Represents individuals like Albert Einstein or Marie Curie</li>
            <li>Organization - Companies, universities, research institutions</li>
            <li>Location - Cities, countries, geographical features</li>
            <li>Concept - Abstract ideas like Machine Learning or Graph Theory</li>
        </ul>

        <h2>Applications</h2>
        <p>Knowledge graphs are used in search engines, recommendation systems,
        and natural language processing. Google's Knowledge Graph and Facebook's
        Social Graph are prominent examples.</p>

        <h3>Technical Implementation</h3>
        <p>Modern knowledge graphs often use graph databases like Neo4j for storage
        and Cypher query language for data retrieval. Vector embeddings can enhance
        semantic search capabilities.</p>
    </body>
    </html>
    """


@pytest.fixture(scope="function")
def test_markdown_content() -> str:
    """Sample Markdown content for testing."""
    return """
# Machine Learning Pipeline Architecture

## Overview
This document describes the architecture of a machine learning pipeline
for processing unstructured data into knowledge graphs.

## Components

### Data Ingestion
- Web crawlers collect data from specified URLs
- Document parsers extract text from various formats
- Content is normalized and cleaned

### Processing Pipeline
1. **Chunking**: Large documents are split into smaller segments
2. **Embedding**: Text chunks are converted to vector representations
3. **Storage**: Vectors are stored in ChromaDB for similarity search

### Knowledge Graph Construction
The pipeline builds a knowledge graph in Neo4j with:
- Document nodes containing text chunks
- Episode nodes grouping related documents
- Relationships linking documents to episodes

## Technologies Used
- Crawl4AI for web scraping
- Docling for document parsing
- LangChain for text chunking
- Sentence Transformers for embeddings
- ChromaDB for vector storage
- Neo4j for graph database
"""


@pytest.fixture(scope="function")
def sample_html_file(temp_dir: Path, test_html_content: str) -> Path:
    """Create a sample HTML file for testing."""
    file_path = temp_dir / "test_document.html"
    file_path.write_text(test_html_content, encoding="utf-8")
    return file_path


@pytest.fixture(scope="function")
def sample_markdown_file(temp_dir: Path, test_markdown_content: str) -> Path:
    """Create a sample Markdown file for testing."""
    file_path = temp_dir / "test_document.md"
    file_path.write_text(test_markdown_content, encoding="utf-8")
    return file_path


@pytest.fixture(scope="function")
def multiple_test_files(temp_dir: Path) -> list[Path]:
    """Create multiple test files for batch processing tests."""
    files = []

    for i in range(3):
        content = f"""
        <!DOCTYPE html>
        <html>
        <head><title>Test Document {i + 1}</title></head>
        <body>
            <h1>Document {i + 1}: Test Content</h1>
            <p>This is test document number {i + 1} for batch processing verification.</p>
            <p>It contains enough content to generate multiple chunks during processing.</p>
            <h2>Section A</h2>
            <p>Additional content for document {i + 1} to ensure proper chunking.</p>
            <h2>Section B</h2>
            <p>More content about topic {i + 1} including technical details.</p>
        </body>
        </html>
        """
        file_path = temp_dir / f"test_doc_{i + 1}.html"
        file_path.write_text(content, encoding="utf-8")
        files.append(file_path)

    return files


@pytest.fixture(scope="function")
def temp_chroma_dir(temp_dir: Path) -> Path:
    """Create a temporary directory for ChromaDB."""
    chroma_path = temp_dir / "chroma_db"
    chroma_path.mkdir(parents=True, exist_ok=True)
    return chroma_path


@pytest.fixture(scope="function")
def vector_store(temp_chroma_dir: Path) -> Generator[VectorStoreService, None, None]:
    """Create a VectorStoreService with temporary storage."""
    store = VectorStoreService(
        persist_path=str(temp_chroma_dir),
        collection_name="test_collection"
    )
    yield store
    # Cleanup handled by temp_dir fixture


@pytest.fixture(scope="function")
def embedder() -> EmbedderService:
    """Create an EmbedderService instance."""
    return EmbedderService()


@pytest.fixture(scope="function")
async def graph_builder() -> AsyncGenerator[GraphBuilderService, None]:
    """Create a GraphBuilderService instance and initialize it."""
    builder = GraphBuilderService()
    yield builder
    # Cleanup
    await builder.close()


@pytest.fixture(scope="function")
def pipeline_manager() -> PipelineManager:
    """Create a PipelineManager instance."""
    return PipelineManager()


@pytest.fixture(scope="function")
async def initialized_graph_builder() -> AsyncGenerator[GraphBuilderService, None]:
    """Create and initialize a GraphBuilderService instance."""
    builder = GraphBuilderService()
    success = await builder.initialize(max_retries=1)
    if not success:
        pytest.skip("Neo4j is not available - skipping graph tests")
    yield builder
    await builder.close()


@pytest.fixture(scope="function")
async def clean_graph(initialized_graph_builder: GraphBuilderService) -> AsyncGenerator[GraphBuilderService, None]:
    """Provide a clean graph for testing (clears existing data)."""
    # Clear the graph before the test
    await initialized_graph_builder.clear_graph()
    yield initialized_graph_builder
    # Clear after the test too
    await initialized_graph_builder.clear_graph()

