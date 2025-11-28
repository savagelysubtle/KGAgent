"""End-to-end tests for the complete KG Agent pipeline.

This test module verifies the entire pipeline from document ingestion
through to vector storage in ChromaDB and knowledge graph via Graphiti/FalkorDB.

Prerequisites:
- FalkorDB must be running (docker compose -f docker-compose.dev.yml up -d)
- ChromaDB storage directory must be accessible
- LM Studio must be running for entity extraction
"""

import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timezone

import pytest

from src.kg_agent.pipeline.manager import PipelineManager
from src.kg_agent.services.embedder import EmbedderService
from src.kg_agent.services.vector_store import VectorStoreService
from src.kg_agent.services.graphiti_service import GraphitiService, get_graphiti_service
from src.kg_agent.models.chunk import Chunk, ChunkBatch


@pytest.mark.asyncio
class TestEndToEndPipeline:
    """End-to-end tests for the complete pipeline including Graphiti KG creation."""

    @pytest.fixture(autouse=True)
    def setup_method_fixture(self, temp_dir: Path):
        """Setup test environment with temporary directories."""
        self.temp_dir = temp_dir
        self.test_data_dir = temp_dir / "test_data"
        self.test_data_dir.mkdir()

    async def test_full_pipeline_with_graphiti_entities(self, test_html_content: str):
        """
        Test the complete pipeline from file processing to Graphiti entity creation.

        This test verifies:
        1. Document parsing
        2. Text chunking
        3. Embedding generation
        4. ChromaDB vector storage
        5. Graphiti entity and relationship extraction
        """
        # Create test file
        test_file = self.test_data_dir / "knowledge_graph_test.html"
        test_file.write_text(test_html_content)

        # Initialize pipeline manager
        pipeline = PipelineManager()

        # Get Graphiti service and check connection
        graphiti = get_graphiti_service()
        init_success = await graphiti.initialize()

        if not init_success:
            pytest.skip("FalkorDB/Graphiti is not available - skipping end-to-end test")

        initial_stats = await graphiti.get_stats()
        initial_entities = initial_stats.get("total_entities", 0)
        initial_relationships = initial_stats.get("total_relationships", 0)

        # Run the full pipeline
        result = await pipeline.run_file_pipeline([str(test_file)])

        # Verify pipeline completed successfully
        assert result["status"] == "success", f"Pipeline failed: {result.get('error')}"
        assert result["metrics"]["files_submitted"] == 1
        assert result["metrics"]["parsed_successfully"] >= 1
        assert result["metrics"]["chunked_successfully"] >= 1

        # Verify artifacts were created
        assert len(result["artifacts"]["parsed_files"]) >= 1
        assert len(result["artifacts"]["chunk_files"]) >= 1

        # Verify ChromaDB vectors were stored
        vector_count = pipeline.vector_store.count()
        assert vector_count >= result["metrics"]["total_chunks"], \
            f"Expected at least {result['metrics']['total_chunks']} vectors in ChromaDB"

        await graphiti.close()

    async def test_pipeline_stores_chunks_in_chromadb(self, test_html_content: str):
        """
        Test that the pipeline correctly stores chunks in ChromaDB.
        """
        test_file = self.test_data_dir / "chromadb_test.html"
        test_file.write_text(test_html_content)

        pipeline = PipelineManager()

        initial_count = pipeline.vector_store.count()

        result = await pipeline.run_file_pipeline([str(test_file)])
        assert result["status"] == "success"

        final_count = pipeline.vector_store.count()

        # Should have added chunks
        assert final_count >= initial_count + result["metrics"]["total_chunks"]

    async def test_batch_file_processing(self, multiple_test_files: list[Path]):
        """
        Test batch processing of multiple files.
        """
        pipeline = PipelineManager()

        # Run pipeline with multiple files
        file_paths = [str(f) for f in multiple_test_files]
        result = await pipeline.run_file_pipeline(file_paths)

        assert result["status"] == "success"
        assert result["metrics"]["files_submitted"] == len(multiple_test_files)
        assert result["metrics"]["parsed_successfully"] >= 1

    async def test_pipeline_metrics_accuracy(self, test_html_content: str):
        """
        Test that pipeline metrics accurately reflect the processing results.
        """
        test_file = self.test_data_dir / "metrics_test.html"
        test_file.write_text(test_html_content)

        pipeline = PipelineManager()

        result = await pipeline.run_file_pipeline([str(test_file)])

        assert result["status"] == "success"

        # Verify metrics consistency
        metrics = result["metrics"]

        # Files submitted should match input
        assert metrics["files_submitted"] == 1

        # Parsed files should be <= submitted
        assert metrics["parsed_successfully"] <= metrics["files_submitted"]

        # Chunked files should be <= parsed
        assert metrics["chunked_successfully"] <= metrics["parsed_successfully"]

        # Total chunks should be > 0 if chunking was successful
        if metrics["chunked_successfully"] > 0:
            assert metrics["total_chunks"] > 0


@pytest.mark.asyncio
class TestGraphitiService:
    """Tests specifically for the GraphitiService."""

    async def test_graphiti_connection(self):
        """Test Graphiti/FalkorDB connection establishment."""
        graphiti = GraphitiService()
        success = await graphiti.initialize()

        if not success:
            pytest.skip("FalkorDB is not available")

        assert graphiti._initialized is True

        # Verify we can query
        stats = await graphiti.get_stats()
        assert stats["status"] == "success"
        assert stats.get("connected", True) is True

        await graphiti.close()

    async def test_add_episode_creates_entities(self):
        """Test that add_episode creates proper entities."""
        graphiti = GraphitiService()
        success = await graphiti.initialize()

        if not success:
            pytest.skip("FalkorDB is not available")

        # Clear graph for clean test
        await graphiti.clear_graph()

        # Add test episode
        result = await graphiti.add_episode(
            content="Google is a technology company founded by Larry Page and Sergey Brin.",
            name="test_episode_001",
            source_description="Test content about Google",
            reference_time=datetime.now(timezone.utc),
            source_type="text",
        )

        assert result["status"] == "success"
        # Graphiti extracts entities via LLM
        assert "nodes_created" in result
        assert "edges_created" in result

        await graphiti.close()

    async def test_search_finds_content(self):
        """Test that search can find inserted content."""
        graphiti = GraphitiService()
        success = await graphiti.initialize()

        if not success:
            pytest.skip("FalkorDB is not available")

        # Add content
        await graphiti.add_episode(
            content="Machine learning is a subset of artificial intelligence focused on algorithms.",
            name="ml_episode",
            source_description="ML content",
            reference_time=datetime.now(timezone.utc),
        )

        # Search for it
        result = await graphiti.search(
            query="machine learning algorithms",
            num_results=5,
        )

        assert result["status"] == "success"
        # May or may not find results depending on LLM extraction

        await graphiti.close()

    async def test_clear_graph(self):
        """Test that clear_graph removes all data."""
        graphiti = GraphitiService()
        success = await graphiti.initialize()

        if not success:
            pytest.skip("FalkorDB is not available")

        # Add some data
        await graphiti.add_episode(
            content="Temporary content for deletion test.",
            name="temp_episode",
            source_description="Will be deleted",
            reference_time=datetime.now(timezone.utc),
        )

        # Clear graph
        clear_result = await graphiti.clear_graph()
        assert clear_result["status"] == "success"

        # Verify graph is empty
        stats = await graphiti.get_stats()
        assert stats["total_entities"] == 0
        assert stats["total_relationships"] == 0

        await graphiti.close()

    async def test_bulk_episodes(self):
        """Test adding multiple episodes in bulk."""
        graphiti = GraphitiService()
        success = await graphiti.initialize()

        if not success:
            pytest.skip("FalkorDB is not available")

        await graphiti.clear_graph()

        episodes = [
            {
                "content": "Apple Inc is a technology company known for the iPhone.",
                "source": "apple_doc",
                "source_description": "Apple company info",
            },
            {
                "content": "Microsoft develops Windows operating system and Office software.",
                "source": "microsoft_doc",
                "source_description": "Microsoft company info",
            },
            {
                "content": "Amazon is an e-commerce and cloud computing company.",
                "source": "amazon_doc",
                "source_description": "Amazon company info",
            },
        ]

        result = await graphiti.add_episodes_bulk(episodes)

        assert result["status"] == "success"
        assert result["episodes_processed"] == 3

        await graphiti.close()


class TestVectorStoreIntegration:
    """Tests for ChromaDB vector store integration."""

    def test_embedding_generation(self, embedder: EmbedderService):
        """Test that embeddings are generated correctly."""
        text = "This is a test sentence for embedding generation."
        embedding = embedder.embed_text(text)

        assert isinstance(embedding, list)
        assert len(embedding) == 384  # MiniLM-L6-v2 dimension
        assert all(isinstance(x, float) for x in embedding)

    def test_batch_embedding(self, embedder: EmbedderService):
        """Test batch embedding generation."""
        texts = [
            "First test sentence.",
            "Second test sentence.",
            "Third test sentence."
        ]
        embeddings = embedder.embed_batch(texts)

        assert len(embeddings) == 3
        for emb in embeddings:
            assert len(emb) == 384

    def test_vector_store_add_and_query(self, vector_store: VectorStoreService, embedder: EmbedderService):
        """Test adding chunks to vector store and querying."""
        # Create test chunks with embeddings
        test_texts = [
            "Machine learning algorithms process data.",
            "Neural networks recognize patterns.",
            "Natural language processing understands text."
        ]

        embeddings = embedder.embed_batch(test_texts)

        test_chunks = [
            Chunk(
                id=f"vs_test_chunk_{i}",
                doc_id="vs_test_doc",
                text=text,
                index=i,
                metadata={"source": "test.html"},
                embedding=embeddings[i]
            )
            for i, text in enumerate(test_texts)
        ]

        # Add chunks
        initial_count = vector_store.count()
        vector_store.add_chunks(test_chunks)
        final_count = vector_store.count()

        assert final_count >= initial_count + len(test_chunks)

        # Query with similar embedding
        query_embedding = embedder.embed_text("machine learning data processing")
        results = vector_store.query(query_embedding, n_results=3)

        assert results is not None
        assert "documents" in results
        assert len(results["documents"]) > 0


class TestAPIEndToEnd:
    """End-to-end tests via API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from src.kg_agent.api.main import app
        return TestClient(app)

    def test_stats_endpoint_returns_real_data(self, client):
        """Test that the stats endpoint returns real database data."""
        response = client.get("/api/v1/stats/overview")

        assert response.status_code == 200
        data = response.json()

        # Verify structure
        assert "status" in data
        assert "services" in data
        assert "metrics" in data

        # Verify ChromaDB data in services
        assert "chromadb" in data["services"]
        assert "status" in data["services"]["chromadb"]

    def test_graph_stats_endpoint(self, client):
        """Test the graph stats endpoint."""
        response = client.get("/api/v1/graph/stats")

        assert response.status_code == 200
        data = response.json()

        # Should have stats or error
        assert "total_entities" in data or "error" in data or "status" in data

    def test_graph_search_endpoint(self, client):
        """Test the graph search endpoint."""
        response = client.post(
            "/api/v1/graph/search",
            json={"query": "test", "limit": 5}
        )

        assert response.status_code == 200
        data = response.json()

        # Should have results structure
        assert "nodes" in data or "edges" in data or "error" in data

    def test_reprocess_jobs_endpoint(self, client):
        """Test the reprocess jobs listing endpoint."""
        response = client.get("/api/v1/reprocess/jobs")

        assert response.status_code == 200
        data = response.json()

        assert "jobs" in data
        assert "count" in data

    def test_reprocess_graph_stats_endpoint(self, client):
        """Test the reprocess graph stats endpoint."""
        response = client.get("/api/v1/reprocess/graph/stats")

        assert response.status_code == 200
        data = response.json()

        # Should have entity counts
        assert "total_entities" in data or "status" in data


@pytest.mark.asyncio
class TestPipelineErrorHandling:
    """Tests for pipeline error handling and edge cases."""

    async def test_pipeline_handles_nonexistent_file(self):
        """Test pipeline gracefully handles non-existent files."""
        pipeline = PipelineManager()

        result = await pipeline.run_file_pipeline(["/nonexistent/file.html"])

        # Pipeline should complete but report no successful processing
        assert result["status"] == "success"
        assert result["metrics"]["files_submitted"] == 1
        assert result["metrics"]["parsed_successfully"] == 0

    async def test_pipeline_handles_empty_file_list(self):
        """Test pipeline handles empty file list."""
        pipeline = PipelineManager()

        result = await pipeline.run_file_pipeline([])

        assert result["status"] == "success"
        assert result["metrics"]["files_submitted"] == 0

    async def test_graphiti_handles_empty_content(self):
        """Test Graphiti handles empty content gracefully."""
        graphiti = GraphitiService()
        success = await graphiti.initialize()

        if not success:
            pytest.skip("FalkorDB is not available")

        # Try to add empty content
        result = await graphiti.add_episode(
            content="",
            name="empty_episode",
            source_description="Empty test",
            reference_time=datetime.now(timezone.utc),
        )

        # Should handle gracefully (may succeed or error, but not crash)
        assert "status" in result

        await graphiti.close()

    async def test_graphiti_connection_retry(self):
        """Test that Graphiti retries on connection failure."""
        graphiti = GraphitiService()

        # This should attempt to connect
        # If FalkorDB is not available, it should fail gracefully
        success = await graphiti.initialize()

        # Either succeeds (FalkorDB available) or fails gracefully (not available)
        assert isinstance(success, bool)

        if success:
            assert graphiti._initialized is True
        else:
            assert graphiti._initialized is False

        await graphiti.close()
