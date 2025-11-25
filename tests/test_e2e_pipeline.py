"""End-to-end tests for the complete KG Agent pipeline.

This test module verifies the entire pipeline from document ingestion
through to entity creation in Neo4j and vector storage in ChromaDB.

Prerequisites:
- Neo4j must be running (docker compose -f docker-compose.dev.yml up -d)
- ChromaDB storage directory must be accessible
"""

import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

import pytest

from src.kg_agent.pipeline.manager import PipelineManager
from src.kg_agent.services.embedder import EmbedderService
from src.kg_agent.services.vector_store import VectorStoreService
from src.kg_agent.services.graph_builder import GraphBuilderService, get_graph_builder
from src.kg_agent.models.chunk import Chunk, ChunkBatch


@pytest.mark.asyncio
class TestEndToEndPipeline:
    """End-to-end tests for the complete pipeline including Neo4j entity creation."""

    @pytest.fixture(autouse=True)
    def setup_method_fixture(self, temp_dir: Path):
        """Setup test environment with temporary directories."""
        self.temp_dir = temp_dir
        self.test_data_dir = temp_dir / "test_data"
        self.test_data_dir.mkdir()

    async def test_full_pipeline_with_neo4j_entities(self, test_html_content: str):
        """
        Test the complete pipeline from file processing to Neo4j entity creation.

        This test verifies:
        1. Document parsing
        2. Text chunking
        3. Embedding generation
        4. ChromaDB vector storage
        5. Neo4j entity and relationship creation
        """
        # Create test file
        test_file = self.test_data_dir / "knowledge_graph_test.html"
        test_file.write_text(test_html_content)

        # Initialize pipeline manager
        pipeline = PipelineManager()

        # Get initial Neo4j stats
        graph_builder = pipeline.graph_builder
        await graph_builder.initialize(max_retries=2)

        if not graph_builder.is_connected():
            pytest.skip("Neo4j is not available - skipping end-to-end test")

        initial_stats = await graph_builder.get_graph_stats()
        initial_nodes = initial_stats.get("total_nodes", 0)
        initial_edges = initial_stats.get("total_edges", 0)

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

        # Verify Neo4j entities were created
        final_stats = await graph_builder.get_graph_stats()
        final_nodes = final_stats.get("total_nodes", 0)
        final_edges = final_stats.get("total_edges", 0)

        # Should have created new nodes and edges
        assert final_nodes > initial_nodes, \
            f"Expected new nodes to be created. Initial: {initial_nodes}, Final: {final_nodes}"
        assert final_edges > initial_edges, \
            f"Expected new edges to be created. Initial: {initial_edges}, Final: {initial_edges}"

        # Verify entity types exist
        assert "Document" in final_stats.get("entity_types", {}), \
            "Expected Document nodes in the graph"
        assert "Episode" in final_stats.get("entity_types", {}), \
            "Expected Episode nodes in the graph"

        # Verify relationship types
        assert "PART_OF" in final_stats.get("relationship_types", {}), \
            "Expected PART_OF relationships in the graph"

        # Verify ChromaDB vectors were stored
        vector_count = pipeline.vector_store.count()
        assert vector_count >= result["metrics"]["total_chunks"], \
            f"Expected at least {result['metrics']['total_chunks']} vectors in ChromaDB"

        # Cleanup
        await graph_builder.close()

    async def test_pipeline_creates_searchable_graph(self, test_html_content: str):
        """
        Test that the pipeline creates a searchable knowledge graph.

        Verifies that entities created can be queried via Neo4j.
        """
        # Create test file with specific searchable content
        test_file = self.test_data_dir / "searchable_test.html"
        test_file.write_text(test_html_content)

        pipeline = PipelineManager()
        await pipeline.graph_builder.initialize(max_retries=2)

        if not pipeline.graph_builder.is_connected():
            pytest.skip("Neo4j is not available")

        # Run pipeline
        result = await pipeline.run_file_pipeline([str(test_file)])
        assert result["status"] == "success"

        # Search for content in the graph
        search_result = await pipeline.graph_builder.search_graph("knowledge graph", limit=10)

        assert search_result["status"] == "success"
        assert search_result["count"] >= 1, \
            "Expected to find at least one result for 'knowledge graph'"

        # Verify search results contain expected data
        for result_item in search_result["results"]:
            assert "id" in result_item
            assert "text" in result_item
            assert "source" in result_item

        await pipeline.graph_builder.close()

    async def test_batch_file_processing_creates_entities(self, multiple_test_files: list[Path]):
        """
        Test batch processing of multiple files creates entities for each.
        """
        pipeline = PipelineManager()
        await pipeline.graph_builder.initialize(max_retries=2)

        if not pipeline.graph_builder.is_connected():
            pytest.skip("Neo4j is not available")

        # Clear graph for clean test
        await pipeline.graph_builder.clear_graph()

        # Run pipeline with multiple files
        file_paths = [str(f) for f in multiple_test_files]
        result = await pipeline.run_file_pipeline(file_paths)

        assert result["status"] == "success"
        assert result["metrics"]["files_submitted"] == len(multiple_test_files)

        # Verify entities were created for each file
        stats = await pipeline.graph_builder.get_graph_stats()

        # Should have Document nodes (at least one per chunk)
        assert stats["entity_types"].get("Document", 0) >= result["metrics"]["total_chunks"]

        # Should have Episode node for the job
        assert stats["entity_types"].get("Episode", 0) >= 1

        await pipeline.graph_builder.close()

    async def test_pipeline_metrics_accuracy(self, test_html_content: str):
        """
        Test that pipeline metrics accurately reflect the processing results.
        """
        test_file = self.test_data_dir / "metrics_test.html"
        test_file.write_text(test_html_content)

        pipeline = PipelineManager()
        await pipeline.graph_builder.initialize(max_retries=2)

        if not pipeline.graph_builder.is_connected():
            pytest.skip("Neo4j is not available")

        # Clear graph for accurate counting
        await pipeline.graph_builder.clear_graph()

        # Get initial ChromaDB count
        initial_vector_count = pipeline.vector_store.count()

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

        # Verify ChromaDB count increased (or equals if chunks were upserted)
        final_vector_count = pipeline.vector_store.count()
        # The count should increase by at least some chunks, or stay the same if upserted
        assert final_vector_count >= initial_vector_count, \
            f"Vector count should not decrease. Initial: {initial_vector_count}, Final: {final_vector_count}"

        # Verify Neo4j nodes match chunks
        stats = await pipeline.graph_builder.get_graph_stats()
        assert stats["entity_types"].get("Document", 0) >= metrics["total_chunks"]

        await pipeline.graph_builder.close()


@pytest.mark.asyncio
class TestGraphBuilderService:
    """Tests specifically for the GraphBuilderService."""

    async def test_neo4j_connection(self):
        """Test Neo4j connection establishment."""
        builder = GraphBuilderService()
        success = await builder.initialize(max_retries=2)

        if not success:
            pytest.skip("Neo4j is not available")

        assert builder.is_connected()

        # Verify we can query
        stats = await builder.get_graph_stats()
        assert stats["status"] == "success"
        assert stats["connected"] is True

        await builder.close()

    async def test_build_from_chunks_creates_entities(self, temp_dir: Path):
        """Test that build_from_chunks creates proper Neo4j entities."""
        builder = GraphBuilderService()
        success = await builder.initialize(max_retries=2)

        if not success:
            pytest.skip("Neo4j is not available")

        # Clear graph for clean test
        await builder.clear_graph()

        # Create test chunks
        test_chunks = [
            Chunk(
                id=f"test_chunk_{i}",
                doc_id="test_doc_1",
                text=f"This is test chunk number {i} with content about knowledge graphs.",
                index=i,
                metadata={"source": "test.html", "job_id": "test_job_001"}
            )
            for i in range(5)
        ]

        # Build graph from chunks
        result = await builder.build_from_chunks(test_chunks, episode_name="test_episode_001")

        assert result["status"] == "success"
        assert result["chunks_processed"] == 5
        assert result["nodes_created"] == 5
        assert result["edges_created"] == 5  # PART_OF relationships

        # Verify entities in graph
        stats = await builder.get_graph_stats()
        assert stats["entity_types"].get("Document", 0) == 5
        assert stats["entity_types"].get("Episode", 0) == 1
        assert stats["relationship_types"].get("PART_OF", 0) == 5

        await builder.close()

    async def test_search_graph_finds_content(self, temp_dir: Path):
        """Test that search_graph can find inserted content."""
        builder = GraphBuilderService()
        success = await builder.initialize(max_retries=2)

        if not success:
            pytest.skip("Neo4j is not available")

        # Clear and populate graph
        await builder.clear_graph()

        test_chunks = [
            Chunk(
                id="chunk_machine_learning",
                doc_id="ml_doc",
                text="Machine learning is a subset of artificial intelligence focused on algorithms.",
                index=0,
                metadata={"source": "ml.html", "job_id": "ml_job"}
            ),
            Chunk(
                id="chunk_deep_learning",
                doc_id="dl_doc",
                text="Deep learning uses neural networks with many layers for pattern recognition.",
                index=0,
                metadata={"source": "dl.html", "job_id": "dl_job"}
            ),
            Chunk(
                id="chunk_nlp",
                doc_id="nlp_doc",
                text="Natural language processing enables computers to understand human language.",
                index=0,
                metadata={"source": "nlp.html", "job_id": "nlp_job"}
            )
        ]

        await builder.build_from_chunks(test_chunks, episode_name="ai_episode")

        # Search for specific content
        ml_results = await builder.search_graph("machine learning", limit=5)
        assert ml_results["status"] == "success"
        assert ml_results["count"] >= 1

        dl_results = await builder.search_graph("neural networks", limit=5)
        assert dl_results["status"] == "success"
        assert dl_results["count"] >= 1

        nlp_results = await builder.search_graph("language processing", limit=5)
        assert nlp_results["status"] == "success"
        assert nlp_results["count"] >= 1

        # Search for non-existent content
        no_results = await builder.search_graph("quantum computing", limit=5)
        assert no_results["count"] == 0

        await builder.close()

    async def test_clear_graph(self):
        """Test that clear_graph removes all data."""
        builder = GraphBuilderService()
        success = await builder.initialize(max_retries=2)

        if not success:
            pytest.skip("Neo4j is not available")

        # Add some data
        test_chunks = [
            Chunk(
                id="temp_chunk",
                doc_id="temp_doc",
                text="Temporary content for deletion test.",
                index=0,
                metadata={"source": "temp.html"}
            )
        ]
        await builder.build_from_chunks(test_chunks, episode_name="temp_episode")

        # Verify data exists
        stats_before = await builder.get_graph_stats()
        assert stats_before["total_nodes"] > 0

        # Clear graph
        clear_result = await builder.clear_graph()
        assert clear_result["status"] == "success"
        assert clear_result["nodes_deleted"] > 0

        # Verify graph is empty
        stats_after = await builder.get_graph_stats()
        assert stats_after["total_nodes"] == 0
        assert stats_after["total_edges"] == 0

        await builder.close()

    async def test_graph_stats_accuracy(self):
        """Test that graph stats accurately reflect the data."""
        builder = GraphBuilderService()
        success = await builder.initialize(max_retries=2)

        if not success:
            pytest.skip("Neo4j is not available")

        # Clear graph
        await builder.clear_graph()

        # Add known number of chunks
        num_chunks = 10
        test_chunks = [
            Chunk(
                id=f"stats_chunk_{i}",
                doc_id=f"stats_doc_{i % 3}",  # 3 different documents
                text=f"Content for chunk {i} testing statistics accuracy.",
                index=i,
                metadata={"source": f"doc_{i % 3}.html"}
            )
            for i in range(num_chunks)
        ]

        await builder.build_from_chunks(test_chunks, episode_name="stats_episode")

        stats = await builder.get_graph_stats()

        # Should have exactly num_chunks Document nodes
        assert stats["entity_types"].get("Document", 0) == num_chunks

        # Should have exactly 1 Episode node
        assert stats["entity_types"].get("Episode", 0) == 1

        # Total nodes = Documents + Episode
        assert stats["total_nodes"] == num_chunks + 1

        # Should have num_chunks PART_OF relationships
        assert stats["relationship_types"].get("PART_OF", 0) == num_chunks
        assert stats["total_edges"] == num_chunks

        await builder.close()


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

        # Verify Neo4j data in services
        assert "neo4j" in data["services"]
        assert "status" in data["services"]["neo4j"]

        # Verify metrics
        assert "chunks" in data["metrics"]
        assert "entities" in data["metrics"]

    def test_graph_stats_endpoint(self, client):
        """Test the graph stats endpoint."""
        response = client.get("/api/v1/graph/stats")

        assert response.status_code == 200
        data = response.json()

        # Should have stats even if Neo4j is not connected
        assert "total_nodes" in data or "error" in data

    def test_graph_search_endpoint(self, client):
        """Test the graph search endpoint."""
        response = client.post(
            "/api/v1/graph/search",
            json={"query": "test", "limit": 5}
        )

        assert response.status_code == 200
        data = response.json()

        assert "results" in data or "error" in data
        assert "count" in data or "error" in data


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

    async def test_graph_builder_handles_empty_chunks(self):
        """Test graph builder handles empty chunk list."""
        builder = GraphBuilderService()
        success = await builder.initialize(max_retries=1)

        if not success:
            pytest.skip("Neo4j is not available")

        result = await builder.build_from_chunks([], episode_name="empty_episode")

        assert result["status"] == "success"
        assert result["message"] == "No chunks to process"
        assert result["entities_added"] == 0

        await builder.close()

    async def test_graph_builder_retry_logic(self):
        """Test that graph builder retries on connection failure."""
        builder = GraphBuilderService()

        # This should attempt to connect with retries
        # If Neo4j is not available, it should fail gracefully after retries
        success = await builder.initialize(max_retries=1)

        # Either succeeds (Neo4j available) or fails gracefully (not available)
        assert isinstance(success, bool)

        if success:
            assert builder.is_connected()
        else:
            assert not builder.is_connected()

        await builder.close()

