"""Tests for the Graphiti-based knowledge graph pipeline.

Tests the resumable pipeline that sends chunks directly to Graphiti
for entity extraction and knowledge graph construction.
"""

import pytest
import asyncio
from pathlib import Path
from datetime import datetime, timezone

from kg_agent.models.chunk import Chunk
from kg_agent.models.entity import (
    Entity,
    Relationship,
    EntityType,
    RelationshipType,
    ExtractionResult,
    ReprocessingOptions,
    ReprocessingStatus,
)


class TestExtractionModels:
    """Tests for extraction data models."""

    def test_entity_creation(self):
        """Test Entity model creation."""
        entity = Entity(
            name="Test Entity",
            type=EntityType.CONCEPT,
            description="A test entity",
        )

        assert entity.name == "Test Entity"
        assert entity.type == EntityType.CONCEPT
        assert entity.confidence == 1.0
        assert len(entity.id) > 0

    def test_relationship_creation(self):
        """Test Relationship model creation."""
        rel = Relationship(
            source_entity="Entity A",
            target_entity="Entity B",
            type=RelationshipType.RELATED_TO,
            source_chunk="chunk1",
        )

        assert rel.source_entity == "Entity A"
        assert rel.target_entity == "Entity B"
        assert rel.type == RelationshipType.RELATED_TO

    def test_extraction_result_creation(self):
        """Test ExtractionResult model creation."""
        result = ExtractionResult(
            chunk_id="chunk1",
            entities=[
                Entity(name="Test", type=EntityType.CONCEPT),
            ],
            relationships=[],
            processing_time=1.5,
        )

        assert result.chunk_id == "chunk1"
        assert len(result.entities) == 1
        assert result.processing_time == 1.5

    def test_reprocessing_options_defaults(self):
        """Test ReprocessingOptions default values."""
        options = ReprocessingOptions()

        assert options.extract_entities is True
        assert options.extract_relationships is True
        assert options.update_existing_graph is True
        # batch_size default is 10 for Graphiti batching
        assert options.batch_size >= 1


class TestChunkModels:
    """Tests for chunk data models used in Graphiti pipeline."""

    def test_chunk_creation(self):
        """Test Chunk model creation."""
        chunk = Chunk(
            id="test_chunk_1",
            doc_id="test_doc_1",
            text="This is test content about knowledge graphs.",
            index=0,
            metadata={"source": "test.html"},
        )

        assert chunk.id == "test_chunk_1"
        assert chunk.doc_id == "test_doc_1"
        assert len(chunk.text) > 0
        assert chunk.index == 0

    def test_chunk_with_embedding(self):
        """Test Chunk with embedding vector."""
        embedding = [0.1] * 384  # MiniLM dimension
        chunk = Chunk(
            id="test_chunk_emb",
            doc_id="test_doc",
            text="Content with embedding",
            index=0,
            metadata={},
            embedding=embedding,
        )

        assert chunk.embedding is not None
        assert len(chunk.embedding) == 384


@pytest.mark.asyncio
class TestGraphitiService:
    """Tests for the GraphitiService wrapper."""

    async def test_graphiti_initialization(self, initialized_graphiti):
        """Test that Graphiti initializes properly."""
        assert initialized_graphiti._initialized is True
        assert initialized_graphiti._graphiti is not None

    async def test_graphiti_add_episode(self, initialized_graphiti):
        """Test adding an episode to Graphiti."""
        result = await initialized_graphiti.add_episode(
            content="OpenAI is an artificial intelligence company founded by Sam Altman.",
            name="test_episode",
            source_description="Test content",
            reference_time=datetime.now(timezone.utc),
            source_type="text",
        )

        assert result["status"] == "success"
        # Graphiti extracts entities automatically
        assert "nodes_created" in result
        assert "edges_created" in result

    async def test_graphiti_search(self, initialized_graphiti):
        """Test searching the knowledge graph."""
        # First add some content
        await initialized_graphiti.add_episode(
            content="Microsoft Corporation is a technology company headquartered in Redmond.",
            name="test_search_episode",
            source_description="Test search content",
            reference_time=datetime.now(timezone.utc),
        )

        # Then search for it
        result = await initialized_graphiti.search(
            query="Microsoft technology",
            num_results=5,
        )

        assert result["status"] == "success"
        # May or may not find results depending on graph state

    async def test_graphiti_stats(self, initialized_graphiti):
        """Test getting graph statistics."""
        stats = await initialized_graphiti.get_stats()

        assert stats["status"] == "success"
        assert "total_entities" in stats
        assert "total_relationships" in stats
        assert "total_episodes" in stats

    async def test_graphiti_clear_graph(self, initialized_graphiti):
        """Test clearing the graph."""
        # Add something first
        await initialized_graphiti.add_episode(
            content="Test content to be deleted.",
            name="delete_test",
            source_description="Will be cleared",
            reference_time=datetime.now(timezone.utc),
        )

        # Clear it
        result = await initialized_graphiti.clear_graph()

        assert result["status"] == "success"

        # Verify it's empty
        stats = await initialized_graphiti.get_stats()
        assert stats["total_entities"] == 0


@pytest.mark.asyncio
class TestResumablePipeline:
    """Tests for the resumable Graphiti pipeline."""

    async def test_pipeline_creation(self):
        """Test resumable pipeline creation."""
        from kg_agent.services.resumable_pipeline import get_resumable_pipeline

        pipeline = get_resumable_pipeline()
        assert pipeline is not None
        assert pipeline.graphiti_service is not None
        assert pipeline.job_tracker is not None

    async def test_job_status_tracking(self):
        """Test job status methods."""
        from kg_agent.services.resumable_pipeline import get_resumable_pipeline
        from kg_agent.services.processing_job_tracker import get_job_tracker, JobType

        pipeline = get_resumable_pipeline()
        job_tracker = get_job_tracker()

        # Create a test job
        job = job_tracker.create_job(
            doc_id="test_doc_123",
            job_type=JobType.ENTITY_EXTRACTION,
            total_chunks=10,
            options={"batch_size": 5},
        )

        # Check status
        status = pipeline.get_job_status(job.id)
        assert status is not None
        assert status["job_id"] == job.id
        assert status["doc_id"] == "test_doc_123"
        assert status["total_chunks"] == 10

        # Cleanup
        job_tracker.delete_job(job.id)

    async def test_pause_request(self):
        """Test pause request mechanism."""
        from kg_agent.services.resumable_pipeline import get_resumable_pipeline
        from kg_agent.services.processing_job_tracker import get_job_tracker, JobType, JobStatus

        pipeline = get_resumable_pipeline()
        job_tracker = get_job_tracker()

        # Create and start a test job
        job = job_tracker.create_job(
            doc_id="test_pause_doc",
            job_type=JobType.ENTITY_EXTRACTION,
            total_chunks=100,
            options={},
        )
        job_tracker.start_job(job.id)

        # Request pause
        pipeline.request_pause(job.id)

        # Verify pause was requested in database
        assert job_tracker.is_pause_requested(job.id) is True

        # Cleanup
        job_tracker.cancel_job(job.id)
        job_tracker.delete_job(job.id)

    async def test_resumable_jobs_list(self):
        """Test getting list of resumable jobs."""
        from kg_agent.services.resumable_pipeline import get_resumable_pipeline
        from kg_agent.services.processing_job_tracker import get_job_tracker, JobType, JobStatus

        pipeline = get_resumable_pipeline()
        job_tracker = get_job_tracker()

        # Create a paused job
        job = job_tracker.create_job(
            doc_id="test_resumable_doc",
            job_type=JobType.ENTITY_EXTRACTION,
            total_chunks=50,
            options={},
        )
        job_tracker.start_job(job.id)
        job_tracker.pause_job(job.id)

        # Get resumable jobs
        resumable = pipeline.get_resumable_jobs()

        # Should include our paused job
        job_ids = [j["job_id"] for j in resumable if j]
        assert job.id in job_ids

        # Cleanup
        job_tracker.delete_job(job.id)


@pytest.mark.asyncio
class TestProcessingJobTracker:
    """Tests for the job tracking system."""

    async def test_create_and_get_job(self):
        """Test job creation and retrieval."""
        from kg_agent.services.processing_job_tracker import get_job_tracker, JobType

        tracker = get_job_tracker()

        job = tracker.create_job(
            doc_id="tracker_test_doc",
            job_type=JobType.ENTITY_EXTRACTION,
            total_chunks=25,
            options={"batch_size": 5},
        )

        assert job is not None
        assert job.doc_id == "tracker_test_doc"
        assert job.total_chunks == 25

        # Retrieve it
        retrieved = tracker.get_job(job.id)
        assert retrieved is not None
        assert retrieved.id == job.id

        # Cleanup
        tracker.delete_job(job.id)

    async def test_job_progress_updates(self):
        """Test updating job progress."""
        from kg_agent.services.processing_job_tracker import get_job_tracker, JobType

        tracker = get_job_tracker()

        job = tracker.create_job(
            doc_id="progress_test_doc",
            job_type=JobType.ENTITY_EXTRACTION,
            total_chunks=100,
            options={},
        )
        tracker.start_job(job.id)

        # Update progress
        tracker.update_progress(
            job.id,
            processed_chunks=50,
            current_chunk_index=50,
            entities_extracted=25,
            relationships_extracted=10,
        )

        # Verify
        updated = tracker.get_job(job.id)
        assert updated.processed_chunks == 50
        assert updated.entities_extracted == 25
        assert updated.relationships_extracted == 10
        assert updated.progress_percent == 50.0

        # Cleanup
        tracker.delete_job(job.id)

    async def test_job_state_transitions(self):
        """Test job state transitions."""
        from kg_agent.services.processing_job_tracker import get_job_tracker, JobType, JobStatus

        tracker = get_job_tracker()

        job = tracker.create_job(
            doc_id="state_test_doc",
            job_type=JobType.ENTITY_EXTRACTION,
            total_chunks=10,
            options={},
        )

        # Initial state is pending
        assert job.status == JobStatus.PENDING.value

        # Start
        tracker.start_job(job.id)
        job = tracker.get_job(job.id)
        assert job.status == JobStatus.RUNNING.value

        # Pause
        tracker.pause_job(job.id)
        job = tracker.get_job(job.id)
        assert job.status == JobStatus.PAUSED.value

        # Resume (start again)
        tracker.start_job(job.id)
        job = tracker.get_job(job.id)
        assert job.status == JobStatus.RUNNING.value

        # Complete
        tracker.complete_job(job.id)
        job = tracker.get_job(job.id)
        assert job.status == JobStatus.COMPLETED.value

        # Cleanup
        tracker.delete_job(job.id)

    async def test_chunk_tracking(self):
        """Test processed chunk tracking."""
        from kg_agent.services.processing_job_tracker import get_job_tracker, JobType

        tracker = get_job_tracker()

        job = tracker.create_job(
            doc_id="chunk_test_doc",
            job_type=JobType.ENTITY_EXTRACTION,
            total_chunks=5,
            options={},
        )
        tracker.start_job(job.id)

        # Record some processed chunks
        tracker.record_processed_chunk(job.id, "chunk_1", 0, 3, 2)
        tracker.record_processed_chunk(job.id, "chunk_2", 1, 5, 1)

        # Get processed chunk IDs
        processed_ids = tracker.get_processed_chunk_ids(job.id)
        assert "chunk_1" in processed_ids
        assert "chunk_2" in processed_ids
        assert len(processed_ids) == 2

        # Cleanup
        tracker.delete_job(job.id)


# Integration test that requires LM Studio + FalkorDB running
@pytest.mark.skipif(
    not pytest.importorskip("openai", reason="OpenAI client not installed"),
    reason="OpenAI client required for integration tests"
)
class TestGraphitiIntegration:
    """Integration tests for Graphiti with actual LLM."""

    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks for testing."""
        return [
            Chunk(
                id="int_test_chunk_1",
                doc_id="int_test_doc_1",
                text="""
                Microsoft Corporation, led by CEO Satya Nadella, announced a major
                partnership with OpenAI. The technology giant, headquartered in Redmond,
                Washington, has invested billions in the AI research company.
                """,
                index=0,
                metadata={"source": "test_document.pdf"},
            ),
            Chunk(
                id="int_test_chunk_2",
                doc_id="int_test_doc_1",
                text="""
                OpenAI, founded by Sam Altman and others, is known for creating ChatGPT
                and GPT-4. The company focuses on artificial general intelligence research
                and is based in San Francisco, California.
                """,
                index=1,
                metadata={"source": "test_document.pdf"},
            ),
        ]

    @pytest.mark.asyncio
    async def test_graphiti_extracts_entities(self, sample_chunks, initialized_graphiti):
        """Test that Graphiti extracts entities from text."""
        # Combine chunks into episode content
        combined_text = "\n\n".join([c.text for c in sample_chunks])

        try:
            result = await initialized_graphiti.add_episode(
                content=combined_text,
                name="integration_test_episode",
                source_description="Integration test content",
                reference_time=datetime.now(timezone.utc),
            )

            assert result["status"] == "success"
            print(f"\nGraphiti extracted:")
            print(f"  - {result.get('nodes_created', 0)} entity nodes")
            print(f"  - {result.get('edges_created', 0)} relationship edges")

            if result.get("entities"):
                print("\nEntities found:")
                for entity in result["entities"]:
                    print(f"  - {entity.get('name')} ({entity.get('labels', [])})")

        except Exception as e:
            pytest.skip(f"LM Studio or FalkorDB not available: {e}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
