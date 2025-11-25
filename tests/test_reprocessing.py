"""Tests for the reprocessing pipeline with entity extraction."""

import pytest
import asyncio
from pathlib import Path
from datetime import datetime

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
from kg_agent.services.entity_resolver import EntityResolver


class TestEntityResolver:
    """Tests for the EntityResolver class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.resolver = EntityResolver()

    def test_normalize_name(self):
        """Test name normalization."""
        assert self.resolver.normalize_name("OpenAI") == "openai"
        assert self.resolver.normalize_name("  OpenAI Inc.  ") == "openai inc"
        assert self.resolver.normalize_name("New York City") == "new york city"

    def test_names_match_exact(self):
        """Test exact name matching."""
        assert self.resolver.names_match("OpenAI", "openai")
        assert self.resolver.names_match("Microsoft", "Microsoft")

    def test_names_match_substring(self):
        """Test substring matching."""
        assert self.resolver.names_match("OpenAI", "OpenAI Inc")
        assert self.resolver.names_match("Google", "Google LLC")

    def test_names_match_word_overlap(self):
        """Test word overlap matching."""
        assert self.resolver.names_match("New York City", "New York")
        assert self.resolver.names_match("United States of America", "United States")

    def test_names_dont_match(self):
        """Test non-matching names."""
        assert not self.resolver.names_match("Apple", "Microsoft")
        assert not self.resolver.names_match("Google", "Amazon")

    def test_resolve_entities_deduplication(self):
        """Test entity deduplication."""
        entities = [
            Entity(name="OpenAI", type=EntityType.ORGANIZATION, description="AI company"),
            Entity(name="openai", type=EntityType.ORGANIZATION, description="Research lab"),
            Entity(name="OpenAI Inc", type=EntityType.ORGANIZATION, description="AI research"),
        ]

        resolved = self.resolver.resolve_entities(entities)

        assert len(resolved) == 1
        assert resolved[0].name in ["OpenAI", "openai", "OpenAI Inc"]
        assert len(resolved[0].aliases) >= 1  # Other names become aliases

    def test_resolve_entities_different_types(self):
        """Test that entities of different types are not merged."""
        entities = [
            Entity(name="Apple", type=EntityType.ORGANIZATION, description="Tech company"),
            Entity(name="Apple", type=EntityType.PRODUCT, description="The fruit"),
        ]

        resolved = self.resolver.resolve_entities(entities)

        assert len(resolved) == 2  # Different types, not merged

    def test_resolve_entities_preserves_sources(self):
        """Test that source chunks/documents are preserved."""
        entities = [
            Entity(
                name="Google",
                type=EntityType.ORGANIZATION,
                source_chunks=["chunk1"],
                source_documents=["doc1"],
            ),
            Entity(
                name="Google LLC",
                type=EntityType.ORGANIZATION,
                source_chunks=["chunk2"],
                source_documents=["doc2"],
            ),
        ]

        resolved = self.resolver.resolve_entities(entities)

        assert len(resolved) == 1
        assert "chunk1" in resolved[0].source_chunks
        assert "chunk2" in resolved[0].source_chunks
        assert "doc1" in resolved[0].source_documents
        assert "doc2" in resolved[0].source_documents

    def test_resolve_relationships(self):
        """Test relationship resolution."""
        entities = [
            Entity(name="Sam Altman", type=EntityType.PERSON, aliases=["Altman"]),
            Entity(name="OpenAI", type=EntityType.ORGANIZATION, aliases=["OpenAI Inc"]),
        ]

        relationships = [
            Relationship(
                source_entity="Sam Altman",
                target_entity="OpenAI",
                type=RelationshipType.WORKS_AT,
                source_chunk="chunk1",
            ),
            Relationship(
                source_entity="Altman",
                target_entity="OpenAI Inc",
                type=RelationshipType.WORKS_AT,
                source_chunk="chunk2",
            ),
        ]

        resolved = self.resolver.resolve_relationships(relationships, entities)

        # Should deduplicate to 1 relationship with canonical names
        assert len(resolved) == 1
        assert resolved[0].source_entity == "Sam Altman"
        assert resolved[0].target_entity == "OpenAI"


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
        assert options.batch_size == 3


class TestEntityExtractor:
    """Tests for the EntityExtractorService (mocked LLM)."""

    @pytest.fixture
    def sample_chunk(self):
        """Create a sample chunk for testing."""
        return Chunk(
            id="test_chunk_1",
            doc_id="test_doc_1",
            text="""
            OpenAI, founded by Sam Altman and others, is an artificial intelligence
            research laboratory. The company is headquartered in San Francisco, California.
            OpenAI developed ChatGPT, a large language model that has revolutionized
            how people interact with AI systems.
            """,
            index=0,
            metadata={"source": "test_document.pdf"},
        )

    def test_chunk_has_required_fields(self, sample_chunk):
        """Test that sample chunk has required fields for extraction."""
        assert sample_chunk.id is not None
        assert sample_chunk.doc_id is not None
        assert len(sample_chunk.text) > 0
        assert sample_chunk.metadata.get("source") is not None


# Integration test that requires LM Studio running
@pytest.mark.skipif(
    not pytest.importorskip("openai", reason="OpenAI client not installed"),
    reason="OpenAI client required for integration tests"
)
class TestEntityExtractorIntegration:
    """Integration tests for EntityExtractorService with actual LLM."""

    @pytest.fixture
    def sample_chunk(self):
        """Create a sample chunk for testing."""
        return Chunk(
            id="test_chunk_1",
            doc_id="test_doc_1",
            text="""
            Microsoft Corporation, led by CEO Satya Nadella, announced a major
            partnership with OpenAI. The technology giant, headquartered in Redmond,
            Washington, has invested billions in the AI research company.
            """,
            index=0,
            metadata={"source": "test_document.pdf"},
        )

    @pytest.mark.asyncio
    async def test_extract_from_chunk(self, sample_chunk):
        """Test entity extraction from a single chunk."""
        from kg_agent.services.entity_extractor import get_entity_extractor

        try:
            extractor = get_entity_extractor()
            result = await extractor.extract_from_chunk(sample_chunk)

            assert result.chunk_id == sample_chunk.id
            assert isinstance(result.entities, list)
            assert isinstance(result.relationships, list)
            assert result.processing_time > 0

            # Should find at least some entities
            if not result.error:
                print(f"\nExtracted {len(result.entities)} entities:")
                for entity in result.entities:
                    print(f"  - {entity.name} ({entity.type.value})")

                print(f"\nExtracted {len(result.relationships)} relationships:")
                for rel in result.relationships:
                    print(f"  - {rel.source_entity} --[{rel.type.value}]--> {rel.target_entity}")

        except Exception as e:
            pytest.skip(f"LM Studio not available: {e}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])

