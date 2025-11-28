"""Entity and Relationship models for knowledge graph extraction."""

import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class EntityType(str, Enum):
    """Types of entities that can be extracted."""
    PERSON = "Person"
    ORGANIZATION = "Organization"
    LOCATION = "Location"
    CONCEPT = "Concept"
    EVENT = "Event"
    TECHNOLOGY = "Technology"
    PRODUCT = "Product"


class RelationshipType(str, Enum):
    """Types of relationships between entities."""
    WORKS_AT = "WORKS_AT"
    LOCATED_IN = "LOCATED_IN"
    FOUNDED = "FOUNDED"
    CREATED = "CREATED"
    RELATED_TO = "RELATED_TO"
    PART_OF = "PART_OF"
    CAUSED_BY = "CAUSED_BY"
    SIMILAR_TO = "SIMILAR_TO"
    MENTIONS = "MENTIONS"
    COLLABORATES_WITH = "COLLABORATES_WITH"
    USES = "USES"
    BELONGS_TO = "BELONGS_TO"
    MANAGES = "MANAGES"
    PRODUCES = "PRODUCES"


class Entity(BaseModel):
    """Extracted entity from document."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="The canonical name of the entity")
    type: EntityType = Field(..., description="The type of entity")
    description: Optional[str] = Field(None, description="Brief description based on context")
    aliases: List[str] = Field(default_factory=list, description="Alternative names or abbreviations")
    source_chunks: List[str] = Field(default_factory=list, description="Chunk IDs where entity was found")
    source_documents: List[str] = Field(default_factory=list, description="Document IDs where entity was found")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction confidence score")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value,
            "description": self.description,
            "aliases": self.aliases,
            "source_chunks": self.source_chunks,
            "source_documents": self.source_documents,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }


class Relationship(BaseModel):
    """Relationship between two entities."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_entity: str = Field(..., description="Source entity name")
    target_entity: str = Field(..., description="Target entity name")
    type: RelationshipType = Field(..., description="Type of relationship")
    description: Optional[str] = Field(None, description="Description of the relationship")
    strength: float = Field(default=1.0, ge=0.0, le=1.0, description="Relationship strength/confidence")
    source_chunk: str = Field(..., description="Chunk ID where relationship was extracted")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "source_entity": self.source_entity,
            "target_entity": self.target_entity,
            "type": self.type.value,
            "description": self.description,
            "strength": self.strength,
            "source_chunk": self.source_chunk,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }


class ExtractionResult(BaseModel):
    """Result of entity extraction for a chunk."""
    chunk_id: str = Field(..., description="ID of the processed chunk")
    entities: List[Entity] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)
    processing_time: float = Field(default=0.0, description="Time taken in seconds")
    token_count: Optional[int] = Field(None, description="Tokens used for extraction")
    error: Optional[str] = Field(None, description="Error message if extraction failed")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "chunk_id": self.chunk_id,
            "entities": [e.to_dict() for e in self.entities],
            "relationships": [r.to_dict() for r in self.relationships],
            "processing_time": self.processing_time,
            "token_count": self.token_count,
            "error": self.error,
        }


class ReprocessingStatus(str, Enum):
    """Status of a reprocessing job."""
    PENDING = "pending"
    LOADING = "loading"
    EXTRACTING = "extracting"
    RESOLVING = "resolving"
    UPDATING_GRAPH = "updating_graph"
    COMPLETED = "completed"
    FAILED = "failed"


class ReprocessingOptions(BaseModel):
    """Options for document reprocessing."""
    extract_entities: bool = Field(default=True, description="Extract named entities")
    extract_relationships: bool = Field(default=True, description="Extract relationships between entities")
    update_existing_graph: bool = Field(default=True, description="Update existing graph vs rebuild")
    batch_size: int = Field(default=10, ge=1, le=20, description="Chunks per LLM call for extraction")
    include_chunk_nodes: bool = Field(default=True, description="Keep original Document chunk nodes")


class ReprocessingResult(BaseModel):
    """Result of a reprocessing job."""
    doc_id: str
    status: ReprocessingStatus
    entities_extracted: int = 0
    entities_after_dedup: int = 0
    relationships_extracted: int = 0
    relationships_after_dedup: int = 0
    nodes_created: int = 0
    edges_created: int = 0
    processing_time: float = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "doc_id": self.doc_id,
            "status": self.status.value,
            "entities_extracted": self.entities_extracted,
            "entities_after_dedup": self.entities_after_dedup,
            "relationships_extracted": self.relationships_extracted,
            "relationships_after_dedup": self.relationships_after_dedup,
            "nodes_created": self.nodes_created,
            "edges_created": self.edges_created,
            "processing_time": self.processing_time,
            "error": self.error,
        }

