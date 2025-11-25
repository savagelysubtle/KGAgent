from typing import Dict, Any, Optional, List
from datetime import datetime
from pydantic import BaseModel, Field

class Chunk(BaseModel):
    """
    Represents a semantic chunk of text from a document.
    """
    id: str = Field(..., description="Unique identifier for the chunk")
    doc_id: str = Field(..., description="ID of the source document")
    text: str = Field(..., description="The text content of the chunk")
    index: int = Field(..., description="Order index of the chunk in the document")

    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata including source, titles, etc.")

    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Embeddings will be added in the next stage, but we can define the field
    embedding: Optional[List[float]] = Field(default=None, description="Vector embedding of the text")

class ChunkBatch(BaseModel):
    """Batch of chunks."""
    chunks: List[Chunk]
    job_id: str
    processed_at: datetime = Field(default_factory=datetime.utcnow)

