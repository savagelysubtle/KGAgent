"""LLM-powered entity and relationship extraction service."""

import json
import asyncio
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime

from openai import OpenAI

from ..core.config import settings
from ..core.logging import logger
from ..models.chunk import Chunk
from ..models.entity import (
    Entity,
    Relationship,
    ExtractionResult,
    EntityType,
    RelationshipType,
)


ENTITY_EXTRACTION_PROMPT = """You are an expert knowledge graph builder specializing in extracting structured information from text.

Your task is to analyze the provided text and extract:

## ENTITIES
Extract all significant named entities. For each entity provide:
- name: The canonical name (properly capitalized)
- type: One of [Person, Organization, Location, Concept, Event, Technology, Product]
- description: A brief 1-2 sentence description based on the context
- aliases: Any alternative names or abbreviations mentioned

Focus on entities that are:
- Specifically named (not generic references)
- Important to understanding the text
- Could have relationships with other entities

## RELATIONSHIPS
Identify meaningful relationships between the extracted entities. For each relationship:
- source: The source entity name (must match an extracted entity)
- target: The target entity name (must match an extracted entity)
- type: One of [WORKS_AT, LOCATED_IN, FOUNDED, CREATED, RELATED_TO, PART_OF, CAUSED_BY, SIMILAR_TO, COLLABORATES_WITH, USES, BELONGS_TO, MANAGES, PRODUCES]
- description: Brief description of how they're related

Only include relationships that are explicitly stated or strongly implied in the text.

## OUTPUT FORMAT
Return valid JSON only, no markdown code blocks:
{
  "entities": [
    {
      "name": "Entity Name",
      "type": "EntityType",
      "description": "Brief description",
      "aliases": ["alias1", "alias2"]
    }
  ],
  "relationships": [
    {
      "source": "Source Entity Name",
      "target": "Target Entity Name",
      "type": "RELATIONSHIP_TYPE",
      "description": "How they relate"
    }
  ]
}

If no entities or relationships are found, return empty arrays."""


class EntityExtractorService:
    """LLM-powered entity and relationship extraction service."""

    def __init__(self):
        """Initialize the entity extractor with LM Studio client."""
        self.client = OpenAI(
            base_url=settings.LLM_BASE_URL,
            api_key=settings.LLM_API_KEY,
        )
        self.model = settings.LLM_MODEL_NAME
        logger.info(f"EntityExtractorService initialized with model: {self.model}")

    async def extract_from_chunk(self, chunk: Chunk) -> ExtractionResult:
        """
        Extract entities and relationships from a single chunk.

        Args:
            chunk: The document chunk to process

        Returns:
            ExtractionResult with entities and relationships
        """
        start_time = datetime.utcnow()

        try:
            # Build the user prompt with chunk context
            user_prompt = f"""Document Source: {chunk.metadata.get('source', 'Unknown')}
Chunk Index: {chunk.index}

TEXT TO ANALYZE:
---
{chunk.text}
---

Extract all entities and relationships from this text. Return only valid JSON."""

            # Make synchronous call in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": ENTITY_EXTRACTION_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.3,  # Lower temperature for more consistent extraction
                    max_tokens=8000,  # Model supports 50K context, allow large entity extractions
                ),
            )

            response_text = response.choices[0].message.content
            token_count = response.usage.total_tokens if response.usage else None

            # Parse JSON response
            data = self._parse_llm_response(response_text)

            # Convert to Entity objects
            entities = self._parse_entities(data.get("entities", []), chunk)

            # Convert to Relationship objects
            relationships = self._parse_relationships(
                data.get("relationships", []), chunk
            )

            processing_time = (datetime.utcnow() - start_time).total_seconds()

            logger.debug(
                f"Extracted {len(entities)} entities, {len(relationships)} relationships from chunk {chunk.id}"
            )

            return ExtractionResult(
                chunk_id=chunk.id,
                entities=entities,
                relationships=relationships,
                processing_time=processing_time,
                token_count=token_count,
            )

        except Exception as e:
            logger.error(f"Entity extraction failed for chunk {chunk.id}: {e}")
            return ExtractionResult(
                chunk_id=chunk.id,
                entities=[],
                relationships=[],
                processing_time=(datetime.utcnow() - start_time).total_seconds(),
                error=str(e),
            )

    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response text into JSON, with recovery for truncated responses."""
        try:
            # Handle potential markdown code blocks
            text = response_text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                parts = text.split("```")
                if len(parts) >= 2:
                    text = parts[1]

            # Try to find JSON object in the text
            text = text.strip()
            if not text.startswith("{"):
                # Try to find the start of JSON
                start_idx = text.find("{")
                if start_idx != -1:
                    text = text[start_idx:]

            return json.loads(text)

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            # Try to salvage truncated JSON by extracting entities array
            try:
                # Find the entities array and try to parse what we can
                if '"entities"' in response_text:
                    # Extract entities section
                    entities_start = response_text.find('"entities"')
                    entities_array_start = response_text.find('[', entities_start)
                    if entities_array_start != -1:
                        # Find matching bracket or truncation point
                        bracket_count = 0
                        last_complete_entity = entities_array_start
                        for i, char in enumerate(response_text[entities_array_start:], entities_array_start):
                            if char == '[':
                                bracket_count += 1
                            elif char == ']':
                                bracket_count -= 1
                                if bracket_count == 0:
                                    # Found complete array
                                    entities_json = response_text[entities_array_start:i+1]
                                    entities = json.loads(entities_json)
                                    logger.info(f"Recovered {len(entities)} entities from truncated response")
                                    return {"entities": entities, "relationships": []}
                            elif char == '}' and bracket_count == 1:
                                # End of an entity object
                                last_complete_entity = i + 1

                        # If we got here, array was truncated - try to salvage complete entities
                        if last_complete_entity > entities_array_start + 1:
                            partial = response_text[entities_array_start:last_complete_entity] + ']'
                            try:
                                entities = json.loads(partial)
                                logger.info(f"Salvaged {len(entities)} entities from truncated response")
                                return {"entities": entities, "relationships": []}
                            except json.JSONDecodeError:
                                pass
            except Exception as recovery_error:
                logger.debug(f"Recovery attempt failed: {recovery_error}")

            logger.debug(f"Response was: {response_text[:500]}")
            return {"entities": [], "relationships": []}

    def _parse_entities(
        self, entity_data: List[Dict], chunk: Chunk
    ) -> List[Entity]:
        """Parse entity dictionaries into Entity objects."""
        entities = []
        for e in entity_data:
            try:
                # Map type string to enum
                type_str = e.get("type", "Concept")
                try:
                    entity_type = EntityType(type_str)
                except ValueError:
                    # Try to match case-insensitively
                    type_lower = type_str.lower()
                    entity_type = next(
                        (t for t in EntityType if t.value.lower() == type_lower),
                        EntityType.CONCEPT,
                    )

                entity = Entity(
                    name=e.get("name", "Unknown").strip(),
                    type=entity_type,
                    description=e.get("description"),
                    aliases=e.get("aliases", []),
                    source_chunks=[chunk.id],
                    source_documents=[chunk.doc_id],
                )
                entities.append(entity)
            except Exception as ex:
                logger.warning(f"Failed to create entity from {e}: {ex}")

        return entities

    def _parse_relationships(
        self, rel_data: List[Dict], chunk: Chunk
    ) -> List[Relationship]:
        """Parse relationship dictionaries into Relationship objects."""
        relationships = []
        for r in rel_data:
            try:
                # Map type string to enum
                type_str = r.get("type", "RELATED_TO")
                try:
                    rel_type = RelationshipType(type_str)
                except ValueError:
                    # Try to match case-insensitively
                    type_upper = type_str.upper().replace(" ", "_")
                    rel_type = next(
                        (t for t in RelationshipType if t.value == type_upper),
                        RelationshipType.RELATED_TO,
                    )

                rel = Relationship(
                    source_entity=r.get("source", "").strip(),
                    target_entity=r.get("target", "").strip(),
                    type=rel_type,
                    description=r.get("description"),
                    source_chunk=chunk.id,
                )

                # Only add if both source and target are non-empty
                if rel.source_entity and rel.target_entity:
                    relationships.append(rel)

            except Exception as ex:
                logger.warning(f"Failed to create relationship from {r}: {ex}")

        return relationships

    async def extract_from_chunks_batch(
        self,
        chunks: List[Chunk],
        batch_size: int = 3,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[ExtractionResult]:
        """
        Extract entities from multiple chunks with batching.

        Args:
            chunks: List of chunks to process
            batch_size: Number of concurrent extractions
            progress_callback: Optional callback(processed, total) for progress updates

        Returns:
            List of ExtractionResults
        """
        results = []
        total = len(chunks)

        logger.info(f"Starting batch extraction for {total} chunks (batch_size={batch_size})")

        for i in range(0, total, batch_size):
            batch = chunks[i : i + batch_size]

            # Process batch concurrently
            batch_tasks = [self.extract_from_chunk(chunk) for chunk in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch extraction error: {result}")
                    # Create empty result for failed extraction
                    results.append(
                        ExtractionResult(
                            chunk_id="unknown",
                            entities=[],
                            relationships=[],
                            error=str(result),
                        )
                    )
                else:
                    results.append(result)

            processed = min(i + batch_size, total)
            if progress_callback:
                progress_callback(processed, total)

            logger.info(f"Processed {processed}/{total} chunks")

        return results


# Singleton instance
_extractor_instance: Optional[EntityExtractorService] = None


def get_entity_extractor() -> EntityExtractorService:
    """Get or create the singleton EntityExtractorService instance."""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = EntityExtractorService()
    return _extractor_instance

