"""Pipeline for agent-enhanced document reprocessing with entity extraction."""

import asyncio
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime

from ..core.logging import logger
from ..models.chunk import Chunk
from ..models.entity import (
    Entity,
    Relationship,
    ExtractionResult,
    ReprocessingStatus,
    ReprocessingOptions,
    ReprocessingResult,
)
from .document_tracker import get_document_tracker
from .vector_store import get_vector_store
from .graph_builder import get_graph_builder
from .entity_extractor import get_entity_extractor
from .entity_resolver import EntityResolver


class ReprocessingPipeline:
    """
    Pipeline for agent-enhanced document reprocessing.
    Extracts entities and relationships using LLM and updates the knowledge graph.
    """

    def __init__(self):
        """Initialize the reprocessing pipeline."""
        self.document_tracker = get_document_tracker()
        self.vector_store = get_vector_store()
        self.graph_builder = get_graph_builder()
        self.entity_extractor = get_entity_extractor()
        self.entity_resolver = EntityResolver()

        self._current_status: Dict[str, ReprocessingStatus] = {}
        self._progress: Dict[str, Dict[str, Any]] = {}

    def get_status(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current reprocessing status for a document.

        Args:
            doc_id: Document ID

        Returns:
            Status dict or None if not processing
        """
        if doc_id not in self._current_status:
            return None
        return {
            "status": self._current_status[doc_id].value,
            "progress": self._progress.get(doc_id, {}),
        }

    async def reprocess_document(
        self,
        doc_id: str,
        options: Optional[ReprocessingOptions] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> ReprocessingResult:
        """
        Reprocess a single document with enhanced entity extraction.

        Args:
            doc_id: Document ID to reprocess
            options: Reprocessing options
            progress_callback: Optional callback(stage, processed, total)

        Returns:
            ReprocessingResult with statistics
        """
        options = options or ReprocessingOptions()
        start_time = datetime.utcnow()

        self._current_status[doc_id] = ReprocessingStatus.PENDING
        self._progress[doc_id] = {"chunks_processed": 0, "total_chunks": 0}

        try:
            # 1. Load document and its chunks
            self._current_status[doc_id] = ReprocessingStatus.LOADING
            logger.info(f"Loading document {doc_id} for reprocessing...")

            doc = self.document_tracker.get_document(doc_id)
            if not doc:
                raise ValueError(f"Document {doc_id} not found")

            # Load chunks from ChromaDB
            vector_ids = doc.vector_ids
            if not vector_ids:
                raise ValueError(f"Document {doc_id} has no associated vectors")

            chunks_data = self.vector_store.get_by_ids(vector_ids)
            chunks = self._reconstruct_chunks(chunks_data, doc_id)

            if not chunks:
                raise ValueError(f"No chunks found for document {doc_id}")

            self._progress[doc_id]["total_chunks"] = len(chunks)
            logger.info(f"Loaded {len(chunks)} chunks for document {doc_id}")

            # 2. Extract entities and relationships
            self._current_status[doc_id] = ReprocessingStatus.EXTRACTING

            all_entities: List[Entity] = []
            all_relationships: List[Relationship] = []

            def extraction_progress(processed: int, total: int):
                self._progress[doc_id]["chunks_processed"] = processed
                if progress_callback:
                    progress_callback("extracting", processed, total)

            extraction_results = await self.entity_extractor.extract_from_chunks_batch(
                chunks,
                batch_size=options.batch_size,
                progress_callback=extraction_progress,
            )

            for result in extraction_results:
                all_entities.extend(result.entities)
                all_relationships.extend(result.relationships)

            logger.info(
                f"Extracted {len(all_entities)} entities, {len(all_relationships)} relationships from {len(chunks)} chunks"
            )

            # 3. Resolve/deduplicate entities
            self._current_status[doc_id] = ReprocessingStatus.RESOLVING

            resolved_entities = self.entity_resolver.resolve_entities(all_entities)
            resolved_relationships = self.entity_resolver.resolve_relationships(
                all_relationships, resolved_entities
            )

            logger.info(
                f"After deduplication: {len(resolved_entities)} entities, {len(resolved_relationships)} relationships"
            )

            # 4. Update knowledge graph
            self._current_status[doc_id] = ReprocessingStatus.UPDATING_GRAPH

            graph_result = await self._update_graph(
                doc_id=doc_id,
                entities=resolved_entities,
                relationships=resolved_relationships,
                options=options,
            )

            # 5. Update document tracker metadata
            current_metadata = doc.metadata or {}
            current_metadata.update({
                "reprocessed_at": datetime.utcnow().isoformat(),
                "entities_count": len(resolved_entities),
                "relationships_count": len(resolved_relationships),
                "reprocessing_options": options.model_dump(),
            })

            self.document_tracker.update_document(
                doc_id,
                metadata=current_metadata,
            )

            self._current_status[doc_id] = ReprocessingStatus.COMPLETED
            processing_time = (datetime.utcnow() - start_time).total_seconds()

            result = ReprocessingResult(
                doc_id=doc_id,
                status=ReprocessingStatus.COMPLETED,
                entities_extracted=len(all_entities),
                entities_after_dedup=len(resolved_entities),
                relationships_extracted=len(all_relationships),
                relationships_after_dedup=len(resolved_relationships),
                nodes_created=graph_result.get("nodes_created", 0),
                edges_created=graph_result.get("edges_created", 0),
                processing_time=processing_time,
            )

            logger.info(f"Reprocessing completed for document {doc_id}: {result.to_dict()}")
            return result

        except Exception as e:
            logger.error(f"Reprocessing failed for {doc_id}: {e}", exc_info=True)
            self._current_status[doc_id] = ReprocessingStatus.FAILED

            return ReprocessingResult(
                doc_id=doc_id,
                status=ReprocessingStatus.FAILED,
                error=str(e),
                processing_time=(datetime.utcnow() - start_time).total_seconds(),
            )

    async def reprocess_batch(
        self,
        doc_ids: List[str],
        options: Optional[ReprocessingOptions] = None,
    ) -> List[ReprocessingResult]:
        """
        Reprocess multiple documents sequentially.

        Args:
            doc_ids: List of document IDs to reprocess
            options: Reprocessing options

        Returns:
            List of ReprocessingResults
        """
        results = []
        total = len(doc_ids)

        logger.info(f"Starting batch reprocessing for {total} documents")

        for i, doc_id in enumerate(doc_ids):
            logger.info(f"Processing document {i + 1}/{total}: {doc_id}")
            result = await self.reprocess_document(doc_id, options)
            results.append(result)

        # Summary
        successful = sum(1 for r in results if r.status == ReprocessingStatus.COMPLETED)
        failed = sum(1 for r in results if r.status == ReprocessingStatus.FAILED)
        total_entities = sum(r.entities_after_dedup for r in results)
        total_relationships = sum(r.relationships_after_dedup for r in results)

        logger.info(
            f"Batch reprocessing complete: {successful} successful, {failed} failed, "
            f"{total_entities} total entities, {total_relationships} total relationships"
        )

        return results

    def _reconstruct_chunks(self, chunks_data: Dict, doc_id: str) -> List[Chunk]:
        """
        Reconstruct Chunk objects from ChromaDB data.

        Args:
            chunks_data: Data from ChromaDB get_by_ids
            doc_id: Document ID

        Returns:
            List of Chunk objects
        """
        chunks = []
        ids = chunks_data.get("ids", [])
        documents = chunks_data.get("documents", [])
        metadatas = chunks_data.get("metadatas", [])

        for i, chunk_id in enumerate(ids):
            text = documents[i] if i < len(documents) else ""
            metadata = metadatas[i] if i < len(metadatas) else {}

            # Skip empty chunks
            if not text or not text.strip():
                continue

            chunk = Chunk(
                id=chunk_id,
                doc_id=doc_id,
                text=text,
                index=i,
                metadata=metadata,
            )
            chunks.append(chunk)

        return chunks

    async def _update_graph(
        self,
        doc_id: str,
        entities: List[Entity],
        relationships: List[Relationship],
        options: ReprocessingOptions,
    ) -> Dict[str, Any]:
        """
        Update FalkorDB graph with extracted entities and relationships.

        Args:
            doc_id: Document ID being processed
            entities: Resolved entities
            relationships: Resolved relationships
            options: Reprocessing options

        Returns:
            Dict with nodes_created and edges_created counts
        """
        await self.graph_builder.initialize()

        if not self.graph_builder.driver:
            logger.warning("FalkorDB not available, skipping graph update")
            return {"nodes_created": 0, "edges_created": 0}

        nodes_created = 0
        edges_created = 0

        try:
            with self.graph_builder.driver.session() as session:
                # Create/update Entity nodes
                for entity in entities:
                    try:
                        result = session.run(
                            """
                            MERGE (e:Entity {name: $name})
                            SET e.type = $type,
                                e.description = $description,
                                e.aliases = $aliases,
                                e.confidence = $confidence,
                                e.updated_at = datetime()
                            RETURN e
                            """,
                            {
                                "name": entity.name,
                                "type": entity.type.value,
                                "description": entity.description or "",
                                "aliases": entity.aliases,
                                "confidence": entity.confidence,
                            },
                        )
                        if result.single():
                            nodes_created += 1

                        # Link entity to source documents
                        for source_doc in entity.source_documents:
                            session.run(
                                """
                                MATCH (e:Entity {name: $entity_name})
                                MATCH (d:Document {doc_id: $doc_id})
                                MERGE (d)-[:MENTIONS]->(e)
                                """,
                                {"entity_name": entity.name, "doc_id": source_doc},
                            )

                    except Exception as e:
                        logger.warning(f"Failed to create entity node for {entity.name}: {e}")

                # Create relationships between entities
                for rel in relationships:
                    try:
                        # Use dynamic relationship type
                        result = session.run(
                            """
                            MATCH (source:Entity {name: $source})
                            MATCH (target:Entity {name: $target})
                            MERGE (source)-[r:RELATES_TO {type: $rel_type}]->(target)
                            SET r.description = $description,
                                r.strength = $strength,
                                r.updated_at = datetime()
                            RETURN r
                            """,
                            {
                                "source": rel.source_entity,
                                "target": rel.target_entity,
                                "rel_type": rel.type.value,
                                "description": rel.description or "",
                                "strength": rel.strength,
                            },
                        )
                        if result.single():
                            edges_created += 1

                    except Exception as e:
                        logger.warning(
                            f"Failed to create relationship {rel.source_entity} -> {rel.target_entity}: {e}"
                        )

            logger.info(f"Graph update complete: {nodes_created} nodes, {edges_created} edges")

        except Exception as e:
            logger.error(f"Graph update failed: {e}")

        return {"nodes_created": nodes_created, "edges_created": edges_created}

    async def get_document_entities(self, doc_id: str) -> List[Dict[str, Any]]:
        """
        Get all entities associated with a document from FalkorDB.

        Args:
            doc_id: Document ID

        Returns:
            List of entity dictionaries
        """
        await self.graph_builder.initialize()

        if not self.graph_builder.driver:
            return []

        try:
            with self.graph_builder.driver.session() as session:
                result = session.run(
                    """
                    MATCH (d:Document {doc_id: $doc_id})-[:MENTIONS]->(e:Entity)
                    RETURN e.name as name, e.type as type, e.description as description,
                           e.aliases as aliases, e.confidence as confidence
                    """,
                    {"doc_id": doc_id},
                )

                entities = []
                for record in result:
                    entities.append({
                        "name": record["name"],
                        "type": record["type"],
                        "description": record["description"],
                        "aliases": record["aliases"],
                        "confidence": record["confidence"],
                    })

                return entities

        except Exception as e:
            logger.error(f"Failed to get entities for document {doc_id}: {e}")
            return []

    async def get_entity_relationships(self, entity_name: str) -> List[Dict[str, Any]]:
        """
        Get all relationships for an entity.

        Args:
            entity_name: Entity name

        Returns:
            List of relationship dictionaries
        """
        await self.graph_builder.initialize()

        if not self.graph_builder.driver:
            return []

        try:
            with self.graph_builder.driver.session() as session:
                result = session.run(
                    """
                    MATCH (e:Entity {name: $name})-[r:RELATES_TO]->(target:Entity)
                    RETURN target.name as target, r.type as type, r.description as description
                    UNION
                    MATCH (source:Entity)-[r:RELATES_TO]->(e:Entity {name: $name})
                    RETURN source.name as target, r.type as type, r.description as description
                    """,
                    {"name": entity_name},
                )

                relationships = []
                for record in result:
                    relationships.append({
                        "related_entity": record["target"],
                        "type": record["type"],
                        "description": record["description"],
                    })

                return relationships

        except Exception as e:
            logger.error(f"Failed to get relationships for entity {entity_name}: {e}")
            return []


# Singleton instance
_pipeline_instance: Optional[ReprocessingPipeline] = None


def get_reprocessing_pipeline() -> ReprocessingPipeline:
    """Get or create the singleton ReprocessingPipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = ReprocessingPipeline()
    return _pipeline_instance

