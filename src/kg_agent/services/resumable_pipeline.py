"""
Resumable pipeline for entity extraction that supports pause/resume.

This pipeline persists progress to SQLite so processing can be paused
and resumed even after application shutdown.
"""

import asyncio
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime

from ..core.logging import logger
from ..models.chunk import Chunk
from ..models.entity import (
    Entity,
    Relationship,
    ExtractionResult,
    ReprocessingOptions,
)
from .document_tracker import get_document_tracker
from .vector_store import get_vector_store
from .graph_builder import get_graph_builder
from .entity_extractor import get_entity_extractor
from .entity_resolver import EntityResolver
from .graphiti_service import get_graphiti_service
from .processing_job_tracker import (
    get_job_tracker,
    ProcessingJob,
    JobStatus,
    JobType,
)

# How often to commit entities to the graph database (every N chunks)
INCREMENTAL_COMMIT_INTERVAL = 20


class ResumablePipeline:
    """
    Resumable entity extraction pipeline.

    Features:
    - Pause and resume processing at any time
    - Persists progress to SQLite
    - Can resume after application shutdown
    - Tracks processed chunks to avoid re-processing
    """

    def __init__(self):
        """Initialize the resumable pipeline."""
        self.document_tracker = get_document_tracker()
        self.vector_store = get_vector_store()
        self.graph_builder = get_graph_builder()
        self.entity_extractor = get_entity_extractor()
        self.entity_resolver = EntityResolver()
        self.job_tracker = get_job_tracker()
        self.graphiti_service = get_graphiti_service()

        # In-memory state for current processing
        self._should_pause: Dict[str, bool] = {}
        self._is_running: Dict[str, bool] = {}

        # Track chunks since last graph commit for incremental updates
        self._chunks_since_commit: Dict[str, int] = {}
        self._pending_entities: Dict[str, List[Entity]] = {}
        self._pending_relationships: Dict[str, List[Relationship]] = {}

    # ==================== Job Control ====================

    def request_pause(self, job_id: str):
        """Request a running job to pause at the next safe point."""
        self._should_pause[job_id] = True
        logger.info(f"Pause requested for job {job_id}")

    def is_paused(self, job_id: str) -> bool:
        """Check if a job is paused."""
        job = self.job_tracker.get_job(job_id)
        return job and job.status == JobStatus.PAUSED.value

    def is_running(self, job_id: str) -> bool:
        """Check if a job is currently running."""
        return self._is_running.get(job_id, False)

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed status for a job."""
        job = self.job_tracker.get_job(job_id)
        if not job:
            return None

        return {
            "job_id": job.id,
            "doc_id": job.doc_id,
            "status": job.status,
            "progress_percent": job.progress_percent,
            "processed_chunks": job.processed_chunks,
            "total_chunks": job.total_chunks,
            "entities_extracted": job.entities_extracted,
            "relationships_extracted": job.relationships_extracted,
            "started_at": job.started_at,
            "paused_at": job.paused_at,
            "can_resume": job.status in [JobStatus.PAUSED.value, JobStatus.PENDING.value],
        }

    # ==================== Main Processing ====================

    async def start_or_resume_extraction(
        self,
        doc_id: str,
        options: Optional[ReprocessingOptions] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Start a new extraction job or resume an existing one.

        Args:
            doc_id: Document ID to process
            options: Processing options
            progress_callback: Called with progress updates

        Returns:
            Job result dictionary
        """
        options = options or ReprocessingOptions()

        # Check for existing resumable job
        existing_job = self.job_tracker.get_active_job_for_document(doc_id)

        if existing_job:
            if existing_job.status == JobStatus.RUNNING.value:
                return {
                    "success": False,
                    "error": "A job is already running for this document",
                    "job_id": existing_job.id,
                }

            # Resume existing job
            logger.info(f"Resuming existing job {existing_job.id} for document {doc_id}")
            return await self._run_extraction(
                existing_job.id,
                doc_id,
                options,
                progress_callback,
                resume_from=existing_job.current_chunk_index,
            )

        # Start new job
        return await self._start_new_extraction(doc_id, options, progress_callback)

    async def _start_new_extraction(
        self,
        doc_id: str,
        options: ReprocessingOptions,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Start a new extraction job."""
        # Load document and get chunk count
        doc = self.document_tracker.get_document(doc_id)
        if not doc:
            return {"success": False, "error": f"Document {doc_id} not found"}

        vector_ids = doc.vector_ids
        if not vector_ids:
            return {"success": False, "error": f"Document {doc_id} has no vectors"}

        # Create job
        job = self.job_tracker.create_job(
            doc_id=doc_id,
            job_type=JobType.ENTITY_EXTRACTION,
            total_chunks=len(vector_ids),
            options=options.model_dump(),
        )

        logger.info(f"Created new extraction job {job.id} for document {doc_id} with {len(vector_ids)} chunks")

        return await self._run_extraction(
            job.id,
            doc_id,
            options,
            progress_callback,
            resume_from=0,
        )

    async def _run_extraction(
        self,
        job_id: str,
        doc_id: str,
        options: ReprocessingOptions,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        resume_from: int = 0,
    ) -> Dict[str, Any]:
        """Run the extraction process."""
        self._should_pause[job_id] = False
        self._is_running[job_id] = True

        try:
            # Start/resume job
            job = self.job_tracker.start_job(job_id)
            if not job:
                return {"success": False, "error": "Failed to start job"}

            # Load document
            doc = self.document_tracker.get_document(doc_id)
            vector_ids = doc.vector_ids

            # Load chunks
            chunks_data = self.vector_store.get_by_ids(vector_ids)
            all_chunks = self._reconstruct_chunks(chunks_data, doc_id)

            if not all_chunks:
                self.job_tracker.fail_job(job_id, "No chunks found")
                return {"success": False, "error": "No chunks found"}

            # Get already processed chunks
            processed_chunk_ids = self.job_tracker.get_processed_chunk_ids(job_id)

            # Filter to unprocessed chunks starting from resume point
            chunks_to_process = [
                c for c in all_chunks
                if c.id not in processed_chunk_ids and c.index >= resume_from
            ]

            logger.info(
                f"Job {job_id}: Processing {len(chunks_to_process)} chunks "
                f"(skipping {len(processed_chunk_ids)} already processed)"
            )

            # Process chunks in batches
            batch_size = options.batch_size
            total_to_process = len(chunks_to_process)
            processed_count = len(processed_chunk_ids)
            total_entities = job.entities_extracted
            total_relationships = job.relationships_extracted

            for i in range(0, total_to_process, batch_size):
                # Check for pause request
                if self._should_pause.get(job_id, False):
                    logger.info(f"Pausing job {job_id} at chunk index {i + resume_from}")
                    self.job_tracker.pause_job(job_id)
                    self.job_tracker.update_progress(
                        job_id,
                        current_chunk_index=i + resume_from,
                    )
                    self._is_running[job_id] = False

                    return {
                        "success": True,
                        "status": "paused",
                        "job_id": job_id,
                        "processed_chunks": processed_count,
                        "total_chunks": len(all_chunks),
                        "entities_extracted": total_entities,
                        "relationships_extracted": total_relationships,
                        "message": f"Job paused. Processed {processed_count}/{len(all_chunks)} chunks.",
                    }

                # Get batch
                batch = chunks_to_process[i:i + batch_size]

                # Process batch
                for chunk in batch:
                    try:
                        result = await self.entity_extractor.extract_from_chunk(chunk)

                        # Store extracted data in job tracker (for persistence)
                        for entity in result.entities:
                            self.job_tracker.store_extracted_entity(
                                job_id, chunk.id, entity.model_dump()
                            )
                            total_entities += 1
                            # Add to pending for incremental commit
                            if job_id not in self._pending_entities:
                                self._pending_entities[job_id] = []
                            self._pending_entities[job_id].append(entity)

                        for rel in result.relationships:
                            self.job_tracker.store_extracted_relationship(
                                job_id, chunk.id, rel.model_dump()
                            )
                            total_relationships += 1
                            # Add to pending for incremental commit
                            if job_id not in self._pending_relationships:
                                self._pending_relationships[job_id] = []
                            self._pending_relationships[job_id].append(rel)

                        # Record chunk as processed
                        self.job_tracker.record_processed_chunk(
                            job_id,
                            chunk.id,
                            chunk.index,
                            entities_found=len(result.entities),
                            relationships_found=len(result.relationships),
                        )

                        processed_count += 1

                        # Track chunks since last commit
                        if job_id not in self._chunks_since_commit:
                            self._chunks_since_commit[job_id] = 0
                        self._chunks_since_commit[job_id] += 1

                        # Incremental commit every N chunks
                        if self._chunks_since_commit[job_id] >= INCREMENTAL_COMMIT_INTERVAL:
                            await self._incremental_commit_to_graphiti(job_id, doc_id)
                            self._chunks_since_commit[job_id] = 0

                    except Exception as e:
                        logger.warning(f"Failed to process chunk {chunk.id}: {e}")

                # Update progress
                self.job_tracker.update_progress(
                    job_id,
                    processed_chunks=processed_count,
                    current_chunk_index=i + batch_size + resume_from,
                    entities_extracted=total_entities,
                    relationships_extracted=total_relationships,
                )

                # Progress callback
                if progress_callback:
                    progress_callback({
                        "job_id": job_id,
                        "status": "running",
                        "processed_chunks": processed_count,
                        "total_chunks": len(all_chunks),
                        "progress_percent": (processed_count / len(all_chunks)) * 100,
                        "entities_extracted": total_entities,
                        "relationships_extracted": total_relationships,
                    })

                # Small delay between batches
                await asyncio.sleep(0.1)

            # All chunks processed - commit any remaining pending entities
            logger.info(f"Job {job_id}: Extraction complete. Committing remaining entities...")

            # Final incremental commit for any remaining entities
            final_commit = await self._incremental_commit_to_graphiti(job_id, doc_id)

            # Clean up tracking state
            self._chunks_since_commit.pop(job_id, None)
            self._pending_entities.pop(job_id, None)
            self._pending_relationships.pop(job_id, None)

            # Get all extracted entities and relationships for stats
            all_entity_data = self.job_tracker.get_extracted_entities(job_id)
            all_rel_data = self.job_tracker.get_extracted_relationships(job_id)

            # Reconstruct Entity and Relationship objects for dedup stats
            from ..models.entity import Entity, Relationship, EntityType, RelationshipType

            entities = []
            for data in all_entity_data:
                try:
                    entities.append(Entity(**data))
                except Exception as e:
                    logger.warning(f"Failed to reconstruct entity: {e}")

            relationships = []
            for data in all_rel_data:
                try:
                    relationships.append(Relationship(**data))
                except Exception as e:
                    logger.warning(f"Failed to reconstruct relationship: {e}")

            # Resolve/deduplicate for stats (Graphiti handles actual dedup)
            resolved_entities = self.entity_resolver.resolve_entities(entities)
            resolved_relationships = self.entity_resolver.resolve_relationships(
                relationships, resolved_entities
            )

            logger.info(
                f"Job {job_id}: Resolved to {len(resolved_entities)} entities, "
                f"{len(resolved_relationships)} relationships"
            )

            # Get final graph stats from Graphiti
            graph_stats = await self.graphiti_service.get_stats()
            graph_result = {
                "nodes_created": graph_stats.get("total_entities", 0),
                "edges_created": graph_stats.get("total_relationships", 0),
            }

            # Mark as inserted
            self.job_tracker.mark_entities_inserted(job_id)
            self.job_tracker.mark_relationships_inserted(job_id)

            # Complete job
            self.job_tracker.complete_job(job_id)

            # Update document metadata
            doc_metadata = doc.metadata or {}
            doc_metadata.update({
                "reprocessed_at": datetime.utcnow().isoformat(),
                "entities_count": len(resolved_entities),
                "relationships_count": len(resolved_relationships),
            })
            self.document_tracker.update_document(doc_id, metadata=doc_metadata)

            self._is_running[job_id] = False

            return {
                "success": True,
                "status": "completed",
                "job_id": job_id,
                "processed_chunks": processed_count,
                "total_chunks": len(all_chunks),
                "entities_extracted": total_entities,
                "entities_after_dedup": len(resolved_entities),
                "relationships_extracted": total_relationships,
                "relationships_after_dedup": len(resolved_relationships),
                "nodes_created": graph_result.get("nodes_created", 0),
                "edges_created": graph_result.get("edges_created", 0),
            }

        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}", exc_info=True)
            self.job_tracker.fail_job(job_id, str(e))
            self._is_running[job_id] = False

            return {
                "success": False,
                "status": "failed",
                "job_id": job_id,
                "error": str(e),
            }

    async def pause_extraction(self, job_id: str) -> Dict[str, Any]:
        """
        Request a job to pause.

        The job will pause at the next safe point (after current batch).
        """
        job = self.job_tracker.get_job(job_id)
        if not job:
            return {"success": False, "error": "Job not found"}

        if job.status != JobStatus.RUNNING.value:
            return {"success": False, "error": f"Job is not running (status: {job.status})"}

        self.request_pause(job_id)

        return {
            "success": True,
            "message": "Pause requested. Job will pause after current batch.",
            "job_id": job_id,
        }

    async def resume_extraction(
        self,
        job_id: str,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Resume a paused job."""
        job = self.job_tracker.get_job(job_id)
        if not job:
            return {"success": False, "error": "Job not found"}

        if job.status not in [JobStatus.PAUSED.value, JobStatus.PENDING.value]:
            return {"success": False, "error": f"Job cannot be resumed (status: {job.status})"}

        options = ReprocessingOptions(**job.options)

        return await self._run_extraction(
            job_id,
            job.doc_id,
            options,
            progress_callback,
            resume_from=job.current_chunk_index,
        )

    async def cancel_extraction(self, job_id: str) -> Dict[str, Any]:
        """Cancel a job."""
        job = self.job_tracker.get_job(job_id)
        if not job:
            return {"success": False, "error": "Job not found"}

        # Request pause if running
        if job.status == JobStatus.RUNNING.value:
            self.request_pause(job_id)
            # Wait a bit for it to pause
            await asyncio.sleep(1)

        self.job_tracker.cancel_job(job_id)

        return {
            "success": True,
            "message": "Job cancelled",
            "job_id": job_id,
        }

    # ==================== Helper Methods ====================

    def _reconstruct_chunks(self, chunks_data: Dict, doc_id: str) -> List[Chunk]:
        """Reconstruct Chunk objects from ChromaDB data."""
        chunks = []
        ids = chunks_data.get("ids", [])
        documents = chunks_data.get("documents", [])
        metadatas = chunks_data.get("metadatas", [])

        for i, chunk_id in enumerate(ids):
            text = documents[i] if i < len(documents) else ""
            metadata = metadatas[i] if i < len(metadatas) else {}

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

    async def _incremental_commit_to_graphiti(self, job_id: str, doc_id: str) -> Dict[str, Any]:
        """
        Commit pending entities and relationships to Graphiti/FalkorDB.

        This allows real-time graph updates during processing instead of
        waiting until the job completes.
        """
        pending_entities = self._pending_entities.get(job_id, [])
        pending_relationships = self._pending_relationships.get(job_id, [])

        if not pending_entities and not pending_relationships:
            return {"nodes_created": 0, "edges_created": 0}

        logger.info(
            f"Job {job_id}: Incremental commit - {len(pending_entities)} entities, "
            f"{len(pending_relationships)} relationships"
        )

        try:
            # Initialize Graphiti if needed
            if not self.graphiti_service._initialized:
                await self.graphiti_service.initialize()

            if not self.graphiti_service._graphiti:
                logger.warning("Graphiti not available, skipping incremental commit")
                return {"nodes_created": 0, "edges_created": 0}

            # Build episode content from entities and relationships
            episode_parts = []

            for entity in pending_entities:
                desc = f" - {entity.description}" if entity.description else ""
                episode_parts.append(f"{entity.name} is a {entity.type.value}{desc}.")

            for rel in pending_relationships:
                desc = f" ({rel.description})" if rel.description else ""
                episode_parts.append(
                    f"{rel.source_entity} {rel.type.value} {rel.target_entity}{desc}."
                )

            if episode_parts:
                from datetime import timezone

                episode_content = " ".join(episode_parts)

                # Add as episode - Graphiti will extract and deduplicate
                result = await self.graphiti_service.add_episode(
                    content=episode_content,
                    name=f"Extraction from {doc_id} (incremental)",
                    source_description=f"Incremental extraction from document {doc_id}",
                    reference_time=datetime.now(timezone.utc),
                    source_type="text",
                )

                # Result is a dict from GraphitiService.add_episode()
                nodes_created = result.get("nodes_created", 0) if result else 0
                edges_created = result.get("edges_created", 0) if result else 0

                logger.info(
                    f"Job {job_id}: Graphiti created {nodes_created} nodes, {edges_created} edges"
                )
            else:
                nodes_created = 0
                edges_created = 0

            # Clear pending
            self._pending_entities[job_id] = []
            self._pending_relationships[job_id] = []

            return {"nodes_created": nodes_created, "edges_created": edges_created}

        except Exception as e:
            logger.error(f"Incremental commit failed: {e}")
            # Don't clear pending on failure - they'll be retried next interval
            return {"nodes_created": 0, "edges_created": 0, "error": str(e)}

    async def _update_graph(
        self,
        doc_id: str,
        entities: List[Entity],
        relationships: List[Relationship],
        options: ReprocessingOptions,
    ) -> Dict[str, Any]:
        """Update FalkorDB graph with extracted entities and relationships (legacy method)."""
        await self.graph_builder.initialize()

        if not self.graph_builder.driver:
            logger.warning("FalkorDB not available, skipping graph update")
            return {"nodes_created": 0, "edges_created": 0}

        nodes_created = 0
        edges_created = 0

        try:
            with self.graph_builder.driver.session() as session:
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

                    except Exception as e:
                        logger.warning(f"Failed to create entity node: {e}")

                for rel in relationships:
                    try:
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
                        logger.warning(f"Failed to create relationship: {e}")

        except Exception as e:
            logger.error(f"Graph update failed: {e}")

        return {"nodes_created": nodes_created, "edges_created": edges_created}

    def get_resumable_jobs(self) -> List[Dict[str, Any]]:
        """Get all jobs that can be resumed."""
        jobs = self.job_tracker.get_resumable_jobs()
        return [self.get_job_status(job.id) for job in jobs]


# Singleton instance
_resumable_pipeline_instance: Optional[ResumablePipeline] = None


def get_resumable_pipeline() -> ResumablePipeline:
    """Get or create the singleton ResumablePipeline instance."""
    global _resumable_pipeline_instance
    if _resumable_pipeline_instance is None:
        _resumable_pipeline_instance = ResumablePipeline()
    return _resumable_pipeline_instance

