"""
Resumable pipeline for knowledge graph building using Graphiti.

This pipeline:
- Sends document chunks directly to Graphiti for entity extraction
- Graphiti handles all extraction, deduplication, and temporal awareness
- Persists progress to SQLite so processing can be paused and resumed
"""

import asyncio
import os
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timezone

from ..core.logging import logger
from ..models.chunk import Chunk
from ..models.entity import ReprocessingOptions
from .document_tracker import get_document_tracker
from .vector_store import get_vector_store
from .graphiti_service import get_graphiti_service
from .processing_job_tracker import (
    get_job_tracker,
    ProcessingJob,
    JobStatus,
    JobType,
)


# How many chunks to batch into a single Graphiti episode
# Larger = fewer LLM calls but more content per call
GRAPHITI_BATCH_SIZE = int(os.environ.get("GRAPHITI_BATCH_SIZE", "10"))


class ResumablePipeline:
    """
    Resumable knowledge graph pipeline using Graphiti.

    Features:
    - Sends raw chunks to Graphiti for entity extraction
    - Graphiti handles all extraction, deduplication, temporal awareness
    - Pause and resume processing at any time
    - Persists progress to SQLite
    - Survives application restarts
    """

    def __init__(self):
        """Initialize the resumable pipeline."""
        self.document_tracker = get_document_tracker()
        self.vector_store = get_vector_store()
        self.job_tracker = get_job_tracker()
        self.graphiti_service = get_graphiti_service()

        # In-memory state for current processing
        self._should_pause: Dict[str, bool] = {}
        self._is_running: Dict[str, bool] = {}

    # ==================== Job Control ====================

    def request_pause(self, job_id: str):
        """Request a running job to pause at the next safe point."""
        self._should_pause[job_id] = True
        self.job_tracker.request_pause(job_id)
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

        This sends raw document chunks to Graphiti, which handles:
        - Entity and relationship extraction (LLM)
        - Deduplication against existing graph
        - Temporal metadata
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
        """
        Run the extraction process using Graphiti.

        Sends batches of raw chunk text to Graphiti, which extracts
        entities and relationships using LLM.
        """
        self._should_pause[job_id] = False
        self._is_running[job_id] = True

        try:
            # Initialize Graphiti
            if not await self.graphiti_service.initialize():
                self.job_tracker.fail_job(job_id, "Failed to initialize Graphiti")
                return {"success": False, "error": "Failed to initialize Graphiti"}

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
            batch_size = options.batch_size or GRAPHITI_BATCH_SIZE
            total_to_process = len(chunks_to_process)
            processed_count = len(processed_chunk_ids)
            total_entities = job.entities_extracted
            total_relationships = job.relationships_extracted

            for i in range(0, total_to_process, batch_size):
                # Check for pause request
                should_pause = self._should_pause.get(job_id, False) or self.job_tracker.is_pause_requested(job_id)
                if should_pause:
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
                batch_num = (i // batch_size) + 1
                total_batches = (total_to_process + batch_size - 1) // batch_size

                logger.info(f"Job {job_id}: Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)")

                try:
                    # Send raw chunks to Graphiti
                    result = await self._send_batch_to_graphiti(batch, doc_id, job_id)

                    # Track entities and relationships from Graphiti's response
                    nodes_created = result.get("nodes_created", 0)
                    edges_created = result.get("edges_created", 0)
                    total_entities += nodes_created
                    total_relationships += edges_created

                    # Record chunks as processed
                    for chunk in batch:
                        self.job_tracker.record_processed_chunk(
                            job_id,
                            chunk.id,
                            chunk.index,
                            entities_found=0,  # Graphiti doesn't report per-chunk
                            relationships_found=0,
                        )
                        processed_count += 1

                    logger.info(
                        f"Job {job_id}: Batch {batch_num} complete - "
                        f"{nodes_created} entities, {edges_created} relationships"
                    )

                except Exception as e:
                    logger.error(f"Failed to process batch {batch_num}: {e}")
                    # Continue with next batch instead of failing entire job

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

            # All chunks processed
            logger.info(f"Job {job_id}: All chunks processed")

            # Get final graph stats
            graph_stats = await self.graphiti_service.get_stats()

            # Complete job
            self.job_tracker.complete_job(job_id)

            # Update document metadata
            doc_metadata = doc.metadata or {}
            doc_metadata.update({
                "reprocessed_at": datetime.now(timezone.utc).isoformat(),
                "entities_count": total_entities,
                "relationships_count": total_relationships,
                "graphiti_processed": True,
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
                "relationships_extracted": total_relationships,
                "graph_total_entities": graph_stats.get("total_entities", 0),
                "graph_total_relationships": graph_stats.get("total_relationships", 0),
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

    async def _send_batch_to_graphiti(
        self,
        chunks: List[Chunk],
        doc_id: str,
        job_id: str,
    ) -> Dict[str, Any]:
        """
        Send a batch of chunks to Graphiti for processing.

        Combines chunk text and sends as a single episode to Graphiti,
        which extracts entities and relationships using LLM.
        """
        if not chunks:
            return {"nodes_created": 0, "edges_created": 0}

        # Combine chunk texts with metadata
        combined_text_parts = []
        for chunk in chunks:
            source = chunk.metadata.get("source", "document")
            combined_text_parts.append(f"[Source: {source}, Chunk {chunk.index}]\n{chunk.text}")

        combined_text = "\n\n---\n\n".join(combined_text_parts)

        # Send to Graphiti
        result = await self.graphiti_service.add_episode(
            content=combined_text,
            name=f"doc_{doc_id}_batch_{chunks[0].index}_{chunks[-1].index}",
            source_description=f"Document {doc_id} chunks {chunks[0].index}-{chunks[-1].index}",
            reference_time=datetime.now(timezone.utc),
            source_type="text",
            group_id=doc_id,  # Group by document for organization
        )

        return {
            "nodes_created": result.get("nodes_created", 0),
            "edges_created": result.get("edges_created", 0),
            "episode_id": result.get("episode_id"),
        }

    async def pause_extraction(self, job_id: str) -> Dict[str, Any]:
        """Request a job to pause."""
        job = self.job_tracker.get_job(job_id)
        if not job:
            return {"success": False, "error": "Job not found"}

        if job.status != JobStatus.RUNNING.value:
            return {"success": False, "error": f"Job is not running (status: {job.status})"}

        self.request_pause(job_id)

        return {
            "success": True,
            "message": "Pause requested. Job will pause after current Graphiti episode.",
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
