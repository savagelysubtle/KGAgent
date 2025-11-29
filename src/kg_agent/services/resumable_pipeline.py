"""
Resumable pipeline for knowledge graph building using Graphiti.

This pipeline:
- Sends document chunks directly to Graphiti for entity extraction
- Graphiti handles all extraction, deduplication, and temporal awareness
- Persists progress to SQLite so processing can be paused and resumed
- Circuit breaker for LLM failures with auto-pause
- Exponential backoff retry for transient failures
- Parallel batch processing for improved throughput
"""

import asyncio
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..core.logging import logger
from ..models.chunk import Chunk
from ..models.entity import ReprocessingOptions
from .document_tracker import get_document_tracker
from .graphiti_service import get_graphiti_service
from .processing_job_tracker import (
    JobStatus,
    JobType,
    get_job_tracker,
)
from .vector_store import get_vector_store

# ==================== Configuration ====================

# How many chunks to batch into a single Graphiti episode/bulk call
# Larger = fewer LLM calls but more content per call
GRAPHITI_BATCH_SIZE = int(os.environ.get("GRAPHITI_BATCH_SIZE", "10"))

# Circuit breaker: auto-pause after this many consecutive failures
MAX_CONSECUTIVE_FAILURES = int(os.environ.get("MAX_CONSECUTIVE_FAILURES", "5"))

# Retry configuration
MAX_RETRIES_PER_BATCH = int(os.environ.get("MAX_RETRIES_PER_BATCH", "3"))
INITIAL_RETRY_DELAY = float(os.environ.get("INITIAL_RETRY_DELAY", "1.0"))  # seconds

# Parallel processing
MAX_CONCURRENT_BATCHES = int(os.environ.get("MAX_CONCURRENT_BATCHES", "3"))

# Bulk API mode: sends each chunk as a separate episode in one API call
# - True: Better entity extraction (each chunk processed independently)
# - False: Combines chunks into one episode (faster but may miss entities)
USE_BULK_API = os.environ.get("USE_BULK_API", "true").lower() in ("true", "1", "yes")


# ==================== Metrics ====================


@dataclass
class JobMetrics:
    """Track job performance metrics."""

    started_at: float = field(default_factory=time.time)
    batches_processed: int = 0
    batches_failed: int = 0
    batches_retried: int = 0
    total_llm_time_ms: float = 0.0
    total_entities: int = 0
    total_relationships: int = 0

    def record_batch(
        self,
        success: bool,
        llm_time_ms: float,
        entities: int = 0,
        relationships: int = 0,
    ):
        """Record a batch result."""
        if success:
            self.batches_processed += 1
            self.total_entities += entities
            self.total_relationships += relationships
        else:
            self.batches_failed += 1
        self.total_llm_time_ms += llm_time_ms

    def record_retry(self):
        """Record a retry attempt."""
        self.batches_retried += 1

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.started_at

    @property
    def throughput(self) -> float:
        """Batches per second."""
        elapsed = self.elapsed_seconds
        return self.batches_processed / elapsed if elapsed > 0 else 0

    @property
    def avg_llm_time_ms(self) -> float:
        """Average LLM response time in ms."""
        total = self.batches_processed + self.batches_failed
        return self.total_llm_time_ms / total if total > 0 else 0

    @property
    def success_rate(self) -> float:
        """Success rate as percentage."""
        total = self.batches_processed + self.batches_failed
        return (self.batches_processed / total * 100) if total > 0 else 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "batches_processed": self.batches_processed,
            "batches_failed": self.batches_failed,
            "batches_retried": self.batches_retried,
            "throughput_per_sec": round(self.throughput, 2),
            "avg_llm_time_ms": round(self.avg_llm_time_ms, 0),
            "success_rate_percent": round(self.success_rate, 1),
            "total_entities": self.total_entities,
            "total_relationships": self.total_relationships,
        }

    def log_summary(self, job_id: str):
        """Log a summary of job metrics."""
        logger.info(
            f"Job {job_id} metrics: "
            f"{self.batches_processed} batches in {self.elapsed_seconds:.1f}s "
            f"({self.throughput:.2f}/sec), "
            f"Avg LLM: {self.avg_llm_time_ms:.0f}ms, "
            f"Success: {self.success_rate:.1f}%, "
            f"Retries: {self.batches_retried}, "
            f"Entities: {self.total_entities}, Rels: {self.total_relationships}"
        )


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

    # ==================== Health Checks ====================

    async def check_llm_health(self) -> Tuple[bool, str]:
        """
        Pre-flight health check to verify LLM is available.

        Returns:
            Tuple of (is_healthy, message)
        """
        try:
            if not await self.graphiti_service.initialize():
                return False, "Failed to initialize Graphiti service"

            # Send a minimal test request
            start_time = time.time()
            result = await self.graphiti_service.add_episode(
                content="Health check: The system is running a pre-flight test.",
                name="health_check",
                source_description="Pipeline health check - can be ignored",
            )
            elapsed_ms = (time.time() - start_time) * 1000

            if result.get("status") == "error":
                error = result.get("error", "Unknown error")
                return False, f"LLM error: {error}"

            logger.info(f"LLM health check passed in {elapsed_ms:.0f}ms")
            return True, f"OK (response time: {elapsed_ms:.0f}ms)"

        except Exception as e:
            return False, f"Health check failed: {str(e)}"

    # ==================== Job Control ====================

    def request_pause(self, job_id: str):
        """Request a running job to pause at the next safe point."""
        self._should_pause[job_id] = True
        self.job_tracker.request_pause(job_id)
        logger.info(f"Pause requested for job {job_id}")

    def is_paused(self, job_id: str) -> bool:
        """Check if a job is paused."""
        job = self.job_tracker.get_job(job_id)
        return bool(job and job.status == JobStatus.PAUSED.value)

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
            "can_resume": job.status
            in [JobStatus.PAUSED.value, JobStatus.PENDING.value],
            "error_message": job.error_message,  # Include error message for UI
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
            logger.info(
                f"Resuming existing job {existing_job.id} for document {doc_id}"
            )
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

        # Pre-flight health check - verify LLM is available before starting
        logger.info("Running pre-flight LLM health check...")
        is_healthy, health_message = await self.check_llm_health()
        if not is_healthy:
            logger.error(f"Pre-flight health check failed: {health_message}")
            return {
                "success": False,
                "error": f"LLM not available: {health_message}",
                "hint": "Please ensure your LLM server is running and has a model loaded.",
            }
        logger.info(f"Pre-flight health check passed: {health_message}")

        # Create job
        job = self.job_tracker.create_job(
            doc_id=doc_id,
            job_type=JobType.ENTITY_EXTRACTION,
            total_chunks=len(vector_ids),
            options=options.model_dump(),
        )

        logger.info(
            f"Created new extraction job {job.id} for document {doc_id} with {len(vector_ids)} chunks"
        )

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
            if not doc:
                self.job_tracker.fail_job(job_id, "Document not found")
                return {"success": False, "error": f"Document {doc_id} not found"}

            vector_ids = doc.vector_ids
            if not vector_ids:
                self.job_tracker.fail_job(job_id, "No vectors for document")
                return {"success": False, "error": f"Document {doc_id} has no vectors"}

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
                c
                for c in all_chunks
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

            # Circuit breaker state
            consecutive_failures = 0
            failed_batch_indices: List[int] = []  # Track failed batches for retry

            # Metrics tracking
            metrics = JobMetrics()

            # Create all batches upfront for parallel processing
            all_batches = []
            for i in range(0, total_to_process, batch_size):
                batch = chunks_to_process[i : i + batch_size]
                batch_num = (i // batch_size) + 1
                all_batches.append((i, batch, batch_num))

            total_batches = len(all_batches)
            api_mode = (
                "bulk API (each chunk separate)"
                if USE_BULK_API
                else "single episode (chunks combined)"
            )
            logger.info(
                f"Job {job_id}: Processing {total_batches} batches "
                f"({MAX_CONCURRENT_BATCHES} concurrent, {api_mode})"
            )

            # Process batches with controlled concurrency
            semaphore = asyncio.Semaphore(MAX_CONCURRENT_BATCHES)

            async def process_single_batch(
                batch_info: Tuple[int, List[Chunk], int],
            ) -> Dict[str, Any]:
                """Process a single batch with semaphore control."""
                i, batch, batch_num = batch_info

                async with semaphore:
                    # Check for pause request before processing
                    should_pause = self._should_pause.get(
                        job_id, False
                    ) or self.job_tracker.is_pause_requested(job_id)
                    if should_pause:
                        return {
                            "status": "paused",
                            "batch_num": batch_num,
                            "chunk_index": i,
                        }

                    logger.info(
                        f"Job {job_id}: Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)"
                    )

                    # Send with retry
                    result = await self._send_batch_with_retry(
                        batch, doc_id, job_id, metrics
                    )
                    result["batch_num"] = batch_num
                    result["chunk_index"] = i
                    result["batch"] = batch
                    return result

            # Process batches in waves to allow for pause/circuit breaker checks
            wave_size = MAX_CONCURRENT_BATCHES * 2  # Process in waves of this size

            for wave_start in range(0, len(all_batches), wave_size):
                # Check for pause before starting wave
                should_pause = self._should_pause.get(
                    job_id, False
                ) or self.job_tracker.is_pause_requested(job_id)
                if should_pause:
                    current_index = (
                        all_batches[wave_start][0]
                        if wave_start < len(all_batches)
                        else total_to_process
                    )
                    logger.info(
                        f"Pausing job {job_id} at batch wave {wave_start // wave_size + 1}"
                    )
                    self.job_tracker.pause_job(job_id)
                    self.job_tracker.update_progress(
                        job_id,
                        current_chunk_index=current_index + resume_from,
                    )
                    self._is_running[job_id] = False
                    metrics.log_summary(job_id)

                    return {
                        "success": True,
                        "status": "paused",
                        "job_id": job_id,
                        "processed_chunks": processed_count,
                        "total_chunks": len(all_chunks),
                        "entities_extracted": total_entities,
                        "relationships_extracted": total_relationships,
                        "message": f"Job paused. Processed {processed_count}/{len(all_chunks)} chunks.",
                        "failed_batches": len(failed_batch_indices),
                        "metrics": metrics.to_dict(),
                    }

                # Get wave of batches
                wave_batches = all_batches[wave_start : wave_start + wave_size]

                # Process wave concurrently
                tasks = [process_single_batch(b) for b in wave_batches]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results
                for raw_result in results:
                    # Handle exceptions from asyncio.gather
                    if isinstance(raw_result, BaseException):
                        consecutive_failures += 1
                        logger.error(f"Job {job_id}: Batch exception: {raw_result}")
                        metrics.record_batch(success=False, llm_time_ms=0)
                        continue

                    # Now we know it's a dict
                    result: Dict[str, Any] = raw_result

                    if result.get("status") == "paused":
                        # Pause was requested during processing
                        continue

                    batch_num: int = result.get("batch_num", 0)
                    batch: List[Chunk] = result.get("batch", [])
                    llm_time_ms: float = result.get("llm_time_ms", 0)

                    if result.get("status") == "error":
                        error_msg = result.get("error", "Unknown error")
                        consecutive_failures += 1
                        failed_batch_indices.append(batch_num)
                        metrics.record_batch(success=False, llm_time_ms=llm_time_ms)

                        logger.warning(
                            f"Job {job_id}: Batch {batch_num} failed after retries ({consecutive_failures}/{MAX_CONSECUTIVE_FAILURES}): {error_msg}"
                        )

                        # Circuit breaker: auto-pause after too many consecutive failures
                        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                            error_detail = f"Auto-paused: {consecutive_failures} consecutive LLM failures. Last error: {error_msg}"
                            logger.error(
                                f"Job {job_id}: Circuit breaker triggered! "
                                f"{consecutive_failures} consecutive failures. Auto-pausing job."
                            )
                            self.job_tracker.pause_job(job_id)
                            self.job_tracker.update_progress(
                                job_id,
                                current_chunk_index=int(result.get("chunk_index", 0))
                                + resume_from,
                                error_message=error_detail,
                            )
                            self._is_running[job_id] = False
                            metrics.log_summary(job_id)

                            return {
                                "success": False,
                                "status": "paused",
                                "job_id": job_id,
                                "processed_chunks": processed_count,
                                "total_chunks": len(all_chunks),
                                "entities_extracted": total_entities,
                                "relationships_extracted": total_relationships,
                                "error": error_detail,
                                "failed_batches": len(failed_batch_indices),
                                "message": "Job auto-paused due to LLM unavailability. Please check your LLM server and resume.",
                                "metrics": metrics.to_dict(),
                            }
                        continue

                    # Success! Reset consecutive failure counter
                    consecutive_failures = 0

                    # Track entities and relationships from Graphiti's response
                    nodes_created: int = result.get("nodes_created", 0)
                    edges_created: int = result.get("edges_created", 0)
                    total_entities += nodes_created
                    total_relationships += edges_created

                    # Record metrics
                    metrics.record_batch(
                        success=True,
                        llm_time_ms=llm_time_ms,
                        entities=nodes_created,
                        relationships=edges_created,
                    )

                    # Record chunks as processed (only on success!)
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
                        f"{nodes_created} entities, {edges_created} rels "
                        f"({llm_time_ms:.0f}ms)"
                    )

                # Update progress after each wave
                self.job_tracker.update_progress(
                    job_id,
                    processed_chunks=processed_count,
                    current_chunk_index=min(wave_start + wave_size, len(all_batches))
                    * batch_size
                    + resume_from,
                    entities_extracted=total_entities,
                    relationships_extracted=total_relationships,
                )

                # Progress callback with metrics
                if progress_callback:
                    progress_callback(
                        {
                            "job_id": job_id,
                            "status": "running",
                            "processed_chunks": processed_count,
                            "total_chunks": len(all_chunks),
                            "progress_percent": (processed_count / len(all_chunks))
                            * 100,
                            "entities_extracted": total_entities,
                            "relationships_extracted": total_relationships,
                            "failed_batches": len(failed_batch_indices),
                            "metrics": metrics.to_dict(),
                        }
                    )

                # Small delay between waves
                await asyncio.sleep(0.05)

            # All chunks processed - log final metrics
            metrics.log_summary(job_id)

            if failed_batch_indices:
                logger.warning(
                    f"Job {job_id}: Completed with {len(failed_batch_indices)} failed batches "
                    f"(batches: {failed_batch_indices[:10]}{'...' if len(failed_batch_indices) > 10 else ''})"
                )
            else:
                logger.info(f"Job {job_id}: All chunks processed successfully")

            # Get final graph stats
            graph_stats = await self.graphiti_service.get_stats()

            # Complete job
            self.job_tracker.complete_job(job_id)

            # Update document metadata
            doc_metadata = doc.metadata or {}
            doc_metadata.update(
                {
                    "reprocessed_at": datetime.now(timezone.utc).isoformat(),
                    "entities_count": total_entities,
                    "relationships_count": total_relationships,
                    "graphiti_processed": True,
                    "failed_batches": len(failed_batch_indices),
                    "processing_time_seconds": metrics.elapsed_seconds,
                    "throughput_per_sec": metrics.throughput,
                }
            )
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
                "failed_batches": len(failed_batch_indices),
                "warning": f"{len(failed_batch_indices)} batches failed and were skipped"
                if failed_batch_indices
                else None,
                "metrics": metrics.to_dict(),
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
        Send a batch of chunks to Graphiti for processing (single attempt).

        Routes to either bulk API or single episode based on USE_BULK_API config.

        Returns:
            Dict with keys:
            - status: "success" or "error"
            - nodes_created: int
            - edges_created: int
            - error: str (only if status="error")
            - llm_time_ms: float (response time)
        """
        if not chunks:
            return {
                "status": "success",
                "nodes_created": 0,
                "edges_created": 0,
                "llm_time_ms": 0,
            }

        if USE_BULK_API:
            return await self._send_batch_bulk(chunks, doc_id, job_id)
        else:
            return await self._send_batch_single_episode(chunks, doc_id, job_id)

    async def _send_batch_single_episode(
        self,
        chunks: List[Chunk],
        doc_id: str,
        job_id: str,
    ) -> Dict[str, Any]:
        """
        Send batch as a single combined episode (legacy mode).

        Combines all chunk text into one episode. Faster but may miss
        entities due to context dilution.
        """
        # Combine chunk texts with metadata
        combined_text_parts = []
        for chunk in chunks:
            source = chunk.metadata.get("source", "document")
            combined_text_parts.append(
                f"[Source: {source}, Chunk {chunk.index}]\n{chunk.text}"
            )

        combined_text = "\n\n---\n\n".join(combined_text_parts)

        # Send to Graphiti with timing
        start_time = time.time()
        result = await self.graphiti_service.add_episode(
            content=combined_text,
            name=f"doc_{doc_id}_batch_{chunks[0].index}_{chunks[-1].index}",
            source_description=f"Document {doc_id} chunks {chunks[0].index}-{chunks[-1].index}",
            reference_time=datetime.now(timezone.utc),
            source_type="text",
            group_id=doc_id,
        )
        llm_time_ms = (time.time() - start_time) * 1000

        return {
            "status": result.get("status", "success"),
            "nodes_created": result.get("nodes_created", 0),
            "edges_created": result.get("edges_created", 0),
            "episode_id": result.get("episode_id"),
            "error": result.get("error"),
            "llm_time_ms": llm_time_ms,
            "mode": "single_episode",
        }

    async def _send_batch_bulk(
        self,
        chunks: List[Chunk],
        doc_id: str,
        job_id: str,
    ) -> Dict[str, Any]:
        """
        Send batch using bulk API - each chunk as a separate episode.

        More efficient API call and better entity extraction since each
        chunk is processed independently with full LLM attention.
        """
        # Prepare episodes for bulk API
        episodes = []
        ref_time = datetime.now(timezone.utc)

        for chunk in chunks:
            source = chunk.metadata.get("source", "document")
            episodes.append(
                {
                    "content": chunk.text,
                    "source": f"doc_{doc_id}_chunk_{chunk.index}",
                    "source_description": f"Document {doc_id} - {source} - Chunk {chunk.index}",
                    "reference_time": ref_time,
                }
            )

        # Send to Graphiti bulk API with timing
        start_time = time.time()
        result = await self.graphiti_service.add_episodes_bulk(
            episodes=episodes,
            group_id=doc_id,
        )
        llm_time_ms = (time.time() - start_time) * 1000

        return {
            "status": result.get("status", "success"),
            "nodes_created": result.get("nodes_created", 0),
            "edges_created": result.get("edges_created", 0),
            "episodes_processed": result.get("episodes_processed", len(chunks)),
            "error": result.get("error"),
            "llm_time_ms": llm_time_ms,
            "mode": "bulk_api",
        }

    async def _send_batch_with_retry(
        self,
        chunks: List[Chunk],
        doc_id: str,
        job_id: str,
        metrics: JobMetrics,
        max_retries: int = MAX_RETRIES_PER_BATCH,
    ) -> Dict[str, Any]:
        """
        Send a batch to Graphiti with exponential backoff retry.

        Retries on failure with increasing delays: 1s, 2s, 4s + jitter.

        Returns:
            Dict with batch result (status, nodes_created, edges_created, etc.)
        """
        last_error = None

        for attempt in range(max_retries):
            result = await self._send_batch_to_graphiti(chunks, doc_id, job_id)

            if result.get("status") != "error":
                # Success!
                return result

            # Failed - store error for later
            last_error = result.get("error", "Unknown error")

            if attempt < max_retries - 1:
                # Calculate delay with exponential backoff + jitter
                delay = (INITIAL_RETRY_DELAY * (2**attempt)) + random.uniform(0, 0.5)
                metrics.record_retry()
                logger.warning(
                    f"Batch failed, retry {attempt + 1}/{max_retries} in {delay:.1f}s: {last_error}"
                )
                await asyncio.sleep(delay)

        # All retries exhausted - return last error result
        logger.error(f"Batch failed after {max_retries} attempts: {last_error}")
        return {
            "status": "error",
            "error": last_error,
            "nodes_created": 0,
            "edges_created": 0,
            "llm_time_ms": 0,
            "retries_exhausted": True,
        }

    async def pause_extraction(self, job_id: str) -> Dict[str, Any]:
        """Request a job to pause."""
        job = self.job_tracker.get_job(job_id)
        if not job:
            return {"success": False, "error": "Job not found"}

        if job.status != JobStatus.RUNNING.value:
            return {
                "success": False,
                "error": f"Job is not running (status: {job.status})",
            }

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
            return {
                "success": False,
                "error": f"Job cannot be resumed (status: {job.status})",
            }

        # Pre-flight health check before resuming
        logger.info("Running pre-flight LLM health check before resume...")
        is_healthy, health_message = await self.check_llm_health()
        if not is_healthy:
            logger.error(f"Pre-flight health check failed: {health_message}")
            return {
                "success": False,
                "error": f"LLM not available: {health_message}",
                "hint": "Please ensure your LLM server is running and has a model loaded before resuming.",
            }
        logger.info(f"Pre-flight health check passed: {health_message}")

        # Clear any previous error message on successful health check
        self.job_tracker.update_progress(job_id, error_message="")

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
        return [
            status
            for job in jobs
            if (status := self.get_job_status(job.id)) is not None
        ]


# Singleton instance
_resumable_pipeline_instance: Optional[ResumablePipeline] = None


def get_resumable_pipeline() -> ResumablePipeline:
    """Get or create the singleton ResumablePipeline instance."""
    global _resumable_pipeline_instance
    if _resumable_pipeline_instance is None:
        _resumable_pipeline_instance = ResumablePipeline()
    return _resumable_pipeline_instance
