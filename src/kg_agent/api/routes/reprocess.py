"""API routes for document reprocessing with entity extraction."""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from typing import List, Optional
from pydantic import BaseModel
import uuid

from ...core.logging import logger
from ...services.reprocessing_pipeline import (
    get_reprocessing_pipeline,
    ReprocessingPipeline,
)
from ...services.resumable_pipeline import get_resumable_pipeline
from ...services.processing_job_tracker import get_job_tracker, JobStatus
from ...models.entity import (
    ReprocessingOptions,
    ReprocessingResult,
    ReprocessingStatus,
)

router = APIRouter()


class ReprocessRequest(BaseModel):
    """Request to reprocess documents."""
    document_ids: List[str]
    options: Optional[ReprocessingOptions] = None


class ReprocessResponse(BaseModel):
    """Response for async reprocess request."""
    job_id: str
    status: str
    documents_queued: int
    message: str


class EntityResponse(BaseModel):
    """Response containing entities."""
    doc_id: str
    entities: List[dict]
    count: int


class RelationshipResponse(BaseModel):
    """Response containing relationships."""
    entity_name: str
    relationships: List[dict]
    count: int


@router.post("/batch", response_model=ReprocessResponse)
async def reprocess_documents_async(
    request: ReprocessRequest,
    background_tasks: BackgroundTasks,
):
    """
    Queue multiple documents for reprocessing with enhanced entity extraction.
    Processing happens in the background.
    """
    if not request.document_ids:
        raise HTTPException(status_code=400, detail="No document IDs provided")

    pipeline = get_reprocessing_pipeline()

    # Generate job ID
    job_id = str(uuid.uuid4())[:8]

    # Queue background task
    background_tasks.add_task(
        pipeline.reprocess_batch,
        request.document_ids,
        request.options,
    )

    logger.info(f"Queued reprocessing job {job_id} for {len(request.document_ids)} documents")

    return ReprocessResponse(
        job_id=job_id,
        status="queued",
        documents_queued=len(request.document_ids),
        message=f"Reprocessing queued for {len(request.document_ids)} documents",
    )


@router.post("/{doc_id}", response_model=dict)
async def reprocess_single_document(
    doc_id: str,
    options: Optional[ReprocessingOptions] = None,
):
    """
    Reprocess a single document synchronously with enhanced entity extraction.

    This will:
    1. Load the document's chunks from ChromaDB
    2. Use LLM to extract entities and relationships from each chunk
    3. Deduplicate and merge entities across chunks
    4. Update the Neo4j knowledge graph with extracted entities

    Returns detailed statistics about the extraction.
    """
    pipeline = get_reprocessing_pipeline()

    try:
        result = await pipeline.reprocess_document(doc_id, options)

        if result.status == ReprocessingStatus.FAILED:
            raise HTTPException(status_code=500, detail=result.error)

        return result.to_dict()

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Reprocessing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{doc_id}/status")
async def get_reprocess_status(doc_id: str):
    """
    Get the current reprocessing status for a document.
    """
    pipeline = get_reprocessing_pipeline()
    status = pipeline.get_status(doc_id)

    if not status:
        return {
            "doc_id": doc_id,
            "status": "not_processing",
            "message": "No active reprocessing job for this document",
        }

    return {
        "doc_id": doc_id,
        **status,
    }


@router.get("/{doc_id}/entities", response_model=EntityResponse)
async def get_document_entities(doc_id: str):
    """
    Get all entities extracted from a document.
    """
    pipeline = get_reprocessing_pipeline()

    try:
        entities = await pipeline.get_document_entities(doc_id)

        return EntityResponse(
            doc_id=doc_id,
            entities=entities,
            count=len(entities),
        )

    except Exception as e:
        logger.error(f"Failed to get entities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/entities/{entity_name}/relationships", response_model=RelationshipResponse)
async def get_entity_relationships(entity_name: str):
    """
    Get all relationships for a specific entity.
    """
    pipeline = get_reprocessing_pipeline()

    try:
        relationships = await pipeline.get_entity_relationships(entity_name)

        return RelationshipResponse(
            entity_name=entity_name,
            relationships=relationships,
            count=len(relationships),
        )

    except Exception as e:
        logger.error(f"Failed to get relationships: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/entities/all")
async def list_all_entities(
    limit: int = Query(100, ge=1, le=1000),
    entity_type: Optional[str] = Query(None, description="Filter by entity type"),
):
    """
    List all entities in the knowledge graph.
    """
    pipeline = get_reprocessing_pipeline()

    await pipeline.graph_builder.initialize()

    if not pipeline.graph_builder.driver:
        return {"entities": [], "count": 0, "message": "Neo4j not available"}

    try:
        with pipeline.graph_builder.driver.session() as session:
            if entity_type:
                result = session.run(
                    """
                    MATCH (e:Entity {type: $type})
                    RETURN e.name as name, e.type as type, e.description as description,
                           e.confidence as confidence
                    LIMIT $limit
                    """,
                    {"type": entity_type, "limit": limit},
                )
            else:
                result = session.run(
                    """
                    MATCH (e:Entity)
                    RETURN e.name as name, e.type as type, e.description as description,
                           e.confidence as confidence
                    LIMIT $limit
                    """,
                    {"limit": limit},
                )

            entities = []
            for record in result:
                entities.append({
                    "name": record["name"],
                    "type": record["type"],
                    "description": record["description"],
                    "confidence": record["confidence"],
                })

            return {"entities": entities, "count": len(entities)}

    except Exception as e:
        logger.error(f"Failed to list entities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph/stats")
async def get_entity_graph_stats():
    """
    Get statistics about the entity knowledge graph.
    """
    pipeline = get_reprocessing_pipeline()

    await pipeline.graph_builder.initialize()

    if not pipeline.graph_builder.driver:
        return {
            "status": "unavailable",
            "message": "Neo4j not available",
        }

    try:
        with pipeline.graph_builder.driver.session() as session:
            # Count entities by type
            type_result = session.run(
                """
                MATCH (e:Entity)
                RETURN e.type as type, count(e) as count
                ORDER BY count DESC
                """
            )
            entities_by_type = {r["type"]: r["count"] for r in type_result}

            # Count total entities
            total_entities = sum(entities_by_type.values())

            # Count relationships by type
            rel_result = session.run(
                """
                MATCH ()-[r:RELATES_TO]->()
                RETURN r.type as type, count(r) as count
                ORDER BY count DESC
                """
            )
            relationships_by_type = {r["type"]: r["count"] for r in rel_result}

            # Count total relationships
            total_relationships = sum(relationships_by_type.values())

            # Count documents with entities
            doc_result = session.run(
                """
                MATCH (d:Document)-[:MENTIONS]->(:Entity)
                RETURN count(DISTINCT d) as count
                """
            )
            docs_with_entities = doc_result.single()["count"]

            return {
                "status": "success",
                "total_entities": total_entities,
                "total_relationships": total_relationships,
                "documents_with_entities": docs_with_entities,
                "entities_by_type": entities_by_type,
                "relationships_by_type": relationships_by_type,
            }

    except Exception as e:
        logger.error(f"Failed to get graph stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Resumable Processing Endpoints ====================

class StartJobRequest(BaseModel):
    """Request to start a resumable extraction job."""
    options: Optional[ReprocessingOptions] = None


@router.post("/jobs/{doc_id}/start")
async def start_extraction_job(
    doc_id: str,
    request: Optional[StartJobRequest] = None,
    background_tasks: BackgroundTasks = None,
):
    """
    Start or resume entity extraction for a document.

    This endpoint uses the resumable pipeline which:
    - Persists progress to SQLite
    - Can be paused and resumed
    - Survives application restarts

    If a paused job exists for this document, it will be resumed.
    """
    pipeline = get_resumable_pipeline()
    options = request.options if request else None

    # Check if already running
    existing_job = get_job_tracker().get_active_job_for_document(doc_id)
    if existing_job and existing_job.status == JobStatus.RUNNING.value:
        return {
            "success": False,
            "error": "A job is already running for this document",
            "job_id": existing_job.id,
            "status": existing_job.status,
        }

    # Start in background
    if background_tasks:
        background_tasks.add_task(
            pipeline.start_or_resume_extraction,
            doc_id,
            options,
        )

        return {
            "success": True,
            "message": "Extraction job started in background",
            "doc_id": doc_id,
            "job_id": existing_job.id if existing_job else "pending",
        }

    # Or run synchronously
    result = await pipeline.start_or_resume_extraction(doc_id, options)
    return result


@router.post("/jobs/{job_id}/pause")
async def pause_extraction_job(job_id: str):
    """
    Pause a running extraction job.

    The job will pause after completing the current batch of chunks.
    Progress is saved and can be resumed later.
    """
    pipeline = get_resumable_pipeline()
    result = await pipeline.pause_extraction(job_id)
    return result


@router.post("/jobs/{job_id}/resume")
async def resume_extraction_job(
    job_id: str,
    background_tasks: BackgroundTasks,
):
    """
    Resume a paused extraction job.

    The job will continue from where it left off.
    """
    pipeline = get_resumable_pipeline()

    # Start in background
    background_tasks.add_task(
        pipeline.resume_extraction,
        job_id,
    )

    job = get_job_tracker().get_job(job_id)

    return {
        "success": True,
        "message": "Job resume started in background",
        "job_id": job_id,
        "resume_from_chunk": job.current_chunk_index if job else 0,
    }


@router.post("/jobs/{job_id}/cancel")
async def cancel_extraction_job(job_id: str):
    """
    Cancel an extraction job.

    The job will be stopped and marked as cancelled.
    Extracted data up to this point is preserved.
    """
    pipeline = get_resumable_pipeline()
    result = await pipeline.cancel_extraction(job_id)
    return result


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """
    Get detailed status of an extraction job.
    """
    pipeline = get_resumable_pipeline()
    status = pipeline.get_job_status(job_id)

    if not status:
        raise HTTPException(status_code=404, detail="Job not found")

    return status


@router.get("/jobs")
async def list_jobs(
    doc_id: Optional[str] = Query(None, description="Filter by document ID"),
    status: Optional[str] = Query(None, description="Filter by status"),
    include_all: bool = Query(False, description="Include completed/failed jobs"),
):
    """
    List extraction jobs.

    By default, returns only active jobs (pending, running, paused).
    Set include_all=true to include completed and failed jobs.
    """
    job_tracker = get_job_tracker()

    if doc_id:
        jobs = job_tracker.get_jobs_for_document(doc_id)
    elif include_all:
        # Get all jobs
        jobs = job_tracker.get_all_jobs()
    else:
        # Get all active jobs (pending, running, paused)
        jobs = job_tracker.get_all_active_jobs()

    # Filter by status if provided
    if status:
        jobs = [j for j in jobs if j.status == status]

    return {
        "jobs": [
            {
                "job_id": j.id,
                "doc_id": j.doc_id,
                "status": j.status,
                "progress_percent": j.progress_percent,
                "processed_chunks": j.processed_chunks,
                "total_chunks": j.total_chunks,
                "entities_extracted": j.entities_extracted,
                "relationships_extracted": j.relationships_extracted,
                "created_at": j.created_at,
                "updated_at": j.updated_at,
            }
            for j in jobs
        ],
        "count": len(jobs),
    }


@router.get("/jobs/resumable")
async def list_resumable_jobs():
    """
    List all jobs that can be resumed (paused or pending).
    """
    pipeline = get_resumable_pipeline()
    jobs = pipeline.get_resumable_jobs()

    return {
        "jobs": jobs,
        "count": len(jobs),
    }


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """
    Delete a job and all its associated data.

    Only completed, failed, or cancelled jobs can be deleted.
    """
    job_tracker = get_job_tracker()
    job = job_tracker.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status in [JobStatus.RUNNING.value, JobStatus.PENDING.value]:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete a running or pending job. Cancel it first."
        )

    job_tracker.delete_job(job_id)

    return {
        "success": True,
        "message": f"Job {job_id} deleted",
    }

