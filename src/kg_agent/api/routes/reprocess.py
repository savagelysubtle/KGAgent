"""API routes for document reprocessing with Graphiti knowledge graph."""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from ...core.logging import logger
from ...services.resumable_pipeline import get_resumable_pipeline
from ...services.processing_job_tracker import get_job_tracker, JobStatus
from ...services.graphiti_service import get_graphiti_service
from ...models.entity import ReprocessingOptions

router = APIRouter()


# ==================== Entity & Graph Stats Endpoints ====================

@router.get("/entities/all")
async def list_all_entities(
    limit: int = Query(100, ge=1, le=1000),
    entity_type: Optional[str] = Query(None, description="Filter by entity type (label)"),
):
    """
    List all entities in the knowledge graph (via Graphiti/FalkorDB).
    """
    graphiti = get_graphiti_service()

    try:
        if not graphiti._initialized:
            await graphiti.initialize()

        if not graphiti._graphiti:
            return {"entities": [], "count": 0, "message": "Graph database not available"}

        driver = graphiti._graphiti.driver

        if entity_type:
            query = f"""
                MATCH (n:Entity)
                WHERE '{entity_type}' IN labels(n)
                RETURN n.name as name, labels(n) as labels, n.summary as summary, n.uuid as uuid
                LIMIT {limit}
            """
        else:
            query = f"""
                MATCH (n:Entity)
                RETURN n.name as name, labels(n) as labels, n.summary as summary, n.uuid as uuid
                LIMIT {limit}
            """

        result = await driver.execute_query(query)

        records = []
        if result and isinstance(result, tuple) and len(result) > 0:
            records = result[0] if result[0] else []
        elif result and isinstance(result, list):
            records = result

        entities = []
        for record in records:
            if isinstance(record, dict):
                labels = record.get("labels", [])
                entity_labels = [l for l in labels if l != "Entity"] if labels else []
                entity_type_name = entity_labels[0] if entity_labels else "Entity"

                entities.append({
                    "name": record.get("name", "Unknown"),
                    "type": entity_type_name,
                    "labels": labels or [],
                    "description": record.get("summary", ""),
                    "uuid": record.get("uuid"),
                    "confidence": 1.0,
                })

        return {"entities": entities, "count": len(entities)}

    except Exception as e:
        logger.error(f"Failed to list entities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph/stats")
async def get_entity_graph_stats():
    """
    Get statistics about the entity knowledge graph (via Graphiti/FalkorDB).
    """
    graphiti = get_graphiti_service()

    try:
        stats = await graphiti.get_stats()

        if stats.get("status") == "error":
            return {
                "status": "unavailable",
                "message": stats.get("error", "Graph database not available"),
            }

        entities_by_type: Dict[str, int] = {}
        relationships_by_type: Dict[str, int] = {}

        if graphiti._graphiti:
            driver = graphiti._graphiti.driver

            try:
                type_result = await driver.execute_query("""
                    MATCH (n:Entity)
                    RETURN labels(n) as labels
                """)

                records = []
                if type_result and isinstance(type_result, tuple) and len(type_result) > 0:
                    records = type_result[0] if type_result[0] else []
                elif type_result and isinstance(type_result, list):
                    records = type_result

                label_counts: Dict[str, int] = {}
                for record in records:
                    if isinstance(record, dict):
                        labels = record.get("labels", [])
                        if labels:
                            for label in labels:
                                if label and label != "Entity":
                                    label_counts[label] = label_counts.get(label, 0) + 1

                entities_by_type = label_counts
            except Exception as e:
                logger.warning(f"Failed to get entity types: {e}")

            try:
                rel_result = await driver.execute_query("""
                    MATCH ()-[r:RELATES_TO]->()
                    RETURN r.name as type
                """)

                records = []
                if rel_result and isinstance(rel_result, tuple) and len(rel_result) > 0:
                    records = rel_result[0] if rel_result[0] else []
                elif rel_result and isinstance(rel_result, list):
                    records = rel_result

                rel_counts: Dict[str, int] = {}
                for record in records:
                    if isinstance(record, dict):
                        type_name = record.get("type")
                        if type_name:
                            rel_counts[type_name] = rel_counts.get(type_name, 0) + 1

                relationships_by_type = rel_counts
            except Exception as e:
                logger.warning(f"Failed to get relationship types: {e}")

        return {
            "status": "success",
            "total_entities": stats.get("total_entities", 0),
            "total_relationships": stats.get("total_relationships", 0),
            "total_episodes": stats.get("total_episodes", 0),
            "documents_with_entities": stats.get("total_episodes", 0),
            "entities_by_type": entities_by_type,
            "relationships_by_type": relationships_by_type,
        }

    except Exception as e:
        logger.error(f"Failed to get graph stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Resumable Processing Jobs ====================

class StartJobRequest(BaseModel):
    """Request to start a resumable extraction job."""
    options: Optional[ReprocessingOptions] = None


class UpdateJobOptionsRequest(BaseModel):
    """Request to update job options."""
    batch_size: Optional[int] = Field(None, ge=1, le=50, description="Chunks to batch per Graphiti episode")


@router.post("/jobs/{doc_id}/start")
async def start_extraction_job(
    doc_id: str,
    request: Optional[StartJobRequest] = None,
    background_tasks: BackgroundTasks = None,
):
    """
    Start or resume entity extraction for a document using Graphiti.

    This uses Graphiti to automatically:
    - Extract entities and relationships from document chunks
    - Deduplicate against existing graph
    - Store with temporal metadata

    If a paused job exists for this document, it will be resumed.
    """
    pipeline = get_resumable_pipeline()
    options = request.options if request else None

    existing_job = get_job_tracker().get_active_job_for_document(doc_id)
    if existing_job and existing_job.status == JobStatus.RUNNING.value:
        return {
            "success": False,
            "error": "A job is already running for this document",
            "job_id": existing_job.id,
            "status": existing_job.status,
        }

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

    result = await pipeline.start_or_resume_extraction(doc_id, options)
    return result


@router.post("/jobs/{job_id}/pause")
async def pause_extraction_job(job_id: str):
    """
    Pause a running extraction job.

    The job will pause after completing the current Graphiti episode.
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
    Data extracted up to this point is preserved in Graphiti.
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
        jobs = job_tracker.get_all_jobs()
    else:
        jobs = job_tracker.get_all_active_jobs()

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


@router.patch("/jobs/{job_id}/options")
async def update_job_options(job_id: str, request: UpdateJobOptionsRequest):
    """
    Update options for a paused or pending job.
    """
    job_tracker = get_job_tracker()
    job = job_tracker.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status not in [JobStatus.PAUSED.value, JobStatus.PENDING.value]:
        raise HTTPException(
            status_code=400,
            detail=f"Can only update options for paused or pending jobs. Current status: {job.status}"
        )

    current_options = job.options or {}
    if request.batch_size is not None:
        current_options["batch_size"] = request.batch_size

    success = job_tracker.update_job_options(job_id, current_options)

    if not success:
        raise HTTPException(status_code=400, detail="Failed to update job options")

    return {
        "success": True,
        "job_id": job_id,
        "options": current_options,
        "message": f"Updated batch_size to {request.batch_size}",
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
