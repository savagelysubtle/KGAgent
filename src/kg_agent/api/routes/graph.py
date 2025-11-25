"""
API endpoints for knowledge graph operations using Graphiti.
"""
import json
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel

from ...core.logging import logger
from ...services.graph_builder import GraphBuilderService
from ...models.chunk import ChunkBatch


class GraphBuildRequest(BaseModel):
    """Request model for graph building operations."""
    job_id: Optional[str] = None
    source_urls: Optional[List[str]] = None
    limit: Optional[int] = 1000  # Limit number of chunks to process


class GraphSearchRequest(BaseModel):
    """Request model for graph search operations."""
    query: str
    limit: int = 10


router = APIRouter()
graph_builder = GraphBuilderService()


@router.post("/build", response_model=dict)
async def build_graph(
    request: GraphBuildRequest,
    background_tasks: BackgroundTasks
):
    """
    Build knowledge graph from processed chunks.

    Either job_id or source_urls must be provided.
    - job_id: Build graph from chunks of a specific pipeline job
    - source_urls: Build graph from chunks matching these source URLs
    """
    if not request.job_id and not request.source_urls:
        raise HTTPException(
            status_code=400,
            detail="Either job_id or source_urls must be provided"
        )

    try:
        # Initialize Graphiti if not already done
        init_success = await graph_builder.initialize()
        if not init_success:
            raise HTTPException(
                status_code=500,
                detail="Failed to initialize Graphiti service"
            )

        # Collect chunks to process
        chunks_to_process = []

        if request.job_id:
            # Load chunks from job directory
            job_dir = Path("./data/chunks") / request.job_id
            if not job_dir.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"Job directory not found: {request.job_id}"
                )

            chunk_files = list(job_dir.glob("*_chunks.json"))
            if not chunk_files:
                raise HTTPException(
                    status_code=404,
                    detail=f"No chunk files found for job: {request.job_id}"
                )

            for chunk_file in chunk_files:
                try:
                    with open(chunk_file, 'r', encoding='utf-8') as f:
                        chunk_data = json.load(f)
                        chunk_batch = ChunkBatch(**chunk_data)
                        chunks_to_process.extend(chunk_batch.chunks)
                except Exception as e:
                    logger.error(f"Failed to load chunk file {chunk_file}: {e}")
                    continue

        elif request.source_urls:
            # Find chunks by source URLs (this would require indexing or search)
            # For now, we'll search through chunk files
            chunks_dir = Path("./data/chunks")
            if not chunks_dir.exists():
                raise HTTPException(
                    status_code=404,
                    detail="No chunk data found"
                )

            for job_dir in chunks_dir.iterdir():
                if not job_dir.is_dir():
                    continue

                for chunk_file in job_dir.glob("*_chunks.json"):
                    try:
                        with open(chunk_file, 'r', encoding='utf-8') as f:
                            chunk_data = json.load(f)
                            chunk_batch = ChunkBatch(**chunk_data)

                            # Filter chunks by source URLs
                            for chunk in chunk_batch.chunks:
                                chunk_source = chunk.metadata.get("source", "")
                                if any(url in chunk_source for url in request.source_urls):
                                    chunks_to_process.append(chunk)
                    except Exception as e:
                        logger.error(f"Failed to load chunk file {chunk_file}: {e}")
                        continue

        if not chunks_to_process:
            raise HTTPException(
                status_code=404,
                detail="No chunks found matching the criteria"
            )

        # Limit chunks if specified
        if request.limit and len(chunks_to_process) > request.limit:
            chunks_to_process = chunks_to_process[:request.limit]
            logger.info(f"Limited processing to {request.limit} chunks out of {len(chunks_to_process)} found")

        # Build graph in background for better UX (can be long-running)
        episode_name = f"graph_build_{request.job_id or 'custom'}_{len(chunks_to_process)}_chunks"

        background_tasks.add_task(
            graph_builder.build_from_chunks,
            chunks_to_process,
            episode_name
        )

        return {
            "status": "accepted",
            "message": "Graph building started in background",
            "chunks_to_process": len(chunks_to_process),
            "episode_name": episode_name
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Graph build request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=dict)
async def get_graph_stats():
    """Get statistics about the current knowledge graph."""
    try:
        # Initialize if needed
        init_success = await graph_builder.initialize()
        if not init_success:
            raise HTTPException(
                status_code=500,
                detail="Failed to initialize Graphiti service"
            )

        stats = await graph_builder.get_graph_stats()
        return stats

    except Exception as e:
        logger.error(f"Failed to get graph stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search", response_model=dict)
async def search_graph(request: GraphSearchRequest):
    """Search the knowledge graph using natural language."""
    try:
        # Initialize if needed
        init_success = await graph_builder.initialize()
        if not init_success:
            raise HTTPException(
                status_code=500,
                detail="Failed to initialize Graphiti service"
            )

        search_results = await graph_builder.search_graph(
            query=request.query,
            limit=request.limit
        )

        return search_results

    except Exception as e:
        logger.error(f"Graph search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/reset")
async def reset_graph():
    """Reset/clear the knowledge graph (dangerous operation)."""
    try:
        # This would require implementing a reset method in GraphBuilderService
        # For now, return not implemented
        raise HTTPException(
            status_code=501,
            detail="Graph reset not yet implemented"
        )

    except Exception as e:
        logger.error(f"Graph reset failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
