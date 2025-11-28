"""
API endpoints for knowledge graph operations using Graphiti.
"""
from typing import List, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from ...core.logging import logger
from ...services.graphiti_service import get_graphiti_service


class GraphSearchRequest(BaseModel):
    """Request model for graph search operations."""
    query: str
    limit: int = 10


router = APIRouter()


@router.get("/stats", response_model=dict)
async def get_graph_stats():
    """Get statistics about the current knowledge graph."""
    try:
        graphiti = get_graphiti_service()
        init_success = await graphiti.initialize()
        if not init_success:
            raise HTTPException(
                status_code=500,
                detail="Failed to initialize Graphiti service"
            )

        stats = await graphiti.get_stats()
        return stats

    except Exception as e:
        logger.error(f"Failed to get graph stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search", response_model=dict)
async def search_graph(request: GraphSearchRequest):
    """Search the knowledge graph using natural language."""
    try:
        graphiti = get_graphiti_service()
        init_success = await graphiti.initialize()
        if not init_success:
            raise HTTPException(
                status_code=500,
                detail="Failed to initialize Graphiti service"
            )

        search_results = await graphiti.search(
            query=request.query,
            num_results=request.limit
        )

        return search_results

    except Exception as e:
        logger.error(f"Graph search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/reset")
async def reset_graph():
    """Reset/clear the knowledge graph (dangerous operation)."""
    try:
        graphiti = get_graphiti_service()
        init_success = await graphiti.initialize()
        if not init_success:
            raise HTTPException(
                status_code=500,
                detail="Failed to initialize Graphiti service"
            )

        result = await graphiti.clear_graph()

        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Graph reset failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
