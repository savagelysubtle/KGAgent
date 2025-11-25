"""
API endpoints for system statistics combining ChromaDB and Neo4j data.
"""
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import json

from fastapi import APIRouter, HTTPException

from ...core.logging import logger
from ...core.config import settings
from ...services.vector_store import VectorStoreService
from ...services.graph_builder import GraphBuilderService, get_graph_builder


router = APIRouter()

# Initialize services lazily
_vector_store: VectorStoreService = None


def get_vector_store() -> VectorStoreService:
    """Get or create VectorStoreService instance."""
    global _vector_store
    if _vector_store is None:
        try:
            _vector_store = VectorStoreService()
        except Exception as e:
            logger.error(f"Failed to initialize VectorStoreService: {e}")
            return None
    return _vector_store


def count_documents() -> int:
    """Count unique documents from chunk files."""
    chunks_dir = Path("./data/chunks")
    if not chunks_dir.exists():
        return 0

    doc_ids = set()
    for job_dir in chunks_dir.iterdir():
        if not job_dir.is_dir():
            continue
        for chunk_file in job_dir.glob("*_chunks.json"):
            try:
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Extract unique doc_ids from chunks
                    for chunk in data.get("chunks", []):
                        doc_id = chunk.get("doc_id")
                        if doc_id:
                            doc_ids.add(doc_id)
            except Exception as e:
                logger.warning(f"Failed to read chunk file {chunk_file}: {e}")
                continue

    return len(doc_ids)


def get_recent_jobs(limit: int = 10) -> List[Dict[str, Any]]:
    """Get recent pipeline jobs with their status."""
    jobs = []

    # Check raw data directory for crawl jobs
    raw_dir = Path("./data/raw")
    if raw_dir.exists():
        for job_dir in sorted(raw_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if job_dir.is_dir() and job_dir.name.startswith(("crawl_job_", "upload_job_", "file_job_")):
                job_type = "crawl" if "crawl" in job_dir.name else "upload"
                jobs.append({
                    "id": job_dir.name,
                    "type": job_type,
                    "stage": "raw",
                    "timestamp": datetime.fromtimestamp(job_dir.stat().st_mtime).isoformat(),
                })

    # Check parsed directory
    parsed_dir = Path("./data/parsed")
    if parsed_dir.exists():
        for job_dir in parsed_dir.iterdir():
            if job_dir.is_dir():
                # Update existing job or add new
                existing = next((j for j in jobs if j["id"] == job_dir.name), None)
                if existing:
                    existing["stage"] = "parsed"
                else:
                    jobs.append({
                        "id": job_dir.name,
                        "type": "unknown",
                        "stage": "parsed",
                        "timestamp": datetime.fromtimestamp(job_dir.stat().st_mtime).isoformat(),
                    })

    # Check chunks directory
    chunks_dir = Path("./data/chunks")
    if chunks_dir.exists():
        for job_dir in chunks_dir.iterdir():
            if job_dir.is_dir():
                existing = next((j for j in jobs if j["id"] == job_dir.name), None)
                if existing:
                    existing["stage"] = "chunked"
                else:
                    jobs.append({
                        "id": job_dir.name,
                        "type": "unknown",
                        "stage": "chunked",
                        "timestamp": datetime.fromtimestamp(job_dir.stat().st_mtime).isoformat(),
                    })

    # Sort by timestamp and limit
    jobs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return jobs[:limit]


@router.get("/overview", response_model=dict)
async def get_system_overview():
    """
    Get comprehensive system statistics from all data stores.

    Returns combined metrics from:
    - File system (documents, chunks)
    - ChromaDB (vector embeddings)
    - Neo4j (knowledge graph)
    """
    stats = {
        "status": "success",
        "timestamp": datetime.utcnow().isoformat(),
        "metrics": {
            "documents": 0,
            "chunks": 0,
            "entities": 0,
            "edges": 0,
            "active_jobs": 0,
        },
        "services": {
            "chromadb": {"status": "unknown", "chunks_stored": 0},
            "neo4j": {"status": "unknown", "nodes": 0, "edges": 0, "entity_types": {}},
        },
        "recent_jobs": [],
    }

    try:
        # Get document count from file system
        stats["metrics"]["documents"] = count_documents()

        # Get recent jobs
        stats["recent_jobs"] = get_recent_jobs(5)
        stats["metrics"]["active_jobs"] = len([j for j in stats["recent_jobs"] if j.get("stage") not in ["chunked", "embedded"]])

    except Exception as e:
        logger.error(f"Error getting file system stats: {e}")

    # Get ChromaDB stats
    try:
        vector_store = get_vector_store()
        if vector_store:
            chunk_count = vector_store.count()
            stats["metrics"]["chunks"] = chunk_count
            stats["services"]["chromadb"] = {
                "status": "connected",
                "chunks_stored": chunk_count,
            }
        else:
            stats["services"]["chromadb"]["status"] = "disconnected"
    except Exception as e:
        logger.error(f"Error getting ChromaDB stats: {e}")
        stats["services"]["chromadb"]["status"] = f"error: {str(e)}"

    # Get Neo4j/Graph stats
    try:
        graph_builder = get_graph_builder()
        if graph_builder:
            init_success = await graph_builder.initialize()

            if init_success:
                graph_stats = await graph_builder.get_graph_stats()

                if graph_stats.get("status") == "success":
                    stats["metrics"]["entities"] = graph_stats.get("total_nodes", 0)
                    stats["metrics"]["edges"] = graph_stats.get("total_edges", 0)
                    stats["services"]["neo4j"] = {
                        "status": "connected",
                        "nodes": graph_stats.get("total_nodes", 0),
                        "edges": graph_stats.get("total_edges", 0),
                        "entity_types": graph_stats.get("entity_types", {}),
                    }
                else:
                    stats["services"]["neo4j"]["status"] = f"error: {graph_stats.get('error', 'unknown')}"
            else:
                stats["services"]["neo4j"]["status"] = "disconnected"
        else:
            stats["services"]["neo4j"]["status"] = "not configured"

    except Exception as e:
        logger.error(f"Error getting Neo4j stats: {e}")
        stats["services"]["neo4j"]["status"] = f"error: {str(e)}"

    return stats


@router.get("/chromadb", response_model=dict)
async def get_chromadb_stats():
    """Get detailed ChromaDB statistics."""
    try:
        vector_store = get_vector_store()
        if not vector_store:
            raise HTTPException(status_code=500, detail="ChromaDB service unavailable")

        return {
            "status": "success",
            "collection_name": settings.CHROMA_COLLECTION_NAME,
            "persist_dir": settings.CHROMA_PERSIST_DIR,
            "chunk_count": vector_store.count(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting ChromaDB stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/neo4j", response_model=dict)
async def get_neo4j_stats():
    """Get detailed Neo4j statistics."""
    try:
        graph_builder = get_graph_builder()
        init_success = await graph_builder.initialize()

        if not init_success:
            raise HTTPException(status_code=500, detail="Neo4j service unavailable")

        return await graph_builder.get_graph_stats()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting Neo4j stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs", response_model=dict)
async def get_pipeline_jobs():
    """Get list of recent pipeline jobs with their status."""
    try:
        jobs = get_recent_jobs(20)
        return {
            "status": "success",
            "jobs": jobs,
            "total": len(jobs),
        }
    except Exception as e:
        logger.error(f"Error getting pipeline jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

