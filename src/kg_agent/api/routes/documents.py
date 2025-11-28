"""API routes for document management."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ...core.logging import logger
from ...services.document_tracker import (
    get_document_tracker,
)
from ...services.graphiti_service import get_graphiti_service
from ...services.vector_store import get_vector_store

router = APIRouter()


class DocumentCreateRequest(BaseModel):
    """Request to create a document record."""

    title: str
    source_type: str
    source_url: Optional[str] = None
    file_path: Optional[str] = None
    content_hash: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class DocumentUpdateRequest(BaseModel):
    """Request to update a document record."""

    status: Optional[str] = None
    chunk_count: Optional[int] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class DeleteDocumentRequest(BaseModel):
    """Request to delete a document and its data."""

    delete_vectors: bool = True
    soft_delete: bool = False


class BulkDeleteRequest(BaseModel):
    """Request to delete multiple documents."""

    document_ids: List[str]
    delete_vectors: bool = True


class DeleteBySourceRequest(BaseModel):
    """Request to delete documents by source."""

    source_pattern: str
    delete_vectors: bool = True


@router.get("/", response_model=Dict[str, Any])
async def list_documents(
    status: Optional[str] = Query(None, description="Filter by status"),
    source_type: Optional[str] = Query(None, description="Filter by source type"),
    search: Optional[str] = Query(None, description="Search in title and source_url"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """List all tracked documents with optional filtering."""
    try:
        tracker = get_document_tracker()
        documents = tracker.list_documents(
            status=status,
            source_type=source_type,
            search=search,
            limit=limit,
            offset=offset,
        )

        return {
            "status": "success",
            "count": len(documents),
            "documents": [doc.to_dict() for doc in documents if doc],
        }

    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=Dict[str, Any])
async def get_document_stats():
    """Get document tracking statistics with live database counts."""
    try:
        tracker = get_document_tracker()
        stats = tracker.get_stats()

        # Get live vector count from ChromaDB
        try:
            vector_store = get_vector_store()
            if vector_store:
                stats["total_vectors"] = vector_store.count()
                stats["chromadb_connected"] = True
        except Exception as e:
            logger.warning(f"Failed to get ChromaDB count: {e}")
            stats["chromadb_connected"] = False

        # Get live graph node count from FalkorDB/Graphiti
        try:
            graphiti = get_graphiti_service()
            if graphiti:
                await graphiti.initialize()
                graph_stats = await graphiti.get_stats()
                if graph_stats.get("status") == "success":
                    stats["total_graph_nodes"] = graph_stats.get("total_entities", 0)
                    stats["total_graph_edges"] = graph_stats.get(
                        "total_relationships", 0
                    )
                    stats["total_episodes"] = graph_stats.get("total_episodes", 0)
                    stats["falkordb_connected"] = True
                else:
                    stats["falkordb_connected"] = False
        except Exception as e:
            logger.warning(f"Failed to get FalkorDB stats: {e}")
            stats["falkordb_connected"] = False

        return {"status": "success", **stats}

    except Exception as e:
        logger.error(f"Error getting document stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{doc_id}", response_model=Dict[str, Any])
async def get_document(doc_id: str):
    """Get a specific document by ID."""
    try:
        tracker = get_document_tracker()
        document = tracker.get_document(doc_id)

        if not document:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

        return {"status": "success", "document": document.to_dict()}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/", response_model=Dict[str, Any])
async def create_document(request: DocumentCreateRequest):
    """Create a new document record."""
    try:
        tracker = get_document_tracker()

        # Check for duplicate by content hash
        if request.content_hash:
            existing = tracker.find_by_content_hash(request.content_hash)
            if existing:
                return {
                    "status": "duplicate",
                    "message": "Document with same content already exists",
                    "existing_document": existing.to_dict(),
                }

        document = tracker.create_document(
            title=request.title,
            source_type=request.source_type,
            source_url=request.source_url,
            file_path=request.file_path,
            content_hash=request.content_hash,
            metadata=request.metadata,
        )

        return {"status": "success", "document": document.to_dict()}

    except Exception as e:
        logger.error(f"Error creating document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{doc_id}", response_model=Dict[str, Any])
async def update_document(doc_id: str, request: DocumentUpdateRequest):
    """Update a document record."""
    try:
        tracker = get_document_tracker()

        # Check if document exists
        existing = tracker.get_document(doc_id)
        if not existing:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

        document = tracker.update_document(
            doc_id=doc_id,
            status=request.status,
            chunk_count=request.chunk_count,
            error_message=request.error_message,
            metadata=request.metadata,
        )

        return {
            "status": "success",
            "document": document.to_dict() if document else None,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{doc_id}", response_model=Dict[str, Any])
async def delete_document(
    doc_id: str,
    delete_vectors: bool = Query(True, description="Delete vectors from ChromaDB"),
    soft_delete: bool = Query(False, description="Soft delete instead of hard delete"),
):
    """
    Delete a document and optionally its associated vector data from ChromaDB.

    Note: Graph data in Graphiti is managed separately through episodes.
    """
    try:
        tracker = get_document_tracker()

        # Get document to find associated IDs
        document = tracker.get_document(doc_id)
        if not document:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

        results = {
            "document_id": doc_id,
            "vectors_deleted": 0,
            "document_deleted": False,
        }

        # Delete vectors from ChromaDB
        if delete_vectors and document.vector_ids:
            try:
                vector_store = get_vector_store()
                results["vectors_deleted"] = vector_store.delete_by_ids(
                    document.vector_ids
                )
            except Exception as e:
                logger.warning(f"Failed to delete vectors: {e}")

        # Delete document record
        results["document_deleted"] = tracker.delete_document(
            doc_id, soft_delete=soft_delete
        )

        return {"status": "success", **results}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bulk-delete", response_model=Dict[str, Any])
async def bulk_delete_documents(request: BulkDeleteRequest):
    """Delete multiple documents and their associated vector data."""
    try:
        tracker = get_document_tracker()
        vector_store = get_vector_store()

        results = {
            "total_requested": len(request.document_ids),
            "documents_deleted": 0,
            "vectors_deleted": 0,
            "errors": [],
        }

        for doc_id in request.document_ids:
            try:
                document = tracker.get_document(doc_id)
                if not document:
                    results["errors"].append(f"Document {doc_id} not found")
                    continue

                # Delete vectors
                if request.delete_vectors and document.vector_ids:
                    try:
                        deleted = vector_store.delete_by_ids(document.vector_ids)
                        results["vectors_deleted"] += deleted
                    except Exception as e:
                        results["errors"].append(
                            f"Failed to delete vectors for {doc_id}: {e}"
                        )

                # Delete document record
                if tracker.delete_document(doc_id):
                    results["documents_deleted"] += 1

            except Exception as e:
                results["errors"].append(f"Error processing {doc_id}: {e}")

        return {"status": "success" if not results["errors"] else "partial", **results}

    except Exception as e:
        logger.error(f"Error in bulk delete: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/delete-by-source", response_model=Dict[str, Any])
async def delete_by_source(request: DeleteBySourceRequest):
    """Delete all documents from a specific source."""
    try:
        tracker = get_document_tracker()
        vector_store = get_vector_store()

        results = {
            "source_pattern": request.source_pattern,
            "documents_deleted": 0,
            "vectors_deleted": 0,
        }

        # Find documents matching the source pattern
        documents = tracker.list_documents(search=request.source_pattern, limit=10000)

        for doc in documents:
            if not doc:
                continue

            # Delete vectors
            if request.delete_vectors and doc.vector_ids:
                try:
                    deleted = vector_store.delete_by_ids(doc.vector_ids)
                    results["vectors_deleted"] += deleted
                except Exception as e:
                    logger.warning(f"Failed to delete vectors for {doc.id}: {e}")

            # Delete document record
            if tracker.delete_document(doc.id):
                results["documents_deleted"] += 1

        return {"status": "success", **results}

    except Exception as e:
        logger.error(f"Error deleting by source: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear-all", response_model=Dict[str, Any])
async def clear_all_data(
    confirm: bool = Query(False, description="Must be true to confirm deletion"),
):
    """
    Clear ALL data from the system. Use with extreme caution!
    Requires confirm=true query parameter.
    """
    if not confirm:
        raise HTTPException(
            status_code=400, detail="Must pass confirm=true to delete all data"
        )

    try:
        tracker = get_document_tracker()
        vector_store = get_vector_store()

        results = {
            "vectors_cleared": 0,
            "graph_cleared": {"nodes_deleted": 0, "edges_deleted": 0},
            "documents_cleared": 0,
            "errors": [],
        }

        # Clear ChromaDB
        try:
            results["vectors_cleared"] = vector_store.clear_collection()
            logger.info(f"Cleared {results['vectors_cleared']} vectors from ChromaDB")
        except Exception as e:
            logger.warning(f"Failed to clear vectors: {e}")
            results["errors"].append(f"ChromaDB: {str(e)}")

        # Clear FalkorDB/Graphiti
        try:
            graphiti_service = get_graphiti_service()
            if graphiti_service:
                await graphiti_service.initialize()
                graph_result = await graphiti_service.clear_graph()
                results["graph_cleared"] = {
                    "nodes_deleted": graph_result.get("nodes_deleted", 0),
                    "edges_deleted": graph_result.get("edges_deleted", 0),
                }
                logger.info(f"Cleared graph: {results['graph_cleared']}")
        except Exception as e:
            logger.warning(f"Failed to clear graph: {e}")
            results["errors"].append(f"FalkorDB: {str(e)}")

        # Clear document tracker (delete all documents)
        try:
            documents = tracker.list_documents(limit=100000)
            for doc in documents:
                if doc:
                    tracker.delete_document(doc.id)
                    results["documents_cleared"] += 1
            logger.info(
                f"Cleared {results['documents_cleared']} documents from tracker"
            )
        except Exception as e:
            logger.warning(f"Failed to clear documents: {e}")
            results["errors"].append(f"Document tracker: {str(e)}")

        return {"status": "success", "message": "All data cleared", **results}

    except Exception as e:
        logger.error(f"Error clearing all data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{doc_id}/history", response_model=Dict[str, Any])
async def get_document_history(doc_id: str):
    """Get processing history for a document."""
    try:
        tracker = get_document_tracker()

        # Check if document exists
        document = tracker.get_document(doc_id)
        if not document:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

        history = tracker.get_document_history(doc_id)

        return {"status": "success", "document_id": doc_id, "history": history}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fix-titles", response_model=Dict[str, Any])
async def fix_document_titles():
    """
    Identify documents with hash-based titles that could be fixed.
    Returns list of documents where original filename is in metadata.
    """
    try:
        tracker = get_document_tracker()
        documents = tracker.list_documents(limit=10000)

        results: Dict[str, Any] = {"checked": 0, "needs_fix": 0, "details": []}

        for doc in documents:
            if not doc:
                continue

            results["checked"] += 1
            old_title = doc.title

            # Check if title looks like a hash
            is_hash_name = (
                len(old_title) == 16 or len(old_title) == 20
            ) and old_title.replace(".html", "").replace(".txt", "").replace(
                ".md", ""
            ).isalnum()

            if is_hash_name and doc.metadata:
                new_title = doc.metadata.get("original_filename")

                if new_title and new_title != old_title:
                    # Store correct title in metadata
                    updated_metadata = dict(doc.metadata) if doc.metadata else {}
                    updated_metadata["correct_title"] = new_title
                    tracker.update_document(doc.id, metadata=updated_metadata)

                    results["needs_fix"] += 1
                    results["details"].append(
                        {
                            "id": doc.id,
                            "old_title": old_title,
                            "correct_title": new_title,
                        }
                    )

        return {"status": "success", **results}

    except Exception as e:
        logger.error(f"Error fixing titles: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/stale", response_model=Dict[str, Any])
async def delete_stale_documents():
    """
    Delete documents that are stuck in pending/processing state with no chunks,
    or documents that failed more than 24 hours ago.
    """
    try:
        tracker = get_document_tracker()
        documents = tracker.list_documents(limit=10000)

        results = {"checked": 0, "deleted": 0, "details": []}

        for doc in documents:
            if not doc:
                continue

            results["checked"] += 1
            should_delete = False
            reason = ""

            # Delete pending/processing documents with no chunks
            if (
                doc.status
                in ["pending", "processing", "parsing", "chunking", "embedding"]
                and doc.chunk_count == 0
            ):
                should_delete = True
                reason = f"Stuck in {doc.status} with no chunks"

            # Delete failed documents with no chunks
            elif doc.status == "failed" and doc.chunk_count == 0:
                should_delete = True
                reason = "Failed with no chunks"

            if should_delete:
                tracker.delete_document(doc.id)
                results["deleted"] += 1
                results["details"].append(
                    {"id": doc.id, "title": doc.title, "reason": reason}
                )

        return {"status": "success", **results}

    except Exception as e:
        logger.error(f"Error deleting stale documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sync", response_model=Dict[str, Any])
async def sync_existing_data():
    """
    Sync existing data from ChromaDB into the document tracker.
    This imports data that was created before the document tracker was added.
    """
    try:
        tracker = get_document_tracker()
        vector_store = get_vector_store()

        results = {"documents_created": 0, "vectors_linked": 0, "errors": []}

        # Get all vectors from ChromaDB
        try:
            all_data = vector_store.collection.get()
            if all_data and all_data.get("ids"):
                # Group by doc_id from metadata
                doc_groups: Dict[str, Dict] = {}

                for i, vec_id in enumerate(all_data["ids"]):
                    metadatas = all_data.get("metadatas")
                    raw_metadata = metadatas[i] if metadatas else {}
                    metadata: Dict[str, Any] = (
                        dict(raw_metadata) if raw_metadata else {}
                    )
                    doc_id = str(metadata.get("doc_id", "unknown"))
                    source = str(metadata.get("source", "unknown"))

                    if doc_id not in doc_groups:
                        doc_groups[doc_id] = {
                            "vector_ids": [],
                            "source": source,
                            "metadata": metadata,
                        }
                    doc_groups[doc_id]["vector_ids"].append(vec_id)

                # Create document records for each group
                for doc_id, group_data in doc_groups.items():
                    # Check if document already exists
                    existing = tracker.list_documents(search=doc_id, limit=1)
                    if existing and len(existing) > 0:
                        # Just link the vectors
                        tracker.add_vector_ids(existing[0].id, group_data["vector_ids"])
                        results["vectors_linked"] += len(group_data["vector_ids"])
                        continue

                    # Create new document
                    source = group_data["source"]
                    # Extract just the filename if source is a path
                    if source and source != "unknown":
                        import os
                        # Handle both URL and file path cases
                        if source.startswith("http"):
                            title = source  # Keep URL as-is
                        else:
                            title = os.path.basename(source)  # Extract just the filename
                    else:
                        title = f"Document {doc_id[:8]}"

                    doc = tracker.create_document(
                        title=title,
                        source_type="web_crawl"
                        if source.startswith("http")
                        else "file_upload",
                        source_url=source if source.startswith("http") else None,
                        metadata={"original_doc_id": doc_id},
                    )

                    # Link vectors
                    tracker.add_vector_ids(doc.id, group_data["vector_ids"])

                    # Update status
                    tracker.update_document(
                        doc.id,
                        status="completed",
                        chunk_count=len(group_data["vector_ids"]),
                        processed_at=datetime.now(timezone.utc).isoformat(),
                    )

                    results["documents_created"] += 1
                    results["vectors_linked"] += len(group_data["vector_ids"])

        except Exception as e:
            results["errors"].append(f"ChromaDB sync error: {str(e)}")
            logger.error(f"ChromaDB sync error: {e}")

        return {"status": "success" if not results["errors"] else "partial", **results}

    except Exception as e:
        logger.error(f"Error syncing data: {e}")
        raise HTTPException(status_code=500, detail=str(e))
