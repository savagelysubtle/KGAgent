"""API routes for document management."""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from datetime import datetime

from ...core.logging import logger
from ...services.document_tracker import (
    get_document_tracker,
    DocumentTrackerService,
    TrackedDocument,
    DocumentStatus,
    DocumentSource,
)
from ...services.vector_store import get_vector_store, VectorStoreService
from ...services.graph_builder import get_graph_builder, GraphBuilderService

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
    delete_graph_nodes: bool = True
    soft_delete: bool = False


class BulkDeleteRequest(BaseModel):
    """Request to delete multiple documents."""
    document_ids: List[str]
    delete_vectors: bool = True
    delete_graph_nodes: bool = True


class DeleteBySourceRequest(BaseModel):
    """Request to delete documents by source."""
    source_pattern: str
    delete_vectors: bool = True
    delete_graph_nodes: bool = True


@router.get("/", response_model=Dict[str, Any])
async def list_documents(
    status: Optional[str] = Query(None, description="Filter by status"),
    source_type: Optional[str] = Query(None, description="Filter by source type"),
    search: Optional[str] = Query(None, description="Search in title and source_url"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """List all tracked documents with optional filtering."""
    try:
        tracker = get_document_tracker()
        documents = tracker.list_documents(
            status=status,
            source_type=source_type,
            search=search,
            limit=limit,
            offset=offset
        )

        return {
            "status": "success",
            "count": len(documents),
            "documents": [doc.to_dict() for doc in documents if doc]
        }

    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=Dict[str, Any])
async def get_document_stats():
    """Get document tracking statistics."""
    try:
        tracker = get_document_tracker()
        stats = tracker.get_stats()

        return {
            "status": "success",
            **stats
        }

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

        return {
            "status": "success",
            "document": document.to_dict()
        }

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
                    "existing_document": existing.to_dict()
                }

        document = tracker.create_document(
            title=request.title,
            source_type=request.source_type,
            source_url=request.source_url,
            file_path=request.file_path,
            content_hash=request.content_hash,
            metadata=request.metadata
        )

        return {
            "status": "success",
            "document": document.to_dict()
        }

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
            metadata=request.metadata
        )

        return {
            "status": "success",
            "document": document.to_dict() if document else None
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{doc_id}", response_model=Dict[str, Any])
async def delete_document(doc_id: str, request: DeleteDocumentRequest):
    """
    Delete a document and optionally its associated data from ChromaDB and Neo4j.
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
            "graph_nodes_deleted": 0,
            "document_deleted": False
        }

        # Delete vectors from ChromaDB
        if request.delete_vectors and document.vector_ids:
            try:
                vector_store = get_vector_store()
                results["vectors_deleted"] = vector_store.delete_by_ids(document.vector_ids)
            except Exception as e:
                logger.warning(f"Failed to delete vectors: {e}")

        # Delete nodes from Neo4j
        if request.delete_graph_nodes and document.graph_node_ids:
            try:
                graph_builder = get_graph_builder()
                graph_result = await graph_builder.delete_by_node_ids(document.graph_node_ids)
                results["graph_nodes_deleted"] = graph_result.get("nodes_deleted", 0)
            except Exception as e:
                logger.warning(f"Failed to delete graph nodes: {e}")

        # Delete document record
        results["document_deleted"] = tracker.delete_document(doc_id, soft_delete=request.soft_delete)

        return {
            "status": "success",
            **results
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bulk-delete", response_model=Dict[str, Any])
async def bulk_delete_documents(request: BulkDeleteRequest):
    """Delete multiple documents and their associated data."""
    try:
        tracker = get_document_tracker()
        vector_store = get_vector_store()
        graph_builder = get_graph_builder()

        results = {
            "total_requested": len(request.document_ids),
            "documents_deleted": 0,
            "vectors_deleted": 0,
            "graph_nodes_deleted": 0,
            "errors": []
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
                        results["errors"].append(f"Failed to delete vectors for {doc_id}: {e}")

                # Delete graph nodes
                if request.delete_graph_nodes and document.graph_node_ids:
                    try:
                        graph_result = await graph_builder.delete_by_node_ids(document.graph_node_ids)
                        results["graph_nodes_deleted"] += graph_result.get("nodes_deleted", 0)
                    except Exception as e:
                        results["errors"].append(f"Failed to delete graph nodes for {doc_id}: {e}")

                # Delete document record
                if tracker.delete_document(doc_id):
                    results["documents_deleted"] += 1

            except Exception as e:
                results["errors"].append(f"Error processing {doc_id}: {e}")

        return {
            "status": "success" if not results["errors"] else "partial",
            **results
        }

    except Exception as e:
        logger.error(f"Error in bulk delete: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/delete-by-source", response_model=Dict[str, Any])
async def delete_by_source(request: DeleteBySourceRequest):
    """Delete all documents from a specific source."""
    try:
        tracker = get_document_tracker()
        vector_store = get_vector_store()
        graph_builder = get_graph_builder()

        results = {
            "source_pattern": request.source_pattern,
            "documents_deleted": 0,
            "vectors_deleted": 0,
            "graph_nodes_deleted": 0
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

            # Delete graph nodes
            if request.delete_graph_nodes and doc.graph_node_ids:
                try:
                    graph_result = await graph_builder.delete_by_node_ids(doc.graph_node_ids)
                    results["graph_nodes_deleted"] += graph_result.get("nodes_deleted", 0)
                except Exception as e:
                    logger.warning(f"Failed to delete graph nodes for {doc.id}: {e}")

            # Delete document record
            if tracker.delete_document(doc.id):
                results["documents_deleted"] += 1

        return {
            "status": "success",
            **results
        }

    except Exception as e:
        logger.error(f"Error deleting by source: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear-all", response_model=Dict[str, Any])
async def clear_all_data(
    confirm: bool = Query(False, description="Must be true to confirm deletion")
):
    """
    Clear ALL data from the system. Use with extreme caution!
    Requires confirm=true query parameter.
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Must pass confirm=true to delete all data"
        )

    try:
        tracker = get_document_tracker()
        vector_store = get_vector_store()
        graph_builder = get_graph_builder()

        results = {
            "vectors_cleared": 0,
            "graph_cleared": {"nodes_deleted": 0, "edges_deleted": 0},
            "documents_cleared": 0
        }

        # Clear ChromaDB
        try:
            results["vectors_cleared"] = vector_store.clear_collection()
        except Exception as e:
            logger.warning(f"Failed to clear vectors: {e}")

        # Clear Neo4j
        try:
            graph_result = await graph_builder.clear_graph()
            results["graph_cleared"] = {
                "nodes_deleted": graph_result.get("nodes_deleted", 0),
                "edges_deleted": graph_result.get("edges_deleted", 0)
            }
        except Exception as e:
            logger.warning(f"Failed to clear graph: {e}")

        # Clear document tracker (delete all documents)
        try:
            documents = tracker.list_documents(limit=100000)
            for doc in documents:
                if doc:
                    tracker.delete_document(doc.id)
                    results["documents_cleared"] += 1
        except Exception as e:
            logger.warning(f"Failed to clear documents: {e}")

        return {
            "status": "success",
            "message": "All data cleared",
            **results
        }

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

        return {
            "status": "success",
            "document_id": doc_id,
            "history": history
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sync", response_model=Dict[str, Any])
async def sync_existing_data():
    """
    Sync existing data from ChromaDB and Neo4j into the document tracker.
    This imports data that was created before the document tracker was added.
    """
    try:
        tracker = get_document_tracker()
        vector_store = get_vector_store()
        graph_builder = get_graph_builder()

        results = {
            "documents_created": 0,
            "vectors_linked": 0,
            "graph_nodes_linked": 0,
            "errors": []
        }

        # Get all vectors from ChromaDB
        try:
            all_data = vector_store.collection.get()
            if all_data and all_data.get("ids"):
                # Group by doc_id from metadata
                doc_groups: Dict[str, List[Dict]] = {}

                for i, vec_id in enumerate(all_data["ids"]):
                    metadata = all_data["metadatas"][i] if all_data.get("metadatas") else {}
                    doc_id = metadata.get("doc_id", "unknown")
                    source = metadata.get("source", "unknown")

                    if doc_id not in doc_groups:
                        doc_groups[doc_id] = {
                            "vector_ids": [],
                            "source": source,
                            "metadata": metadata
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
                    title = source if source != "unknown" else f"Document {doc_id[:8]}"

                    doc = tracker.create_document(
                        title=title,
                        source_type="web_crawl" if source.startswith("http") else "file_upload",
                        source_url=source if source.startswith("http") else None,
                        metadata={"original_doc_id": doc_id}
                    )

                    # Link vectors
                    tracker.add_vector_ids(doc.id, group_data["vector_ids"])

                    # Update status
                    tracker.update_document(
                        doc.id,
                        status="completed",
                        chunk_count=len(group_data["vector_ids"]),
                        processed_at=datetime.utcnow().isoformat()
                    )

                    results["documents_created"] += 1
                    results["vectors_linked"] += len(group_data["vector_ids"])

        except Exception as e:
            results["errors"].append(f"ChromaDB sync error: {str(e)}")
            logger.error(f"ChromaDB sync error: {e}")

        # Get graph nodes from Neo4j
        try:
            await graph_builder.initialize()
            if graph_builder.is_connected():
                with graph_builder.driver.session() as session:
                    # Get all Document nodes
                    result = session.run("""
                        MATCH (d:Document)
                        RETURN d.id as id, d.doc_id as doc_id, d.source as source
                    """)

                    for record in result:
                        node_id = record["id"]
                        doc_id = record["doc_id"]
                        source = record["source"]

                        if not node_id:
                            continue

                        # Find matching document in tracker
                        docs = tracker.list_documents(search=source or doc_id, limit=1)
                        if docs and len(docs) > 0:
                            tracker.add_graph_node_ids(docs[0].id, [node_id])
                            results["graph_nodes_linked"] += 1

        except Exception as e:
            results["errors"].append(f"Neo4j sync error: {str(e)}")
            logger.error(f"Neo4j sync error: {e}")

        return {
            "status": "success" if not results["errors"] else "partial",
            **results
        }

    except Exception as e:
        logger.error(f"Error syncing data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

