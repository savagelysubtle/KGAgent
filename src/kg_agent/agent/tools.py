"""RAG Tools for the Pydantic AI Knowledge Graph Agent."""

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..core.config import settings
from ..core.logging import logger
from ..services.document_tracker import (
    get_document_tracker,
)
from ..services.embedder import EmbedderService
from ..services.graphiti_service import GraphitiService, get_graphiti_service
from ..services.vector_store import VectorStoreService, get_vector_store


@dataclass
class SearchResult:
    """A search result from the knowledge base."""

    text: str
    source: str
    score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class GraphStats:
    """Statistics about the knowledge graph."""

    total_nodes: int
    total_edges: int
    entity_types: Dict[str, int]
    relationship_types: Dict[str, int]
    connected: bool


@dataclass
class VectorStats:
    """Statistics about the vector store."""

    total_chunks: int
    collection_name: str


@dataclass
class DocumentInfo:
    """Information about a tracked document."""

    id: str
    title: str
    source_url: Optional[str]
    source_type: str
    status: str
    chunk_count: int
    created_at: str


@dataclass
class DocumentStats:
    """Statistics about tracked documents."""

    total_documents: int
    by_status: Dict[str, int]
    by_source_type: Dict[str, int]
    total_vectors: int
    total_graph_nodes: int


@dataclass
class DeleteResult:
    """Result of a delete operation."""

    success: bool
    documents_deleted: int
    vectors_deleted: int
    graph_nodes_deleted: int
    message: str


@dataclass
class EntityCreateResult:
    """Result of creating an entity."""

    success: bool
    entity_id: Optional[str]
    message: str


class RAGTools:
    """
    RAG (Retrieval-Augmented Generation) tools for the Knowledge Graph Agent.

    Provides tools for:
    - Vector search in ChromaDB
    - Graph search in FalkorDB via Graphiti
    - Database statistics
    """

    def __init__(self):
        self._vector_store: Optional[VectorStoreService] = None
        self._embedder: Optional[EmbedderService] = None
        self._graphiti: Optional[GraphitiService] = None
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize all services."""
        if self._initialized:
            return True

        try:
            # Initialize embedder
            self._embedder = EmbedderService()
            logger.info("EmbedderService initialized")

            # Initialize vector store
            self._vector_store = VectorStoreService()
            logger.info("VectorStoreService initialized")

            # Initialize Graphiti service
            self._graphiti = get_graphiti_service()
            await self._graphiti.initialize()
            logger.info("GraphitiService initialized")

            self._initialized = True
            return True

        except Exception as e:
            logger.error(f"Failed to initialize RAG tools: {e}")
            return False

    async def search_vectors(
        self, query: str, n_results: int = 5
    ) -> List[SearchResult]:
        """
        Search the vector database (ChromaDB) for semantically similar content.

        Args:
            query: The search query text
            n_results: Number of results to return (default: 5)

        Returns:
            List of SearchResult objects with matching content
        """
        if not self._initialized:
            await self.initialize()

        if not self._embedder or not self._vector_store:
            logger.warning("Vector search services not available")
            return []

        try:
            # Generate embedding for the query (async to avoid blocking)
            query_embedding = await self._embedder.embed_text_async(query)

            # Search ChromaDB
            results = self._vector_store.query(query_embedding, n_results=n_results)

            search_results = []
            if results and "documents" in results and results["documents"]:
                documents = results["documents"][0] if results["documents"] else []
                metadatas = results["metadatas"][0] if results.get("metadatas") else []
                distances = results["distances"][0] if results.get("distances") else []

                for i, doc in enumerate(documents):
                    metadata = metadatas[i] if i < len(metadatas) else {}
                    distance = distances[i] if i < len(distances) else None

                    # Convert distance to similarity score (cosine distance -> similarity)
                    score = 1 - distance if distance is not None else None

                    search_results.append(
                        SearchResult(
                            text=doc,
                            source=metadata.get("source", "unknown"),
                            score=score,
                            metadata=metadata,
                        )
                    )

            logger.info(
                f"Vector search for '{query}' returned {len(search_results)} results"
            )
            return search_results

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    async def search_graph(self, query: str, limit: int = 10) -> List[SearchResult]:
        """
        Search the knowledge graph (FalkorDB via Graphiti) for structured information.

        Uses hybrid search (BM25 + cosine similarity) for best results.

        Args:
            query: The search query text
            limit: Maximum number of results to return (default: 10)

        Returns:
            List of SearchResult objects with matching graph content
        """
        if not self._initialized:
            await self.initialize()

        if not self._graphiti:
            logger.warning("Graph search service not available")
            return []

        try:
            result = await self._graphiti.search(query, num_results=limit)

            search_results = []
            if result.get("status") == "success":
                # Add edges (relationships/facts) as results
                for edge in result.get("edges", []):
                    search_results.append(
                        SearchResult(
                            text=edge.get("fact", ""),
                            source=f"relationship:{edge.get('name', 'unknown')}",
                            score=None,
                            metadata={
                                "source_node": edge.get("source_node"),
                                "target_node": edge.get("target_node"),
                                "created_at": edge.get("created_at"),
                                "valid_at": edge.get("valid_at"),
                            },
                        )
                    )

                # Add nodes (entities) as results
                for node in result.get("nodes", []):
                    search_results.append(
                        SearchResult(
                            text=f"{node.get('name', 'Unknown')}: {node.get('summary', '')}",
                            source=f"entity:{','.join(node.get('labels', ['Unknown']))}",
                            score=None,
                            metadata={
                                "uuid": node.get("uuid"),
                                "labels": node.get("labels"),
                            },
                        )
                    )

            logger.info(
                f"Graph search for '{query}' returned {len(search_results)} results"
            )
            return search_results

        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            return []

    async def hybrid_search(
        self, query: str, vector_results: int = 3, graph_results: int = 3
    ) -> Dict[str, List[SearchResult]]:
        """
        Perform a hybrid search combining vector and graph search.

        Args:
            query: The search query text
            vector_results: Number of vector search results
            graph_results: Number of graph search results

        Returns:
            Dictionary with 'vector' and 'graph' search results
        """
        # Run both searches concurrently
        vector_task = self.search_vectors(query, n_results=vector_results)
        graph_task = self.search_graph(query, limit=graph_results)

        vector_results_list, graph_results_list = await asyncio.gather(
            vector_task, graph_task
        )

        return {"vector": vector_results_list, "graph": graph_results_list}

    async def get_graph_stats(self) -> GraphStats:
        """
        Get statistics about the knowledge graph.

        Returns:
            GraphStats object with current graph statistics
        """
        if not self._initialized:
            await self.initialize()

        if not self._graphiti:
            return GraphStats(
                total_nodes=0,
                total_edges=0,
                entity_types={},
                relationship_types={},
                connected=False,
            )

        try:
            stats = await self._graphiti.get_stats()

            return GraphStats(
                total_nodes=stats.get("total_entities", 0),
                total_edges=stats.get("total_relationships", 0),
                entity_types={"episodes": stats.get("total_episodes", 0)},
                relationship_types={},
                connected=stats.get("connected", False),
            )

        except Exception as e:
            logger.error(f"Failed to get graph stats: {e}")
            return GraphStats(
                total_nodes=0,
                total_edges=0,
                entity_types={},
                relationship_types={},
                connected=False,
            )

    async def get_vector_stats(self) -> VectorStats:
        """
        Get statistics about the vector store.

        Returns:
            VectorStats object with current vector store statistics
        """
        if not self._initialized:
            await self.initialize()

        if not self._vector_store:
            return VectorStats(
                total_chunks=0, collection_name=settings.CHROMA_COLLECTION_NAME
            )

        try:
            count = self._vector_store.count()
            return VectorStats(
                total_chunks=count, collection_name=settings.CHROMA_COLLECTION_NAME
            )

        except Exception as e:
            logger.error(f"Failed to get vector stats: {e}")
            return VectorStats(
                total_chunks=0, collection_name=settings.CHROMA_COLLECTION_NAME
            )

    # ==================== Document Management Tools ====================

    async def list_documents(
        self,
        status: Optional[str] = None,
        source_type: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 20,
    ) -> List[DocumentInfo]:
        """
        List tracked documents with optional filtering.

        Args:
            status: Filter by status (pending, processing, completed, failed, deleted)
            source_type: Filter by source type (web_crawl, file_upload, api)
            search: Search in title and source URL
            limit: Maximum number of results (default: 20)

        Returns:
            List of DocumentInfo objects
        """
        try:
            tracker = get_document_tracker()
            documents = tracker.list_documents(
                status=status, source_type=source_type, search=search, limit=limit
            )

            return [
                DocumentInfo(
                    id=doc.id,
                    title=doc.title,
                    source_url=doc.source_url,
                    source_type=doc.source_type,
                    status=doc.status,
                    chunk_count=doc.chunk_count,
                    created_at=doc.created_at,
                )
                for doc in documents
                if doc
            ]

        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return []

    async def get_document_stats(self) -> DocumentStats:
        """
        Get statistics about tracked documents.

        Returns:
            DocumentStats object with current document statistics
        """
        try:
            tracker = get_document_tracker()
            stats = tracker.get_stats()

            return DocumentStats(
                total_documents=stats.get("total_documents", 0),
                by_status=stats.get("by_status", {}),
                by_source_type=stats.get("by_source_type", {}),
                total_vectors=stats.get("total_vectors", 0),
                total_graph_nodes=stats.get("total_graph_nodes", 0),
            )

        except Exception as e:
            logger.error(f"Failed to get document stats: {e}")
            return DocumentStats(
                total_documents=0,
                by_status={},
                by_source_type={},
                total_vectors=0,
                total_graph_nodes=0,
            )

    async def delete_document(
        self, doc_id: str, delete_vectors: bool = True, delete_graph_nodes: bool = True
    ) -> DeleteResult:
        """
        Delete a document and its associated data from all databases.

        Note: Graph node deletion is not fully supported with Graphiti.
        The document tracker record and vectors will be deleted.

        Args:
            doc_id: The document ID to delete
            delete_vectors: Whether to delete vectors from ChromaDB (default: True)
            delete_graph_nodes: Whether to delete nodes from graph (currently limited)

        Returns:
            DeleteResult with details of the deletion
        """
        try:
            tracker = get_document_tracker()
            document = tracker.get_document(doc_id)

            if not document:
                return DeleteResult(
                    success=False,
                    documents_deleted=0,
                    vectors_deleted=0,
                    graph_nodes_deleted=0,
                    message=f"Document {doc_id} not found",
                )

            vectors_deleted = 0
            graph_nodes_deleted = 0

            # Delete vectors from ChromaDB
            if delete_vectors and document.vector_ids:
                try:
                    vector_store = get_vector_store()
                    vectors_deleted = vector_store.delete_by_ids(document.vector_ids)
                except Exception as e:
                    logger.warning(f"Failed to delete vectors: {e}")

            # Note: Graphiti doesn't expose direct node deletion
            # Graph data is managed through temporal invalidation
            if delete_graph_nodes and document.graph_node_ids:
                logger.info(
                    "Graph node deletion requested but not supported with Graphiti"
                )

            # Delete document record
            tracker.delete_document(doc_id)

            return DeleteResult(
                success=True,
                documents_deleted=1,
                vectors_deleted=vectors_deleted,
                graph_nodes_deleted=graph_nodes_deleted,
                message=f"Successfully deleted document '{document.title}'",
            )

        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            return DeleteResult(
                success=False,
                documents_deleted=0,
                vectors_deleted=0,
                graph_nodes_deleted=0,
                message=f"Error: {str(e)}",
            )

    async def delete_by_source(
        self,
        source_pattern: str,
        delete_vectors: bool = True,
        delete_graph_nodes: bool = True,
    ) -> DeleteResult:
        """
        Delete all documents from a specific source.

        Note: Graph node deletion is not fully supported with Graphiti.

        Args:
            source_pattern: Pattern to match source URLs (e.g., "example.com")
            delete_vectors: Whether to delete vectors from ChromaDB
            delete_graph_nodes: Whether to delete nodes from graph (currently limited)

        Returns:
            DeleteResult with details of the deletion
        """
        try:
            tracker = get_document_tracker()
            documents = tracker.list_documents(search=source_pattern, limit=10000)

            if not documents:
                return DeleteResult(
                    success=True,
                    documents_deleted=0,
                    vectors_deleted=0,
                    graph_nodes_deleted=0,
                    message=f"No documents found matching '{source_pattern}'",
                )

            total_vectors = 0
            total_docs = 0

            for doc in documents:
                if not doc:
                    continue

                # Delete vectors
                if delete_vectors and doc.vector_ids:
                    try:
                        vector_store = get_vector_store()
                        total_vectors += vector_store.delete_by_ids(doc.vector_ids)
                    except Exception as e:
                        logger.warning(f"Failed to delete vectors for {doc.id}: {e}")

                # Delete document record
                if tracker.delete_document(doc.id):
                    total_docs += 1

            return DeleteResult(
                success=True,
                documents_deleted=total_docs,
                vectors_deleted=total_vectors,
                graph_nodes_deleted=0,
                message=f"Deleted {total_docs} documents matching '{source_pattern}'",
            )

        except Exception as e:
            logger.error(f"Failed to delete by source: {e}")
            return DeleteResult(
                success=False,
                documents_deleted=0,
                vectors_deleted=0,
                graph_nodes_deleted=0,
                message=f"Error: {str(e)}",
            )

    async def create_entity(
        self,
        name: str,
        entity_type: str,
        properties: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
    ) -> EntityCreateResult:
        """
        Create or update an entity in the knowledge graph using Graphiti.

        This creates an episode describing the entity, which Graphiti processes
        to extract and deduplicate the entity against the existing graph.

        Args:
            name: The name of the entity (e.g., "Steve", "Python", "OpenAI")
            entity_type: The type of entity (e.g., "Person", "Technology", "Organization")
            properties: Additional properties for the entity
            description: A description of the entity

        Returns:
            EntityCreateResult with details of the creation
        """
        if not self._initialized:
            await self.initialize()

        if not self._graphiti:
            return EntityCreateResult(
                success=False, entity_id=None, message="Graph database not available"
            )

        try:
            # Build a natural language description for Graphiti to process
            props_str = ""
            if properties:
                props_str = ". ".join([f"{k}: {v}" for k, v in properties.items()])

            content = f"{name} is a {entity_type}."
            if description:
                content += f" {description}"
            if props_str:
                content += f" Additional information: {props_str}"

            # Add as an episode - Graphiti will extract and deduplicate
            result = await self._graphiti.add_episode(
                content=content,
                name=f"entity_creation_{name}",
                source_description=f"Manual entity creation for {name}",
            )

            if result.get("status") == "success":
                entity_id = result.get("episode_id")
                nodes_created = result.get("nodes_created", 0)
                logger.info(f"Created entity via Graphiti: {name} ({entity_type})")
                return EntityCreateResult(
                    success=True,
                    entity_id=entity_id,
                    message=f"Successfully created entity '{name}' of type '{entity_type}' ({nodes_created} nodes)",
                )
            else:
                return EntityCreateResult(
                    success=False,
                    entity_id=None,
                    message=result.get("error", "Failed to create entity"),
                )

        except Exception as e:
            logger.error(f"Failed to create entity: {e}")
            return EntityCreateResult(
                success=False,
                entity_id=None,
                message=f"Error creating entity: {str(e)}",
            )

    async def create_relationship(
        self,
        source_entity: str,
        target_entity: str,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> EntityCreateResult:
        """
        Create a relationship between two entities in the knowledge graph using Graphiti.

        This creates an episode describing the relationship, which Graphiti processes
        to extract and deduplicate against the existing graph.

        Args:
            source_entity: The name of the source entity
            target_entity: The name of the target entity
            relationship_type: The type of relationship (e.g., "KNOWS", "WORKS_AT", "CREATED")
            properties: Additional properties for the relationship

        Returns:
            EntityCreateResult with details of the creation
        """
        if not self._initialized:
            await self.initialize()

        if not self._graphiti:
            return EntityCreateResult(
                success=False, entity_id=None, message="Graph database not available"
            )

        try:
            # Build a natural language description for Graphiti to process
            props_str = ""
            if properties:
                props_str = ". ".join([f"{k}: {v}" for k, v in properties.items()])

            # Convert relationship type to natural language
            rel_phrase = relationship_type.lower().replace("_", " ")
            content = f"{source_entity} {rel_phrase} {target_entity}."
            if props_str:
                content += f" Additional context: {props_str}"

            # Add as an episode - Graphiti will extract and deduplicate
            result = await self._graphiti.add_episode(
                content=content,
                name=f"relationship_{source_entity}_{target_entity}",
                source_description=f"Manual relationship creation: {source_entity} -> {target_entity}",
            )

            if result.get("status") == "success":
                episode_id = result.get("episode_id")
                edges_created = result.get("edges_created", 0)
                logger.info(
                    f"Created relationship via Graphiti: {source_entity} -[{relationship_type}]-> {target_entity}"
                )
                return EntityCreateResult(
                    success=True,
                    entity_id=episode_id,
                    message=f"Successfully created relationship '{source_entity}' -[{relationship_type}]-> '{target_entity}' ({edges_created} edges)",
                )
            else:
                return EntityCreateResult(
                    success=False,
                    entity_id=None,
                    message=result.get("error", "Failed to create relationship"),
                )

        except Exception as e:
            logger.error(f"Failed to create relationship: {e}")
            return EntityCreateResult(
                success=False,
                entity_id=None,
                message=f"Error creating relationship: {str(e)}",
            )

    async def clear_all_data(self, confirm: bool = False) -> DeleteResult:
        """
        Clear ALL data from all databases. Use with extreme caution!

        Note: Graph data clearing requires manual FalkorDB reset.

        Args:
            confirm: Must be True to actually perform the deletion

        Returns:
            DeleteResult with details of the deletion
        """
        if not confirm:
            return DeleteResult(
                success=False,
                documents_deleted=0,
                vectors_deleted=0,
                graph_nodes_deleted=0,
                message="Deletion not confirmed. Set confirm=True to delete all data.",
            )

        try:
            vectors_cleared = 0
            docs_cleared = 0

            # Clear ChromaDB
            try:
                vector_store = get_vector_store()
                vectors_cleared = vector_store.clear_collection()
            except Exception as e:
                logger.warning(f"Failed to clear vectors: {e}")

            # Note: Graphiti/FalkorDB graph clearing requires direct database access
            # For now, we just clear the document tracker
            logger.warning(
                "Graph data clearing not implemented - reset FalkorDB manually if needed"
            )

            # Clear document tracker
            try:
                tracker = get_document_tracker()
                documents = tracker.list_documents(limit=100000)
                for doc in documents:
                    if doc and tracker.delete_document(doc.id):
                        docs_cleared += 1
            except Exception as e:
                logger.warning(f"Failed to clear documents: {e}")

            return DeleteResult(
                success=True,
                documents_deleted=docs_cleared,
                vectors_deleted=vectors_cleared,
                graph_nodes_deleted=0,
                message=f"Cleared data: {docs_cleared} documents, {vectors_cleared} vectors. Graph requires manual reset.",
            )

        except Exception as e:
            logger.error(f"Failed to clear all data: {e}")
            return DeleteResult(
                success=False,
                documents_deleted=0,
                vectors_deleted=0,
                graph_nodes_deleted=0,
                message=f"Error: {str(e)}",
            )


# Singleton instance
_rag_tools_instance: Optional[RAGTools] = None


def get_rag_tools() -> RAGTools:
    """Get or create the singleton RAGTools instance."""
    global _rag_tools_instance
    if _rag_tools_instance is None:
        _rag_tools_instance = RAGTools()
    return _rag_tools_instance
