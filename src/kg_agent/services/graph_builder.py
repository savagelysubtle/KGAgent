"""Service for building knowledge graphs using Neo4j backend."""

import asyncio
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..models.chunk import Chunk
from ..core.config import settings
from ..core.logging import logger


class GraphBuilderService:
    """Service for building knowledge graphs using Neo4j with connection pooling and retry logic."""

    def __init__(self):
        self.driver = None
        self._initialized = False
        self._max_retries = 3
        self._retry_delay = 2  # seconds

    async def initialize(self, max_retries: int = None) -> bool:
        """
        Initialize Neo4j connection with retry logic.

        Args:
            max_retries: Number of connection retry attempts

        Returns:
            bool: True if initialization successful, False otherwise
        """
        if self._initialized and self.driver:
            # Verify connection is still valid
            try:
                with self.driver.session() as session:
                    session.run("RETURN 1")
                return True
            except Exception:
                self._initialized = False
                self.driver = None

        retries = max_retries or self._max_retries

        for attempt in range(retries):
            try:
                from neo4j import GraphDatabase
                from neo4j.exceptions import ServiceUnavailable, AuthError

                logger.info(f"Initializing Neo4j connection (attempt {attempt + 1}/{retries})...")
                logger.info(f"Neo4j URI: {settings.NEO4J_URI}")

                # Initialize Neo4j driver with connection pool settings
                self.driver = GraphDatabase.driver(
                    settings.NEO4J_URI,
                    auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD),
                    max_connection_lifetime=3600,  # 1 hour
                    max_connection_pool_size=50,
                    connection_acquisition_timeout=30,
                )

                # Test Neo4j connection
                with self.driver.session() as session:
                    result = session.run("RETURN 'Neo4j connection successful' as message")
                    record = result.single()
                    logger.info(f"Neo4j: {record['message']}")

                    # Get server version
                    version_result = session.run("CALL dbms.components() YIELD name, versions RETURN name, versions[0] as version")
                    for rec in version_result:
                        logger.info(f"Connected to {rec['name']} v{rec['version']}")

                self._initialized = True
                logger.info("Neo4j initialization completed successfully")
                return True

            except ImportError as e:
                logger.warning(f"Neo4j driver not installed: {e}")
                return False

            except AuthError as e:
                logger.error(f"Neo4j authentication failed: {e}")
                logger.error("Check NEO4J_USERNAME and NEO4J_PASSWORD in your .env file")
                return False

            except ServiceUnavailable as e:
                logger.warning(f"Neo4j service unavailable (attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    logger.info(f"Retrying in {self._retry_delay} seconds...")
                    await asyncio.sleep(self._retry_delay)
                else:
                    logger.error("Neo4j service is not available. Make sure Neo4j is running.")
                    logger.error("Run 'docker compose -f docker-compose.dev.yml up -d' to start Neo4j")
                    return False

            except Exception as e:
                logger.warning(f"Failed to initialize Neo4j (attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(self._retry_delay)
                else:
                    logger.error(f"Failed to initialize Neo4j after {retries} attempts: {e}")
                    return False

        return False

    def is_connected(self) -> bool:
        """Check if Neo4j connection is active."""
        if not self._initialized or not self.driver:
            return False

        try:
            with self.driver.session() as session:
                session.run("RETURN 1")
            return True
        except Exception:
            return False

    async def build_from_chunks(self, chunks: List[Chunk], episode_name: str = None) -> Dict[str, Any]:
        """
        Build knowledge graph from document chunks.

        Args:
            chunks: List of document chunks to process
            episode_name: Optional name for the episode/grouping

        Returns:
            Dict containing build results and statistics
        """
        if not self._initialized:
            init_success = await self.initialize()
            if not init_success:
                return {"status": "error", "error": "Failed to initialize Neo4j"}

        if not chunks:
            return {"status": "success", "message": "No chunks to process", "entities_added": 0, "edges_added": 0}

        try:
            logger.info(f"Building knowledge graph from {len(chunks)} chunks")

            total_nodes = 0
            total_edges = 0
            episode = episode_name or f"episode_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

            with self.driver.session() as session:
                # Create Episode node to group related chunks
                session.run("""
                    MERGE (e:Episode {name: $episode_name})
                    SET e.created_at = datetime(),
                        e.chunk_count = $chunk_count
                """, {"episode_name": episode, "chunk_count": len(chunks)})

                for chunk in chunks:
                    # Create Document node for each chunk
                    result = session.run("""
                        MERGE (d:Document {id: $chunk_id})
                        SET d.text = $text,
                            d.doc_id = $doc_id,
                            d.source = $source,
                            d.created_at = datetime()
                        WITH d
                        MATCH (e:Episode {name: $episode_name})
                        MERGE (d)-[:PART_OF]->(e)
                        RETURN d
                    """, {
                        "chunk_id": chunk.id,
                        "text": chunk.text[:2000],  # Truncate for storage
                        "doc_id": chunk.doc_id,
                        "source": chunk.metadata.get("source", "unknown"),
                        "episode_name": episode
                    })

                    if result.single():
                        total_nodes += 1
                        total_edges += 1  # For PART_OF relationship

            result = {
                "status": "success",
                "chunks_processed": len(chunks),
                "nodes_created": total_nodes,
                "edges_created": total_edges,
                "episode": episode
            }

            logger.info(f"Knowledge graph build completed: {result}")
            return result

        except Exception as e:
            logger.error(f"Failed to build knowledge graph: {e}")
            return {"status": "error", "error": str(e)}

    async def get_graph_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current knowledge graph.

        Returns:
            Dict containing graph statistics
        """
        if not self._initialized:
            init_success = await self.initialize()
            if not init_success:
                return {
                    "status": "error",
                    "error": "Neo4j not initialized",
                    "total_nodes": 0,
                    "total_edges": 0,
                    "entity_types": {},
                    "connected": False
                }

        try:
            with self.driver.session() as session:
                # Count nodes
                node_result = session.run("MATCH (n) RETURN count(n) as node_count")
                node_count = node_result.single()["node_count"]

                # Count relationships
                edge_result = session.run("MATCH ()-[r]->() RETURN count(r) as edge_count")
                edge_count = edge_result.single()["edge_count"]

                # Count by label
                label_result = session.run("""
                    MATCH (n)
                    UNWIND labels(n) as label
                    RETURN label, count(*) as count
                    ORDER BY count DESC
                """)
                entity_stats = {}
                for record in label_result:
                    entity_stats[record["label"]] = record["count"]

                # Get relationship types
                rel_result = session.run("""
                    MATCH ()-[r]->()
                    RETURN type(r) as rel_type, count(*) as count
                    ORDER BY count DESC
                """)
                rel_stats = {}
                for record in rel_result:
                    rel_stats[record["rel_type"]] = record["count"]

                return {
                    "status": "success",
                    "total_nodes": node_count,
                    "total_edges": edge_count,
                    "entity_types": entity_stats,
                    "relationship_types": rel_stats,
                    "connected": True
                }

        except Exception as e:
            logger.error(f"Failed to get graph stats: {e}")
            return {
                "status": "error",
                "error": str(e),
                "total_nodes": 0,
                "total_edges": 0,
                "entity_types": {},
                "connected": False
            }

    async def search_graph(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """
        Search the knowledge graph using text matching.

        Args:
            query: Search query
            limit: Maximum number of results to return

        Returns:
            Dict containing search results
        """
        if not self._initialized:
            init_success = await self.initialize()
            if not init_success:
                return {"status": "error", "error": "Neo4j not initialized", "results": [], "count": 0}

        try:
            with self.driver.session() as session:
                # Case-insensitive text search on Document nodes
                result = session.run("""
                    MATCH (d:Document)
                    WHERE toLower(d.text) CONTAINS toLower($query)
                       OR toLower(d.source) CONTAINS toLower($query)
                    RETURN d.id as id,
                           d.text as text,
                           d.source as source,
                           d.doc_id as doc_id,
                           d.created_at as created_at
                    ORDER BY d.created_at DESC
                    LIMIT $limit
                """, {"query": query, "limit": limit})

                results = []
                for record in result:
                    results.append({
                        "id": record["id"],
                        "text": record["text"][:500] if record["text"] else "",
                        "source": record["source"],
                        "doc_id": record["doc_id"],
                        "created_at": str(record["created_at"]) if record["created_at"] else None
                    })

                return {
                    "status": "success",
                    "query": query,
                    "results": results,
                    "count": len(results)
                }

        except Exception as e:
            logger.error(f"Failed to search graph: {e}")
            return {"status": "error", "error": str(e), "results": [], "count": 0}

    async def clear_graph(self) -> Dict[str, Any]:
        """
        Clear all data from the graph. Use with caution!

        Returns:
            Dict containing operation result
        """
        if not self._initialized:
            return {"status": "error", "error": "Neo4j not initialized"}

        try:
            with self.driver.session() as session:
                # Get counts before deletion
                before_nodes = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
                before_edges = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]

                # Delete all nodes and relationships
                session.run("MATCH (n) DETACH DELETE n")

                return {
                    "status": "success",
                    "message": "Graph cleared",
                    "nodes_deleted": before_nodes,
                    "edges_deleted": before_edges
                }

        except Exception as e:
            logger.error(f"Failed to clear graph: {e}")
            return {"status": "error", "error": str(e)}

    async def delete_by_doc_id(self, doc_id: str) -> Dict[str, Any]:
        """
        Delete all nodes and relationships associated with a document ID.

        Args:
            doc_id: Document ID to delete

        Returns:
            Dict containing deletion results
        """
        if not self._initialized:
            init_success = await self.initialize()
            if not init_success:
                return {"status": "error", "error": "Neo4j not initialized"}

        try:
            with self.driver.session() as session:
                # Count nodes before deletion
                count_result = session.run("""
                    MATCH (d:Document {doc_id: $doc_id})
                    RETURN count(d) as count
                """, {"doc_id": doc_id})
                before_count = count_result.single()["count"]

                # Delete nodes with the doc_id and their relationships
                session.run("""
                    MATCH (d:Document {doc_id: $doc_id})
                    DETACH DELETE d
                """, {"doc_id": doc_id})

                return {
                    "status": "success",
                    "doc_id": doc_id,
                    "nodes_deleted": before_count
                }

        except Exception as e:
            logger.error(f"Failed to delete by doc_id: {e}")
            return {"status": "error", "error": str(e)}

    async def delete_by_node_ids(self, node_ids: List[str]) -> Dict[str, Any]:
        """
        Delete specific nodes by their IDs.

        Args:
            node_ids: List of node IDs to delete

        Returns:
            Dict containing deletion results
        """
        if not self._initialized:
            init_success = await self.initialize()
            if not init_success:
                return {"status": "error", "error": "Neo4j not initialized"}

        if not node_ids:
            return {"status": "success", "nodes_deleted": 0}

        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (d:Document)
                    WHERE d.id IN $node_ids
                    WITH d, count(d) as cnt
                    DETACH DELETE d
                    RETURN sum(cnt) as deleted
                """, {"node_ids": node_ids})

                deleted = result.single()["deleted"] or 0

                return {
                    "status": "success",
                    "nodes_deleted": deleted
                }

        except Exception as e:
            logger.error(f"Failed to delete by node IDs: {e}")
            return {"status": "error", "error": str(e)}

    async def delete_by_source(self, source_pattern: str) -> Dict[str, Any]:
        """
        Delete all nodes from a specific source.

        Args:
            source_pattern: Source URL pattern to match

        Returns:
            Dict containing deletion results
        """
        if not self._initialized:
            init_success = await self.initialize()
            if not init_success:
                return {"status": "error", "error": "Neo4j not initialized"}

        try:
            with self.driver.session() as session:
                # Count before deletion
                count_result = session.run("""
                    MATCH (d:Document)
                    WHERE d.source CONTAINS $source_pattern
                    RETURN count(d) as count
                """, {"source_pattern": source_pattern})
                before_count = count_result.single()["count"]

                # Delete matching nodes
                session.run("""
                    MATCH (d:Document)
                    WHERE d.source CONTAINS $source_pattern
                    DETACH DELETE d
                """, {"source_pattern": source_pattern})

                return {
                    "status": "success",
                    "source_pattern": source_pattern,
                    "nodes_deleted": before_count
                }

        except Exception as e:
            logger.error(f"Failed to delete by source: {e}")
            return {"status": "error", "error": str(e)}

    async def delete_episode(self, episode_name: str) -> Dict[str, Any]:
        """
        Delete an episode and all its associated documents.

        Args:
            episode_name: Name of the episode to delete

        Returns:
            Dict containing deletion results
        """
        if not self._initialized:
            init_success = await self.initialize()
            if not init_success:
                return {"status": "error", "error": "Neo4j not initialized"}

        try:
            with self.driver.session() as session:
                # Count documents in episode
                count_result = session.run("""
                    MATCH (d:Document)-[:PART_OF]->(e:Episode {name: $episode_name})
                    RETURN count(d) as doc_count
                """, {"episode_name": episode_name})
                doc_count = count_result.single()["doc_count"]

                # Delete episode and all related documents
                session.run("""
                    MATCH (e:Episode {name: $episode_name})
                    OPTIONAL MATCH (d:Document)-[:PART_OF]->(e)
                    DETACH DELETE d, e
                """, {"episode_name": episode_name})

                return {
                    "status": "success",
                    "episode": episode_name,
                    "documents_deleted": doc_count,
                    "episode_deleted": 1
                }

        except Exception as e:
            logger.error(f"Failed to delete episode: {e}")
            return {"status": "error", "error": str(e)}

    async def list_episodes(self) -> List[Dict[str, Any]]:
        """
        List all episodes in the graph.

        Returns:
            List of episode information
        """
        if not self._initialized:
            init_success = await self.initialize()
            if not init_success:
                return []

        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (e:Episode)
                    OPTIONAL MATCH (d:Document)-[:PART_OF]->(e)
                    RETURN e.name as name,
                           e.created_at as created_at,
                           count(d) as document_count
                    ORDER BY e.created_at DESC
                """)

                episodes = []
                for record in result:
                    episodes.append({
                        "name": record["name"],
                        "created_at": str(record["created_at"]) if record["created_at"] else None,
                        "document_count": record["document_count"]
                    })

                return episodes

        except Exception as e:
            logger.error(f"Failed to list episodes: {e}")
            return []

    async def get_document_ids_by_source(self, source_pattern: str) -> List[str]:
        """
        Get document IDs from a specific source.

        Args:
            source_pattern: Source URL pattern to match

        Returns:
            List of document IDs
        """
        if not self._initialized:
            init_success = await self.initialize()
            if not init_success:
                return []

        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (d:Document)
                    WHERE d.source CONTAINS $source_pattern
                    RETURN d.doc_id as doc_id
                """, {"source_pattern": source_pattern})

                return [record["doc_id"] for record in result if record["doc_id"]]

        except Exception as e:
            logger.error(f"Failed to get document IDs by source: {e}")
            return []

    async def close(self):
        """Clean up resources."""
        if self.driver:
            try:
                self.driver.close()
                logger.info("Neo4j connection closed")
            except Exception as e:
                logger.warning(f"Error closing Neo4j connection: {e}")

        self._initialized = False
        self.driver = None


# Singleton instance for reuse across the application
_graph_builder_instance: Optional[GraphBuilderService] = None


def get_graph_builder() -> GraphBuilderService:
    """Get or create the singleton GraphBuilderService instance."""
    global _graph_builder_instance
    if _graph_builder_instance is None:
        _graph_builder_instance = GraphBuilderService()
    return _graph_builder_instance
