"""
Graphiti Service - Wrapper for Graphiti temporal knowledge graph.

This service provides integration with Graphiti for:
- Automatic entity extraction from text
- Entity deduplication against existing graph
- Temporal awareness (when facts were learned)
- Hybrid search (semantic + BM25)
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from collections.abc import Iterable

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.embedder.client import EmbedderClient, EmbedderConfig
from graphiti_core.search.search_config import (
    SearchResults,
    SearchConfig,
    EdgeSearchConfig,
    EdgeSearchMethod,
    EdgeReranker,
    NodeSearchConfig,
    NodeSearchMethod,
    NodeReranker,
    EpisodeSearchConfig,
    EpisodeSearchMethod,
    EpisodeReranker,
)

from sentence_transformers import SentenceTransformer

from ..core.config import settings
from ..core.logging import logger


# BM25-only search config for Neo4j Community Edition (no vector search)
# Neo4j Community doesn't support vector.similarity.cosine
BM25_ONLY_SEARCH = SearchConfig(
    edge_config=EdgeSearchConfig(
        search_methods=[EdgeSearchMethod.bm25],
        reranker=EdgeReranker.rrf,
    ),
    node_config=NodeSearchConfig(
        search_methods=[NodeSearchMethod.bm25],
        reranker=NodeReranker.rrf,
    ),
    episode_config=EpisodeSearchConfig(
        search_methods=[EpisodeSearchMethod.bm25],
        reranker=EpisodeReranker.rrf,
    ),
)


class SentenceTransformerEmbedder(EmbedderClient):
    """
    Custom embedder that uses sentence-transformers for local embeddings.
    This allows Graphiti to use the same embeddings as ChromaDB.
    """

    def __init__(self, model_name: str = settings.HF_EMBEDDING_MODEL):
        import os
        os.environ['HF_HOME'] = settings.HF_HOME

        logger.info(f"Initializing SentenceTransformerEmbedder with model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")

    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        """Create embedding for a single text or first text in list."""
        if isinstance(input_data, str):
            text = input_data
        elif isinstance(input_data, list) and len(input_data) > 0 and isinstance(input_data[0], str):
            text = input_data[0]
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")

        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None, lambda: self.model.encode(text).tolist()
        )
        return embedding

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        """Create embeddings for a batch of texts."""
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: self.model.encode(input_data_list).tolist()
        )
        return embeddings


class GraphitiService:
    """
    Service wrapper for Graphiti temporal knowledge graph.

    Provides:
    - Episode ingestion with automatic entity extraction
    - Hybrid search (semantic + BM25)
    - Entity and relationship management
    - Graph statistics
    """

    def __init__(self):
        self._graphiti: Optional[Graphiti] = None
        self._initialized = False
        self._embedder: Optional[SentenceTransformerEmbedder] = None
        self._llm_client: Optional[OpenAIGenericClient] = None

    async def initialize(self) -> bool:
        """
        Initialize Graphiti with LM Studio LLM and sentence-transformers embedder.

        Returns:
            bool: True if initialization successful
        """
        if self._initialized:
            return True

        try:
            logger.info("Initializing GraphitiService...")

            # Create custom embedder using sentence-transformers
            logger.info(f"Creating embedder with model: {settings.HF_EMBEDDING_MODEL}")
            self._embedder = SentenceTransformerEmbedder(settings.HF_EMBEDDING_MODEL)

            # Create LLM client for LM Studio using OpenAIGenericClient
            logger.info(f"Creating LLM client for LM Studio at: {settings.LLM_BASE_URL}")
            llm_config = LLMConfig(
                api_key=settings.LLM_API_KEY,
                model=settings.LLM_MODEL_NAME,
                base_url=settings.LLM_BASE_URL,
                temperature=settings.LLM_TEMPERATURE,
            )
            self._llm_client = OpenAIGenericClient(
                config=llm_config,
                max_tokens=8192,  # Higher limit for local models
            )

            # Initialize Graphiti with appropriate graph driver
            if settings.GRAPH_DRIVER == "falkordb":
                logger.info(f"Connecting to FalkorDB at: {settings.FALKORDB_HOST}:{settings.FALKORDB_PORT}")
                from graphiti_core.driver.falkordb_driver import FalkorDriver

                graph_driver = FalkorDriver(
                    host=settings.FALKORDB_HOST,
                    port=settings.FALKORDB_PORT,
                    password=settings.FALKORDB_PASSWORD,
                )

                self._graphiti = Graphiti(
                    llm_client=self._llm_client,
                    embedder=self._embedder,
                    store_raw_episode_content=True,
                    graph_driver=graph_driver,
                )
            else:
                # Default to Neo4j (requires Enterprise for vector search)
                logger.info(f"Connecting to Neo4j at: {settings.NEO4J_URI}")
                self._graphiti = Graphiti(
                    uri=settings.NEO4J_URI,
                    user=settings.NEO4J_USERNAME,
                    password=settings.NEO4J_PASSWORD,
                    llm_client=self._llm_client,
                    embedder=self._embedder,
                    store_raw_episode_content=True,
                )

            # Build indices and constraints (idempotent)
            # Note: Graphiti may throw errors if indices already exist, which is fine
            logger.info("Building Graphiti indices and constraints...")
            try:
                await self._graphiti.build_indices_and_constraints()
            except Exception as idx_error:
                # Indices already exist - this is fine
                if "EquivalentSchemaRuleAlreadyExists" in str(idx_error):
                    logger.info("Graphiti indices already exist, continuing...")
                else:
                    raise idx_error

            self._initialized = True
            logger.info("GraphitiService initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize GraphitiService: {e}")
            self._initialized = False
            return False

    async def add_episode(
        self,
        content: str,
        name: str = "document",
        source_description: str = "",
        source_type: str = "text",
        reference_time: Optional[datetime] = None,
        group_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Add a single episode (text content) to the knowledge graph.

        Graphiti will automatically:
        - Extract entities and relationships using LLM
        - Deduplicate against existing graph
        - Store with temporal metadata

        Args:
            content: The text content to process
            name: Episode name/identifier (e.g., document ID)
            source_description: Human-readable source description
            source_type: Type of episode ("text", "message", "json")
            reference_time: When the content was created/occurred
            group_id: Optional group for multi-tenant scenarios

        Returns:
            Dict with episode info and extracted entities/relationships
        """
        if not self._initialized:
            await self.initialize()

        if not self._graphiti:
            raise RuntimeError("GraphitiService not initialized")

        try:
            # Map source type to EpisodeType
            ep_type = EpisodeType.text
            if source_type == "message":
                ep_type = EpisodeType.message
            elif source_type == "json":
                ep_type = EpisodeType.json

            # Use current time if not specified
            if reference_time is None:
                reference_time = datetime.now(timezone.utc)

            logger.debug(f"Adding episode: {name}")

            result = await self._graphiti.add_episode(
                name=name,
                episode_body=content,
                source_description=source_description or f"Content from {name}",
                reference_time=reference_time,
                source=ep_type,  # Parameter is 'source', not 'episode_type'
                group_id=group_id,
            )

            return {
                "status": "success",
                "episode_id": str(result.episode.uuid) if result.episode else None,
                "nodes_created": len(result.nodes),
                "edges_created": len(result.edges),
                "entities": [
                    {
                        "name": n.name,
                        "labels": n.labels,  # Entity types are in labels
                        "uuid": str(n.uuid),
                        "summary": n.summary,
                    }
                    for n in result.nodes
                ],
                "relationships": [
                    {
                        "fact": e.fact,
                        "source": e.source_node_uuid,
                        "target": e.target_node_uuid,
                    }
                    for e in result.edges
                ],
            }

        except Exception as e:
            logger.error(f"Failed to add episode: {e}")
            return {
                "status": "error",
                "error": str(e),
                "nodes_created": 0,
                "edges_created": 0,
            }

    async def add_episodes_bulk(
        self,
        episodes: List[Dict[str, Any]],
        group_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Add multiple episodes in bulk (more efficient than individual calls).

        Args:
            episodes: List of episode dicts with keys:
                - content: str (required)
                - source: str (optional)
                - source_description: str (optional)
                - reference_time: datetime (optional)
            group_id: Optional group for multi-tenant scenarios

        Returns:
            Dict with bulk processing results
        """
        if not self._initialized:
            await self.initialize()

        if not self._graphiti:
            raise RuntimeError("GraphitiService not initialized")

        try:
            from graphiti_core.utils.bulk_utils import RawEpisode

            raw_episodes = []
            for ep in episodes:
                ref_time = ep.get("reference_time", datetime.now(timezone.utc))
                if isinstance(ref_time, str):
                    ref_time = datetime.fromisoformat(ref_time)

                raw_episodes.append(
                    RawEpisode(
                        name=ep.get("source", "document"),
                        content=ep["content"],
                        source_description=ep.get("source_description", ""),
                        reference_time=ref_time,
                        episode_type=EpisodeType.text,
                    )
                )

            logger.info(f"Adding {len(raw_episodes)} episodes in bulk")

            result = await self._graphiti.add_episode_bulk(
                episodes=raw_episodes,
                group_id=group_id,
            )

            return {
                "status": "success",
                "episodes_processed": len(result.episodes),
                "nodes_created": len(result.nodes),
                "edges_created": len(result.edges),
            }

        except Exception as e:
            logger.error(f"Failed to add episodes in bulk: {e}")
            return {
                "status": "error",
                "error": str(e),
                "nodes_created": 0,
                "edges_created": 0,
                "episodes_processed": 0,
            }

    async def search(
        self,
        query: str,
        num_results: int = 10,
        group_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Perform BM25 search on the knowledge graph.

        Note: Using BM25-only search because Neo4j Community Edition
        doesn't support vector similarity functions. For hybrid search,
        upgrade to Neo4j Enterprise or use a different graph driver.

        Args:
            query: Search query
            num_results: Maximum number of results
            group_ids: Optional list of group IDs to filter by

        Returns:
            Dict with search results (entities and relationships)
        """
        if not self._initialized:
            await self.initialize()

        if not self._graphiti:
            raise RuntimeError("GraphitiService not initialized")

        try:
            logger.debug(f"Searching for: {query}")

            # Use search_ method with hybrid search config
            # FalkorDB supports vector search, so we can use full hybrid
            search_config = SearchConfig(
                edge_config=EdgeSearchConfig(
                    search_methods=[EdgeSearchMethod.bm25, EdgeSearchMethod.cosine_similarity],
                    reranker=EdgeReranker.rrf,
                ),
                node_config=NodeSearchConfig(
                    search_methods=[NodeSearchMethod.bm25, NodeSearchMethod.cosine_similarity],
                    reranker=NodeReranker.rrf,
                ),
                limit=num_results,
            )

            results: SearchResults = await self._graphiti.search_(
                query=query,
                config=search_config,
                group_ids=group_ids,
            )

            return {
                "status": "success",
                "query": query,
                "edges": [
                    {
                        "fact": edge.fact,
                        "name": edge.name,
                        "source_node": edge.source_node_uuid,
                        "target_node": edge.target_node_uuid,
                        "created_at": edge.created_at.isoformat() if edge.created_at else None,
                        "valid_at": edge.valid_at.isoformat() if edge.valid_at else None,
                        "invalid_at": edge.invalid_at.isoformat() if edge.invalid_at else None,
                    }
                    for edge in results.edges
                ],
                "nodes": [
                    {
                        "name": node.name,
                        "labels": node.labels,
                        "uuid": str(node.uuid),
                        "summary": node.summary,
                    }
                    for node in results.nodes
                ],
            }

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "edges": [],
                "nodes": [],
            }

    async def get_entity(self, entity_name: str) -> Optional[Dict[str, Any]]:
        """
        Get an entity by name.

        Args:
            entity_name: Name of the entity to retrieve

        Returns:
            Entity dict or None if not found
        """
        if not self._initialized:
            await self.initialize()

        if not self._graphiti:
            raise RuntimeError("GraphitiService not initialized")

        try:
            # Use search to find the entity
            results = await self.search(entity_name, num_results=5)

            if results["status"] == "success" and results["nodes"]:
                # Find exact match
                for node in results["nodes"]:
                    if node["name"].lower() == entity_name.lower():
                        return node
                # Return first result if no exact match
                return results["nodes"][0]

            return None

        except Exception as e:
            logger.error(f"Failed to get entity: {e}")
            return None

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge graph.

        Returns:
            Dict with node and edge counts
        """
        if not self._initialized:
            await self.initialize()

        if not self._graphiti:
            return {
                "status": "error",
                "error": "GraphitiService not initialized",
            }

        try:
            # Query graph directly for stats using the driver
            driver = self._graphiti.driver

            def extract_count(result: Any) -> int:
                """Extract count from FalkorDB result tuple format."""
                if result is None:
                    return 0
                # FalkorDB returns (records, header, None) tuple
                if isinstance(result, tuple) and len(result) > 0:
                    records = result[0]
                    if records and len(records) > 0:
                        record = records[0]
                        if isinstance(record, dict):
                            return record.get("count", 0)
                # Fallback for list format
                elif isinstance(result, list) and len(result) > 0:
                    record = result[0]
                    if isinstance(record, dict):
                        return record.get("count", 0)
                return 0

            # Count entity nodes
            node_result = await driver.execute_query(
                "MATCH (n:Entity) RETURN count(n) as count"
            )
            node_count = extract_count(node_result)

            # Count edges
            edge_result = await driver.execute_query(
                "MATCH ()-[r:RELATES_TO]->() RETURN count(r) as count"
            )
            edge_count = extract_count(edge_result)

            # Count episodes
            episode_result = await driver.execute_query(
                "MATCH (e:Episodic) RETURN count(e) as count"
            )
            episode_count = extract_count(episode_result)

            return {
                "status": "success",
                "connected": True,
                "total_entities": node_count,
                "total_relationships": edge_count,
                "total_episodes": episode_count,
            }

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {
                "status": "error",
                "error": str(e),
                "connected": False,
            }

    async def clear_graph(self) -> Dict[str, Any]:
        """
        Clear all data from the graph database.

        WARNING: This will delete ALL nodes, edges, and episodes!
        """
        if not self._initialized:
            await self.initialize()

        if not self._graphiti:
            return {
                "status": "error",
                "error": "GraphitiService not initialized",
                "nodes_deleted": 0,
                "edges_deleted": 0,
            }

        try:
            driver = self._graphiti.driver

            # Get counts before deletion for reporting
            node_count = 0
            edge_count = 0

            try:
                node_result, _, _ = await driver.execute_query(
                    "MATCH (n) RETURN count(n) as count"
                )
                if node_result and len(node_result) > 0:
                    node_count = node_result[0].get("count", 0)

                edge_result, _, _ = await driver.execute_query(
                    "MATCH ()-[r]->() RETURN count(r) as count"
                )
                if edge_result and len(edge_result) > 0:
                    edge_count = edge_result[0].get("count", 0)
            except Exception as e:
                logger.warning(f"Failed to get counts before clear: {e}")

            # Delete all relationships first
            logger.info("Deleting all relationships from graph...")
            await driver.execute_query("MATCH ()-[r]->() DELETE r")

            # Delete all nodes
            logger.info("Deleting all nodes from graph...")
            await driver.execute_query("MATCH (n) DELETE n")

            logger.info(f"Graph cleared: {node_count} nodes, {edge_count} edges deleted")

            return {
                "status": "success",
                "nodes_deleted": node_count,
                "edges_deleted": edge_count,
            }

        except Exception as e:
            logger.error(f"Failed to clear graph: {e}")
            return {
                "status": "error",
                "error": str(e),
                "nodes_deleted": 0,
                "edges_deleted": 0,
            }

    async def close(self):
        """Clean up connections."""
        if self._graphiti:
            try:
                await self._graphiti.close()
                logger.info("GraphitiService closed")
            except Exception as e:
                logger.error(f"Error closing GraphitiService: {e}")

        self._initialized = False
        self._graphiti = None


# Singleton instance
_graphiti_service: Optional[GraphitiService] = None


def get_graphiti_service() -> GraphitiService:
    """Get or create the singleton GraphitiService instance."""
    global _graphiti_service
    if _graphiti_service is None:
        _graphiti_service = GraphitiService()
    return _graphiti_service

