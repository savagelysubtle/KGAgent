import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Any, Optional
from ..models.chunk import Chunk
from ..core.config import settings
from ..core.logging import logger


class VectorStoreService:
    """
    Service to interact with ChromaDB for vector storage.
    """

    def __init__(self, persist_path: str = settings.CHROMA_PERSIST_DIR, collection_name: str = settings.CHROMA_COLLECTION_NAME):
        logger.info(f"Initializing VectorStoreService at {persist_path}, collection: {collection_name}")
        try:
            self.client = chromadb.PersistentClient(path=persist_path)
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"} # Cosine similarity
            )
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    # ChromaDB max batch size limit
    MAX_BATCH_SIZE = 5000  # Leave some margin below 5461 limit

    def add_chunks(self, chunks: List[Chunk], embeddings: Optional[List[List[float]]] = None):
        """
        Add chunks to the vector store with automatic batching.
        If embeddings are provided separately, they are used.
        Otherwise, it expects chunks to have the 'embedding' field set, or relies on Chroma's default embedder (if not provided).
        However, since we use EmbedderService, we expect embeddings to be passed or in the object.
        """
        if not chunks:
            return

        # Prepare embeddings
        final_embeddings = None
        if embeddings:
            final_embeddings = embeddings
        else:
            # Try to extract from chunks if available
            extracted_embeddings = []
            all_have_embeddings = True
            for c in chunks:
                if c.embedding:
                    extracted_embeddings.append(c.embedding)
                else:
                    all_have_embeddings = False
                    break

            if all_have_embeddings and extracted_embeddings:
                final_embeddings = extracted_embeddings

        try:
            total_chunks = len(chunks)

            # Process in batches if needed
            if total_chunks <= self.MAX_BATCH_SIZE:
                # Small enough for single batch
                self._upsert_batch(chunks, final_embeddings)
            else:
                # Need to batch
                logger.info(f"Batching {total_chunks} chunks into batches of {self.MAX_BATCH_SIZE}")
                for i in range(0, total_chunks, self.MAX_BATCH_SIZE):
                    batch_end = min(i + self.MAX_BATCH_SIZE, total_chunks)
                    batch_chunks = chunks[i:batch_end]
                    batch_embeddings = final_embeddings[i:batch_end] if final_embeddings else None

                    logger.info(f"Processing batch {i // self.MAX_BATCH_SIZE + 1}: chunks {i} to {batch_end}")
                    self._upsert_batch(batch_chunks, batch_embeddings)

            logger.info(f"Upserted {total_chunks} chunks to ChromaDB")

        except Exception as e:
            logger.error(f"Error adding chunks to ChromaDB: {e}")
            raise

    def _upsert_batch(self, chunks: List[Chunk], embeddings: Optional[List[List[float]]] = None):
        """
        Upsert a single batch of chunks to ChromaDB.

        Args:
            chunks: List of chunks (must be <= MAX_BATCH_SIZE)
            embeddings: Optional list of embeddings matching chunks
        """
        ids = [c.id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [c.metadata for c in chunks]

        if embeddings:
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )
        else:
            # Fallback: let Chroma embed (if configured, but we prefer our own)
            logger.warning("Adding chunks without explicit embeddings. ChromaDB default embedder might be used.")
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )

    def query(self, query_embedding: List[float], n_results: int = 5) -> Dict[str, Any]:
        """
        Query the vector store.
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            return results
        except Exception as e:
            logger.error(f"Error querying ChromaDB: {e}")
            return {}

    def count(self) -> int:
        return self.collection.count()

    def delete_by_ids(self, ids: List[str]) -> int:
        """
        Delete vectors by their IDs.

        Args:
            ids: List of vector IDs to delete

        Returns:
            int: Number of vectors deleted
        """
        if not ids:
            return 0

        try:
            # Get count before deletion
            before_count = self.collection.count()

            # Delete by IDs
            self.collection.delete(ids=ids)

            # Get count after deletion
            after_count = self.collection.count()
            deleted = before_count - after_count

            logger.info(f"Deleted {deleted} vectors from ChromaDB")
            return deleted

        except Exception as e:
            logger.error(f"Error deleting vectors from ChromaDB: {e}")
            raise

    def delete_by_metadata(self, where: Dict[str, Any]) -> int:
        """
        Delete vectors by metadata filter.

        Args:
            where: Metadata filter (e.g., {"source": "example.com"})

        Returns:
            int: Number of vectors deleted
        """
        try:
            before_count = self.collection.count()

            # Delete by metadata filter
            self.collection.delete(where=where)

            after_count = self.collection.count()
            deleted = before_count - after_count

            logger.info(f"Deleted {deleted} vectors matching filter {where}")
            return deleted

        except Exception as e:
            logger.error(f"Error deleting vectors by metadata: {e}")
            raise

    def delete_by_doc_id(self, doc_id: str) -> int:
        """
        Delete all vectors associated with a document ID.

        Args:
            doc_id: Document ID to delete vectors for

        Returns:
            int: Number of vectors deleted
        """
        return self.delete_by_metadata({"doc_id": doc_id})

    def get_by_ids(self, ids: List[str]) -> Dict[str, Any]:
        """
        Get vectors by their IDs.

        Args:
            ids: List of vector IDs to retrieve

        Returns:
            Dict containing documents, metadatas, and ids
        """
        if not ids:
            return {"ids": [], "documents": [], "metadatas": []}

        try:
            results = self.collection.get(ids=ids)
            return results
        except Exception as e:
            logger.error(f"Error getting vectors by IDs: {e}")
            return {"ids": [], "documents": [], "metadatas": []}

    def get_all_ids(self) -> List[str]:
        """
        Get all vector IDs in the collection.

        Returns:
            List of all vector IDs
        """
        try:
            results = self.collection.get()
            return results.get("ids", [])
        except Exception as e:
            logger.error(f"Error getting all vector IDs: {e}")
            return []

    def clear_collection(self) -> int:
        """
        Clear all vectors from the collection.

        Returns:
            int: Number of vectors deleted
        """
        try:
            count = self.collection.count()
            if count > 0:
                all_ids = self.get_all_ids()
                if all_ids:
                    self.collection.delete(ids=all_ids)
            logger.info(f"Cleared {count} vectors from collection")
            return count
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            raise

    async def initialize(self):
        """Initialize method for compatibility with async patterns."""
        # ChromaDB client is already initialized in __init__
        pass


# Singleton instance
_vector_store_instance: Optional[VectorStoreService] = None


def get_vector_store() -> VectorStoreService:
    """Get or create the singleton VectorStoreService instance."""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStoreService()
    return _vector_store_instance

