import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from ..core.config import settings
from ..core.logging import logger


# Global thread pool for CPU-bound embedding operations
# This prevents blocking the async event loop
_embedding_executor: Optional[ThreadPoolExecutor] = None


def get_embedding_executor() -> ThreadPoolExecutor:
    """Get or create the embedding thread pool executor."""
    global _embedding_executor
    if _embedding_executor is None:
        # Use 2 threads - embeddings are GPU/CPU bound, more threads won't help much
        _embedding_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="embedder")
        logger.info("Created embedding ThreadPoolExecutor with 2 workers")
    return _embedding_executor


class EmbedderService:
    """
    Service to generate vector embeddings for text using HuggingFace models.
    Supports both sync and async operations for better concurrency.
    """

    def __init__(self, model_name: str = settings.HF_EMBEDDING_MODEL):
        logger.info(f"Initializing EmbedderService with model: {model_name}")
        logger.info(f"Using HF_HOME: {settings.HF_HOME}")

        # Set HF_HOME to use local model cache
        os.environ['HF_HOME'] = settings.HF_HOME

        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            logger.error(f"Failed to load embedding model {model_name}: {e}")
            raise

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text string (synchronous).
        """
        try:
            embedding = self.model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            return []

    def embed_batch(self, texts: List[str], show_progress: bool = True) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts (synchronous).
        """
        try:
            embeddings = self.model.encode(texts, show_progress_bar=show_progress)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error embedding batch: {e}")
            return []

    async def embed_text_async(self, text: str) -> List[float]:
        """
        Generate embedding for a single text string (async - runs in thread pool).
        Use this in async contexts to avoid blocking the event loop.
        """
        loop = asyncio.get_event_loop()
        executor = get_embedding_executor()
        try:
            embedding = await loop.run_in_executor(
                executor,
                lambda: self.model.encode(text).tolist()
            )
            return embedding
        except Exception as e:
            logger.error(f"Error embedding text async: {e}")
            return []

    async def embed_batch_async(self, texts: List[str], show_progress: bool = True) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts (async - runs in thread pool).
        Use this in async contexts to avoid blocking the event loop.
        """
        loop = asyncio.get_event_loop()
        executor = get_embedding_executor()
        try:
            embeddings = await loop.run_in_executor(
                executor,
                lambda: self.model.encode(texts, show_progress_bar=show_progress).tolist()
            )
            return embeddings
        except Exception as e:
            logger.error(f"Error embedding batch async: {e}")
            return []

