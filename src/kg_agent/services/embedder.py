import os
from typing import List
from sentence_transformers import SentenceTransformer
from ..core.config import settings
from ..core.logging import logger

class EmbedderService:
    """
    Service to generate vector embeddings for text using HuggingFace models.
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
        Generate embedding for a single text string.
        """
        try:
            embedding = self.model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            return []

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.
        """
        try:
            embeddings = self.model.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error embedding batch: {e}")
            return []

