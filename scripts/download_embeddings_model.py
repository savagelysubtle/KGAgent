#!/usr/bin/env python3
"""
Download HuggingFace embedding models to local directory.
This script downloads the specified embedding model to models/embeddings/ for offline use.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from kg_agent.core.config import settings
from kg_agent.core.logging import logger

def download_model():
    """Download the configured embedding model to local directory."""
    model_name = settings.HF_EMBEDDING_MODEL
    models_dir = Path("./models/embeddings")

    # Create models directory if it doesn't exist
    models_dir.mkdir(parents=True, exist_ok=True)

    # Set HF_HOME to our models directory
    os.environ['HF_HOME'] = str(models_dir)

    logger.info(f"Downloading embedding model: {model_name}")
    logger.info(f"Model will be saved to: {models_dir}")

    try:
        # Download the model using sentence-transformers (recommended for embedding models)
        logger.info("Downloading with SentenceTransformer...")
        model = SentenceTransformer(model_name)

        # Test the model with a simple embedding
        test_text = "This is a test to verify the model works."
        embedding = model.encode(test_text)

        logger.info(f"Model downloaded successfully!")
        logger.info(f"Model type: {type(model)}")
        logger.info(f"Embedding dimension: {len(embedding)}")
        logger.info(f"Test embedding shape: {embedding.shape}")

        # Also download the base transformers components (optional but good practice)
        logger.info("Downloading base transformers components...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        base_model = AutoModel.from_pretrained(model_name)

        logger.info("All components downloaded successfully!")
        logger.info(f"Total model size: ~{get_model_size(models_dir)} MB")

        return True

    except Exception as e:
        logger.error(f"Failed to download model {model_name}: {e}")
        return False

def get_model_size(directory: Path) -> float:
    """Calculate the total size of the model directory in MB."""
    total_size = 0
    for file_path in directory.rglob('*'):
        if file_path.is_file():
            total_size += file_path.stat().st_size

    return round(total_size / (1024 * 1024), 2)

if __name__ == "__main__":
    success = download_model()
    if success:
        print("✅ Model download completed successfully!")
        sys.exit(0)
    else:
        print("❌ Model download failed!")
        sys.exit(1)
