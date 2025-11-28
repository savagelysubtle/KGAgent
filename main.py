"""Main entry point for KG Agent."""

import os
import uvicorn

from src.kg_agent.core.config import settings
from src.kg_agent.core.logging import logger


def main():
    """Run the KG Agent API server."""
    logger.info("Starting KG Agent API server")

    # Number of workers - use env var or default to 4 for better concurrency
    # More workers = better concurrent request handling
    # But each worker loads the embedding model, so balance memory usage
    workers = int(os.environ.get("UVICORN_WORKERS", "4"))

    # Disable auto-reload - watchfiles detects constant changes from ChromaDB/SQLite
    # For development, manually restart the server when code changes
    uvicorn.run(
        "src.kg_agent.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=workers,  # Multiple workers for concurrent request handling
        log_level=settings.LOG_LEVEL.lower(),
    )


if __name__ == "__main__":
    main()
