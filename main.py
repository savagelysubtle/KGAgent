"""Main entry point for KG Agent."""

import uvicorn

from src.kg_agent.core.config import settings
from src.kg_agent.core.logging import logger


def main():
    """Run the KG Agent API server."""
    logger.info("Starting KG Agent API server")

    # Disable auto-reload - watchfiles detects constant changes from ChromaDB/SQLite
    # For development, manually restart the server when code changes
    uvicorn.run(
        "src.kg_agent.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level=settings.LOG_LEVEL.lower(),
    )


if __name__ == "__main__":
    main()
