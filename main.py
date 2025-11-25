"""Main entry point for KG Agent."""
import uvicorn
from src.kg_agent.core.config import settings
from src.kg_agent.core.logging import logger


def main():
    """Run the KG Agent API server."""
    logger.info("Starting KG Agent API server")

    # Directories to exclude from file watching (prevents reload loops)
    watch_dirs_exclude = [
        "data",
        ".venv",
        "__pycache__",
        ".git",
        "node_modules",
        "dashboard",
        "*.db",
        "*.sqlite",
        "*.log",
    ]

    uvicorn.run(
        "src.kg_agent.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        reload_excludes=watch_dirs_exclude if settings.DEBUG else None,
        log_level=settings.LOG_LEVEL.lower()
    )


if __name__ == "__main__":
    main()
