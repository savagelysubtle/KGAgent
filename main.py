"""Main entry point for KG Agent."""

import os
import uvicorn

from src.kg_agent.core.config import settings
from src.kg_agent.core.logging import logger


def main():
    """Run the KG Agent API server."""
    logger.info("Starting KG Agent API server")

    # Dev mode: enable auto-reload (set DEV_MODE=1 or UVICORN_RELOAD=1)
    dev_mode = os.environ.get("DEV_MODE", "0") == "1" or os.environ.get("UVICORN_RELOAD", "0") == "1"

    if dev_mode:
        # Dev mode: single worker with auto-reload (uvicorn requires workers=1 for reload)
        logger.info("Running in DEV MODE with auto-reload enabled")
        uvicorn.run(
            "src.kg_agent.api.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            reload_dirs=["src"],  # Only watch source code
            reload_excludes=[
                "*.db",
                "*.sqlite",
                "*.sqlite3",
                "*.log",
                "*.pyc",
                "__pycache__",
                "data/*",
                "storage/*",
                "cache/*",
                "logs/*",
                "models/*",
                ".git/*",
            ],
            log_level=settings.LOG_LEVEL.lower(),
        )
    else:
        # Production mode: multiple workers, no reload
        workers = int(os.environ.get("UVICORN_WORKERS", "4"))
        logger.info(f"Running in PRODUCTION MODE with {workers} workers")
        uvicorn.run(
            "src.kg_agent.api.main:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            workers=workers,
            log_level=settings.LOG_LEVEL.lower(),
        )


if __name__ == "__main__":
    main()
