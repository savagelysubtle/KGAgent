"""
FastAPI application for web crawling API.
"""
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import time
from contextlib import asynccontextmanager

from ..core.config import settings
from ..core.logging import logger
from .routes import crawl, health, session, upload, graph, stats, agent, documents, chat, reprocess
from .middleware import add_process_time_header


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager."""
    logger.info("Starting KG Agent Crawler API")
    yield
    logger.info("Shutting down KG Agent Crawler API")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    description="Web-to-Knowledge-Graph ETL Pipeline with Crawl4AI",
    version=settings.APP_VERSION,
    debug=settings.DEBUG,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

# Add custom middleware
app.middleware("http")(add_process_time_header)


# Include routers
app.include_router(
    crawl.router,
    prefix=settings.API_V1_PREFIX,
    tags=["crawling"]
)

app.include_router(
    graph.router,
    prefix=f"{settings.API_V1_PREFIX}/graph",
    tags=["graph"]
)

app.include_router(
    health.router,
    prefix=settings.API_V1_PREFIX,
    tags=["health"]
)

app.include_router(
    session.router,
    prefix=f"{settings.API_V1_PREFIX}/session",
    tags=["session"]
)

app.include_router(
    upload.router,
    prefix=settings.API_V1_PREFIX,
    tags=["upload"]
)

app.include_router(
    stats.router,
    prefix=f"{settings.API_V1_PREFIX}/stats",
    tags=["stats"]
)

app.include_router(
    agent.router,
    prefix=f"{settings.API_V1_PREFIX}/agent",
    tags=["agent"]
)

app.include_router(
    documents.router,
    prefix=f"{settings.API_V1_PREFIX}/documents",
    tags=["documents"]
)

app.include_router(
    chat.router,
    prefix=f"{settings.API_V1_PREFIX}/chat",
    tags=["chat"]
)

app.include_router(
    reprocess.router,
    prefix=f"{settings.API_V1_PREFIX}/reprocess",
    tags=["reprocess"]
)


# Health check endpoint
@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": settings.APP_VERSION
    }


# Proxy /v1/models to LM Studio (stops 404 noise from CopilotKit)
@app.get("/v1/models", tags=["llm-proxy"])
async def proxy_models():
    """Proxy models endpoint to LM Studio."""
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.LLM_BASE_URL}/models",
                timeout=5.0
            )
            return response.json()
    except Exception:
        return {"data": [], "object": "list"}


# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "description": "Web-to-Knowledge-Graph ETL Pipeline with Crawl4AI",
        "docs": "/docs",
        "health": "/health"
    }


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "type": "internal_error"
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "kg_agent.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
