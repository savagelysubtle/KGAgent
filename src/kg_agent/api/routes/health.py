"""Health check endpoints."""
from fastapi import APIRouter, Depends
from typing import Dict, Any
import psutil
import time

from ...core.config import settings
from ...core.security import verify_api_key
from ...crawler.cache import cache


router = APIRouter()


@router.get("/health", summary="Basic health check")
async def health_check() -> Dict[str, Any]:
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": settings.APP_VERSION,
        "service": settings.APP_NAME
    }


@router.get("/health/detailed", summary="Detailed health check")
async def detailed_health_check(
    api_key: str = Depends(verify_api_key)
) -> Dict[str, Any]:
    """Detailed health check with system metrics."""
    # System metrics
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')

    # Cache stats
    cache_stats = cache.get_stats()

    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": settings.APP_VERSION,
        "service": settings.APP_NAME,
        "system": {
            "memory_percent": memory.percent,
            "cpu_percent": psutil.cpu_percent(interval=1),
            "disk_percent": disk.percent
        },
        "cache": cache_stats,
        "configuration": {
            "debug": settings.DEBUG,
            "max_concurrent": settings.CRAWLER_MAX_CONCURRENT,
            "timeout": settings.CRAWLER_TIMEOUT
        }
    }


@router.get("/health/ready", summary="Readiness check")
async def readiness_check() -> Dict[str, Any]:
    """Readiness check for load balancers."""
    # Add more comprehensive checks here (database, redis, etc.)
    return {
        "status": "ready",
        "timestamp": time.time()
    }
