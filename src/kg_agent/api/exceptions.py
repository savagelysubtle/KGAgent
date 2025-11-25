"""Custom exception handlers."""
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from typing import Dict, Any

from ..core.logging import logger


class CrawlerException(Exception):
    """Base exception for crawler-related errors."""

    def __init__(self, message: str, status_code: int = 500, details: Dict[str, Any] = None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class CrawlTimeoutException(CrawlerException):
    """Exception raised when crawling times out."""

    def __init__(self, url: str, timeout: int):
        super().__init__(
            f"Crawl timeout for URL: {url}",
            status_code=408,
            details={"url": url, "timeout": timeout}
        )


class InvalidURLError(CrawlerException):
    """Exception raised for invalid URLs."""

    def __init__(self, url: str):
        super().__init__(
            f"Invalid URL: {url}",
            status_code=400,
            details={"url": url}
        )


class RateLimitException(CrawlerException):
    """Exception raised when rate limit is exceeded."""

    def __init__(self, retry_after: int = 60):
        super().__init__(
            "Rate limit exceeded",
            status_code=429,
            details={"retry_after": retry_after}
        )


async def crawler_exception_handler(request: Request, exc: CrawlerException):
    """Handle crawler-specific exceptions."""
    logger.error(f"Crawler error: {exc.message}", extra=exc.details)

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.message,
            "type": "crawler_error",
            "details": exc.details
        }
    )


async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with proper logging."""
    logger.warning(
        f"HTTP error: {exc.status_code} - {exc.detail}",
        extra={"path": str(request.url), "method": request.method}
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "type": "http_error"
        }
    )


# Exception handler registry
exception_handlers = {
    CrawlerException: crawler_exception_handler,
    CrawlTimeoutException: crawler_exception_handler,
    InvalidURLError: crawler_exception_handler,
    RateLimitException: crawler_exception_handler,
    HTTPException: http_exception_handler,
}
