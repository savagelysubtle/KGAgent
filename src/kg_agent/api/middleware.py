"""Request/response middleware."""
import time
from fastapi import Request, Response
from ..core.logging import logger


async def add_process_time_header(request: Request, call_next):
    """
    Middleware to add processing time header to responses.

    Args:
        request: FastAPI request object
        call_next: Next middleware callable

    Returns:
        Response with added processing time header
    """
    start_time = time.time()

    try:
        response = await call_next(request)
        process_time = time.time() - start_time

        # Add processing time header
        response.headers["X-Process-Time"] = str(process_time)

        # Log request details
        logger.info(
            f"Request: {request.method} {request.url.path} "
            f"Status: {response.status_code} "
            f"Time: {process_time:.3f}s"
        )

        return response

    except Exception as exc:
        process_time = time.time() - start_time
        logger.error(
            f"Request failed: {request.method} {request.url.path} "
            f"Time: {process_time:.3f}s Error: {str(exc)}"
        )
        raise
