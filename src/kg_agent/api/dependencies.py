"""FastAPI dependencies."""
from fastapi import Depends, HTTPException, status, Header
import secrets

from ..core.config import settings
from ..core.logging import logger


def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    """
    Verify API key from X-API-Key header.

    Args:
        x_api_key: API key from X-API-Key header

    Returns:
        API key if valid

    Raises:
        HTTPException: If API key is invalid
    """
    if not secrets.compare_digest(x_api_key, settings.API_KEY_SECRET):
        logger.warning(f"Invalid API key attempt: {x_api_key[:8]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    return x_api_key
