"""API authentication and security utilities."""
import secrets
from typing import Optional
from fastapi import Depends, HTTPException, status, Header

from .config import settings


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
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    return x_api_key


def generate_api_key() -> str:
    """
    Generate a new API key.

    Returns:
        A new randomly generated API key
    """
    return secrets.token_urlsafe(32)


def hash_api_key(api_key: str) -> str:
    """
    Hash an API key for storage.

    Args:
        api_key: The API key to hash

    Returns:
        Hashed version of the API key
    """
    # For production, use proper password hashing
    # For now, just return as-is (not recommended for production)
    return api_key


def validate_request_origin(origin: str) -> bool:
    """
    Validate request origin against allowed CORS origins.

    Args:
        origin: Request origin header

    Returns:
        True if origin is allowed, False otherwise
    """
    if "*" in settings.CORS_ORIGINS:
        return True

    return origin in settings.CORS_ORIGINS
