"""Retry decorators and utilities."""
import asyncio
import functools
from typing import Callable, Any, Optional
import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

from ..core.logging import logger


def retry_async(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 60.0,
    backoff_multiplier: float = 2.0,
    retry_exceptions: Optional[tuple] = None
):
    """
    Decorator for retrying async functions with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        min_wait: Minimum wait time between retries (seconds)
        max_wait: Maximum wait time between retries (seconds)
        backoff_multiplier: Exponential backoff multiplier
        retry_exceptions: Tuple of exception types to retry on (default: all exceptions)
    """
    if retry_exceptions is None:
        retry_exceptions = (Exception,)

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            retry_config = retry(
                stop=stop_after_attempt(max_attempts),
                wait=wait_exponential(
                    multiplier=backoff_multiplier,
                    min=min_wait,
                    max=max_wait
                ),
                retry=retry_if_exception_type(retry_exceptions),
                before_sleep=before_sleep_log(logger, logging.WARNING),
                reraise=True
            )

            async def async_func():
                return await func(*args, **kwargs)

            return await retry_config(async_func)()

        return wrapper
    return decorator


def retry_sync(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 60.0,
    backoff_multiplier: float = 2.0,
    retry_exceptions: Optional[tuple] = None
):
    """
    Decorator for retrying sync functions with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        min_wait: Minimum wait time between retries (seconds)
        max_wait: Maximum wait time between retries (seconds)
        backoff_multiplier: Exponential backoff multiplier
        retry_exceptions: Tuple of exception types to retry on (default: all exceptions)
    """
    if retry_exceptions is None:
        retry_exceptions = (Exception,)

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            retry_config = retry(
                stop=stop_after_attempt(max_attempts),
                wait=wait_exponential(
                    multiplier=backoff_multiplier,
                    min=min_wait,
                    max=max_wait
                ),
                retry=retry_if_exception_type(retry_exceptions),
                before_sleep=before_sleep_log(logger, logging.WARNING),
                reraise=True
            )

            return retry_config(func)(*args, **kwargs)

        return wrapper
    return decorator


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_attempts: int = 3,
        min_wait: float = 1.0,
        max_wait: float = 60.0,
        backoff_multiplier: float = 2.0,
        retry_exceptions: Optional[tuple] = None
    ):
        self.max_attempts = max_attempts
        self.min_wait = min_wait
        self.max_wait = max_wait
        self.backoff_multiplier = backoff_multiplier
        self.retry_exceptions = retry_exceptions or (Exception,)
