"""Performance optimizations for the multi-agent system.

Provides:
- Query caching for common operations
- State emission optimization
- Utility functions for efficiency
"""

import hashlib
import json
import time
from typing import Any, Optional

from ...core.logging import logger


class QueryCache:
    """Simple cache for common queries with TTL support.

    Useful for caching:
    - Database stats (don't change frequently)
    - Recent search results
    - User profile lookups
    """

    def __init__(self, max_size: int = 100, ttl_seconds: int = 300):
        """Initialize the cache.

        Args:
            max_size: Maximum number of cached items
            ttl_seconds: Time-to-live for cached items (default 5 minutes)
        """
        self._cache: dict[str, tuple[Any, float]] = {}
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._hits = 0
        self._misses = 0

    def _hash_query(self, query: str, params: Optional[dict] = None) -> str:
        """Create cache key from query and params."""
        content = json.dumps({"q": query, "p": params or {}}, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, query: str, params: Optional[dict] = None) -> Optional[Any]:
        """Get cached result if valid.

        Args:
            query: The query string
            params: Optional parameters dict

        Returns:
            Cached value if valid, None otherwise
        """
        key = self._hash_query(query, params)
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < self._ttl:
                self._hits += 1
                logger.debug(f"Cache hit for query: {query[:50]}")
                return value
            else:
                # Expired - remove it
                del self._cache[key]

        self._misses += 1
        return None

    def set(self, query: str, value: Any, params: Optional[dict] = None) -> None:
        """Cache a result.

        Args:
            query: The query string
            value: Value to cache
            params: Optional parameters dict
        """
        # Evict oldest if at capacity
        if len(self._cache) >= self._max_size:
            oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]
            logger.debug(f"Cache evicted oldest entry, size: {len(self._cache)}")

        key = self._hash_query(query, params)
        self._cache[key] = (value, time.time())

    def invalidate(self, query: str, params: Optional[dict] = None) -> bool:
        """Invalidate a specific cache entry.

        Args:
            query: The query string
            params: Optional parameters dict

        Returns:
            True if entry was found and removed
        """
        key = self._hash_query(query, params)
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> int:
        """Clear all cached entries.

        Returns:
            Number of entries cleared
        """
        count = len(self._cache)
        self._cache.clear()
        logger.info(f"Cache cleared: {count} entries removed")
        return count

    @property
    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0

        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "ttl_seconds": self._ttl,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_percent": round(hit_rate, 2),
        }


# Global cache instances
_query_cache: Optional[QueryCache] = None
_stats_cache: Optional[QueryCache] = None


def get_query_cache() -> QueryCache:
    """Get the global query cache for general queries."""
    global _query_cache
    if _query_cache is None:
        _query_cache = QueryCache(max_size=100, ttl_seconds=300)
    return _query_cache


def get_stats_cache() -> QueryCache:
    """Get the stats cache with longer TTL (stats don't change often)."""
    global _stats_cache
    if _stats_cache is None:
        _stats_cache = QueryCache(max_size=20, ttl_seconds=600)
    return _stats_cache


# === State Optimization ===


def optimize_thinking_steps(steps: list[dict], max_steps: int = 20) -> list[dict]:
    """Optimize thinking steps for state emission.

    Keeps only the most recent steps to avoid large state payloads.

    Args:
        steps: List of thinking step dicts
        max_steps: Maximum number of steps to keep

    Returns:
        Trimmed list of steps
    """
    if len(steps) <= max_steps:
        return steps

    # Keep first step (context) and last N-1 steps (recent activity)
    return [steps[0]] + steps[-(max_steps - 1) :]


def should_emit_state(
    current_state: dict,
    previous_state: Optional[dict],
    force_fields: Optional[set[str]] = None,
) -> bool:
    """Determine if state should be emitted (has meaningful changes).

    Args:
        current_state: Current agent state
        previous_state: Previous emitted state
        force_fields: Fields that always trigger emission if changed

    Returns:
        True if state should be emitted
    """
    if previous_state is None:
        return True

    # Always emit if these fields changed
    force_fields = force_fields or {
        "current_agent",
        "should_end",
        "final_response",
        "last_error",
    }

    for field in force_fields:
        if current_state.get(field) != previous_state.get(field):
            return True

    # Check if thinking_steps grew
    current_steps = len(current_state.get("thinking_steps", []))
    previous_steps = len(previous_state.get("thinking_steps", []))
    if current_steps > previous_steps:
        return True

    return False


def create_state_diff(
    current_state: dict,
    previous_state: Optional[dict],
) -> dict:
    """Create a diff of state changes for efficient emission.

    Instead of sending the full state, only send changed fields.

    Args:
        current_state: Current agent state
        previous_state: Previous emitted state

    Returns:
        Dict containing only changed fields
    """
    if previous_state is None:
        return current_state

    diff = {}
    for key, value in current_state.items():
        if key not in previous_state or previous_state[key] != value:
            diff[key] = value

    return diff


# === Async Optimization ===


async def batch_tool_calls(
    tools: list[tuple[callable, tuple, dict]],
    max_concurrent: int = 3,
) -> list[Any]:
    """Execute multiple tool calls with controlled concurrency.

    Args:
        tools: List of (func, args, kwargs) tuples
        max_concurrent: Maximum concurrent executions

    Returns:
        List of results in same order as input
    """
    import asyncio

    semaphore = asyncio.Semaphore(max_concurrent)
    results = [None] * len(tools)

    async def run_with_semaphore(index: int, func, args, kwargs):
        async with semaphore:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            results[index] = result

    tasks = [
        run_with_semaphore(i, func, args, kwargs)
        for i, (func, args, kwargs) in enumerate(tools)
    ]

    await asyncio.gather(*tasks, return_exceptions=True)
    return results
