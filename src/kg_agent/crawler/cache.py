"""Caching layer for crawled content."""
import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import asyncio
from crawl4ai import CacheMode

from ..core.config import settings
from ..core.logging import logger


class CrawlCache:
    """
    Intelligent caching layer for crawled content.

    Features:
    - Content-based caching with TTL
    - Metadata tracking
    - Cache invalidation strategies
    - Memory-efficient storage
    """

    def __init__(self, cache_dir: Optional[str] = None, ttl_hours: int = 24):
        self.cache_dir = Path(cache_dir or settings.CRAWLER_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_hours = ttl_hours
        self._lock = asyncio.Lock()

    def _get_cache_key(self, url: str, config_hash: str) -> str:
        """Generate cache key from URL and configuration."""
        content = f"{url}:{config_hash}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _get_config_hash(self, config: Dict[str, Any]) -> str:
        """Generate hash of crawl configuration."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()

    async def get(self, url: str, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached crawl result if available and not expired.

        Args:
            url: Target URL
            config: Crawl configuration

        Returns:
            Cached result or None if not found/expired
        """
        config_hash = self._get_config_hash(config)
        cache_key = self._get_cache_key(url, config_hash)
        cache_file = self.cache_dir / f"{cache_key}.json"

        if not cache_file.exists():
            return None

        try:
            async with self._lock:
                cache_data = json.loads(cache_file.read_text())

            # Check TTL
            cached_at = datetime.fromisoformat(cache_data["cached_at"])
            if datetime.utcnow() - cached_at > timedelta(hours=self.ttl_hours):
                logger.debug(f"Cache expired for {url}")
                await self.invalidate(url, config)
                return None

            logger.debug(f"Cache hit for {url}")
            return cache_data["result"]

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Invalid cache file {cache_file}: {e}")
            await self.invalidate(url, config)
            return None

    async def set(self, url: str, config: Dict[str, Any], result: Dict[str, Any]):
        """
        Store crawl result in cache.

        Args:
            url: Target URL
            config: Crawl configuration
            result: Crawl result to cache
        """
        config_hash = self._get_config_hash(config)
        cache_key = self._get_cache_key(url, config_hash)
        cache_file = self.cache_dir / f"{cache_key}.json"

        cache_data = {
            "url": url,
            "config_hash": config_hash,
            "cached_at": datetime.utcnow().isoformat() + "Z",
            "result": result
        }

        try:
            async with self._lock:
                cache_file.write_text(json.dumps(cache_data, indent=2))
            logger.debug(f"Cached result for {url}")
        except Exception as e:
            logger.error(f"Failed to cache result for {url}: {e}")

    async def invalidate(self, url: str, config: Dict[str, Any]):
        """
        Remove cached result for URL and configuration.

        Args:
            url: Target URL
            config: Crawl configuration
        """
        config_hash = self._get_config_hash(config)
        cache_key = self._get_cache_key(url, config_hash)
        cache_file = self.cache_dir / f"{cache_key}.json"

        try:
            if cache_file.exists():
                cache_file.unlink()
                logger.debug(f"Invalidated cache for {url}")
        except Exception as e:
            logger.error(f"Failed to invalidate cache for {url}: {e}")

    async def clear_expired(self):
        """Remove all expired cache entries."""
        logger.info("Clearing expired cache entries")

        expired_files = []
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_data = json.loads(cache_file.read_text())
                cached_at = datetime.fromisoformat(cache_data["cached_at"])

                if datetime.utcnow() - cached_at > timedelta(hours=self.ttl_hours):
                    expired_files.append(cache_file)
            except Exception as e:
                logger.warning(f"Invalid cache file {cache_file}: {e}")
                expired_files.append(cache_file)

        for cache_file in expired_files:
            try:
                cache_file.unlink()
                logger.debug(f"Removed expired cache file: {cache_file}")
            except Exception as e:
                logger.error(f"Failed to remove cache file {cache_file}: {e}")

        logger.info(f"Cleared {len(expired_files)} expired cache entries")

    async def clear_all(self):
        """Remove all cache entries."""
        logger.info("Clearing all cache entries")

        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
                count += 1
            except Exception as e:
                logger.error(f"Failed to remove cache file {cache_file}: {e}")

        logger.info(f"Cleared {count} cache entries")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_files = len(list(self.cache_dir.glob("*.json")))
        total_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.json"))

        return {
            "total_entries": total_files,
            "total_size_bytes": total_size,
            "cache_dir": str(self.cache_dir),
            "ttl_hours": self.ttl_hours
        }


# Global cache instance
cache = CrawlCache()
