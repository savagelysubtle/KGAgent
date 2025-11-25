"""
Crawl4AI service wrapper providing async web crawling capabilities.
"""
from typing import List, Optional, Dict, Any
import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.extraction_strategy import LLMExtractionStrategy, CosineStrategy
from crawl4ai.chunking_strategy import RegexChunking, SlidingWindowChunking

from ..models.crawl_result import CrawlResult, CrawlMetadata
from ..core.config import settings
from ..core.logging import logger


class CrawlerService:
    """
    Main crawler service using Crawl4AI for async web content extraction.

    Features:
    - Async crawling with configurable concurrency
    - Session management for authenticated sites
    - Custom extraction strategies
    - Intelligent caching
    - Proxy rotation support
    """

    def __init__(
        self,
        max_concurrent: int = 5,
        headless: bool = True,
        proxy_config: Optional[Dict[str, str]] = None
    ):
        self.max_concurrent = max_concurrent
        self.browser_config = BrowserConfig(
            headless=headless,
            viewport_width=1920,
            viewport_height=1080,
            browser_type="chromium"  # chromium, firefox, webkit
        )
        self.proxy_config = proxy_config
        self._crawler: Optional[AsyncWebCrawler] = None

    async def __aenter__(self):
        """Async context manager entry."""
        self._crawler = AsyncWebCrawler(config=self.browser_config)
        await self._crawler.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._crawler:
            await self._crawler.__aexit__(exc_type, exc_val, exc_tb)

    async def crawl_single(
        self,
        url: str,
        wait_for: str = "domcontentloaded",
        word_count_threshold: int = 10,
        cache_mode: CacheMode = CacheMode.ENABLED,
        extraction_strategy: Optional[str] = None,
        screenshot: bool = False,
        pdf: bool = False
    ) -> CrawlResult:
        """
        Crawl a single URL with configurable options.

        Args:
            url: Target URL to crawl
            wait_for: Wait strategy ('networkidle', 'domcontentloaded', 'load')
            word_count_threshold: Minimum word count for content blocks
            cache_mode: Caching strategy (ENABLED, DISABLED, BYPASS)
            extraction_strategy: Custom extraction strategy name
            screenshot: Capture screenshot
            pdf: Generate PDF

        Returns:
            CrawlResult with extracted content and metadata
        """
        try:
            logger.info(f"Crawling URL: {url}")

            # Configure run parameters
            run_config = CrawlerRunConfig(
                word_count_threshold=word_count_threshold,
                wait_for=wait_for,
                cache_mode=cache_mode,
                screenshot=screenshot,
                pdf=pdf,
                verbose=settings.LOG_LEVEL == "DEBUG"
            )

            # Apply extraction strategy if specified
            if extraction_strategy:
                run_config.extraction_strategy = self._get_extraction_strategy(
                    extraction_strategy
                )

            # Execute crawl
            result = await self._crawler.arun(url=url, config=run_config)

            # Determine actual success based on having content
            has_content = bool(result.html or result.markdown)
            actual_success = result.success and has_content

            # Transform to internal model
            metadata_dict = result.metadata or {}
            crawl_result = CrawlResult(
                url=url,
                success=actual_success,
                html=result.html,
                markdown=result.markdown,
                cleaned_html=result.cleaned_html,
                media=result.media,
                links=result.links,
                metadata=CrawlMetadata(
                    title=metadata_dict.get("title", ""),
                    description=metadata_dict.get("description", ""),
                    keywords=metadata_dict.get("keywords", []),
                    author=metadata_dict.get("author", ""),
                    published_date=metadata_dict.get("published_date"),
                    language=metadata_dict.get("language", "en"),
                    word_count=len(result.markdown.split()) if result.markdown else 0,
                    crawl_timestamp=metadata_dict.get("crawl_timestamp")
                ),
                screenshot_path=result.screenshot if screenshot else None,
                pdf_path=result.pdf if pdf else None,
                error_message=result.error_message if not actual_success else None
            )

            logger.info(f"Successfully crawled: {url}")
            return crawl_result

        except Exception as e:
            logger.error(f"Error crawling {url}: {str(e)}")
            return CrawlResult(
                url=url,
                success=False,
                error_message=str(e)
            )

    async def crawl_batch(
        self,
        urls: List[str],
        max_concurrent: Optional[int] = None,
        **kwargs
    ) -> List[CrawlResult]:
        """
        Crawl multiple URLs concurrently with memory-adaptive dispatching.

        Args:
            urls: List of URLs to crawl
            max_concurrent: Override default concurrency limit
            **kwargs: Additional arguments passed to crawl_single

        Returns:
            List of CrawlResult objects
        """
        max_concurrent = max_concurrent or self.max_concurrent
        logger.info(f"Starting batch crawl of {len(urls)} URLs with concurrency={max_concurrent}")

        semaphore = asyncio.Semaphore(max_concurrent)

        async def crawl_with_semaphore(url: str) -> CrawlResult:
            async with semaphore:
                return await self.crawl_single(url, **kwargs)

        results = await asyncio.gather(
            *[crawl_with_semaphore(url) for url in urls],
            return_exceptions=True
        )

        # Handle exceptions
        crawl_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Exception crawling {urls[i]}: {str(result)}")
                crawl_results.append(CrawlResult(
                    url=urls[i],
                    success=False,
                    error_message=str(result)
                ))
            else:
                logger.debug(f"Result for {urls[i]}: success={result.success}, has_html={bool(result.html)}")
                crawl_results.append(result)

        success_count = sum(1 for r in crawl_results if r.success)
        logger.info(f"Batch crawl complete: {success_count}/{len(urls)} successful")

        return crawl_results

    async def deep_crawl(
        self,
        start_url: str,
        max_depth: int = 2,
        same_domain_only: bool = True,
        url_filter_pattern: Optional[str] = None,
        max_pages: int = 100
    ) -> List[CrawlResult]:
        """
        Perform deep crawling starting from a URL and following links.

        Args:
            start_url: Starting URL
            max_depth: Maximum link depth to follow
            same_domain_only: Only crawl URLs from same domain
            url_filter_pattern: Regex pattern to filter URLs
            max_pages: Maximum pages to crawl

        Returns:
            List of CrawlResult objects from all crawled pages
        """
        from urllib.parse import urlparse
        import re

        logger.info(f"Starting deep crawl from {start_url} (max_depth={max_depth})")

        visited = set()
        results = []
        queue = [(start_url, 0)]  # (url, depth)
        start_domain = urlparse(start_url).netloc

        while queue and len(visited) < max_pages:
            url, depth = queue.pop(0)

            if url in visited or depth > max_depth:
                continue

            # Apply domain filter
            if same_domain_only and urlparse(url).netloc != start_domain:
                continue

            # Apply URL filter pattern
            if url_filter_pattern and not re.match(url_filter_pattern, url):
                continue

            visited.add(url)

            # Crawl page
            result = await self.crawl_single(url)
            results.append(result)

            # Extract and queue links if depth allows
            if result.success and depth < max_depth:
                for link in result.links.get("internal", []):
                    if link not in visited:
                        queue.append((link, depth + 1))

        logger.info(f"Deep crawl complete: crawled {len(results)} pages")
        return results

    async def crawl_with_session(
        self,
        urls: List[str],
        session_id: str,
        auth_config: Optional[Dict[str, Any]] = None
    ) -> List[CrawlResult]:
        """
        Crawl URLs using a persistent session for authenticated sites.

        Args:
            urls: List of URLs to crawl
            session_id: Unique session identifier
            auth_config: Authentication configuration (login_url, credentials, etc.)

        Returns:
            List of CrawlResult objects
        """
        logger.info(f"Starting session-based crawl with session_id={session_id}")

        # Configure session-aware browser
        session_config = BrowserConfig(
            **self.browser_config.__dict__,
            user_data_dir=f"./sessions/{session_id}"
        )

        async with AsyncWebCrawler(config=session_config) as crawler:
            # Perform authentication if config provided
            if auth_config:
                await self._authenticate_session(crawler, auth_config)

            # Crawl URLs with session
            results = []
            for url in urls:
                result = await self.crawl_single(url)
                results.append(result)

        return results

    def _get_extraction_strategy(self, strategy_name: str):
        """Get extraction strategy by name."""
        strategies = {
            "llm": LLMExtractionStrategy(),
            "cosine": CosineStrategy(),
            # Add custom strategies here
        }
        return strategies.get(strategy_name)

    def _get_user_agent(self) -> str:
        """Get user agent string."""
        return settings.CRAWLER_USER_AGENT or \
               "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

    def _get_extra_headers(self) -> Dict[str, str]:
        """Get extra HTTP headers."""
        return {
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive"
        }

    async def _authenticate_session(
        self,
        crawler: AsyncWebCrawler,
        auth_config: Dict[str, Any]
    ):
        """Perform authentication for session-based crawling."""
        # Implementation depends on auth type (form, OAuth, etc.)
        pass
