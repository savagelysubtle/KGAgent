# Web Crawler & API Implementation Plan

## Using Crawl4AI for LLM-Ready Web Content Extraction

---

## 1. Executive Summary

This plan details the implementation of the web crawling service using

**Crawl4AI** as the core crawling engine, integrated with a FastAPI-based REST

API. The crawler will provide async, LLM-ready content extraction with support

for JavaScript rendering, session management, and intelligent content

extraction.

### Key Capabilities

- **Async crawling** with configurable concurrency
- **LLM-ready Markdown output** with structured content
- **JavaScript rendering** via Playwright
- **Session persistence** for authenticated sites
- **Batch processing** with memory-adaptive dispatching
- **Deep crawling** for nested page exploration
- **Proxy support** with rotation
- **Custom extraction strategies** via hooks and filters

---

## 2. Technology Stack

| Component          | Technology | Version          | Purpose                          |

| ------------------ | ---------- | ---------------- | -------------------------------- |

| Crawler Engine     | Crawl4AI   | Latest (v0.7.6+) | Core web crawling and extraction |

| Web Framework      | FastAPI    | 0.109+           | REST API endpoints               |

| Task Queue         | Celery     | 5.3+             | Async job processing             |

| Message Broker     | Redis      | 7+               | Celery broker and result backend |

| Browser Automation | Playwright | Via Crawl4AI     | JavaScript rendering             |

| Data Models        | Pydantic   | 2.5+             | Request/response validation      |

| HTTP Client        | httpx      | 0.26+            | Async HTTP operations            |

---

## 3. Project Structure

```
src/kg_agent/
├── crawler/
│   ├── __init__.py
│   ├── service.py              # Main Crawl4AI service wrapper
│   ├── config.py               # Crawler configuration models
│   ├── strategies.py           # Custom extraction strategies
│   ├── filters.py              # Content filtering logic
│   ├── hooks.py                # Pre/post crawl hooks
│   ├── cache.py                # Caching layer for crawled content
│   └── validators.py           # URL and content validators
├── api/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app initialization
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── crawl.py            # Crawl endpoints
│   │   ├── session.py          # Session management endpoints
│   │   └── health.py           # Health check endpoints
│   ├── dependencies.py         # FastAPI dependencies
│   ├── middleware.py           # Request/response middleware
│   └── exceptions.py           # Custom exception handlers
├── models/
│   ├── __init__.py
│   ├── requests.py             # API request schemas
│   ├── responses.py            # API response schemas
│   ├── crawl_result.py         # Crawl result data models
│   └── job.py                  # Job status models
├── tasks/
│   ├── __init__.py
│   ├── celery_app.py           # Celery configuration
│   ├── crawl_tasks.py          # Async crawling tasks
│   └── batch_tasks.py          # Batch processing tasks
├── core/
│   ├── __init__.py
│   ├── config.py               # Application settings
│   ├── logging.py              # Logging configuration
│   └── security.py             # API authentication
└── utils/
    ├── __init__.py
    ├── retry.py                # Retry decorators
    ├── url_utils.py            # URL parsing and validation
    └── content_utils.py        # Content cleaning utilities
```

---

## 4. Core Service Implementation

### 4.1 Crawler Service (`crawler/service.py`)

```python
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
            browser_type="chromium",  # chromium, firefox, webkit
            use_persistent_context=True,
            user_agent=self._get_user_agent(),
            extra_headers=self._get_extra_headers()
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
        wait_for: str = "networkidle",
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

            # Transform to internal model
            crawl_result = CrawlResult(
                url=url,
                success=result.success,
                html=result.html,
                markdown=result.markdown,
                cleaned_html=result.cleaned_html,
                media=result.media,
                links=result.links,
                metadata=CrawlMetadata(
                    title=result.metadata.get("title", ""),
                    description=result.metadata.get("description", ""),
                    keywords=result.metadata.get("keywords", []),
                    author=result.metadata.get("author", ""),
                    published_date=result.metadata.get("published_date"),
                    language=result.metadata.get("language", "en"),
                    word_count=len(result.markdown.split()),
                    crawl_timestamp=result.metadata.get("crawl_timestamp")
                ),
                screenshot_path=result.screenshot if screenshot else None,
                pdf_path=result.pdf if pdf else None,
                error_message=result.error_message if not result.success else None
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
```

### 4.2 Configuration Models (`crawler/config.py`)

```python
"""
Configuration models for crawler service.
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, HttpUrl, field_validator
from enum import Enum


class WaitStrategy(str, Enum):
    """Page load wait strategies."""
    NETWORKIDLE = "networkidle"
    DOMCONTENTLOADED = "domcontentloaded"
    LOAD = "load"


class CacheMode(str, Enum):
    """Caching strategies."""
    ENABLED = "enabled"
    DISABLED = "disabled"
    BYPASS = "bypass"


class ExtractionStrategy(str, Enum):
    """Content extraction strategies."""
    DEFAULT = "default"
    LLM = "llm"
    COSINE = "cosine"
    CUSTOM = "custom"


class ProxyConfig(BaseModel):
    """Proxy configuration."""
    server: str = Field(..., description="Proxy server URL")
    username: Optional[str] = None
    password: Optional[str] = None
    bypass: Optional[List[str]] = Field(
        default_factory=list,
        description="Domains to bypass proxy"
    )


class CrawlConfig(BaseModel):
    """Configuration for single crawl operation."""
    url: HttpUrl = Field(..., description="Target URL to crawl")
    wait_for: WaitStrategy = Field(
        default=WaitStrategy.NETWORKIDLE,
        description="Wait strategy before content extraction"
    )
    word_count_threshold: int = Field(
        default=10,
        ge=1,
        description="Minimum word count for content blocks"
    )
    cache_mode: CacheMode = Field(
        default=CacheMode.ENABLED,
        description="Caching strategy"
    )
    extraction_strategy: Optional[ExtractionStrategy] = Field(
        default=ExtractionStrategy.DEFAULT,
        description="Content extraction strategy"
    )
    screenshot: bool = Field(
        default=False,
        description="Capture page screenshot"
    )
    pdf: bool = Field(
        default=False,
        description="Generate PDF of page"
    )
    timeout: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Timeout in seconds"
    )
    custom_headers: Optional[Dict[str, str]] = None
    cookies: Optional[Dict[str, str]] = None


class BatchCrawlConfig(BaseModel):
    """Configuration for batch crawl operation."""
    urls: List[HttpUrl] = Field(..., min_length=1, max_length=1000)
    max_concurrent: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum concurrent requests"
    )
    crawl_config: Optional[CrawlConfig] = Field(
        default_factory=CrawlConfig,
        description="Shared crawl configuration for all URLs"
    )
    continue_on_error: bool = Field(
        default=True,
        description="Continue batch even if some URLs fail"
    )


class DeepCrawlConfig(BaseModel):
    """Configuration for deep crawling operation."""
    start_url: HttpUrl = Field(..., description="Starting URL")
    max_depth: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Maximum link depth to follow"
    )
    same_domain_only: bool = Field(
        default=True,
        description="Only crawl URLs from same domain"
    )
    url_filter_pattern: Optional[str] = Field(
        default=None,
        description="Regex pattern to filter URLs"
    )
    max_pages: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum pages to crawl"
    )
    exclude_patterns: Optional[List[str]] = Field(
        default_factory=list,
        description="URL patterns to exclude"
    )
    include_patterns: Optional[List[str]] = Field(
        default_factory=list,
        description="URL patterns to include (if specified, only these are crawled)"
    )
    crawl_config: Optional[CrawlConfig] = None


class SessionConfig(BaseModel):
    """Configuration for session-based crawling."""
    session_id: str = Field(..., description="Unique session identifier")
    auth_type: Optional[str] = Field(
        default=None,
        description="Authentication type (form, oauth, token)"
    )
    auth_credentials: Optional[Dict[str, str]] = None
    persist: bool = Field(
        default=True,
        description="Persist session for reuse"
    )
```

---

## 5. API Endpoints

### 5.1 Crawl Routes (`api/routes/crawl.py`)

```python
"""
FastAPI routes for web crawling operations.
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status
from typing import List
import uuid

from ...crawler.service import CrawlerService
from ...crawler.config import (
    CrawlConfig, BatchCrawlConfig, DeepCrawlConfig, SessionConfig
)
from ...models.requests import (
    CrawlRequest, BatchCrawlRequest, DeepCrawlRequest, SessionCrawlRequest
)
from ...models.responses import (
    CrawlResponse, BatchCrawlResponse, JobResponse, JobStatusResponse
)
from ...models.job import JobStatus
from ...tasks.crawl_tasks import (
    crawl_single_task, crawl_batch_task, deep_crawl_task
)
from ...core.security import verify_api_key
from ...core.logging import logger


router = APIRouter()


@router.post(
    "/single",
    response_model=CrawlResponse,
    status_code=status.HTTP_200_OK,
    summary="Crawl a single URL",
    description="Synchronously crawl a single URL and return the extracted content"
)
async def crawl_single(
    request: CrawlRequest,
    api_key: str = Depends(verify_api_key)
) -> CrawlResponse:
    """
    Crawl a single URL synchronously.

    - **url**: Target URL to crawl
    - **config**: Optional crawl configuration

    Returns extracted content including:
    - HTML and Markdown versions
    - Metadata (title, description, etc.)
    - Links and media references
    - Optional screenshot/PDF
    """
    try:
        async with CrawlerService() as crawler:
            result = await crawler.crawl_single(
                url=str(request.url),
                **request.config.dict(exclude_none=True)
            )

        return CrawlResponse(
            success=result.success,
            data=result,
            message="Crawl completed successfully" if result.success else "Crawl failed"
        )

    except Exception as e:
        logger.error(f"Error in crawl_single endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Crawl failed: {str(e)}"
        )


@router.post(
    "/batch",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Crawl multiple URLs",
    description="Asynchronously crawl multiple URLs and return a job ID for status tracking"
)
async def crawl_batch(
    request: BatchCrawlRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
) -> JobResponse:
    """
    Crawl multiple URLs asynchronously.

    - **urls**: List of URLs to crawl (max 1000)
    - **config**: Optional shared crawl configuration
    - **max_concurrent**: Concurrency limit (default: 5)

    Returns a job ID for tracking the batch crawl progress.
    Use GET /crawl/job/{job_id} to check status.
    """
    job_id = str(uuid.uuid4())

    try:
        # Queue async task
        task = crawl_batch_task.delay(
            job_id=job_id,
            urls=[str(url) for url in request.urls],
            config=request.config.dict(exclude_none=True) if request.config else {}
        )

        logger.info(f"Batch crawl job created: {job_id} for {len(request.urls)} URLs")

        return JobResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            message=f"Batch crawl job created for {len(request.urls)} URLs",
            total_urls=len(request.urls)
        )

    except Exception as e:
        logger.error(f"Error creating batch crawl job: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create batch crawl job: {str(e)}"
        )


@router.post(
    "/deep",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Deep crawl from a starting URL",
    description="Recursively crawl pages following links from a starting URL"
)
async def deep_crawl(
    request: DeepCrawlRequest,
    api_key: str = Depends(verify_api_key)
) -> JobResponse:
    """
    Perform deep crawling starting from a URL.

    - **start_url**: Starting URL
    - **max_depth**: Maximum link depth (default: 2)
    - **max_pages**: Maximum pages to crawl (default: 100)
    - **same_domain_only**: Restrict to same domain (default: true)

    Returns a job ID for tracking the deep crawl progress.
    """
    job_id = str(uuid.uuid4())

    try:
        task = deep_crawl_task.delay(
            job_id=job_id,
            start_url=str(request.start_url),
            config=request.config.dict(exclude_none=True) if request.config else {}
        )

        logger.info(f"Deep crawl job created: {job_id} from {request.start_url}")

        return JobResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            message=f"Deep crawl job created from {request.start_url}"
        )

    except Exception as e:
        logger.error(f"Error creating deep crawl job: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create deep crawl job: {str(e)}"
        )


@router.get(
    "/job/{job_id}",
    response_model=JobStatusResponse,
    summary="Get crawl job status",
    description="Check the status and results of a crawl job"
)
async def get_job_status(
    job_id: str,
    api_key: str = Depends(verify_api_key)
) -> JobStatusResponse:
    """
    Get the status of a crawl job.

    Returns:
    - Job status (pending, running, completed, failed)
    - Progress information
    - Results (if completed)
    - Error details (if failed)
    """
    from celery.result import AsyncResult
    from ...tasks.celery_app import celery_app

    try:
        result = AsyncResult(job_id, app=celery_app)

        if result.ready():
            if result.successful():
                return JobStatusResponse(
                    job_id=job_id,
                    status=JobStatus.COMPLETED,
                    progress=100,
                    results=result.result,
                    message="Job completed successfully"
                )
            else:
                return JobStatusResponse(
                    job_id=job_id,
                    status=JobStatus.FAILED,
                    error=str(result.info),
                    message="Job failed"
                )
        else:
            return JobStatusResponse(
                job_id=job_id,
                status=JobStatus.RUNNING,
                progress=50,  # Could implement progress tracking
                message="Job is running"
            )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}"
        )


@router.delete(
    "/job/{job_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Cancel a crawl job",
    description="Cancel a running or pending crawl job"
)
async def cancel_job(
    job_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Cancel a running or pending crawl job."""
    from celery.result import AsyncResult
    from ...tasks.celery_app import celery_app

    try:
        result = AsyncResult(job_id, app=celery_app)
        result.revoke(terminate=True)
        logger.info(f"Job cancelled: {job_id}")

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}"
        )
```

### 5.2 API Models (`models/requests.py` & `models/responses.py`)

```python
# models/requests.py
"""API request schemas."""
from typing import Optional
from pydantic import BaseModel, HttpUrl, Field
from ..crawler.config import CrawlConfig, DeepCrawlConfig


class CrawlRequest(BaseModel):
    """Request schema for single URL crawl."""
    url: HttpUrl = Field(..., description="URL to crawl")
    config: Optional[CrawlConfig] = Field(
        default_factory=CrawlConfig,
        description="Crawl configuration"
    )


class BatchCrawlRequest(BaseModel):
    """Request schema for batch crawl."""
    urls: list[HttpUrl] = Field(..., min_length=1, max_length=1000)
    config: Optional[CrawlConfig] = None
    max_concurrent: int = Field(default=5, ge=1, le=20)


class DeepCrawlRequest(BaseModel):
    """Request schema for deep crawl."""
    start_url: HttpUrl
    config: Optional[DeepCrawlConfig] = None


# models/responses.py
"""API response schemas."""
from typing import Optional, Any, List
from pydantic import BaseModel, Field
from datetime import datetime
from ..models.crawl_result import CrawlResult
from ..models.job import JobStatus


class CrawlResponse(BaseModel):
    """Response schema for crawl operations."""
    success: bool
    data: Optional[CrawlResult] = None
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class JobResponse(BaseModel):
    """Response schema for async job creation."""
    job_id: str
    status: JobStatus
    message: str
    total_urls: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class JobStatusResponse(BaseModel):
    """Response schema for job status query."""
    job_id: str
    status: JobStatus
    progress: Optional[int] = Field(None, ge=0, le=100)
    results: Optional[List[CrawlResult]] = None
    error: Optional[str] = None
    message: str
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
```

---

## 6. Async Task Processing

### 6.1 Celery Configuration (`tasks/celery_app.py`)

```python
"""Celery application configuration."""
from celery import Celery
from ..core.config import settings

celery_app = Celery(
    "kg_agent_crawler",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["kg_agent.tasks.crawl_tasks"]
)

# Configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour
    task_soft_time_limit=3000,  # 50 minutes
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=50
)
```

### 6.2 Crawl Tasks (`tasks/crawl_tasks.py`)

```python
"""Celery tasks for async crawling operations."""
from typing import List, Dict, Any
from .celery_app import celery_app
from ..crawler.service import CrawlerService
from ..core.logging import logger


@celery_app.task(bind=True, max_retries=3)
async def crawl_single_task(self, url: str, config: Dict[str, Any]):
    """Task for crawling a single URL."""
    try:
        async with CrawlerService() as crawler:
            result = await crawler.crawl_single(url, **config)
        return result.dict()
    except Exception as e:
        logger.error(f"Task failed for {url}: {str(e)}")
        raise self.retry(exc=e, countdown=60)


@celery_app.task(bind=True)
async def crawl_batch_task(
    self,
    job_id: str,
    urls: List[str],
    config: Dict[str, Any]
):
    """Task for batch crawling multiple URLs."""
    logger.info(f"Starting batch crawl task {job_id} for {len(urls)} URLs")

    try:
        async with CrawlerService() as crawler:
            results = await crawler.crawl_batch(urls, **config)

        logger.info(f"Batch crawl task {job_id} completed")
        return [r.dict() for r in results]

    except Exception as e:
        logger.error(f"Batch crawl task {job_id} failed: {str(e)}")
        raise


@celery_app.task(bind=True)
async def deep_crawl_task(
    self,
    job_id: str,
    start_url: str,
    config: Dict[str, Any]
):
    """Task for deep crawling from a starting URL."""
    logger.info(f"Starting deep crawl task {job_id} from {start_url}")

    try:
        async with CrawlerService() as crawler:
            results = await crawler.deep_crawl(start_url, **config)

        logger.info(f"Deep crawl task {job_id} completed: {len(results)} pages")
        return [r.dict() for r in results]

    except Exception as e:
        logger.error(f"Deep crawl task {job_id} failed: {str(e)}")
        raise
```

---

## 7. Configuration Management

### 7.1 Application Settings (`core/config.py`)

```python
"""Application configuration using Pydantic Settings."""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, List


class Settings(BaseSettings):
    """Application settings."""

    # Application
    APP_NAME: str = "KG Agent Crawler"
    APP_VERSION: str = "1.0.0"
    APP_ENV: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"

    # API
    API_V1_PREFIX: str = "/api/v1"
    API_KEY_NAME: str = "X-API-Key"
    API_KEY_SECRET: str = "changeme"
    CORS_ORIGINS: List[str] = ["*"]

    # Crawler
    CRAWLER_MAX_CONCURRENT: int = 5
    CRAWLER_TIMEOUT: int = 30
    CRAWLER_USER_AGENT: Optional[str] = None
    CRAWLER_HEADLESS: bool = True
    CRAWLER_CACHE_DIR: str = "./cache/crawler"

    # Celery
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0

    # Storage
    STORAGE_DIR: str = "./storage"
    SCREENSHOT_DIR: str = "./storage/screenshots"
    PDF_DIR: str = "./storage/pdfs"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True
    )


settings = Settings()
```

### 7.2 Environment Variables (`.env`)

```bash
# Application
APP_ENV=development
DEBUG=true
LOG_LEVEL=INFO

# API
API_KEY_SECRET=your_secret_key_here
CORS_ORIGINS=["http://localhost:3000","http://localhost:8000"]

# Crawler
CRAWLER_MAX_CONCURRENT=5
CRAWLER_TIMEOUT=30
CRAWLER_HEADLESS=true

# Celery & Redis
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/1
REDIS_HOST=localhost
REDIS_PORT=6379

# Storage
STORAGE_DIR=./storage
SCREENSHOT_DIR=./storage/screenshots
PDF_DIR=./storage/pdfs
```

---

## 8. Implementation Roadmap

### Phase 1: Foundation (Week 1)

- [ ] Set up project structure
- [ ] Install Crawl4AI and dependencies
- [ ] Configure Pydantic settings
- [ ] Set up logging infrastructure
- [ ] Create base data models

### Phase 2: Core Crawler Service (Week 2)

- [ ] Implement `CrawlerService` class
- [ ] Add single URL crawling
- [ ] Add batch crawling with concurrency
- [ ] Add deep crawling functionality
- [ ] Implement caching layer
- [ ] Add retry logic with exponential backoff

### Phase 3: API Layer (Week 3)

- [ ] Initialize FastAPI application
- [ ] Create API route handlers
- [ ] Implement request/response models
- [ ] Add API key authentication
- [ ] Set up CORS middleware
- [ ] Create custom exception handlers

### Phase 4: Async Task Processing (Week 4)

- [ ] Configure Celery with Redis
- [ ] Implement crawl tasks
- [ ] Add job status tracking
- [ ] Implement progress monitoring
- [ ] Add task result persistence
- [ ] Create job cancellation endpoint

### Phase 5: Advanced Features (Week 5)

- [ ] Session management for authenticated sites
- [ ] Proxy rotation support
- [ ] Custom extraction strategies
- [ ] Content filtering hooks
- [ ] Screenshot and PDF generation
- [ ] Rate limiting per domain

### Phase 6: Testing & Documentation (Week 6)

- [ ] Write unit tests for crawler service
- [ ] Write integration tests for API
- [ ] Create API documentation (OpenAPI/Swagger)
- [ ] Add usage examples
- [ ] Performance benchmarking
- [ ] Load testing

### Phase 7: Deployment (Week 7)

- [ ] Create Docker containers
- [ ] Set up docker-compose
- [ ] Configure production settings
- [ ] Set up monitoring and logging
- [ ] Create deployment documentation
- [ ] CI/CD pipeline setup

---

## 9. Testing Strategy

### 9.1 Unit Tests

```python
# tests/test_crawler_service.py
import pytest
from kg_agent.crawler.service import CrawlerService


@pytest.mark.asyncio
async def test_crawl_single_url():
    """Test crawling a single URL."""
    async with CrawlerService() as crawler:
        result = await crawler.crawl_single("https://example.com")

    assert result.success is True
    assert result.html is not None
    assert result.markdown is not None
    assert result.metadata.title is not None


@pytest.mark.asyncio
async def test_crawl_batch_urls():
    """Test batch crawling multiple URLs."""
    urls = [
        "https://example.com",
        "https://example.org",
        "https://example.net"
    ]

    async with CrawlerService() as crawler:
        results = await crawler.crawl_batch(urls)

    assert len(results) == 3
    assert all(r.success for r in results)


@pytest.mark.asyncio
async def test_deep_crawl():
    """Test deep crawling functionality."""
    async with CrawlerService() as crawler:
        results = await crawler.deep_crawl(
            "https://example.com",
            max_depth=2,
            max_pages=10
        )

    assert len(results) > 0
    assert len(results) <= 10
```

### 9.2 API Integration Tests

```python
# tests/test_api_endpoints.py
from fastapi.testclient import TestClient
from kg_agent.api.main import app


client = TestClient(app)


def test_crawl_single_endpoint():
    """Test single URL crawl endpoint."""
    response = client.post(
        "/api/v1/crawl/single",
        json={"url": "https://example.com"},
        headers={"X-API-Key": "test_key"}
    )

    assert response.status_code == 200
    assert response.json()["success"] is True


def test_crawl_batch_endpoint():
    """Test batch crawl endpoint."""
    response = client.post(
        "/api/v1/crawl/batch",
        json={
            "urls": [
                "https://example.com",
                "https://example.org"
            ]
        },
        headers={"X-API-Key": "test_key"}
    )

    assert response.status_code == 202
    assert "job_id" in response.json()
```

---

## 10. Deployment Configuration

### 10.1 Docker Compose (`docker-compose.yml`)

```yaml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - '8000:8000'
    environment:
      - APP_ENV=production
      - CELERY_BROKER_URL=redis://redis:6379/0
      - CELERY_RESULT_BACKEND=redis://redis:6379/1
    depends_on:
      - redis
    volumes:
      - ./storage:/app/storage
      - ./cache:/app/cache
    command: uvicorn kg_agent.api.main:app --host 0.0.0.0 --port 8000

  worker:
    build:
      context: .
      dockerfile: Dockerfile
    environment:
      - APP_ENV=production
      - CELERY_BROKER_URL=redis://redis:6379/0
      - CELERY_RESULT_BACKEND=redis://redis:6379/1
    depends_on:
      - redis
    volumes:
      - ./storage:/app/storage
      - ./cache:/app/cache
    command: celery -A kg_agent.tasks.celery_app worker --loglevel=info

  redis:
    image: redis:7-alpine
    ports:
      - '6379:6379'
    volumes:
      - redis_data:/data

volumes:
  redis_data:
```

### 10.2 Dockerfile

```dockerfile
FROM python:3.11-slim

# Install system dependencies for Playwright
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files
COPY pyproject.toml ./
COPY README.md ./

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Install Playwright browsers
RUN playwright install chromium
RUN playwright install-deps chromium

# Copy application code
COPY src/ ./src/

# Create directories
RUN mkdir -p /app/storage/screenshots /app/storage/pdfs /app/cache

EXPOSE 8000

CMD ["uvicorn", "kg_agent.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 11. Usage Examples

### 11.1 Python Client

```python
import httpx
import asyncio


class CrawlerClient:
    """Client for interacting with the Crawler API."""

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.headers = {"X-API-Key": api_key}
        self.client = httpx.AsyncClient()

    async def crawl_single(self, url: str) -> dict:
        """Crawl a single URL."""
        response = await self.client.post(
            f"{self.base_url}/api/v1/crawl/single",
            json={"url": url},
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()

    async def crawl_batch(self, urls: list[str]) -> str:
        """Start a batch crawl job."""
        response = await self.client.post(
            f"{self.base_url}/api/v1/crawl/batch",
            json={"urls": urls},
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()["job_id"]

    async def get_job_status(self, job_id: str) -> dict:
        """Get job status."""
        response = await self.client.get(
            f"{self.base_url}/api/v1/crawl/job/{job_id}",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()


# Usage
async def main():
    client = CrawlerClient(
        base_url="http://localhost:8000",
        api_key="your_api_key"
    )

    # Single crawl
    result = await client.crawl_single("https://example.com")
    print(f"Title: {result['data']['metadata']['title']}")

    # Batch crawl
    job_id = await client.crawl_batch([
        "https://example.com",
        "https://example.org"
    ])

    # Poll for results
    while True:
        status = await client.get_job_status(job_id)
        if status["status"] == "completed":
            print(f"Results: {len(status['results'])} pages crawled")
            break
        await asyncio.sleep(2)


asyncio.run(main())
```

### 11.2 cURL Examples

```bash
# Single URL crawl
curl -X POST http://localhost:8000/api/v1/crawl/single \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'

# Batch crawl
curl -X POST http://localhost:8000/api/v1/crawl/batch \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "urls": [
      "https://example.com",
      "https://example.org"
    ],
    "max_concurrent": 3
  }'

# Check job status
curl -X GET http://localhost:8000/api/v1/crawl/job/{job_id} \
  -H "X-API-Key: your_api_key"

# Deep crawl
curl -X POST http://localhost:8000/api/v1/crawl/deep \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "start_url": "https://example.com",
    "config": {
      "max_depth": 2,
      "max_pages": 50,
      "same_domain_only": true
    }
  }'
```

---

## 12. Monitoring & Observability

### 12.1 Logging Configuration

```python
# core/logging.py
import logging
import sys
from pathlib import Path
from .config import settings


def setup_logging():
    """Configure application logging."""

    # Create logs directory
    log_dir = Path("./logs")
    log_dir.mkdir(exist_ok=True)

    # Configure formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # File handler
    file_handler = logging.FileHandler(log_dir / "crawler.log")
    file_handler.setFormatter(formatter)

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, settings.LOG_LEVEL))
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


logger = setup_logging()
```

### 12.2 Metrics Collection

```python
# utils/metrics.py
from prometheus_client import Counter, Histogram, Gauge

# Counters
crawl_requests_total = Counter(
    'crawl_requests_total',
    'Total number of crawl requests',
    ['endpoint', 'status']
)

crawl_errors_total = Counter(
    'crawl_errors_total',
    'Total number of crawl errors',
    ['error_type']
)

# Histograms
crawl_duration_seconds = Histogram(
    'crawl_duration_seconds',
    'Crawl duration in seconds',
    ['url_type']
)

# Gauges
active_crawl_jobs = Gauge(
    'active_crawl_jobs',
    'Number of active crawl jobs'
)
```

---

## 13. Next Steps

1. **Review and approve this plan**
2. **Set up development environment** (Python, Redis, dependencies)
3. **Create project structure** following the defined layout
4. **Implement Phase 1**: Foundation and core models
5. **Begin Phase 2**: Crawler service implementation
6. **Iterate and test** each phase before moving to the next

---

## Appendix: Dependencies

### A.1 `pyproject.toml`

```toml
[project]
name = "kg-agent"
version = "0.1.0"
description = "Web-to-Knowledge-Graph ETL Pipeline with Crawl4AI"
requires-python = ">=3.11"

dependencies = [
    "crawl4ai>=0.7.6",
    "fastapi>=0.109.0",
    "uvicorn[standard]>=0.27.0",
    "pydantic>=2.5.0",
    "pydantic-settings>=2.1.0",
    "celery[redis]>=5.3.0",
    "redis>=5.0.0",
    "httpx>=0.26.0",
    "python-multipart>=0.0.6",
    "python-dotenv>=1.0.0",
    "playwright>=1.40.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.1.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
    "ty",
]

[build-system]
requires = ["uv_build"]
build-backend = "uv_build"
```

---

**End of Plan**

This comprehensive plan provides a production-ready blueprint for implementing

the web crawler service with Crawl4AI. The architecture is modular, scalable,

and follows best practices for async Python development with FastAPI and Celery.