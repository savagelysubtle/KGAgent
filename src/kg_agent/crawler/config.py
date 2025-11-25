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
