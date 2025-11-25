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


class SessionCrawlRequest(BaseModel):
    """Request schema for session-based crawling."""
    urls: list[HttpUrl] = Field(..., min_length=1, max_length=1000)
    session_id: str = Field(..., description="Unique session identifier")
    auth_type: Optional[str] = Field(
        default=None,
        description="Authentication type (form, oauth, token)"
    )
    auth_credentials: Optional[dict] = None
