"""Data models for crawl results."""
from typing import Optional, Dict, List, Any
from datetime import datetime
from pydantic import BaseModel, Field


class CrawlMetadata(BaseModel):
    """Metadata extracted from crawled pages."""
    title: Optional[str] = Field(default=None, description="Page title")
    description: Optional[str] = Field(default=None, description="Page description")
    keywords: List[str] = Field(default_factory=list, description="Page keywords")
    author: Optional[str] = Field(default=None, description="Page author")
    published_date: Optional[datetime] = Field(default=None, description="Publication date")
    language: str = Field(default="en", description="Page language")
    word_count: int = Field(default=0, description="Word count")
    crawl_timestamp: Optional[datetime] = Field(default=None, description="Crawl timestamp")


class CrawlResult(BaseModel):
    """Result of a single crawl operation."""
    url: str = Field(..., description="Crawled URL")
    success: bool = Field(default=False, description="Whether crawl was successful")
    html: Optional[str] = Field(default=None, description="Raw HTML content")
    markdown: Optional[str] = Field(default=None, description="Markdown content")
    cleaned_html: Optional[str] = Field(default=None, description="Cleaned HTML")
    media: Dict[str, List[str]] = Field(default_factory=dict, description="Media references")
    links: Dict[str, List[str]] = Field(default_factory=dict, description="Extracted links")
    metadata: CrawlMetadata = Field(default_factory=CrawlMetadata, description="Page metadata")
    screenshot_path: Optional[str] = Field(default=None, description="Screenshot file path")
    pdf_path: Optional[str] = Field(default=None, description="PDF file path")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    crawl_timestamp: datetime = Field(default_factory=datetime.utcnow, description="When crawl completed")
