"""API response schemas."""
from typing import Optional, Any, List
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from .crawl_result import CrawlResult
from .job import JobStatus


class CrawlResponse(BaseModel):
    """Response schema for crawl operations."""
    success: bool
    data: Optional[CrawlResult] = None
    message: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class JobResponse(BaseModel):
    """Response schema for async job creation."""
    job_id: str
    status: JobStatus
    message: str
    total_urls: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class BatchCrawlResponse(BaseModel):
    """Response schema for batch crawl operations."""
    success: bool
    data: Optional[List[CrawlResult]] = None
    message: str
    total_urls: int
    successful_urls: int
    failed_urls: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


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
