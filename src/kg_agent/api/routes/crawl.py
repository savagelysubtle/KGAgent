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
