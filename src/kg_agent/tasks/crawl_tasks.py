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
