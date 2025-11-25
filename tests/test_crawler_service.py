"""Unit tests for crawler service."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.kg_agent.crawler.service import CrawlerService


@pytest.mark.asyncio
async def test_crawler_service_initialization():
    """Test CrawlerService initialization."""
    service = CrawlerService()
    assert service.max_concurrent == 5
    assert service.browser_config is not None
    assert service._crawler is None


@pytest.mark.asyncio
async def test_crawler_single_success():
    """Test successful single URL crawl."""
    with patch("src.kg_agent.crawler.service.AsyncWebCrawler") as mock_crawler_class:
        # Mock the crawler instance
        mock_crawler = AsyncMock()
        mock_crawler_class.return_value = mock_crawler

        # Mock the crawl result
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.html = "<html><body>Test</body></html>"
        mock_result.markdown = "# Test"
        mock_result.cleaned_html = "<div>Test</div>"
        mock_result.media = {"images": ["test.jpg"]}
        mock_result.links = {"internal": ["http://example.com/page2"]}
        mock_result.metadata = {"title": "Test Page", "description": "Test description"}
        mock_result.screenshot = None
        mock_result.pdf = None
        mock_result.error_message = None

        mock_crawler.arun.return_value = mock_result

        service = CrawlerService()

        async with service:
            result = await service.crawl_single("http://example.com")

        assert result.success is True
        assert result.url == "http://example.com"
        assert result.html == "<html><body>Test</body></html>"
        assert result.markdown == "# Test"
        assert result.metadata.title == "Test Page"
        assert result.metadata.description == "Test description"


@pytest.mark.asyncio
async def test_crawler_single_failure():
    """Test failed single URL crawl."""
    with patch("src.kg_agent.crawler.service.AsyncWebCrawler") as mock_crawler_class:
        mock_crawler = AsyncMock()
        mock_crawler_class.return_value = mock_crawler

        mock_crawler.arun.side_effect = Exception("Network error")

        service = CrawlerService()

        async with service:
            result = await service.crawl_single("http://example.com")

        assert result.success is False
        assert result.url == "http://example.com"
        assert "Network error" in result.error_message


@pytest.mark.asyncio
async def test_crawler_batch():
    """Test batch URL crawling."""
    with patch("src.kg_agent.crawler.service.AsyncWebCrawler") as mock_crawler_class:
        mock_crawler = AsyncMock()
        mock_crawler_class.return_value = mock_crawler

        # Mock successful results
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.html = "<html>Test</html>"
        mock_result.markdown = "# Test"
        mock_result.cleaned_html = "<div>Test</div>"
        mock_result.media = {}
        mock_result.links = {}
        mock_result.metadata = {"title": "Test"}
        mock_result.screenshot = None
        mock_result.pdf = None
        mock_result.error_message = None

        mock_crawler.arun.return_value = mock_result

        service = CrawlerService()

        urls = ["http://example.com/1", "http://example.com/2"]

        async with service:
            results = await service.crawl_batch(urls)

        assert len(results) == 2
        assert all(r.success for r in results)
        assert all(r.url in urls for r in results)


@pytest.mark.asyncio
async def test_crawler_deep_crawl():
    """Test deep crawling functionality."""
    with patch("src.kg_agent.crawler.service.AsyncWebCrawler") as mock_crawler_class:
        mock_crawler = AsyncMock()
        mock_crawler_class.return_value = mock_crawler

        # Mock result with links
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.html = "<html><a href='/page2'>Link</a></html>"
        mock_result.markdown = "# Test"
        mock_result.cleaned_html = "<div>Test</div>"
        mock_result.media = {}
        mock_result.links = {"internal": ["http://example.com/page2"]}
        mock_result.metadata = {"title": "Test"}
        mock_result.screenshot = None
        mock_result.pdf = None
        mock_result.error_message = None

        # First call returns page with link, second call returns linked page
        mock_crawler.arun.side_effect = [mock_result, mock_result]

        service = CrawlerService()

        async with service:
            results = await service.deep_crawl(
                "http://example.com", max_depth=1, max_pages=2
            )

        assert len(results) >= 1
        assert all(r.success for r in results)
