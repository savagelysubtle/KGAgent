# KG Agent - Web-to-Knowledge-Graph ETL Pipeline

A comprehensive web crawling service built with Crawl4AI, FastAPI, and Celery for extracting LLM-ready content from websites.

## Features

- **Async Web Crawling**: High-performance crawling using Crawl4AI and Playwright
- **LLM-Ready Content**: Markdown output optimized for language models
- **Batch Processing**: Concurrent crawling with configurable limits
- **Deep Crawling**: Recursive page exploration with link following
- **Session Management**: Persistent browser sessions for authenticated sites
- **Caching**: Intelligent content caching with TTL
- **REST API**: FastAPI-based REST endpoints with OpenAPI documentation
- **Async Tasks**: Celery-based background job processing
- **Docker Support**: Containerized deployment with docker-compose

## Quick Start

### Prerequisites

- Python 3.11+
- Redis (for Celery broker and result backend)
- Playwright browsers (installed automatically)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd kg-agent
```

2. Install dependencies:
```bash
uv sync --all-groups
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Install Playwright browsers:
```bash
playwright install chromium
```

### Running Locally

1. Start Redis (if not already running):
```bash
redis-server
```

2. Start the API server:
```bash
python main.py
```

3. In another terminal, start the Celery worker:
```bash
celery -A kg_agent.tasks.celery_app worker --loglevel=info
```

The API will be available at `http://localhost:8000`

## API Usage

### Authentication

All API endpoints require an API key header:
```
X-API-Key: your_secret_key_here
```

### Single URL Crawl

```bash
curl -X POST http://localhost:8000/api/v1/crawl/single \
  -H "X-API-Key: your_secret_key_here" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'
```

### Batch Crawl

```bash
curl -X POST http://localhost:8000/api/v1/crawl/batch \
  -H "X-API-Key: your_secret_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "urls": [
      "https://example.com",
      "https://example.org"
    ],
    "max_concurrent": 3
  }'
```

### Deep Crawl

```bash
curl -X POST http://localhost:8000/api/v1/crawl/deep \
  -H "X-API-Key: your_secret_key_here" \
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

### Check Job Status

```bash
curl -X GET http://localhost:8000/api/v1/crawl/job/{job_id} \
  -H "X-API-Key: your_secret_key_here"
```

## Python Client

```python
import httpx
import asyncio

class CrawlerClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.headers = {"X-API-Key": api_key}
        self.client = httpx.AsyncClient()

    async def crawl_single(self, url: str) -> dict:
        response = await self.client.post(
            f"{self.base_url}/api/v1/crawl/single",
            json={"url": url},
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()

# Usage
async def main():
    client = CrawlerClient(
        base_url="http://localhost:8000",
        api_key="your_secret_key_here"
    )

    result = await client.crawl_single("https://example.com")
    print(f"Title: {result['data']['metadata']['title']}")

asyncio.run(main())
```

## Docker Deployment

### Using Docker Compose

1. Build and start services:
```bash
docker-compose up --build
```

2. The API will be available at `http://localhost:8000`

### Manual Docker Build

```bash
# Build the image
docker build -t kg-agent .

# Run the container
docker run -p 8000:8000 \
  -v ./storage:/app/storage \
  -v ./cache:/app/cache \
  kg-agent
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_ENV` | development | Application environment |
| `DEBUG` | true | Enable debug mode |
| `LOG_LEVEL` | INFO | Logging level |
| `API_KEY_SECRET` | changeme | API authentication key |
| `CRAWLER_MAX_CONCURRENT` | 5 | Maximum concurrent crawls |
| `CRAWLER_TIMEOUT` | 30 | Crawl timeout in seconds |
| `CELERY_BROKER_URL` | redis://localhost:6379/0 | Celery broker URL |
| `CELERY_RESULT_BACKEND` | redis://localhost:6379/1 | Celery result backend |

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/kg_agent

# Run specific test
pytest tests/test_crawler_service.py::test_crawler_service_initialization
```

### Code Quality

```bash
# Format code
black src/

# Lint code
ruff check src/

# Type checking
mypy src/
```

### API Documentation

When running locally, visit `http://localhost:8000/docs` for interactive API documentation.

## Architecture

### Core Components

- **Crawler Service** (`crawler/service.py`): Main crawling logic using Crawl4AI
- **API Layer** (`api/`): FastAPI routes and request handling
- **Task Processing** (`tasks/`): Celery-based async job processing
- **Data Models** (`models/`): Pydantic schemas for validation
- **Utilities** (`utils/`): Helper functions for URL processing, content cleaning

### Data Flow

1. API receives crawl request
2. Request validated and queued as Celery task
3. Worker processes task using CrawlerService
4. Results stored and made available via API
5. Client polls for completion status

## Monitoring

### Health Checks

```bash
# Basic health check
curl http://localhost:8000/health

# Detailed health check
curl http://localhost:8000/api/v1/health/detailed
```

### Logs

Logs are written to `./logs/crawler.log` and stdout/stderr.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions and support, please open an issue on GitHub.
