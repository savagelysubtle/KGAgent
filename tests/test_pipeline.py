import asyncio
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from kg_agent.pipeline.manager import PipelineManager
from kg_agent.crawler.service import CrawlerService
from kg_agent.models.crawl_result import CrawlResult, CrawlMetadata
from kg_agent.core.logging import setup_logging

async def test_pipeline_with_mock_data():
    """Test the pipeline with mock crawl data."""
    setup_logging()
    import logging
    logging.getLogger().setLevel(logging.INFO)

    # Create a mock CrawlResult with actual content
    mock_result = CrawlResult(
        url="https://example.com",
        success=True,
        html="""<!DOCTYPE html>
<html>
<head>
    <title>Example Domain</title>
    <meta charset="utf-8" />
    <meta http-equiv="Content-type" content="text/html; charset=utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <style type="text/css">
    body {
        background-color: #f0f0f2;
        margin: 0;
        padding: 0;
        font-family: -apple-system, system-ui, BlinkMacSystemFont, "Segoe UI", "Open Sans", "Helvetica Neue", Helvetica, Arial, sans-serif;

    }
    div {
        width: 600px;
        margin: 5em auto;
        padding: 2em;
        background-color: #fdfdff;
        border-radius: 0.5em;
        box-shadow: 2px 3px 7px 2px rgba(0,0,0,0.02);
    }
    a:link, a:visited {
        color: #38488f;
        text-decoration: none;
    }
    @media (max-width: 700px) {
        div {
            margin: 0 auto;
            width: auto;
        }
    }
    </style>
</head>

<body>
<div>
    <h1>Example Domain</h1>
    <p>This domain is for use in illustrative examples in documents. You may use this
    domain in literature without prior coordination or asking for permission.</p>
    <p><a href="https://www.iana.org/domains/example">More information...</a></p>
</div>
</body>
</html>""",
        markdown="# Example Domain\n\nThis domain is for use in illustrative examples in documents. You may use this domain in literature without prior coordination or asking for permission.\n\n[More information...](https://www.iana.org/domains/example)",
        cleaned_html="""<html><head><title>Example Domain</title></head><body><div><h1>Example Domain</h1><p>This domain is for use in illustrative examples in documents. You may use this domain in literature without prior coordination or asking for permission.</p><p><a href="https://www.iana.org/domains/example">More information...</a></p></div></body></html>""",
        metadata=CrawlMetadata(
            title="Example Domain",
            description="",
            word_count=42
        )
    )

    # Test the pipeline steps manually
    from kg_agent.crawler.storage import StorageService
    from kg_agent.parser.service import ParserService
    from kg_agent.chunker.service import ChunkerService

    storage = StorageService()
    parser = ParserService()
    chunker = ChunkerService()

    print("Testing Storage Service...")
    saved_path = await storage.save_raw_content(mock_result, "test_job")
    print(f"Saved to: {saved_path}")

    if saved_path:
        print("Testing Parser Service...")
        parsed_path = parser.parse_file(saved_path, "test_job")
        print(f"Parsed to: {parsed_path}")

        if parsed_path:
            print("Testing Chunker Service...")
            chunked_path = chunker.chunk_file(parsed_path, "test_job")
            print(f"Chunked to: {chunked_path}")

            if chunked_path:
                print("✅ Pipeline test successful!")
                return True

    print("❌ Pipeline test failed!")
    return False

async def main():
    print("Running pipeline test with mock data...")
    success = await test_pipeline_with_mock_data()

    if success:
        print("\n🎉 All pipeline components are working correctly!")
    else:
        print("\n💥 Pipeline has issues that need to be fixed.")

if __name__ == "__main__":
    asyncio.run(main())

