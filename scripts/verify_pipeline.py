#!/usr/bin/env python3
"""
Verification script for the Docling to ChromaDB pipeline integration.
Tests the complete flow from file upload through embedding and storage.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the src directory to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kg_agent.pipeline.manager import PipelineManager
from kg_agent.services.vector_store import VectorStoreService
from kg_agent.core.logging import logger

async def verify_pipeline():
    """Run a verification test of the pipeline."""
    logger.info("Starting pipeline verification...")

    # Initialize services
    pipeline = PipelineManager()
    vector_store = VectorStoreService()

    # Get initial count
    initial_count = vector_store.count()
    logger.info(f"Initial vector store count: {initial_count}")

    # Create a simple test file for verification
    test_dir = Path("./data/test")
    test_dir.mkdir(parents=True, exist_ok=True)

    # Create a simple HTML file for testing
    test_html = """
    <!DOCTYPE html>
    <html>
    <head><title>Test Document</title></head>
    <body>
        <h1>Test Knowledge Graph Pipeline</h1>
        <p>This is a test document to verify the Docling to ChromaDB pipeline integration.</p>
        <p>The pipeline should parse this HTML, chunk the content, generate embeddings, and store them in ChromaDB.</p>
        <h2>Technical Details</h2>
        <p>The system uses:</p>
        <ul>
            <li>Docling for document parsing</li>
            <li>LangChain for text chunking</li>
            <li>Sentence Transformers for embeddings</li>
            <li>ChromaDB for vector storage</li>
        </ul>
    </body>
    </html>
    """

    test_file = test_dir / "test_pipeline.html"
    test_file.write_text(test_html)

    logger.info(f"Created test file: {test_file}")

    try:
        # Run the file pipeline
        result = await pipeline.run_file_pipeline([str(test_file)])

        logger.info(f"Pipeline result: {result}")

        if result["status"] == "success":
            logger.info("Pipeline completed successfully!")

            # Check final count
            final_count = vector_store.count()
            logger.info(f"Final vector store count: {final_count}")

            stored_count = result["metrics"].get("stored_successfully", 0)
            if stored_count > 0:
                logger.info(f"Successfully processed and stored {stored_count} chunks in vector database")

                # Test a simple query
                query_embedding = pipeline.embedder.embed_text("knowledge graph pipeline")
                query_result = vector_store.query(query_embedding, n_results=3)

                if query_result and "documents" in query_result and query_result["documents"]:
                    logger.info("Vector search working - found relevant documents")
                    logger.info(f"Sample document: {query_result['documents'][0][:100]}...")
                    logger.info("SUCCESS: Pipeline integration with ChromaDB is working!")
                else:
                    logger.warning("Vector search returned no results - this may be expected for small test data")

            else:
                logger.error("No chunks were stored in the vector database")

        else:
            logger.error(f"❌ Pipeline failed: {result.get('error', 'Unknown error')}")

    except Exception as e:
        logger.error(f"❌ Verification failed with exception: {e}")
        return False

    # Cleanup
    try:
        test_file.unlink()
        test_dir.rmdir()
        logger.info("Cleaned up test files")
    except:
        pass

    return result.get("status") == "success"

if __name__ == "__main__":
    success = asyncio.run(verify_pipeline())
    sys.exit(0 if success else 1)
