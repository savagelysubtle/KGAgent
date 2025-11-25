"""
Reprocess a GPT chat export through the full pipeline.
This script:
1. Parses the HTML with the GPT parser
2. Chunks the text
3. Embeds chunks into ChromaDB
4. Updates the document tracker
"""
import sys
import asyncio
sys.path.insert(0, ".")

from pathlib import Path
from datetime import datetime
import uuid

from src.kg_agent.parser.service import ParserService
from src.kg_agent.chunker.service import ChunkerService
from src.kg_agent.services.vector_store import get_vector_store
from src.kg_agent.services.document_tracker import get_document_tracker, DocumentStatus
from src.kg_agent.core.logging import logger


async def reprocess_gpt_export(html_file_path: str, chunk_size: int = 2000, chunk_overlap: int = 400):
    """
    Reprocess a GPT chat export file.

    Args:
        html_file_path: Path to the GPT export HTML file
        chunk_size: Size of each chunk (larger for better context)
        chunk_overlap: Overlap between chunks
    """
    path = Path(html_file_path)
    if not path.exists():
        print(f"File not found: {html_file_path}")
        return

    job_id = f"gpt_reprocess_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    doc_id = str(uuid.uuid4())

    print(f"=== Reprocessing GPT Export ===")
    print(f"File: {html_file_path}")
    print(f"Job ID: {job_id}")
    print(f"Doc ID: {doc_id}")
    print(f"Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
    print()

    # Initialize services
    parser = ParserService()
    chunker = ChunkerService(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    vector_store = get_vector_store()
    doc_tracker = get_document_tracker()

    # 1. Parse the file
    print("Step 1: Parsing HTML...")
    parsed_path = parser.parse_file(str(path), job_id)
    if not parsed_path:
        print("ERROR: Parsing failed!")
        return
    print(f"  Parsed to: {parsed_path}")

    # 2. Chunk the parsed content
    print("\nStep 2: Chunking content...")
    chunks_path = chunker.chunk_file(parsed_path, job_id)
    if not chunks_path:
        print("ERROR: Chunking failed!")
        return
    print(f"  Chunks saved to: {chunks_path}")

    # Load chunks to count them
    chunk_batch = chunker.load_chunks(chunks_path)
    if not chunk_batch:
        print("ERROR: Could not load chunks!")
        return

    num_chunks = len(chunk_batch.chunks)
    print(f"  Created {num_chunks} chunks")

    # 3. Embed and store in ChromaDB
    print("\nStep 3: Embedding and storing in ChromaDB...")

    # Update chunk metadata with doc_id
    for chunk in chunk_batch.chunks:
        chunk.metadata["doc_id"] = doc_id
        chunk.metadata["source_type"] = "gpt_chat_export"

    # Add to vector store in batches
    # ChromaDB will use its default embedder if we don't provide embeddings
    batch_size = 100
    vector_ids = []
    total_chunks = len(chunk_batch.chunks)

    for i in range(0, total_chunks, batch_size):
        batch_chunks = chunk_batch.chunks[i:i+batch_size]

        # ChromaDB's add_chunks will use default embedder
        vector_store.add_chunks(batch_chunks)
        vector_ids.extend([c.id for c in batch_chunks])

        batch_num = i // batch_size + 1
        total_batches = (total_chunks + batch_size - 1) // batch_size
        print(f"  Embedded batch {batch_num}/{total_batches} ({len(vector_ids)}/{total_chunks} chunks)")

    print(f"  Total vectors stored: {len(vector_ids)}")

    # 4. Register in document tracker
    print("\nStep 4: Registering document...")
    doc = doc_tracker.create_document(
        title=path.name,
        source_type="gpt_chat_export",
        file_path=str(path),
        metadata={
            "job_id": job_id,
            "original_file_size": path.stat().st_size,
            "conversation_count": 1257,  # From earlier parsing
        }
    )

    # Update status and chunk count
    doc_tracker.update_document(
        doc.id,
        status=DocumentStatus.COMPLETED,
        chunk_count=num_chunks,
    )

    # Add vector IDs
    doc_tracker.add_vector_ids(doc.id, vector_ids)

    print(f"  Document registered with ID: {doc.id}")

    print("\n=== Reprocessing Complete! ===")
    print(f"Document ID: {doc.id}")
    print(f"Chunks: {num_chunks}")
    print(f"Vectors: {len(vector_ids)}")
    print("\nYou can now run entity extraction on this document from the dashboard!")

    return doc.id


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Reprocess GPT chat export")
    parser.add_argument("file", help="Path to GPT export HTML file")
    parser.add_argument("--chunk-size", type=int, default=2000, help="Chunk size (default: 2000)")
    parser.add_argument("--chunk-overlap", type=int, default=400, help="Chunk overlap (default: 400)")

    args = parser.parse_args()

    asyncio.run(reprocess_gpt_export(args.file, args.chunk_size, args.chunk_overlap))

