import uuid
import asyncio
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..crawler.service import CrawlerService
from ..crawler.storage import StorageService
from ..parser.service import ParserService
from ..chunker.service import ChunkerService
from ..services.embedder import EmbedderService
from ..services.vector_store import VectorStoreService
from ..services.graphiti_service import get_graphiti_service
from ..services.document_tracker import get_document_tracker, DocumentStatus, DocumentSource
from ..core.logging import logger


# Thread pool for CPU-bound pipeline operations (parsing, chunking)
_pipeline_executor: Optional[ThreadPoolExecutor] = None


def get_pipeline_executor() -> ThreadPoolExecutor:
    """Get or create the pipeline thread pool executor."""
    global _pipeline_executor
    if _pipeline_executor is None:
        _pipeline_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="pipeline")
        logger.info("Created pipeline ThreadPoolExecutor with 2 workers")
    return _pipeline_executor

class PipelineManager:
    """
    Orchestrator for the Web-to-KG pipeline.
    """

    def __init__(self):
        self.crawler = CrawlerService()
        self.storage = StorageService()
        self.parser = ParserService()
        self.chunker = ChunkerService()
        self.embedder = EmbedderService()
        self.vector_store = VectorStoreService()
        self.graphiti_service = get_graphiti_service()
        self.document_tracker = get_document_tracker()

    async def run_pipeline(self, urls: List[str], job_id: str = None) -> Dict[str, Any]:
        """
        Run the full pipeline from crawling to graph building.

        Args:
            urls: List of URLs to crawl.
            job_id: Optional job identifier.

        Returns:
            Dictionary containing paths to generated artifacts.
        """
        if not job_id:
            job_id = f"job_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

        logger.info(f"Starting pipeline job {job_id} for {len(urls)} URLs")

        # Track documents for each URL
        tracked_doc_ids = []
        url_to_doc_id = {}

        try:
            # Create document records for each URL
            for url in urls:
                doc = self.document_tracker.create_document(
                    title=url,  # Will be updated with actual title if available
                    source_type=DocumentSource.WEB_CRAWL.value,
                    source_url=url,
                    file_path=None,
                    content_hash=None,
                    metadata={
                        "job_id": job_id,
                        "original_url": url
                    }
                )
                tracked_doc_ids.append(doc.id)
                url_to_doc_id[url] = doc.id

                # Update status to processing
                self.document_tracker.update_document(doc.id, status=DocumentStatus.PROCESSING.value)

            # 1. Crawl
            logger.info("Step 1: Crawling...")
            async with self.crawler as crawler:
                crawl_results = await crawler.crawl_batch(urls)

            # 2. Save Raw Data
            logger.info("Step 2: Saving raw content...")
            raw_paths = []
            url_to_path = {}
            for result in crawl_results:
                path = await self.storage.save_raw_content(result, job_id)
                if path:
                    raw_paths.append(path)
                    url_to_path[result.url] = path

                    # Update document with file path and content hash
                    if result.url in url_to_doc_id:
                        doc_id = url_to_doc_id[result.url]
                        content_hash = None
                        if result.html:
                            content_hash = hashlib.sha256(result.html.encode()).hexdigest()
                        self.document_tracker.update_document(
                            doc_id,
                            metadata={
                                "job_id": job_id,
                                "original_url": result.url,
                                "title": result.title or result.url,
                                "file_path": path
                            }
                        )

            if not raw_paths:
                logger.warning("No content saved from crawl. Aborting pipeline.")
                # Mark documents as failed
                for doc_id in tracked_doc_ids:
                    self.document_tracker.update_document(
                        doc_id,
                        status=DocumentStatus.FAILED.value,
                        error_message="Crawl failed - no content retrieved"
                    )
                return {"status": "failed", "job_id": job_id, "error": "No content crawled", "document_ids": tracked_doc_ids}

            # 3. Parse (Docling)
            logger.info(f"Step 3: Parsing {len(raw_paths)} documents...")
            parsed_paths = self.parser.process_batch(raw_paths, job_id)

            if not parsed_paths:
                logger.warning("Parsing yielded no results.")
                for doc_id in tracked_doc_ids:
                    self.document_tracker.update_document(
                        doc_id,
                        status=DocumentStatus.FAILED.value,
                        error_message="Parsing failed - no content extracted"
                    )
                return {"status": "failed", "job_id": job_id, "error": "Parsing failed", "document_ids": tracked_doc_ids}

            # 4. Chunk
            logger.info(f"Step 4: Chunking {len(parsed_paths)} parsed documents...")
            chunk_files = self.chunker.process_batch(parsed_paths, job_id)

            all_chunks = []
            doc_chunks_map = {}  # Map doc_id to its chunks

            # Load chunks from files to process them
            for chunk_file in chunk_files:
                batch = self.chunker.load_chunks(chunk_file)
                if batch:
                    for chunk in batch.chunks:
                        all_chunks.append(chunk)
                        # Try to map chunk back to document via source URL
                        source_path = chunk.metadata.get("source", "")
                        for url, doc_id in url_to_doc_id.items():
                            if url in source_path or (url in url_to_path and url_to_path[url] in source_path):
                                if doc_id not in doc_chunks_map:
                                    doc_chunks_map[doc_id] = []
                                doc_chunks_map[doc_id].append(chunk)
                                break

            logger.info(f"Generated {len(all_chunks)} total chunks")

            # 5. Embed & Store Vectors in streaming batches to avoid memory exhaustion
            EMBED_BATCH_SIZE = 2000
            if all_chunks:
                total_chunks = len(all_chunks)
                logger.info(f"Step 5: Generating embeddings and storing {total_chunks} chunks in batches of {EMBED_BATCH_SIZE}...")

                loop = asyncio.get_event_loop()

                for batch_start in range(0, total_chunks, EMBED_BATCH_SIZE):
                    batch_end = min(batch_start + EMBED_BATCH_SIZE, total_chunks)
                    batch_num = batch_start // EMBED_BATCH_SIZE + 1
                    total_batches = (total_chunks + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE

                    logger.info(f"Processing batch {batch_num}/{total_batches}: chunks {batch_start} to {batch_end}")

                    # Get batch of chunks
                    batch_chunks = all_chunks[batch_start:batch_end]
                    batch_texts = [chunk.text for chunk in batch_chunks]

                    # Embed this batch (async, no progress bar for sub-batches)
                    batch_embeddings = await self.embedder.embed_batch_async(batch_texts, show_progress=False)

                    # Assign embeddings
                    for i, chunk in enumerate(batch_chunks):
                        if i < len(batch_embeddings):
                            chunk.embedding = batch_embeddings[i]

                    # Store immediately
                    await loop.run_in_executor(
                        get_pipeline_executor(),
                        lambda chunks=batch_chunks: self.vector_store.add_chunks(chunks)
                    )

                    # Clear embeddings from memory
                    for chunk in batch_chunks:
                        chunk.embedding = None

                    logger.info(f"Batch {batch_num}/{total_batches} complete")

            # 6. Build Knowledge Graph via Graphiti
            graph_node_ids = []
            if all_chunks:
                logger.info("Step 7: Building Knowledge Graph in FalkorDB via Graphiti...")
                # Initialize Graphiti service
                await self.graphiti_service.initialize()

                # Build graph by adding chunks as episodes
                from datetime import timezone

                # Combine chunks into episodes (batch by source document)
                episode_content = "\n\n".join([chunk.text for chunk in all_chunks[:50]])  # Limit for initial processing

                try:
                    result = await self.graphiti_service.add_episode(
                        content=episode_content,
                        name=f"Crawl job {job_id}",
                        source_description=f"Web crawl job {job_id}",
                        reference_time=datetime.now(timezone.utc),
                        source_type="text",
                    )

                    if result:
                        # Result is a dict from GraphitiService
                        graph_node_ids = [e.get("uuid") for e in result.get("entities", [])]
                        logger.info(f"Graph build result: {result.get('nodes_created', 0)} nodes, {result.get('edges_created', 0)} edges")
                    else:
                        logger.warning("No graph results returned from Graphiti")
                except Exception as e:
                    logger.warning(f"Graph building failed: {e}")

            # 7. Update document tracker with results
            for doc_id in tracked_doc_ids:
                doc_chunks = doc_chunks_map.get(doc_id, [])
                doc_vector_ids = [c.id for c in doc_chunks]

                # Add vector IDs
                if doc_vector_ids:
                    self.document_tracker.add_vector_ids(doc_id, doc_vector_ids)

                # Add graph node IDs
                if graph_node_ids and len(tracked_doc_ids) > 0:
                    doc_graph_nodes = graph_node_ids[:len(graph_node_ids)//len(tracked_doc_ids) + 1]
                    self.document_tracker.add_graph_node_ids(doc_id, doc_graph_nodes)

                # Update document status
                self.document_tracker.update_document(
                    doc_id,
                    status=DocumentStatus.COMPLETED.value,
                    chunk_count=len(doc_chunks),
                    processed_at=datetime.utcnow().isoformat() + "Z"
                )

            summary = {
                "status": "success",
                "job_id": job_id,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "document_ids": tracked_doc_ids,
                "metrics": {
                    "urls_submitted": len(urls),
                    "crawled_successfully": len(raw_paths),
                    "parsed_successfully": len(parsed_paths),
                    "chunked_successfully": len(chunk_files),
                    "total_chunks": len(all_chunks),
                    "vectors_stored": len(all_chunks),
                    "documents_tracked": len(tracked_doc_ids)
                },
                "artifacts": {
                    "raw_files": raw_paths,
                    "parsed_files": parsed_paths,
                    "chunk_files": chunk_files
                }
            }

            logger.info(f"Pipeline completed for job {job_id}")
            return summary

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            # Mark all documents as failed
            for doc_id in tracked_doc_ids:
                try:
                    self.document_tracker.update_document(
                        doc_id,
                        status=DocumentStatus.FAILED.value,
                        error_message=str(e)
                    )
                except Exception:
                    pass
            return {
                "status": "error",
                "job_id": job_id,
                "error": str(e),
                "document_ids": tracked_doc_ids
            }

    async def run_file_pipeline(
        self,
        file_paths: List[str],
        job_id: str = None,
        original_filenames: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Run the pipeline for uploaded files (skipping crawl step).

        Args:
            file_paths: List of paths to already saved raw files.
            job_id: Optional job identifier.
            original_filenames: Optional mapping of file_path -> original filename.

        Returns:
            Dictionary containing paths to generated artifacts.
        """
        if not job_id:
            job_id = f"job_upload_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

        logger.info(f"Starting file pipeline job {job_id} for {len(file_paths)} files")

        # Track documents for each file
        tracked_doc_ids = []
        file_to_doc_id = {}

        try:
            # Create document records for each file
            for file_path in file_paths:
                path_obj = Path(file_path)
                # Use original filename if provided, otherwise fall back to path name
                filename = (
                    original_filenames.get(file_path, path_obj.name)
                    if original_filenames
                    else path_obj.name
                )

                # Generate content hash for deduplication
                try:
                    with open(file_path, 'rb') as f:
                        content_hash = hashlib.sha256(f.read()).hexdigest()
                except Exception:
                    content_hash = None

                # Create document record with original filename as title
                doc = self.document_tracker.create_document(
                    title=filename,
                    source_type=DocumentSource.FILE_UPLOAD.value,
                    source_url=None,
                    file_path=str(file_path),
                    content_hash=content_hash,
                    metadata={
                        "job_id": job_id,
                        "original_filename": filename,
                        "file_extension": path_obj.suffix
                    }
                )
                tracked_doc_ids.append(doc.id)
                file_to_doc_id[file_path] = doc.id

                # Update status to processing
                self.document_tracker.update_document(doc.id, status=DocumentStatus.PROCESSING.value)

            # 1. Parse (Docling) - Run in thread pool to avoid blocking
            logger.info(f"Step 1: Parsing {len(file_paths)} uploaded documents...")
            # Update status to parsing
            for doc_id in tracked_doc_ids:
                self.document_tracker.update_document(doc_id, status="parsing")

            # Run parsing in thread pool (CPU-bound operation)
            loop = asyncio.get_event_loop()
            parsed_paths = await loop.run_in_executor(
                get_pipeline_executor(),
                lambda: self.parser.process_batch(file_paths, job_id)
            )

            if not parsed_paths:
                logger.warning("Parsing yielded no results.")
                # Mark documents as failed
                for doc_id in tracked_doc_ids:
                    self.document_tracker.update_document(
                        doc_id,
                        status=DocumentStatus.FAILED.value,
                        error_message="Parsing failed - no content extracted"
                    )
                return {
                    "status": "failed",
                    "job_id": job_id,
                    "error": "Parsing yielded no results",
                    "document_ids": tracked_doc_ids
                }

            # 2. Chunk - Run in thread pool to avoid blocking
            logger.info(f"Step 2: Chunking {len(parsed_paths)} parsed documents...")
            # Update status to chunking
            for doc_id in tracked_doc_ids:
                self.document_tracker.update_document(doc_id, status="chunking")

            # Run chunking in thread pool (CPU-bound operation)
            chunk_files = await loop.run_in_executor(
                get_pipeline_executor(),
                lambda: self.chunker.process_batch(parsed_paths, job_id)
            )

            all_chunks = []
            doc_chunks_map = {}  # Map doc_id to its chunks

            # Load chunks from files
            for chunk_file in chunk_files:
                batch = self.chunker.load_chunks(chunk_file)
                if batch:
                    for chunk in batch.chunks:
                        all_chunks.append(chunk)
                        # Try to map chunk back to document
                        source_path = chunk.metadata.get("source", "")
                        for file_path, doc_id in file_to_doc_id.items():
                            if file_path in source_path or Path(file_path).stem in chunk.doc_id:
                                if doc_id not in doc_chunks_map:
                                    doc_chunks_map[doc_id] = []
                                doc_chunks_map[doc_id].append(chunk)
                                break

            logger.info(f"Generated {len(all_chunks)} total chunks")

            # 3. Embed & Store Vectors in streaming batches to avoid memory exhaustion
            # Process in batches of 2000 chunks to keep memory usage reasonable
            EMBED_BATCH_SIZE = 2000
            vector_ids_stored = []

            if all_chunks:
                total_chunks = len(all_chunks)
                logger.info(f"Step 3: Generating embeddings and storing {total_chunks} chunks in batches of {EMBED_BATCH_SIZE}...")

                # Update status to embedding
                for doc_id in tracked_doc_ids:
                    self.document_tracker.update_document(doc_id, status="embedding")

                loop = asyncio.get_event_loop()

                for batch_start in range(0, total_chunks, EMBED_BATCH_SIZE):
                    batch_end = min(batch_start + EMBED_BATCH_SIZE, total_chunks)
                    batch_num = batch_start // EMBED_BATCH_SIZE + 1
                    total_batches = (total_chunks + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE

                    logger.info(f"Processing batch {batch_num}/{total_batches}: chunks {batch_start} to {batch_end}")

                    # Get batch of chunks
                    batch_chunks = all_chunks[batch_start:batch_end]
                    batch_texts = [chunk.text for chunk in batch_chunks]

                    # Embed this batch (async to avoid blocking)
                    batch_embeddings = await self.embedder.embed_batch_async(batch_texts, show_progress=False)

                    # Assign embeddings to chunks
                    for i, chunk in enumerate(batch_chunks):
                        if i < len(batch_embeddings):
                            chunk.embedding = batch_embeddings[i]

                    # Store this batch to ChromaDB immediately (don't hold in memory)
                    await loop.run_in_executor(
                        get_pipeline_executor(),
                        lambda chunks=batch_chunks: self.vector_store.add_chunks(chunks)
                    )

                    # Track stored IDs
                    vector_ids_stored.extend([chunk.id for chunk in batch_chunks])

                    # Clear embeddings from memory after storing
                    for chunk in batch_chunks:
                        chunk.embedding = None

                    logger.info(f"Batch {batch_num}/{total_batches} complete: {len(batch_chunks)} chunks embedded and stored")

            # 4. Build Knowledge Graph via Graphiti
            graph_node_ids = []
            if all_chunks:
                logger.info("Step 5: Building Knowledge Graph in FalkorDB via Graphiti...")
                # Update status to graphing
                for doc_id in tracked_doc_ids:
                    self.document_tracker.update_document(doc_id, status="graphing")
                await self.graphiti_service.initialize()

                # Build graph by adding chunks as episodes
                from datetime import timezone

                # Combine chunks into episodes (batch by source document)
                episode_content = "\n\n".join([chunk.text for chunk in all_chunks[:50]])  # Limit for initial processing

                try:
                    result = await self.graphiti_service.add_episode(
                        content=episode_content,
                        name=f"Upload job {job_id}",
                        source_description=f"File upload job {job_id}",
                        reference_time=datetime.now(timezone.utc),
                        source_type="text",
                    )

                    if result:
                        # Result is a dict from GraphitiService
                        graph_node_ids = [e.get("uuid") for e in result.get("entities", [])]
                        logger.info(f"Graph build result: {result.get('nodes_created', 0)} nodes, {result.get('edges_created', 0)} edges")
                    else:
                        logger.warning("No graph results returned from Graphiti")
                except Exception as e:
                    logger.warning(f"Graph building failed: {e}")

            # 5. Update document tracker with results
            for doc_id in tracked_doc_ids:
                doc_chunks = doc_chunks_map.get(doc_id, [])
                doc_vector_ids = [c.id for c in doc_chunks]

                # Add vector IDs
                if doc_vector_ids:
                    self.document_tracker.add_vector_ids(doc_id, doc_vector_ids)

                # Add graph node IDs (distribute among documents)
                if graph_node_ids and len(tracked_doc_ids) > 0:
                    # Simple distribution - in reality you'd want to track which nodes belong to which doc
                    doc_graph_nodes = graph_node_ids[:len(graph_node_ids)//len(tracked_doc_ids) + 1]
                    self.document_tracker.add_graph_node_ids(doc_id, doc_graph_nodes)

                # Update document status
                self.document_tracker.update_document(
                    doc_id,
                    status=DocumentStatus.COMPLETED.value,
                    chunk_count=len(doc_chunks),
                    processed_at=datetime.utcnow().isoformat() + "Z"
                )

            summary = {
                "status": "success",
                "job_id": job_id,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "document_ids": tracked_doc_ids,
                "metrics": {
                    "files_submitted": len(file_paths),
                    "parsed_successfully": len(parsed_paths),
                    "chunked_successfully": len(chunk_files),
                    "total_chunks": len(all_chunks),
                    "vectors_stored": len(all_chunks),
                    "documents_tracked": len(tracked_doc_ids)
                },
                "artifacts": {
                    "raw_files": file_paths,
                    "parsed_files": parsed_paths,
                    "chunk_files": chunk_files
                }
            }

            logger.info(f"File pipeline completed for job {job_id}")
            return summary

        except Exception as e:
            logger.error(f"File pipeline failed: {e}")
            # Mark all documents as failed
            for doc_id in tracked_doc_ids:
                try:
                    self.document_tracker.update_document(
                        doc_id,
                        status=DocumentStatus.FAILED.value,
                        error_message=str(e)
                    )
                except Exception:
                    pass
            return {
                "status": "error",
                "job_id": job_id,
                "error": str(e),
                "document_ids": tracked_doc_ids
            }
