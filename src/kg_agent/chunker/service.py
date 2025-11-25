import json
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..models.chunk import Chunk, ChunkBatch
from ..core.logging import logger

class ChunkerService:
    """
    Service to split parsed documents into semantic chunks.
    """

    def __init__(self,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 output_dir: str = "data/chunks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def _extract_text_from_docling_json(self, doc_data: Dict[str, Any]) -> str:
        """
        Extract the main text content from Docling's JSON output.
        Docling 2.x export structure usually has 'main_text' or we can iterate body items.
        """
        # Try to get simple text representation first
        # Note: The exact field depends on Docling version, but 'main_text' is common in exports
        # If not, we might need to iterate 'body' -> 'text'

        # Attempt to reconstruct from paragraphs if main_text is missing
        text_parts = []

        # Check for 'body' structure (Docling 2.0+)
        if "body" in doc_data and "children" in doc_data["body"]:
            # Recursive traversal could be needed, but let's try flat iteration if available
            # Or just use the export's text field if it exists
            pass

        # Fallback/Simpler approach: Check for 'texts' or similar, or just look for the 'text' field if at root
        # DoclingDocument often has a Markdown export which is good for chunking.
        # If we saved the export_to_dict, it might have 'markdown' or 'main_text'.

        # Let's assume the parser might have saved a "markdown" field or we use a generic traversal.
        # For now, let's look for 'main_text' or traverse 'texts'.

        # Docling v2:
        texts = []
        if "texts" in doc_data:
            for t in doc_data["texts"]:
                if isinstance(t, dict) and "text" in t:
                    texts.append(t["text"])

        if not texts and "paragraphs" in doc_data: # Legacy or alternative
             for p in doc_data["paragraphs"]:
                 if isinstance(p, dict):
                     texts.append(p.get("text", ""))
                 elif isinstance(p, str):
                     texts.append(p)

        return "\n\n".join(texts) if texts else ""

    def chunk_file(self, parsed_file_path: str, job_id: str = "default") -> Optional[str]:
        """
        Chunk a parsed document file.

        Args:
            parsed_file_path: Path to the JSON file from ParserService.
            job_id: Job identifier.

        Returns:
            Path to the saved chunks JSON file.
        """
        path_obj = Path(parsed_file_path)
        if not path_obj.exists():
            logger.error(f"Parsed file not found: {parsed_file_path}")
            return None

        try:
            with open(path_obj, "r", encoding="utf-8") as f:
                doc_data = json.load(f)

            # Extract metadata
            source_meta = doc_data.get("_metadata", {})
            original_source = source_meta.get("source_path", str(path_obj))

            # Extract text
            # We might want to prefer Markdown if Docling produced it, as it preserves headers
            # For this implementation, let's try to find a text field or extract it.
            # If Docling's export_to_dict was used, we need to navigate it.
            # Simpler hack: If 'main_text' exists (it often does in exports), use it.
            # Otherwise, we might need to rely on the parser saving markdown content specifically.

            # Let's update the Parser to ensure we have a good text source, or implement robust extraction here.
            # I'll implement a heuristic here.

            full_text = ""

            # Check common Docling export fields
            if "main_text" in doc_data:
                full_text = doc_data["main_text"]
            elif "markdown" in doc_data:
                full_text = doc_data["markdown"]
            else:
                full_text = self._extract_text_from_docling_json(doc_data)

            if not full_text:
                logger.warning(f"No text extracted from {parsed_file_path}")
                return None

            # Split text
            text_chunks = self.splitter.split_text(full_text)

            # Create Chunk objects
            chunks = []
            doc_id = path_obj.stem # Use filename as doc_id

            for i, text in enumerate(text_chunks):
                chunk_id = f"{doc_id}_{i}"
                chunk = Chunk(
                    id=chunk_id,
                    doc_id=doc_id,
                    text=text,
                    index=i,
                    metadata={
                        "source": original_source,
                        "job_id": job_id,
                        "chunk_size": len(text)
                    }
                )
                chunks.append(chunk)

            # Save chunks
            batch = ChunkBatch(chunks=chunks, job_id=job_id)

            job_dir = self.output_dir / job_id
            job_dir.mkdir(parents=True, exist_ok=True)

            output_filename = f"{doc_id}_chunks.json"
            output_path = job_dir / output_filename

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(batch.model_dump_json(indent=2))

            logger.info(f"Saved {len(chunks)} chunks to {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Error chunking {parsed_file_path}: {e}")
            return None

    def process_batch(self, file_paths: List[str], job_id: str = "default") -> List[str]:
        """Process a batch of parsed files."""
        results = []
        for path in file_paths:
            if path:
                res = self.chunk_file(path, job_id)
                if res:
                    results.append(res)
        return results

    def load_chunks(self, chunk_file_path: str) -> Optional[ChunkBatch]:
        """
        Load chunks from a previously saved chunk file.

        Args:
            chunk_file_path: Path to the JSON file containing chunks.

        Returns:
            ChunkBatch object if successful, None otherwise.
        """
        path_obj = Path(chunk_file_path)
        if not path_obj.exists():
            logger.error(f"Chunk file not found: {chunk_file_path}")
            return None

        try:
            with open(path_obj, "r", encoding="utf-8") as f:
                data = json.load(f)

            return ChunkBatch(**data)

        except Exception as e:
            logger.error(f"Error loading chunks from {chunk_file_path}: {e}")
            return None

