import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import DoclingDocument

from ..core.logging import logger
from .gpt_chat_parser import get_gpt_chat_parser

class ParserService:
    """
    Service to parse raw documents using Docling.
    Also handles special formats like GPT chat exports.
    """

    def __init__(self, output_dir: str = "data/parsed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.converter = DocumentConverter()
        self.gpt_parser = get_gpt_chat_parser()

    def _is_gpt_export(self, file_path: Path) -> bool:
        """Check if file is a GPT chat export by examining content."""
        if file_path.suffix.lower() != '.html':
            return False

        try:
            # Read first 50KB to check for GPT export indicators
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(50000)
            return self.gpt_parser.is_gpt_export(content)
        except:
            return False

    def parse_file(self, file_path: str, job_id: str = "default") -> Optional[str]:
        """
        Parse a file and save the structured output.

        Args:
            file_path: Path to the raw file (HTML/PDF).
            job_id: Batch/Job identifier.

        Returns:
            Path to the saved parsed JSON file.
        """
        path_obj = Path(file_path)
        if not path_obj.exists():
            logger.error(f"File not found: {file_path}")
            return None

        try:
            # Check if this is a GPT chat export
            if self._is_gpt_export(path_obj):
                logger.info(f"Detected GPT chat export: {file_path}")
                return self.gpt_parser.parse_file(file_path, job_id)

            logger.info(f"Parsing file with Docling: {file_path}")

            # Convert with Docling
            result = self.converter.convert(path_obj)
            doc: DoclingDocument = result.document

            # Export to dictionary
            # Docling's export_to_dict() provides a structured representation
            doc_data = doc.export_to_dict()

            # Add some metadata about the source
            doc_data["_metadata"] = {
                "source_path": str(file_path),
                "job_id": job_id,
                "parsed_at": str(doc_data.get("time", ""))
            }

            # Save to output dir
            job_dir = self.output_dir / job_id
            job_dir.mkdir(parents=True, exist_ok=True)

            output_filename = f"{path_obj.stem}.json"
            output_path = job_dir / output_filename

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(doc_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved parsed document to {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return None

    def process_batch(self, file_paths: List[str], job_id: str = "default") -> List[str]:
        """Process a batch of files."""
        results = []
        for path in file_paths:
            if path:
                res = self.parse_file(path, job_id)
                if res:
                    results.append(res)
        return results

