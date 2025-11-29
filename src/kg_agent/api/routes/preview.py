"""
API endpoints for file preview (parse without storage).

This endpoint parses files and returns their content for immediate use
in chat context, WITHOUT storing them in the knowledge base.
"""
import tempfile
import os
from typing import List, Dict, Any
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from ...core.logging import logger
from ...parser.service import ParserService
from ...chunker.service import ChunkerService

router = APIRouter()

# Initialize services (lightweight, no DB connections)
parser_service = ParserService(output_dir=tempfile.gettempdir())
chunker_service = ChunkerService(output_dir=tempfile.gettempdir())


class PreviewedFile(BaseModel):
    """A previewed file with its content."""
    filename: str
    content: str
    chunk_count: int
    file_type: str
    size_bytes: int


class PreviewResponse(BaseModel):
    """Response from the preview endpoint."""
    status: str
    files: List[PreviewedFile]
    total_content_length: int
    message: str


def extract_text_from_parsed(parsed_data: Dict[str, Any]) -> str:
    """
    Extract plain text from a Docling parsed document.

    Args:
        parsed_data: The JSON output from Docling parser

    Returns:
        Plain text content from the document
    """
    text_parts = []

    # Try to get text from various Docling structures
    if "texts" in parsed_data:
        for text_item in parsed_data["texts"]:
            if isinstance(text_item, dict):
                text_parts.append(text_item.get("text", ""))
            elif isinstance(text_item, str):
                text_parts.append(text_item)

    # Also check for markdown export
    if "main_text" in parsed_data:
        text_parts.append(parsed_data["main_text"])

    # Check body content
    if "body" in parsed_data:
        body = parsed_data["body"]
        if isinstance(body, str):
            text_parts.append(body)
        elif isinstance(body, list):
            for item in body:
                if isinstance(item, dict):
                    text_parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    text_parts.append(item)

    # For GPT chat exports, look for conversations
    if "conversations" in parsed_data:
        for conv in parsed_data["conversations"]:
            if "messages" in conv:
                for msg in conv["messages"]:
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    text_parts.append(f"[{role}]: {content}")

    return "\n\n".join(filter(None, text_parts))


def read_text_file(file_path: Path) -> str:
    """Read a plain text file directly."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        logger.warning(f"Could not read as text: {e}")
        return ""


@router.post("/preview", response_model=PreviewResponse)
async def preview_files(
    files: List[UploadFile] = File(...)
) -> PreviewResponse:
    """
    Preview uploaded files by parsing and returning their content.

    This does NOT store files in the knowledge base - it only extracts
    text content for immediate use in chat context.

    Supported formats:
    - PDF (via Docling)
    - HTML (via Docling, including GPT chat exports)
    - Plain text files (.txt, .md, .json, .csv, etc.)

    Returns:
        PreviewResponse with file contents for agent context
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    previewed_files: List[PreviewedFile] = []
    total_content_length = 0

    for file in files:
        temp_path = None
        try:
            # Save to temp file
            suffix = Path(file.filename or "file").suffix or ".txt"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                content = await file.read()
                tmp.write(content)
                temp_path = tmp.name

            file_size = len(content)
            file_type = suffix.lstrip('.')

            # Determine parsing strategy based on file type
            text_extensions = {'.txt', '.md', '.json', '.csv', '.xml', '.yaml', '.yml', '.log', '.py', '.js', '.ts', '.jsx', '.tsx', '.css', '.html'}

            extracted_text = ""
            chunk_count = 0

            if suffix.lower() in text_extensions and suffix.lower() != '.html':
                # Plain text - read directly
                extracted_text = read_text_file(Path(temp_path))
                chunk_count = 1
            else:
                # Use Docling for PDF, HTML, etc.
                import json
                parsed_path = parser_service.parse_file(temp_path, job_id="preview")

                if parsed_path and Path(parsed_path).exists():
                    with open(parsed_path, 'r', encoding='utf-8') as f:
                        parsed_data = json.load(f)

                    extracted_text = extract_text_from_parsed(parsed_data)

                    # Clean up parsed file
                    try:
                        os.unlink(parsed_path)
                    except:
                        pass

            # If we got text, count approximate chunks
            if extracted_text:
                # Rough estimate: ~500 chars per chunk
                chunk_count = max(1, len(extracted_text) // 500)
                total_content_length += len(extracted_text)

                previewed_files.append(PreviewedFile(
                    filename=file.filename or "unnamed",
                    content=extracted_text,
                    chunk_count=chunk_count,
                    file_type=file_type,
                    size_bytes=file_size
                ))
            else:
                # Still add file info even if we couldn't extract text
                previewed_files.append(PreviewedFile(
                    filename=file.filename or "unnamed",
                    content=f"[Could not extract text from {file_type} file]",
                    chunk_count=0,
                    file_type=file_type,
                    size_bytes=file_size
                ))

        except Exception as e:
            logger.error(f"Error previewing file {file.filename}: {e}")
            previewed_files.append(PreviewedFile(
                filename=file.filename or "unnamed",
                content=f"[Error processing file: {str(e)}]",
                chunk_count=0,
                file_type="unknown",
                size_bytes=0
            ))
        finally:
            # Clean up temp file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass

    return PreviewResponse(
        status="success",
        files=previewed_files,
        total_content_length=total_content_length,
        message=f"Previewed {len(previewed_files)} files ({total_content_length:,} chars total)"
    )


@router.post("/preview/text", response_model=PreviewResponse)
async def preview_text_content(
    content: str,
    filename: str = "pasted_content.txt"
) -> PreviewResponse:
    """
    Preview pasted text content (no file upload needed).

    Useful for when users paste content directly into chat.
    """
    if not content:
        raise HTTPException(status_code=400, detail="No content provided")

    return PreviewResponse(
        status="success",
        files=[PreviewedFile(
            filename=filename,
            content=content,
            chunk_count=max(1, len(content) // 500),
            file_type="txt",
            size_bytes=len(content.encode('utf-8'))
        )],
        total_content_length=len(content),
        message=f"Previewed text content ({len(content):,} chars)"
    )

