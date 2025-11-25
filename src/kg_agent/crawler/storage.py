import hashlib
import os
from pathlib import Path
from typing import Optional
from datetime import datetime

from ..models.crawl_result import CrawlResult
from ..core.logging import logger

class StorageService:
    """
    Service to store raw crawl artifacts for downstream processing.
    """

    def __init__(self, base_dir: str = "data/raw"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_safe_filename(self, url: str) -> str:
        """Generate a safe filename from URL."""
        return hashlib.sha256(url.encode()).hexdigest()

    async def save_raw_content(self, crawl_result: CrawlResult, job_id: str = "default") -> Optional[str]:
        """
        Save raw HTML or PDF content to storage.

        Args:
            crawl_result: The result from the crawler.
            job_id: Identifier for the current crawl job/batch.

        Returns:
            Path to the saved file, or None if no content to save.
        """
        if not crawl_result.success:
            logger.warning(f"Skipping storage for failed crawl: {crawl_result.url}")
            return None

        job_dir = self.base_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        filename_base = self._get_safe_filename(crawl_result.url)

        # Prioritize PDF if available (e.g. if it was a PDF URL or PDF generation was requested)
        # But actually, if we downloaded a PDF, it might be in a temp location.
        # CrawlResult.pdf_path usually points to a temp file or cache file.
        # If we have HTML, we save that.

        saved_path = None

        try:
            if crawl_result.pdf_path and os.path.exists(crawl_result.pdf_path):
                # Copy PDF
                extension = ".pdf"
                target_path = job_dir / f"{filename_base}{extension}"
                # We read and write to ensure it's in our controlled dir
                with open(crawl_result.pdf_path, "rb") as src, open(target_path, "wb") as dst:
                    dst.write(src.read())
                saved_path = str(target_path)

            elif crawl_result.html:
                # Save HTML
                extension = ".html"
                target_path = job_dir / f"{filename_base}{extension}"
                with open(target_path, "w", encoding="utf-8") as f:
                    f.write(crawl_result.html)
                saved_path = str(target_path)

            if saved_path:
                logger.info(f"Saved raw content for {crawl_result.url} to {saved_path}")

                # Also save metadata sidecar
                meta_path = job_dir / f"{filename_base}.meta.json"
                with open(meta_path, "w", encoding="utf-8") as f:
                    f.write(crawl_result.json())

            return saved_path

        except Exception as e:
            logger.error(f"Failed to save raw content for {crawl_result.url}: {e}")
            return None

    async def save_uploaded_file(self, file_content: bytes, filename: str, job_id: str = "default") -> Optional[str]:
        """
        Save an uploaded file to storage.

        Args:
            file_content: The binary content of the file.
            filename: Original filename.
            job_id: Identifier for the current job.

        Returns:
            Path to the saved file.
        """
        try:
            job_dir = self.base_dir / job_id
            job_dir.mkdir(parents=True, exist_ok=True)

            # Sanitize filename to be safe but preserve extension
            name, ext = os.path.splitext(filename)
            safe_name = hashlib.sha256(name.encode()).hexdigest()[:16]
            target_filename = f"{safe_name}{ext}"

            target_path = job_dir / target_filename

            with open(target_path, "wb") as f:
                f.write(file_content)

            logger.info(f"Saved uploaded file {filename} to {target_path}")
            return str(target_path)

        except Exception as e:
            logger.error(f"Failed to save uploaded file {filename}: {e}")
            return None
