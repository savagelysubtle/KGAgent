"""
API endpoints for file uploads.
"""
import uuid
import asyncio
from datetime import datetime
from typing import List, Dict

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks

from ...core.logging import logger
from ...pipeline.manager import PipelineManager
from ...crawler.storage import StorageService

router = APIRouter()
pipeline_manager = PipelineManager()
storage_service = StorageService()


def run_async_pipeline(saved_paths: List[str], job_id: str, original_filenames: Dict[str, str]):
    """
    Wrapper to run async pipeline in a background task.

    FastAPI's BackgroundTasks works best with sync functions, so we create
    a new event loop for the async pipeline execution.
    """
    try:
        # Create a new event loop for this background task
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                pipeline_manager.run_file_pipeline(saved_paths, job_id, original_filenames)
            )
            logger.info(f"Background pipeline completed for job {job_id}: {result.get('status', 'unknown')}")
            return result
        finally:
            loop.close()
    except Exception as e:
        logger.error(f"Background pipeline failed for job {job_id}: {e}", exc_info=True)

@router.post("/upload", response_model=dict, status_code=202)
async def upload_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    """
    Upload files for processing (Parse -> Chunk -> Graph).
    Files are processed asynchronously.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    job_id = f"upload_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

    saved_paths = []
    original_filenames = {}  # Map saved path -> original filename

    try:
        for file in files:
            content = await file.read()
            result = await storage_service.save_uploaded_file(content, file.filename, job_id)
            if result:
                saved_path, original_name = result
                saved_paths.append(saved_path)
                original_filenames[saved_path] = original_name
            else:
                logger.error(f"Failed to save uploaded file: {file.filename}")

        if not saved_paths:
            raise HTTPException(status_code=500, detail="Failed to save any files")

        # Trigger pipeline in background with original filenames
        # Use the sync wrapper to properly run the async pipeline
        background_tasks.add_task(
            run_async_pipeline,
            saved_paths,
            job_id,
            original_filenames
        )

        return {
            "job_id": job_id,
            "status": "accepted",
            "files_received": len(files),
            "files_saved": len(saved_paths),
            "message": "Files uploaded and processing started"
        }

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

