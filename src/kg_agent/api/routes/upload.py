"""
API endpoints for file uploads.
"""
import uuid
from datetime import datetime
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks

from ...core.logging import logger
from ...pipeline.manager import PipelineManager
from ...crawler.storage import StorageService

router = APIRouter()
pipeline_manager = PipelineManager()
storage_service = StorageService()

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

    try:
        for file in files:
            content = await file.read()
            path = await storage_service.save_uploaded_file(content, file.filename, job_id)
            if path:
                saved_paths.append(path)
            else:
                logger.error(f"Failed to save uploaded file: {file.filename}")

        if not saved_paths:
            raise HTTPException(status_code=500, detail="Failed to save any files")

        # Trigger pipeline in background
        background_tasks.add_task(pipeline_manager.run_file_pipeline, saved_paths, job_id)

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

