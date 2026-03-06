"""
Video API - Endpoints for video processing and tracking
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any
from core.video_manager import video_manager

router = APIRouter()


class FrameExtractionRequest(BaseModel):
    video_path: str
    output_dir: str
    interval: int = 1
    max_frames: Optional[int] = None


class TrackingRequest(BaseModel):
    video_path: str
    model_path: str
    output_path: Optional[str] = None
    conf_threshold: float = 0.5
    tracker: str = "bytetrack"


# Store tracking job status
tracking_jobs: Dict[str, Dict[str, Any]] = {}


@router.get("/info")
async def get_video_info(video_path: str):
    """Get metadata about a video file"""
    result = video_manager.get_video_info(video_path)
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result


@router.post("/extract-frames")
async def extract_frames(request: FrameExtractionRequest):
    """Extract frames from a video at specified intervals"""
    result = video_manager.extract_frames(
        video_path=request.video_path,
        output_dir=request.output_dir,
        interval=request.interval,
        max_frames=request.max_frames
    )
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    
    return result


@router.post("/track")
async def run_tracking(request: TrackingRequest, background_tasks: BackgroundTasks):
    """Run object tracking on a video (async)"""
    import uuid
    
    job_id = str(uuid.uuid4())
    
    # Initialize job
    tracking_jobs[job_id] = {
        "status": "running",
        "progress": 0,
        "total_frames": 0,
        "result": None
    }
    
    def run_track():
        def progress_callback(current: int, total: int):
            tracking_jobs[job_id]["progress"] = current
            tracking_jobs[job_id]["total_frames"] = total
        
        result = video_manager.run_tracking(
            video_path=request.video_path,
            model_path=request.model_path,
            output_path=request.output_path,
            conf_threshold=request.conf_threshold,
            tracker=request.tracker,
            progress_callback=progress_callback
        )
        
        tracking_jobs[job_id]["status"] = "completed" if "error" not in result else "failed"
        tracking_jobs[job_id]["result"] = result
    
    background_tasks.add_task(run_track)
    
    return {"job_id": job_id, "status": "started"}


@router.get("/track/{job_id}/status")
async def get_tracking_status(job_id: str):
    """Get status of a tracking job"""
    if job_id not in tracking_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return tracking_jobs[job_id]


@router.get("/supported-formats")
async def get_supported_formats():
    """Get list of supported video formats"""
    return {
        "formats": video_manager.supported_formats,
        "trackers": ["bytetrack", "botsort"]
    }
