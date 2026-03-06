"""
Video Frame Labeling API - Endpoints for frame-by-frame video annotation
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from core.video_frame_labeler import video_frame_labeler

router = APIRouter()


class FrameAnnotationRequest(BaseModel):
    project_path: str
    video_name: str
    frame_index: int
    annotations: List[Dict[str, Any]]


class InterpolationRequest(BaseModel):
    start_annotation: Dict[str, Any]
    end_annotation: Dict[str, Any]
    start_frame: int
    end_frame: int
    target_frame: int


@router.get("/info")
async def get_video_info(video_path: str, start_frame: int = 0, frame_count: int = 10):
    """Get video frame information"""
    result = video_frame_labeler.get_video_frames(video_path, start_frame, frame_count)
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result


@router.post("/extract-frame")
async def extract_frame(video_path: str, frame_index: int, output_dir: str):
    """Extract a single frame from video"""
    result = video_frame_labeler.extract_frame(video_path, frame_index, output_dir)
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result


@router.post("/extract-keyframes")
async def extract_keyframes(
    video_path: str,
    output_dir: str,
    interval: int = 30,
    max_frames: int = 100
):
    """Extract keyframes at regular intervals for labeling"""
    result = video_frame_labeler.extract_keyframes(
        video_path=video_path,
        output_dir=output_dir,
        interval=interval,
        max_frames=max_frames
    )
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result


@router.post("/save")
async def save_frame_annotation(request: FrameAnnotationRequest):
    """Save annotations for a specific video frame"""
    result = video_frame_labeler.save_frame_annotation(
        project_path=request.project_path,
        video_name=request.video_name,
        frame_index=request.frame_index,
        annotations=request.annotations
    )
    return result


@router.get("/load")
async def load_frame_annotation(project_path: str, video_name: str, frame_index: int):
    """Load annotations for a specific video frame"""
    return video_frame_labeler.load_frame_annotation(
        project_path=project_path,
        video_name=video_name,
        frame_index=frame_index
    )


@router.post("/interpolate")
async def interpolate_annotation(request: InterpolationRequest):
    """Interpolate annotation between two keyframes"""
    result = video_frame_labeler.interpolate_annotations(
        start_annotation=request.start_annotation,
        end_annotation=request.end_annotation,
        start_frame=request.start_frame,
        end_frame=request.end_frame,
        target_frame=request.target_frame
    )
    return result


@router.post("/export")
async def export_video_annotations(
    project_path: str,
    video_name: str,
    output_format: str = "yolo"
):
    """Export all video frame annotations"""
    result = video_frame_labeler.export_video_annotations(
        project_path=project_path,
        video_name=video_name,
        output_format=output_format
    )
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result
