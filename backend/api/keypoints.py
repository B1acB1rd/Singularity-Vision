"""
Keypoints API - Endpoints for keypoint/pose annotation
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from core.keypoint_manager import keypoint_manager

router = APIRouter()


class KeypointData(BaseModel):
    x: float
    y: float
    visible: int = 2  # 0=not labeled, 1=labeled but occluded, 2=labeled and visible


class SaveKeypointsRequest(BaseModel):
    project_path: str
    image_id: str
    skeleton_type: str
    keypoints: List[KeypointData]
    class_name: str = "person"


@router.get("/presets")
async def get_skeleton_presets():
    """Get available skeleton presets"""
    return {"presets": keypoint_manager.get_presets()}


@router.get("/presets/{preset_id}")
async def get_skeleton_preset(preset_id: str):
    """Get a specific skeleton preset with full details"""
    preset = keypoint_manager.get_preset(preset_id)
    if not preset:
        raise HTTPException(status_code=404, detail="Preset not found")
    return preset


@router.post("/validate")
async def validate_keypoints(skeleton_type: str, keypoints: List[KeypointData]):
    """Validate keypoint annotation"""
    kp_dicts = [{"x": kp.x, "y": kp.y, "visible": kp.visible} for kp in keypoints]
    result = keypoint_manager.validate_keypoints(kp_dicts, skeleton_type)
    return result


@router.post("/save")
async def save_keypoints(request: SaveKeypointsRequest):
    """Save keypoint annotations for an image"""
    kp_dicts = [{"x": kp.x, "y": kp.y, "visible": kp.visible} for kp in request.keypoints]
    
    annotation = keypoint_manager.create_annotation(
        image_id=request.image_id,
        skeleton_type=request.skeleton_type,
        keypoints=kp_dicts,
        class_name=request.class_name
    )
    
    result = keypoint_manager.save_annotations(
        project_path=request.project_path,
        image_id=request.image_id,
        annotations=[annotation]
    )
    
    return result


@router.get("/load")
async def load_keypoints(project_path: str, image_id: str):
    """Load keypoint annotations for an image"""
    annotations = keypoint_manager.load_annotations(project_path, image_id)
    return {"annotations": annotations}


@router.post("/export/coco")
async def export_coco_keypoints(project_path: str, output_path: str):
    """Export all keypoints in COCO format"""
    result = keypoint_manager.export_coco_keypoints(project_path, output_path)
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result
