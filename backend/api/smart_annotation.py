"""
Smart Annotation API - Advanced annotation assistance endpoints
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from core.smart_annotation import smart_annotation_tools

router = APIRouter()


class BboxInput(BaseModel):
    x: float
    y: float
    width: float
    height: float


class SnapRequest(BaseModel):
    image_path: str
    bbox: BboxInput
    snap_radius: int = 10


class ValidationRequest(BaseModel):
    annotations: List[Dict[str, Any]]
    image_width: int
    image_height: int
    rules: Optional[Dict[str, Any]] = None


class QualityRequest(BaseModel):
    annotations: List[Dict[str, Any]]
    image_path: str


@router.post("/snap-to-edges")
async def snap_to_edges(request: SnapRequest):
    """Snap bounding box edges to detected image edges"""
    bbox_dict = {
        "x": request.bbox.x,
        "y": request.bbox.y,
        "width": request.bbox.width,
        "height": request.bbox.height
    }
    
    result = smart_annotation_tools.snap_to_edges(
        image_path=request.image_path,
        bbox=bbox_dict,
        snap_radius=request.snap_radius
    )
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result


@router.post("/validate")
async def validate_annotations(request: ValidationRequest):
    """Validate annotations against quality rules"""
    return smart_annotation_tools.validate_annotations(
        annotations=request.annotations,
        image_width=request.image_width,
        image_height=request.image_height,
        rules=request.rules
    )


@router.post("/quality-score")
async def calculate_quality_score(request: QualityRequest):
    """Calculate annotation quality score"""
    return smart_annotation_tools.calculate_quality_score(
        annotations=request.annotations,
        image_path=request.image_path
    )


@router.post("/find-duplicates")
async def find_duplicates(
    annotations: List[Dict[str, Any]],
    iou_threshold: float = 0.9
):
    """Find potential duplicate annotations"""
    return {
        "duplicates": smart_annotation_tools.find_duplicate_annotations(
            annotations=annotations,
            iou_threshold=iou_threshold
        )
    }


@router.post("/suggest")
async def suggest_annotations(
    image_path: str,
    existing_annotations: List[Dict[str, Any]] = [],
    model_path: Optional[str] = None
):
    """Get AI-powered annotation suggestions"""
    return smart_annotation_tools.suggest_annotations(
        image_path=image_path,
        existing_annotations=existing_annotations,
        model_path=model_path
    )
