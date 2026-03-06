"""
Change Detection API - Endpoints for comparing images and detecting changes
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import os

router = APIRouter()


class ChangeDetectionRequest(BaseModel):
    before_path: str
    after_path: str
    threshold: Optional[float] = 30  # For simple: 0-255, for SSIM: 0-1
    min_area: Optional[int] = 100
    method: Optional[str] = "simple"  # "simple" or "ssim"
    align: Optional[bool] = False
    output_dir: Optional[str] = None


@router.post("/detect")
async def detect_changes(request: ChangeDetectionRequest):
    """Detect changes between two images."""
    from core.change_detection import ChangeDetectionEngine
    
    # Validate paths
    if not os.path.exists(request.before_path):
        raise HTTPException(status_code=404, detail="Before image not found")
    if not os.path.exists(request.after_path):
        raise HTTPException(status_code=404, detail="After image not found")
    
    try:
        engine = ChangeDetectionEngine(output_dir=request.output_dir)
        
        if request.method == "ssim":
            result = engine.detect_changes_ssim(
                before_path=request.before_path,
                after_path=request.after_path,
                threshold=min(1.0, request.threshold) if request.threshold <= 1 else 0.9,
                min_area=request.min_area
            )
        else:
            result = engine.detect_changes_simple(
                before_path=request.before_path,
                after_path=request.after_path,
                threshold=int(request.threshold) if request.threshold > 1 else 30,
                min_area=request.min_area,
                align=request.align
            )
        
        # Convert regions to serializable format
        regions = []
        for r in result.regions:
            regions.append({
                "x": r.x,
                "y": r.y,
                "width": r.width,
                "height": r.height,
                "change_type": r.change_type.value,
                "confidence": r.confidence,
                "area_percentage": r.area_percentage
            })
        
        return {
            "status": "success",
            "has_changes": result.has_changes,
            "change_percentage": result.change_percentage,
            "regions_count": len(regions),
            "regions": regions,
            "diff_image_path": result.diff_image_path,
            "overlay_image_path": result.overlay_image_path
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/methods")
async def get_available_methods():
    """Get list of available change detection methods."""
    return {
        "methods": [
            {
                "id": "simple",
                "name": "Simple Difference",
                "description": "Fast pixel-by-pixel comparison. Best for controlled environments.",
                "parameters": {
                    "threshold": "Pixel difference threshold (0-255)",
                    "align": "Align images before comparison"
                }
            },
            {
                "id": "ssim",
                "name": "Structural Similarity (SSIM)",
                "description": "Robust to lighting changes. Better for outdoor/variable conditions.",
                "parameters": {
                    "threshold": "SSIM threshold (0-1, higher = more similar required)"
                }
            }
        ]
    }
