"""
OCR API - Endpoints for text detection and extraction
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from core.ocr_manager import ocr_manager

router = APIRouter()


class OCRDetectRequest(BaseModel):
    image_path: str
    languages: List[str] = ['en']
    conf_threshold: float = 0.5


class OCRBatchRequest(BaseModel):
    image_paths: List[str]
    languages: List[str] = ['en']
    conf_threshold: float = 0.5


class OCRExportRequest(BaseModel):
    results: dict
    output_path: str
    format: str = "json"  # json, txt, csv


@router.post("/detect")
async def detect_text(request: OCRDetectRequest):
    """Detect and extract text from an image"""
    result = ocr_manager.detect_text(
        image_path=request.image_path,
        languages=request.languages,
        conf_threshold=request.conf_threshold
    )
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result


@router.post("/extract")
async def extract_text(image_path: str, languages: List[str] = ['en']):
    """Extract just the text content from an image"""
    text = ocr_manager.extract_text_only(image_path, languages)
    return {"text": text, "image_path": image_path}


@router.post("/batch")
async def batch_detect(request: OCRBatchRequest):
    """Run OCR on multiple images"""
    result = ocr_manager.batch_detect(
        image_paths=request.image_paths,
        languages=request.languages,
        conf_threshold=request.conf_threshold
    )
    return result


@router.post("/visualize")
async def visualize_detections(
    image_path: str,
    project_path: str
):
    """Generate an annotated image with text detections"""
    # First detect text
    detection_result = ocr_manager.detect_text(image_path)
    
    if "error" in detection_result:
        raise HTTPException(status_code=400, detail=detection_result["error"])
    
    # Generate visualization
    import os
    output_dir = os.path.join(project_path, "exports", "ocr")
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"ocr_{base_name}")
    
    viz_path = ocr_manager.visualize_detections(
        image_path=image_path,
        detections=detection_result.get("detections", []),
        output_path=output_path
    )
    
    return {
        "visualization_path": viz_path,
        "detection_count": detection_result.get("detection_count", 0),
        "full_text": detection_result.get("full_text", "")
    }


@router.post("/export")
async def export_results(request: OCRExportRequest):
    """Export OCR results to file"""
    result = ocr_manager.export_results(
        results=request.results,
        output_path=request.output_path,
        format=request.format
    )
    return result


@router.get("/languages")
async def get_supported_languages():
    """Get list of supported OCR languages"""
    return {
        "languages": ocr_manager.supported_languages,
        "default": "en"
    }
