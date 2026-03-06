"""
Export API - Endpoints for exporting trained models
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from core.export_manager import export_manager
from core.annotation_manager import annotation_manager

router = APIRouter()

class ExportRequest(BaseModel):
    model_path: str
    output_dir: str
    format: str = 'onnx'
    imgsz: int = 640
    half: bool = False
    dynamic: bool = False

class AnnotationExportRequest(BaseModel):
    project_path: str
    format: str = 'voc'  # 'voc', 'coco', 'yolo'

@router.get("/models")
async def get_exportable_models(project_path: str):
    """Get list of trained models that can be exported"""
    try:
        models = export_manager.get_exportable_models(project_path)
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/formats")
async def get_supported_formats():
    """Get list of supported export formats"""
    return {
        "model_formats": [
            {"id": "onnx", "name": "ONNX", "description": "Open Neural Network Exchange - cross-platform"},
            {"id": "torchscript", "name": "TorchScript", "description": "PyTorch optimized format"},
            {"id": "openvino", "name": "OpenVINO", "description": "Intel optimized inference"},
            {"id": "coreml", "name": "CoreML", "description": "Apple devices (iOS/macOS)"},
            {"id": "tflite", "name": "TensorFlow Lite", "description": "Mobile and embedded devices"},
        ],
        "annotation_formats": [
            {"id": "yolo", "name": "YOLO", "description": "YOLO txt format (class x y w h)"},
            {"id": "voc", "name": "Pascal VOC", "description": "XML format with absolute coordinates"},
            {"id": "coco", "name": "COCO", "description": "JSON format with annotations list"},
        ]
    }

@router.post("/run")
async def export_model(request: ExportRequest):
    """Export a model to the specified format"""
    try:
        result = export_manager.export_model(
            model_path=request.model_path,
            output_dir=request.output_dir,
            format=request.format,
            imgsz=request.imgsz,
            half=request.half,
            dynamic=request.dynamic
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/annotations")
async def export_annotations(request: AnnotationExportRequest):
    """Export annotations to specified format (VOC, COCO, etc.)"""
    import os
    import json
    from PIL import Image
    
    try:
        project_path = request.project_path
        images_dir = os.path.join(project_path, "datasets", "images")
        labels_dir = os.path.join(project_path, "datasets", "labels")
        
        if not os.path.exists(images_dir):
            raise HTTPException(status_code=404, detail="Images directory not found")
        
        # Get classes
        classes = annotation_manager.get_classes(project_path)
        exported_files = []
        
        # Find all images
        image_files = [f for f in os.listdir(images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for image_file in image_files:
            image_path = os.path.join(images_dir, image_file)
            
            # Get image size
            with Image.open(image_path) as img:
                width, height = img.size
            
            # Get annotations
            annotations = annotation_manager.get_annotation(project_path, image_file)
            
            if annotations and request.format == 'voc':
                xml_path = annotation_manager.export_to_voc(
                    project_path=project_path,
                    image_name=image_file,
                    annotations=annotations,
                    image_size={"width": width, "height": height},
                    classes=classes
                )
                exported_files.append(xml_path)
        
        return {
            "status": "success",
            "format": request.format,
            "exported_count": len(exported_files),
            "output_dir": os.path.join(project_path, "exports", request.format)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

