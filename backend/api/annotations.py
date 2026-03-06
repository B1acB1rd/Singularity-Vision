from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from core.annotation_manager import annotation_manager

router = APIRouter()

class ClassRequest(BaseModel):
    project_path: str
    class_name: str

class SaveAnnotationRequest(BaseModel):
    project_path: str
    image_id: str
    annotations: List[Dict[str, Any]]
    image_size: Dict[str, int]

@router.get("/classes")
async def get_classes(project_path: str):
    return dataset_manager.get_classes(project_path)
    # Note: Using dataset_manager vs annotation_manager? 
    # Logic is in annotation_manager now, so let's use that.
    return annotation_manager.get_classes(project_path)

@router.post("/classes")
async def add_class(request: ClassRequest):
    try:
        classes = annotation_manager.add_class(request.project_path, request.class_name)
        return {"classes": classes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/classes")
async def remove_class(request: ClassRequest):
    try:
        classes = annotation_manager.remove_class(request.project_path, request.class_name)
        return {"classes": classes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/save")
async def save_annotation(request: SaveAnnotationRequest):
    try:
        annotation_manager.save_annotation(
            request.project_path, 
            request.image_id, 
            request.annotations, 
            request.image_size
        )
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/load")
async def load_annotation(project_path: str, image_id: str):
    try:
        annotations = annotation_manager.get_annotation(project_path, image_id)
        return {"annotations": annotations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
