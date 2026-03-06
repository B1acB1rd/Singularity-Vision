"""
Model Hub API - Endpoints for browsing and downloading pre-trained models
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from core.model_hub import model_hub

router = APIRouter()


class DownloadRequest(BaseModel):
    model_id: str
    destination: Optional[str] = None


@router.get("/featured")
async def get_featured_models(task_type: Optional[str] = None):
    """Get list of featured/recommended models."""
    models = model_hub.get_featured_models(task_type)
    return {
        "status": "success",
        "count": len(models),
        "models": models
    }


@router.get("/search")
async def search_models(
    query: str = "",
    task: str = "object-detection",
    limit: int = 20
):
    """Search Hugging Face Hub for models."""
    try:
        models = model_hub.search_huggingface(query, task, limit)
        return {
            "status": "success",
            "count": len(models),
            "models": [m.to_dict() for m in models]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/info/{model_id:path}")
async def get_model_info(model_id: str):
    """Get detailed information about a specific model."""
    try:
        info = model_hub.get_model_info(model_id)
        if not info:
            raise HTTPException(status_code=404, detail="Model not found")
        
        return {
            "status": "success",
            "model": info.to_dict()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/download")
async def download_model(request: DownloadRequest):
    """Download a model from the hub."""
    try:
        path = model_hub.download_model(
            model_id=request.model_id,
            destination=request.destination
        )
        
        if not path:
            raise HTTPException(status_code=500, detail="Download failed")
        
        return {
            "status": "success",
            "model_id": request.model_id,
            "path": path
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/local")
async def get_local_models():
    """List models in the local cache."""
    models = model_hub.get_local_models()
    return {
        "status": "success",
        "count": len(models),
        "models": models
    }


class YoloDownloadRequest(BaseModel):
    model_name: str  # e.g., 'yolov8n.pt'
    project_path: str
    download_url: str


@router.post("/download-yolo")
async def download_yolo_model(request: YoloDownloadRequest):
    """Download a YOLO model to the project's models folder."""
    import requests
    import os
    
    try:
        # Create models directory in project
        models_dir = os.path.join(request.project_path, "models")
        os.makedirs(models_dir, exist_ok=True)
        
        model_path = os.path.join(models_dir, request.model_name)
        
        # Check if already downloaded
        if os.path.exists(model_path):
            return {
                "status": "success",
                "model_name": request.model_name,
                "path": model_path,
                "message": "Model already exists"
            }
        
        # Download model
        print(f"[ModelHub] Downloading {request.model_name} to {model_path}")
        response = requests.get(request.download_url, stream=True, timeout=300)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
        
        print(f"[ModelHub] Downloaded {request.model_name} ({downloaded} bytes)")
        
        return {
            "status": "success",
            "model_name": request.model_name,
            "path": model_path,
            "size": downloaded
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


@router.get("/project-models")
async def get_project_models(project_path: str):
    """List models downloaded to a specific project."""
    import os
    
    models_dir = os.path.join(project_path, "models")
    if not os.path.exists(models_dir):
        return {"models": []}
    
    models = []
    for f in os.listdir(models_dir):
        if f.endswith('.pt'):
            full_path = os.path.join(models_dir, f)
            models.append({
                "name": f,
                "path": full_path,
                "size": os.path.getsize(full_path)
            })
    
    return {"models": models}


@router.get("/tasks")
async def get_supported_tasks():
    """Get list of supported task types for model search."""
    return {
        "tasks": [
            {"id": "object-detection", "name": "Object Detection"},
            {"id": "image-segmentation", "name": "Image Segmentation"},
            {"id": "image-classification", "name": "Image Classification"},
            {"id": "zero-shot-object-detection", "name": "Zero-Shot Detection"},
            {"id": "image-to-image", "name": "Image-to-Image"}
        ]
    }
