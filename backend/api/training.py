from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional, Any
from core.training_manager import training_manager

router = APIRouter()

class TrainRequest(BaseModel):
    project_path: str
    model_name: str
    config: Dict[str, Any]
    task_type: str = "detection"  # "detection", "classification", "segmentation"

@router.post("/start")
async def start_training(request: TrainRequest):
    try:
        if request.task_type == "classification":
            job_id = training_manager.start_classification_training(
                request.project_path,
                request.model_name,
                request.config
            )
        elif request.task_type == "segmentation":
            job_id = training_manager.start_segmentation_training(
                request.project_path,
                request.model_name,
                request.config
            )
        else:
            # Default to detection
            job_id = training_manager.start_training(
                request.project_path,
                request.model_name,
                request.config
            )
        return {"job_id": job_id, "status": "started", "task_type": request.task_type}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{job_id}/stop")
async def stop_training(job_id: str):
    training_manager.stop_training(job_id)
    return {"status": "stopping"}

@router.get("/{job_id}/status")
async def get_training_status(job_id: str):
    status = training_manager.get_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    return status

@router.get("/available")
async def get_available_models(task_type: str = "detection"):
    """Get available models by task type"""
    if task_type == "detection":
        return {
            "models": [
                {"id": "yolov8n.pt", "name": "YOLOv8 Nano (Fastest)", "size": "6.2 MB"},
                {"id": "yolov8s.pt", "name": "YOLOv8 Small", "size": "21.5 MB"},
                {"id": "yolov8m.pt", "name": "YOLOv8 Medium", "size": "49.7 MB"},
                {"id": "yolov8l.pt", "name": "YOLOv8 Large", "size": "83.7 MB"},
                {"id": "yolov8x.pt", "name": "YOLOv8 XLarge (Most Accurate)", "size": "168.4 MB"},
            ]
        }
    elif task_type == "classification":
        return {
            "models": [
                {"id": "yolov8n-cls.pt", "name": "YOLOv8 Nano Classification", "size": "5.8 MB"},
                {"id": "yolov8s-cls.pt", "name": "YOLOv8 Small Classification", "size": "12.8 MB"},
                {"id": "yolov8m-cls.pt", "name": "YOLOv8 Medium Classification", "size": "33.4 MB"},
                {"id": "yolov8l-cls.pt", "name": "YOLOv8 Large Classification", "size": "55.6 MB"},
                {"id": "yolov8x-cls.pt", "name": "YOLOv8 XLarge Classification", "size": "98.5 MB"},
            ]
        }
    elif task_type == "segmentation":
        return {
            "models": [
                {"id": "yolov8n-seg.pt", "name": "YOLOv8 Nano Segmentation", "size": "7.1 MB"},
                {"id": "yolov8s-seg.pt", "name": "YOLOv8 Small Segmentation", "size": "23.8 MB"},
                {"id": "yolov8m-seg.pt", "name": "YOLOv8 Medium Segmentation", "size": "54.8 MB"},
                {"id": "yolov8l-seg.pt", "name": "YOLOv8 Large Segmentation", "size": "91.5 MB"},
                {"id": "yolov8x-seg.pt", "name": "YOLOv8 XLarge Segmentation", "size": "182.3 MB"},
            ]
        }
    return {"models": []}

@router.get("/devices")
async def get_device_info():
    """Get available compute devices (CPU/GPU) with auto-selection recommendation"""
    try:
        device_info = training_manager.get_device_info()
        return device_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ResumeRequest(BaseModel):
    project_path: str
    run_name: str
    config: Dict[str, Any]

@router.post("/resume")
async def resume_training(request: ResumeRequest):
    """Resume an interrupted training job from checkpoint"""
    try:
        job_id = training_manager.resume_training(
            request.project_path,
            request.run_name,
            request.config
        )
        return {"job_id": job_id, "status": "resuming"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
