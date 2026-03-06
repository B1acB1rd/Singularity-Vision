"""
Task Orchestrator API Endpoints

Provides hardware assessment, task evaluation, and optimization suggestions.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List, Any

from core.task_orchestrator import task_orchestrator

router = APIRouter()


class TaskAssessmentRequest(BaseModel):
    """Request to assess a task before execution."""
    task_type: str  # "training", "inference", "3d_reconstruction", "change_detection"
    dataset_size: int
    image_resolution: Optional[List[int]] = None  # [width, height]
    project_profile: str = "general"
    config: Optional[Dict[str, Any]] = None


@router.get("/hardware")
async def get_hardware_info(force_refresh: bool = False):
    """
    Get current hardware capabilities.
    
    Returns:
        Hardware report including CPU, GPU, RAM, disk info and scores.
    """
    try:
        hardware = task_orchestrator.get_hardware_info(force_refresh=force_refresh)
        return hardware.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/assess")
async def assess_task(request: TaskAssessmentRequest):
    """
    Assess a task and get execution recommendations.
    
    This should be called BEFORE starting any heavy task like training,
    inference on large datasets, or 3D reconstruction.
    
    Returns:
        ExecutionDecision with mode, warnings, suggestions, and time estimate.
    """
    try:
        image_resolution = tuple(request.image_resolution) if request.image_resolution else None
        
        decision = task_orchestrator.assess_task(
            task_type=request.task_type,
            dataset_size=request.dataset_size,
            image_resolution=image_resolution,
            project_profile=request.project_profile,
            config=request.config
        )
        
        return decision.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/suggestions/{task_type}")
async def get_optimization_suggestions(
    task_type: str,
    dataset_size: int = 100,
    project_profile: str = "general"
):
    """
    Get optimization suggestions for a task type.
    
    Args:
        task_type: Type of task
        dataset_size: Number of images
        project_profile: Industry profile ID
    
    Returns:
        List of optimization suggestions.
    """
    try:
        # Assess task to generate suggestions
        decision = task_orchestrator.assess_task(
            task_type=task_type,
            dataset_size=dataset_size,
            project_profile=project_profile
        )
        
        return {
            "task_type": task_type,
            "dataset_size": dataset_size,
            "suggestions": [
                {
                    "type": s.type,
                    "title": s.title,
                    "description": s.description,
                    "action": s.action,
                    "impact": s.impact
                } for s in decision.suggestions
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/estimate")
async def estimate_execution_time(
    task_type: str,
    dataset_size: int,
    epochs: int = 50,
    batch_size: int = 16
):
    """
    Estimate execution time for a task.
    
    Args:
        task_type: Type of task
        dataset_size: Number of images
        epochs: Number of training epochs (for training tasks)
        batch_size: Batch size (for training tasks)
    
    Returns:
        Time estimate with min/max seconds and confidence.
    """
    try:
        hardware = task_orchestrator.get_hardware_info()
        config = {"epochs": epochs, "batch_size": batch_size}
        
        estimate = task_orchestrator.estimate_execution_time(
            task_type=task_type,
            dataset_size=dataset_size,
            hardware=hardware,
            config=config
        )
        
        return estimate.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
