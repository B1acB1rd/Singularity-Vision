"""
Experiment Tracking API Endpoints for Singularity Vision

Provides REST API for training reproducibility:
- Create experiments with version binding
- Track training progress
- Compare experiments
- Check reproducibility
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import os

router = APIRouter(prefix="/experiments", tags=["experiments"])


# Request/Response Models

class ExperimentConfigRequest(BaseModel):
    model_name: str
    model_version: Optional[str] = None
    pretrained_weights: Optional[str] = None
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 0.01
    optimizer: str = "SGD"
    augmentations: List[str] = []
    image_size: int = 640
    num_classes: Optional[int] = None
    class_names: List[str] = []
    device: str = "auto"
    seed: Optional[int] = None
    extra: Dict = {}


class CreateExperimentRequest(BaseModel):
    project_path: str
    dataset_path: str
    config: ExperimentConfigRequest
    profile_id: str = "general"


class UpdateResultRequest(BaseModel):
    experiment_id: str
    project_path: str
    status: Optional[str] = None
    metrics: Optional[Dict] = None
    checkpoint_path: Optional[str] = None
    error_message: Optional[str] = None


class CompareExperimentsRequest(BaseModel):
    project_path: str
    experiment_ids: List[str]


class CheckReproducibilityRequest(BaseModel):
    project_path: str
    experiment_id: str
    dataset_path: str
    config: ExperimentConfigRequest


# Endpoints

@router.post("/create")
async def create_experiment(request: CreateExperimentRequest):
    """
    Create a new experiment with version bindings.
    
    CRITICAL: This locks the dataset version and config hash.
    """
    try:
        from core.experiment_tracker import ExperimentTracker, ExperimentConfig
        
        # Convert request to ExperimentConfig
        config = ExperimentConfig(
            model_name=request.config.model_name,
            model_version=request.config.model_version,
            pretrained_weights=request.config.pretrained_weights,
            epochs=request.config.epochs,
            batch_size=request.config.batch_size,
            learning_rate=request.config.learning_rate,
            optimizer=request.config.optimizer,
            augmentations=request.config.augmentations,
            image_size=request.config.image_size,
            num_classes=request.config.num_classes,
            class_names=request.config.class_names,
            device=request.config.device,
            seed=request.config.seed,
            extra=request.config.extra
        )
        
        # Create tracker and experiment
        tracker = ExperimentTracker(request.project_path, request.profile_id)
        binding = tracker.create_experiment(config, request.dataset_path)
        
        return {
            "success": True,
            "experiment_id": binding.experiment_id,
            "dataset_version": binding.dataset_version,
            "config_hash": binding.config_hash,
            "created_at": binding.created_at
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/get/{experiment_id}")
async def get_experiment(experiment_id: str, project_path: str):
    """Get full experiment data including binding and results."""
    try:
        from core.experiment_tracker import ExperimentTracker
        
        tracker = ExperimentTracker(project_path)
        exp = tracker.get_experiment(experiment_id)
        
        if not exp:
            raise HTTPException(status_code=404, detail="Experiment not found")
        
        return {
            "success": True,
            "experiment": exp
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list")
async def list_experiments(
    project_path: str,
    status: Optional[str] = None,
    limit: int = 50
):
    """List experiments with optional status filter."""
    try:
        from core.experiment_tracker import ExperimentTracker
        
        tracker = ExperimentTracker(project_path)
        experiments = tracker.list_experiments(status=status, limit=limit)
        
        return {
            "success": True,
            "count": len(experiments),
            "experiments": experiments
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/update-result")
async def update_result(request: UpdateResultRequest):
    """Update experiment results during/after training."""
    try:
        from core.experiment_tracker import ExperimentTracker
        
        tracker = ExperimentTracker(request.project_path)
        success = tracker.update_result(
            experiment_id=request.experiment_id,
            status=request.status,
            metrics=request.metrics,
            checkpoint_path=request.checkpoint_path,
            error_message=request.error_message
        )
        
        return {"success": success}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare")
async def compare_experiments(request: CompareExperimentsRequest):
    """
    Compare multiple experiments.
    
    Shows config differences, dataset versions, and results.
    """
    try:
        from core.experiment_tracker import ExperimentTracker
        
        tracker = ExperimentTracker(request.project_path)
        comparison = tracker.compare_experiments(request.experiment_ids)
        
        return {
            "success": True,
            "comparison": comparison
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/check-reproducibility")
async def check_reproducibility(request: CheckReproducibilityRequest):
    """
    Check if current inputs match a previous experiment.
    
    Use case: "Will this produce the same results?"
    """
    try:
        from core.experiment_tracker import ExperimentTracker, ExperimentConfig
        
        # Convert request config
        config = ExperimentConfig(
            model_name=request.config.model_name,
            model_version=request.config.model_version,
            pretrained_weights=request.config.pretrained_weights,
            epochs=request.config.epochs,
            batch_size=request.config.batch_size,
            learning_rate=request.config.learning_rate,
            optimizer=request.config.optimizer,
            augmentations=request.config.augmentations,
            image_size=request.config.image_size,
            num_classes=request.config.num_classes,
            class_names=request.config.class_names,
            device=request.config.device,
            seed=request.config.seed,
            extra=request.config.extra
        )
        
        tracker = ExperimentTracker(request.project_path)
        result = tracker.check_reproducibility(
            experiment_id=request.experiment_id,
            dataset_path=request.dataset_path,
            config=config
        )
        
        return {
            "success": True,
            "reproducibility": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/find-matching")
async def find_matching_experiments(
    project_path: str,
    dataset_version: Optional[str] = None,
    config_hash: Optional[str] = None
):
    """
    Find experiments matching the given version/config.
    
    Use case: "Have I run this exact experiment before?"
    """
    try:
        from core.experiment_tracker import ExperimentTracker
        
        tracker = ExperimentTracker(project_path)
        matches = tracker.find_matching_experiments(
            dataset_version=dataset_version,
            config_hash=config_hash
        )
        
        return {
            "success": True,
            "count": len(matches),
            "experiment_ids": matches
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config-hash")
async def compute_config_hash(
    model_name: str,
    epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 0.01,
    image_size: int = 640
):
    """
    Compute config hash for given parameters.
    
    Useful for checking before creating an experiment.
    """
    try:
        from core.experiment_tracker import ExperimentConfig
        
        config = ExperimentConfig(
            model_name=model_name,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            image_size=image_size
        )
        
        return {
            "config_hash": config.compute_hash(),
            "config": config.to_dict()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
