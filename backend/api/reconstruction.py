"""
3D Reconstruction API Endpoints

3D is a CORE DIFFERENTIATOR - not optional.
These endpoints provide access to the reconstruction engine.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
import os

from core.reconstruction_engine import (
    reconstruction_engine, 
    ReconstructionConfig,
    Point3D
)

router = APIRouter()


class StartReconstructionRequest(BaseModel):
    """Request to start a 3D reconstruction job."""
    images_dir: str
    output_dir: str
    project_path: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


class MeasureDistanceRequest(BaseModel):
    """Request to measure distance between two points."""
    point_a: Dict[str, float]  # {x, y, z}
    point_b: Dict[str, float]  # {x, y, z}


class CalculateVolumeRequest(BaseModel):
    """Request to calculate volume from point cloud path."""
    point_cloud_path: str


@router.post("/start")
async def start_reconstruction(request: StartReconstructionRequest):
    """
    Start a 3D reconstruction job.
    
    This starts reconstruction in the background. Poll /status/{job_id}
    to check progress.
    
    Args:
        images_dir: Directory containing input images
        output_dir: Where to save reconstruction results
        project_path: Optional project path
        config: Optional reconstruction configuration
    
    Returns:
        {job_id: str, status: str}
    """
    try:
        # Validate directories
        if not os.path.exists(request.images_dir):
            raise HTTPException(status_code=400, detail="Images directory does not exist")
        
        # Parse config if provided
        config = None
        if request.config:
            config = ReconstructionConfig(
                feature_detector=request.config.get("feature_detector", "ORB"),
                max_features=request.config.get("max_features", 5000),
                match_ratio=request.config.get("match_ratio", 0.75),
                min_matches=request.config.get("min_matches", 50)
            )
        
        job_id = reconstruction_engine.start_reconstruction(
            images_dir=request.images_dir,
            output_dir=request.output_dir,
            config=config,
            project_path=request.project_path
        )
        
        return {
            "job_id": job_id,
            "status": "started",
            "message": "Reconstruction job started. Poll /3d/status/{job_id} for progress."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{job_id}")
async def get_reconstruction_status(job_id: str):
    """
    Get the status of a reconstruction job.
    
    Args:
        job_id: The job ID returned from /start
    
    Returns:
        Job status including progress, message, and results when complete.
    """
    status = reconstruction_engine.get_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    return status


@router.get("/jobs")
async def list_reconstruction_jobs():
    """
    List all reconstruction jobs.
    
    Returns:
        List of all jobs with their current status.
    """
    return {
        "jobs": reconstruction_engine.list_jobs()
    }


@router.get("/point-cloud/{job_id}")
async def get_point_cloud(job_id: str, format: str = "json"):
    """
    Get point cloud data for a completed reconstruction.
    
    Args:
        job_id: The job ID
        format: "json" for parsed data, "file" for raw PLY file
    
    Returns:
        Point cloud data or file.
    """
    status = reconstruction_engine.get_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    if status["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Job not completed. Current status: {status['status']}"
        )
    
    result = status.get("result", {})
    point_cloud_path = result.get("point_cloud_path")
    
    if not point_cloud_path or not os.path.exists(point_cloud_path):
        raise HTTPException(status_code=404, detail="Point cloud file not found")
    
    if format == "file":
        return FileResponse(
            point_cloud_path,
            media_type="application/octet-stream",
            filename=os.path.basename(point_cloud_path)
        )
    
    # Parse and return as JSON
    points = reconstruction_engine.load_point_cloud(point_cloud_path)
    return {
        "num_points": len(points),
        "points": [p.to_dict() for p in points[:10000]]  # Limit for performance
    }


@router.post("/measure/distance")
async def measure_distance(request: MeasureDistanceRequest):
    """
    Measure the distance between two 3D points.
    
    Args:
        point_a: First point {x, y, z}
        point_b: Second point {x, y, z}
    
    Returns:
        Distance in reconstruction units.
    """
    try:
        a = Point3D(
            x=request.point_a["x"],
            y=request.point_a["y"],
            z=request.point_a["z"]
        )
        b = Point3D(
            x=request.point_b["x"],
            y=request.point_b["y"],
            z=request.point_b["z"]
        )
        
        distance = reconstruction_engine.measure_distance(a, b)
        
        return {
            "distance": distance,
            "unit": "reconstruction_units",
            "note": "Scale to real-world units requires calibration"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/measure/volume")
async def calculate_volume(request: CalculateVolumeRequest):
    """
    Calculate the volume of a point cloud (convex hull).
    
    Useful for mining volume estimation.
    
    Args:
        point_cloud_path: Path to PLY file
    
    Returns:
        Volume and bounding box information.
    """
    try:
        if not os.path.exists(request.point_cloud_path):
            raise HTTPException(status_code=404, detail="Point cloud not found")
        
        points = reconstruction_engine.load_point_cloud(request.point_cloud_path)
        
        if not points:
            raise HTTPException(status_code=400, detail="No points in cloud")
        
        volume = reconstruction_engine.calculate_volume(points)
        bbox = reconstruction_engine.calculate_bounding_box(points)
        
        return {
            "volume": volume,
            "bounding_box": bbox,
            "num_points": len(points),
            "unit": "reconstruction_units_cubed"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/bounding-box/{job_id}")
async def get_bounding_box(job_id: str):
    """
    Get the bounding box of a completed reconstruction.
    
    Args:
        job_id: The job ID
    
    Returns:
        Bounding box with min/max coordinates and dimensions.
    """
    status = reconstruction_engine.get_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    if status["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Job not completed. Current status: {status['status']}"
        )
    
    result = status.get("result", {})
    point_cloud_path = result.get("point_cloud_path")
    
    if not point_cloud_path:
        raise HTTPException(status_code=404, detail="Point cloud not found")
    
    points = reconstruction_engine.load_point_cloud(point_cloud_path)
    bbox = reconstruction_engine.calculate_bounding_box(points)
    
    return bbox
