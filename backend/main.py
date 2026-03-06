from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import sys
import logging

# Initialize logging first
from core.logging_config import setup_logging, get_logger
setup_logging()
logger = get_logger("main")

# Security initialization
from core.security import add_allowed_dir

import psutil
from api import (
    projects, datasets, training, inference, export, annotations, 
    spatial, versioning, change_detection, model_hub, bundle, video, 
    user_profile, tasks, resources, keypoints, ocr, video_frames, smart_annotation,
    opencv_lab, orchestrator, profiles, reconstruction, augmentation, evaluation
)

app = FastAPI(
    title="Singularity Vision API",
    description="Backend API for Singularity Vision Desktop Platform",
    version="1.0.0"
)

# CORS - Strict in production, open for dev
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "app://."
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize security and resources on startup"""
    logger.info("Singularity Vision API starting...")
    
    # Set allowed directories for file operations
    # Add user's home directory and common project locations
    home_dir = os.path.expanduser("~")
    add_allowed_dir(home_dir)
    add_allowed_dir(os.path.join(home_dir, "Documents"))
    add_allowed_dir(os.path.join(home_dir, "Desktop"))
    add_allowed_dir(os.path.join(home_dir, ".singularity-vision"))
    
    # Add current working directory
    add_allowed_dir(os.getcwd())
    
    logger.info("Security policies initialized")


# Include Routers
app.include_router(projects.router, prefix="/projects", tags=["projects"])
app.include_router(datasets.router, prefix="/datasets", tags=["datasets"])
app.include_router(annotations.router, prefix="/annotations", tags=["annotations"])
app.include_router(training.router, prefix="/training", tags=["training"])
app.include_router(inference.router, prefix="/inference", tags=["inference"])
app.include_router(export.router, prefix="/export", tags=["export"])
app.include_router(spatial.router, prefix="/spatial", tags=["spatial"])
app.include_router(versioning.router, prefix="/versioning", tags=["versioning"])
app.include_router(change_detection.router, prefix="/changes", tags=["changes"])
app.include_router(model_hub.router, prefix="/models", tags=["models"])
app.include_router(bundle.router, prefix="/bundle", tags=["bundle"])
app.include_router(video.router, prefix="/video", tags=["video"])
app.include_router(user_profile.router, prefix="/profile", tags=["profile"])
app.include_router(tasks.router, prefix="/tasks", tags=["tasks"])
app.include_router(resources.router, prefix="/resources", tags=["resources"])
app.include_router(keypoints.router, prefix="/keypoints", tags=["keypoints"])
app.include_router(ocr.router, prefix="/ocr", tags=["ocr"])
app.include_router(video_frames.router, prefix="/video-frames", tags=["video-frames"])
app.include_router(smart_annotation.router, prefix="/smart-annotation", tags=["smart-annotation"])
app.include_router(opencv_lab.router, prefix="/opencv", tags=["opencv-lab"])
app.include_router(orchestrator.router, prefix="/orchestrator", tags=["orchestrator"])
app.include_router(profiles.router, prefix="/profiles", tags=["industry-profiles"])
app.include_router(reconstruction.router, prefix="/3d", tags=["3d-reconstruction"])
app.include_router(augmentation.router, prefix="/augmentation", tags=["augmentation"])
app.include_router(evaluation.router, prefix="/evaluation", tags=["evaluation"])

logger.info(f"Registered {26} API routers")


@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/system/info")
async def system_info():
    import torch
    
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
    
    return {
        "platform": sys.platform,
        "python_version": sys.version,
        "cpu_count": psutil.cpu_count(),
        "memory_total": psutil.virtual_memory().total,
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "torch_version": torch.__version__
    }


if __name__ == "__main__":
    logger.info("Starting development server...")
    uvicorn.run("main:app", host="127.0.0.1", port=8765, reload=True)

