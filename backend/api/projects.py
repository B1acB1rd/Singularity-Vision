from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Literal
import os
import json
import shutil
from datetime import datetime

router = APIRouter()

# Industry profiles for specialized workflows
IndustryProfile = Literal['general', 'defense', 'aviation', 'mining', 'sports', 'health']
SecurityMode = Literal['local', 'hybrid', 'cloud']
TaskType = Literal['classification', 'detection', 'segmentation', 'change_detection', 'tracking', 'pose']

class ProjectCreate(BaseModel):
    name: str
    path: str
    task_type: str
    framework: str
    industry_profile: IndustryProfile = 'general'
    security_mode: SecurityMode = 'local'

class Project(ProjectCreate):
    id: str
    created_at: str
    updated_at: str
    
@router.post("/", response_model=Project)
async def create_project(project: ProjectCreate):
    # Normalize path
    project_path = os.path.normpath(project.path)
    
    if os.path.exists(project_path):
        # Check if it's already a project
        if os.path.exists(os.path.join(project_path, "config.json")):
            raise HTTPException(status_code=400, detail="Project already exists at this location")
    else:
        try:
            os.makedirs(project_path)
        except OSError as e:
            raise HTTPException(status_code=500, detail=f"Failed to create directory: {str(e)}")

    # Create directory structure
    subdirs = ["datasets", "labels", "models", "exports", "logs"]
    for d in subdirs:
        os.makedirs(os.path.join(project_path, d), exist_ok=True)

    # Create config file
    config = {
        "id": os.path.basename(project_path), # Simple ID for now
        "name": project.name,
        "description": "",
        "createdAt": datetime.now().isoformat(),
        "updatedAt": datetime.now().isoformat(),
        "taskType": project.task_type,
        "framework": project.framework,
        "industryProfile": project.industry_profile,
        "securityMode": project.security_mode,
        "datasetInfo": None,
        "modelInfo": None,
        "trainingConfig": None
    }
    
    config_path = os.path.join(project_path, "config.json")
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write config: {str(e)}")

    return {
        **project.dict(),
        "id": config["id"],
        "created_at": config["createdAt"],
        "updated_at": config["updatedAt"]
    }

@router.get("/load")
async def load_project(path: str):
    config_path = os.path.join(path, "config.json")
    if not os.path.exists(config_path):
        raise HTTPException(status_code=404, detail="Project config not found")
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load project: {str(e)}")

@router.post("/save")
async def save_project(payload: dict):
    path = payload.get("path")
    project_data = payload.get("project")
    
    if not path or not project_data:
        raise HTTPException(status_code=400, detail="Missing path or project data")

    config_path = os.path.join(path, "config.json")
    try:
        with open(config_path, 'w') as f:
            json.dump(project_data, f, indent=2)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save project: {str(e)}")
