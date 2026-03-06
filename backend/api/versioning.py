"""
Dataset Versioning API - Endpoints for managing dataset versions
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from core.dataset_version import DatasetVersionManager

router = APIRouter()


class CreateVersionRequest(BaseModel):
    project_path: str
    name: str
    description: Optional[str] = ""


class CompareVersionsRequest(BaseModel):
    project_path: str
    version_a_id: str
    version_b_id: str


@router.post("/create")
async def create_version(request: CreateVersionRequest):
    """Create a new version snapshot of the dataset."""
    try:
        manager = DatasetVersionManager(request.project_path)
        version = manager.create_version(
            name=request.name,
            description=request.description or ""
        )
        return {
            "status": "success",
            "version": version.to_dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list")
async def list_versions(project_path: str):
    """Get list of all versions for a project."""
    try:
        manager = DatasetVersionManager(project_path)
        versions = manager.get_versions()
        return {
            "status": "success",
            "versions": versions,
            "current_head": manager.manifest.get("current_head")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/get")
async def get_version(project_path: str, version_id: str):
    """Get details of a specific version."""
    try:
        manager = DatasetVersionManager(project_path)
        version = manager.get_version(version_id)
        
        if not version:
            raise HTTPException(status_code=404, detail="Version not found")
        
        return {
            "status": "success",
            "version": version.to_dict()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare")
async def compare_versions(request: CompareVersionsRequest):
    """Compare two versions and return differences."""
    try:
        manager = DatasetVersionManager(request.project_path)
        diff = manager.compare_versions(
            request.version_a_id,
            request.version_b_id
        )
        
        if not diff:
            raise HTTPException(status_code=404, detail="One or both versions not found")
        
        return {
            "status": "success",
            "diff": {
                "added_images": diff.added_images,
                "removed_images": diff.removed_images,
                "modified_images": diff.modified_images,
                "added_annotations": diff.added_annotations,
                "removed_annotations": diff.removed_annotations
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/changes")
async def get_current_changes(project_path: str):
    """Get uncommitted changes since last version."""
    try:
        manager = DatasetVersionManager(project_path)
        diff = manager.get_current_changes()
        
        if not diff:
            return {
                "status": "success",
                "has_changes": False,
                "message": "No versions created yet"
            }
        
        has_changes = bool(
            diff.added_images or 
            diff.removed_images or 
            diff.modified_images or
            diff.added_annotations or
            diff.removed_annotations
        )
        
        return {
            "status": "success",
            "has_changes": has_changes,
            "diff": {
                "added_images": diff.added_images,
                "removed_images": diff.removed_images,
                "modified_images": diff.modified_images,
                "added_annotations": diff.added_annotations,
                "removed_annotations": diff.removed_annotations
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
