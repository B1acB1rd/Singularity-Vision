"""
User Profile API - Endpoints for user settings and preferences
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
from core.user_profile import user_profile_manager

router = APIRouter()


class PreferenceUpdate(BaseModel):
    category: str
    updates: Dict[str, Any]


class RecentProject(BaseModel):
    project_path: str
    project_name: str


class ApiKeyRequest(BaseModel):
    key_name: str
    key_value: str


@router.get("/")
async def get_profile(profile_id: str = "default"):
    """Get user profile"""
    profile = user_profile_manager.get_profile(profile_id)
    
    if "error" in profile:
        raise HTTPException(status_code=404, detail=profile["error"])
    
    # Don't return API keys in response
    profile_safe = profile.copy()
    profile_safe["api_keys"] = list(profile.get("api_keys", {}).keys())
    
    return profile_safe


@router.get("/list")
async def list_profiles():
    """List all user profiles"""
    return {"profiles": user_profile_manager.list_profiles()}


@router.put("/preferences")
async def update_preferences(profile_id: str, update: PreferenceUpdate):
    """Update user preferences"""
    result = user_profile_manager.update_preferences(
        profile_id=profile_id,
        category=update.category,
        updates=update.updates
    )
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result


@router.post("/recent")
async def add_recent_project(profile_id: str, project: RecentProject):
    """Add project to recent list"""
    result = user_profile_manager.add_recent_project(
        profile_id=profile_id,
        project_path=project.project_path,
        project_name=project.project_name
    )
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result


@router.post("/api-keys")
async def set_api_key(profile_id: str, key_data: ApiKeyRequest):
    """Store an API key"""
    result = user_profile_manager.set_api_key(
        profile_id=profile_id,
        key_name=key_data.key_name,
        key_value=key_data.key_value
    )
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result


@router.get("/api-keys/{key_name}")
async def get_api_key(profile_id: str, key_name: str):
    """Check if an API key exists (doesn't return value)"""
    value = user_profile_manager.get_api_key(profile_id, key_name)
    
    return {
        "key_name": key_name,
        "exists": value is not None
    }
