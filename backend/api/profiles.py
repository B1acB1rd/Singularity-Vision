"""
Industry Profiles API Endpoints

Provides profile listing, constraints checking, and feature validation.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List, Any

from core.industry_profiles import profile_manager

router = APIRouter()


class ConstraintCheckRequest(BaseModel):
    """Request to check a constraint."""
    profile_id: str
    action: str
    context: Optional[Dict[str, Any]] = None


class ApplyDefaultsRequest(BaseModel):
    """Request to apply profile defaults to a config."""
    profile_id: str
    config: Dict[str, Any]


@router.get("/list")
async def list_profiles():
    """
    List all available industry profiles.
    
    Returns:
        List of all profiles with their configurations.
    """
    try:
        return {
            "profiles": profile_manager.list_profiles()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{profile_id}")
async def get_profile(profile_id: str):
    """
    Get a specific industry profile.
    
    Args:
        profile_id: Profile identifier (e.g., "defense", "mining", "health")
    
    Returns:
        Profile configuration.
    """
    try:
        profile = profile_manager.get_profile(profile_id)
        return profile.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{profile_id}/features")
async def get_profile_features(profile_id: str):
    """
    Get feature permissions for a profile.
    
    Args:
        profile_id: Profile identifier
    
    Returns:
        Allowed and disabled features.
    """
    try:
        profile = profile_manager.get_profile(profile_id)
        return {
            "profile_id": profile_id,
            "allowed_features": profile.allowed_features,
            "disabled_features": profile.disabled_features
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{profile_id}/constraints")
async def get_profile_constraints(profile_id: str):
    """
    Get mandatory constraints for a profile.
    
    Args:
        profile_id: Profile identifier
    
    Returns:
        Constraint settings (cloud_allowed, encryption_required, etc.)
    """
    try:
        profile = profile_manager.get_profile(profile_id)
        return {
            "profile_id": profile_id,
            "profile_name": profile.name,
            "constraints": {
                "cloud_allowed": profile.constraints.cloud_allowed,
                "offload_allowed": profile.constraints.offload_allowed,
                "encryption_required": profile.constraints.encryption_required,
                "anonymization_required": profile.constraints.anonymization_required,
                "offline_only": profile.constraints.offline_only,
                "max_resolution": profile.constraints.max_resolution,
                "audit_logging": profile.constraints.audit_logging
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/check-constraint")
async def check_constraint(request: ConstraintCheckRequest):
    """
    Check if an action is allowed under profile constraints.
    
    Use this before performing actions that may be restricted.
    
    Args:
        profile_id: Profile identifier
        action: Action to check (e.g., "cloud_upload", "start_training")
        context: Optional context (e.g., {"encryption_enabled": True})
    
    Returns:
        {allowed: bool, error: str|null}
    """
    try:
        allowed, error = profile_manager.check_constraint(
            profile_id=request.profile_id,
            action=request.action,
            context=request.context
        )
        return {
            "allowed": allowed,
            "error": error
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{profile_id}/feature/{feature}")
async def check_feature(profile_id: str, feature: str):
    """
    Check if a specific feature is allowed for a profile.
    
    Args:
        profile_id: Profile identifier
        feature: Feature to check (e.g., "cloud_training", "3d_reconstruction")
    
    Returns:
        {allowed: bool}
    """
    try:
        allowed = profile_manager.is_feature_allowed(profile_id, feature)
        return {
            "profile_id": profile_id,
            "feature": feature,
            "allowed": allowed
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/apply-defaults")
async def apply_defaults(request: ApplyDefaultsRequest):
    """
    Apply profile default settings to a configuration.
    User settings override profile defaults.
    
    Args:
        profile_id: Profile identifier
        config: User configuration
    
    Returns:
        Merged configuration with defaults applied.
    """
    try:
        result = profile_manager.apply_defaults(
            profile_id=request.profile_id,
            config=request.config
        )
        return {
            "profile_id": request.profile_id,
            "config": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{profile_id}/disabled-features")
async def get_disabled_features(profile_id: str):
    """
    Get list of disabled features for UI filtering.
    
    The frontend should hide or disable these features in the UI.
    
    Args:
        profile_id: Profile identifier
    
    Returns:
        List of disabled feature IDs.
    """
    try:
        disabled = profile_manager.get_disabled_features(profile_id)
        return {
            "profile_id": profile_id,
            "disabled_features": disabled
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
