"""
User Profile Manager - Local user settings and preferences
"""
import os
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
from core.encryption import encryption_manager
from core.security import sanitize_filename, PathValidationError

logger = logging.getLogger("singularity.profile")


class UserProfileManager:
    """
    Manages local user profiles and preferences.
    
    Stores:
    - User preferences (theme, defaults)
    - Recent projects
    - Custom shortcuts
    - API keys (encrypted)
    """
    
    DEFAULT_PROFILE = {
        "id": "default",
        "name": "Default User",
        "created_at": None,
        "preferences": {
            "theme": "dark",
            "default_task_type": "detection",
            "default_model": "yolov8n.pt",
            "auto_save": True,
            "auto_backup": True,
            "backup_interval_hours": 24,
            "show_tips": True,
            "language": "en"
        },
        "training": {
            "default_epochs": 50,
            "default_batch_size": 16,
            "default_imgsz": 640,
            "auto_device_selection": True,
            "preferred_device": "auto"
        },
        "inference": {
            "default_confidence": 0.5,
            "default_iou": 0.45,
            "save_results": True,
            "annotate_images": True
        },
        "export": {
            "default_format": "onnx",
            "include_inference_script": True
        },
        "recent_projects": [],
        "favorites": [],
        "api_keys": {}
    }
    
    def __init__(self, profiles_dir: Optional[str] = None):
        if profiles_dir:
            self.profiles_dir = profiles_dir
        else:
            # Default to user's app data directory
            self.profiles_dir = os.path.join(
                os.path.expanduser("~"),
                ".singularity-vision",
                "profiles"
            )
        os.makedirs(self.profiles_dir, exist_ok=True)
        
        # Ensure default profile exists
        self._ensure_default_profile()
    
    def _ensure_default_profile(self):
        """Create default profile if it doesn't exist"""
        default_path = os.path.join(self.profiles_dir, "default.json")
        if not os.path.exists(default_path):
            profile = self.DEFAULT_PROFILE.copy()
            profile["created_at"] = datetime.now().isoformat()
            self._save_profile("default", profile)
    
    def _get_profile_path(self, profile_id: str) -> str:
        # Sanitize profile_id to prevent path traversal
        safe_id = sanitize_filename(profile_id)
        return os.path.join(self.profiles_dir, f"{safe_id}.json")
    
    def _save_profile(self, profile_id: str, profile: Dict[str, Any]):
        """Save profile to disk"""
        try:
            path = self._get_profile_path(profile_id)
            with open(path, 'w') as f:
                json.dump(profile, f, indent=2)
            logger.info(f"Profile saved: {profile_id}")
        except Exception as e:
            logger.error(f"Failed to save profile {profile_id}: {e}")
            raise
    
    def get_profile(self, profile_id: str = "default") -> Dict[str, Any]:
        """Load a user profile"""
        path = self._get_profile_path(profile_id)
        
        if not os.path.exists(path):
            return {"error": f"Profile '{profile_id}' not found"}
        
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            return {"error": str(e)}
    
    def update_preferences(
        self,
        profile_id: str,
        category: str,
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update specific preferences in a profile.
        
        Args:
            profile_id: Profile to update
            category: Category (preferences, training, inference, export)
            updates: Dict of key-value pairs to update
        """
        profile = self.get_profile(profile_id)
        if "error" in profile:
            return profile
        
        if category not in profile:
            return {"error": f"Unknown category: {category}"}
        
        profile[category].update(updates)
        profile["updated_at"] = datetime.now().isoformat()
        
        self._save_profile(profile_id, profile)
        return {"status": "success", "profile": profile}
    
    def add_recent_project(
        self,
        profile_id: str,
        project_path: str,
        project_name: str
    ) -> Dict[str, Any]:
        """Add a project to recent list"""
        profile = self.get_profile(profile_id)
        if "error" in profile:
            return profile
        
        recents = profile.get("recent_projects", [])
        
        # Remove if already exists
        recents = [r for r in recents if r.get("path") != project_path]
        
        # Add to front
        recents.insert(0, {
            "path": project_path,
            "name": project_name,
            "opened_at": datetime.now().isoformat()
        })
        
        # Keep only last 20
        profile["recent_projects"] = recents[:20]
        
        self._save_profile(profile_id, profile)
        return {"status": "success", "recent_count": len(recents)}
    
    def list_profiles(self) -> List[Dict[str, Any]]:
        """List all user profiles"""
        profiles = []
        
        for filename in os.listdir(self.profiles_dir):
            if filename.endswith('.json'):
                profile_id = filename[:-5]
                path = self._get_profile_path(profile_id)
                
                try:
                    with open(path, 'r') as f:
                        profile = json.load(f)
                    profiles.append({
                        "id": profile_id,
                        "name": profile.get("name", profile_id),
                        "created_at": profile.get("created_at")
                    })
                except:
                    pass
        
        return profiles
    
    def set_api_key(
        self,
        profile_id: str,
        key_name: str,
        key_value: str
    ) -> Dict[str, Any]:
        """Store an API key (encrypted)"""
        profile = self.get_profile(profile_id)
        if "error" in profile:
            return profile
        
        # Encrypt the API key
        try:
            encrypted_value = encryption_manager.encrypt(key_value)
        except Exception as e:
            logger.error(f"Failed to encrypt API key: {e}")
            return {"error": "Failed to encrypt API key"}
        
        profile["api_keys"][key_name] = {
            "value": encrypted_value,
            "encrypted": True,
            "added_at": datetime.now().isoformat()
        }
        
        self._save_profile(profile_id, profile)
        logger.info(f"API key stored: {key_name} (encrypted)")
        return {"status": "success", "key_name": key_name}
    
    def get_api_key(self, profile_id: str, key_name: str) -> Optional[str]:
        """Retrieve and decrypt an API key"""
        profile = self.get_profile(profile_id)
        if "error" in profile:
            return None
        
        key_data = profile.get("api_keys", {}).get(key_name)
        if not key_data:
            return None
        
        encrypted_value = key_data.get("value")
        if not encrypted_value:
            return None
        
        # Decrypt if encrypted
        if key_data.get("encrypted", False):
            try:
                return encryption_manager.decrypt(encrypted_value)
            except Exception as e:
                logger.error(f"Failed to decrypt API key {key_name}: {e}")
                return None
        else:
            # Legacy plaintext key - migrate it
            logger.warning(f"Migrating plaintext API key: {key_name}")
            self.set_api_key(profile_id, key_name, encrypted_value)
            return encrypted_value


# Singleton instance
user_profile_manager = UserProfileManager()
