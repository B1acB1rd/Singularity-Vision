"""
Industry Profiles System for Singularity Vision

Industry profiles control MORE than just UI visibility:
- Feature enabling/disabling
- Offloading constraints (cloud allowed, hybrid allowed)
- Security requirements (encryption, anonymization)
- Default settings per industry

Philosophy: Profiles are behavioral configs, not code forks.
"""

import os
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MandatoryConstraints:
    """
    Constraints that are ENFORCED, not optional.
    These control compliance and security.
    """
    # Offloading Control
    cloud_allowed: bool = True
    offload_allowed: bool = True
    
    # Security
    encryption_required: bool = False
    anonymization_required: bool = False
    
    # Execution
    offline_only: bool = False
    max_resolution: Optional[int] = None  # Max image resolution allowed
    audit_logging: bool = False  # Log all actions for compliance


@dataclass
class IndustryProfile:
    """
    Complete industry profile configuration.
    
    Affects:
    - UI (which features are visible)
    - Engines (which operations are allowed)
    - Security (encryption, anonymization)
    - Offloading (cloud/hybrid/local)
    """
    profile_id: str
    name: str
    description: str
    
    # Feature Control
    allowed_features: List[str] = field(default_factory=list)
    disabled_features: List[str] = field(default_factory=list)
    
    # Default Settings
    default_settings: Dict[str, Any] = field(default_factory=dict)
    
    # Mandatory Constraints
    constraints: MandatoryConstraints = field(default_factory=MandatoryConstraints)
    
    # UI Customization
    accent_color: Optional[str] = None
    icon: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "profile_id": self.profile_id,
            "name": self.name,
            "description": self.description,
            "allowed_features": self.allowed_features,
            "disabled_features": self.disabled_features,
            "default_settings": self.default_settings,
            "constraints": {
                "cloud_allowed": self.constraints.cloud_allowed,
                "offload_allowed": self.constraints.offload_allowed,
                "encryption_required": self.constraints.encryption_required,
                "anonymization_required": self.constraints.anonymization_required,
                "offline_only": self.constraints.offline_only,
                "max_resolution": self.constraints.max_resolution,
                "audit_logging": self.constraints.audit_logging
            },
            "accent_color": self.accent_color,
            "icon": self.icon
        }


class IndustryProfileManager:
    """
    Manages industry profiles for the platform.
    
    Responsibilities:
    - Load profiles from JSON config files
    - Check feature permissions
    - Enforce mandatory constraints
    - Apply default settings
    """
    
    # All available features in the platform
    ALL_FEATURES = [
        "detection",
        "classification", 
        "segmentation",
        "training",
        "inference",
        "cloud_training",
        "remote_inference",
        "model_sharing",
        "3d_reconstruction",
        "volume_tools",
        "change_detection",
        "mapping",
        "spatial_analysis",
        "ocr",
        "video_processing",
        "webcam_inference",
        "export_onnx",
        "export_tflite",
        "annotation_ai_assist"
    ]
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize profile manager.
        
        Args:
            config_dir: Directory containing profile JSON files.
                       If None, uses built-in profiles.
        """
        self.config_dir = config_dir
        self._profiles: Dict[str, IndustryProfile] = {}
        self._load_builtin_profiles()
        
        if config_dir and os.path.exists(config_dir):
            self._load_profiles_from_dir(config_dir)
    
    def _load_builtin_profiles(self):
        """Load built-in default profiles."""
        
        # General (Default) - All features enabled
        self._profiles["general"] = IndustryProfile(
            profile_id="general",
            name="General",
            description="All features enabled. Suitable for most use cases.",
            allowed_features=self.ALL_FEATURES.copy(),
            disabled_features=[],
            default_settings={
                "execution_mode": "hybrid",
                "default_model": "yolov8n.pt"
            },
            constraints=MandatoryConstraints(
                cloud_allowed=True,
                offload_allowed=True
            ),
            accent_color="#6366f1",
            icon="sparkles"
        )
        
        # Defense & Security - Maximum restrictions
        self._profiles["defense"] = IndustryProfile(
            profile_id="defense",
            name="Defense & Security",
            description="Offline-only, encrypted, fully local. No cloud features.",
            allowed_features=[
                "detection", "classification", "segmentation",
                "training", "inference",
                "3d_reconstruction", "mapping", "change_detection",
                "spatial_analysis", "video_processing",
                "export_onnx"
            ],
            disabled_features=[
                "cloud_training", "remote_inference", "model_sharing",
                "annotation_ai_assist"  # May require cloud
            ],
            default_settings={
                "execution_mode": "local",
                "encryption_enabled": True,
                "audit_logging": True
            },
            constraints=MandatoryConstraints(
                cloud_allowed=False,
                offload_allowed=False,
                encryption_required=True,
                offline_only=True,
                audit_logging=True
            ),
            accent_color="#dc2626",
            icon="shield"
        )
        
        # Mining & Resources - 3D and volume tools
        self._profiles["mining"] = IndustryProfile(
            profile_id="mining",
            name="Mining & Resources",
            description="3D reconstruction, volume estimation, terrain analysis.",
            allowed_features=self.ALL_FEATURES.copy(),
            disabled_features=[],
            default_settings={
                "execution_mode": "hybrid",
                "enable_volume_tools": True,
                "enable_3d": True,
                "default_model": "yolov8m.pt"  # Larger model for accuracy
            },
            constraints=MandatoryConstraints(
                cloud_allowed=True,
                offload_allowed=True
            ),
            accent_color="#f59e0b",
            icon="mountain"
        )
        
        # Healthcare & Medical - Privacy focused
        self._profiles["health"] = IndustryProfile(
            profile_id="health",
            name="Healthcare & Medical",
            description="Anonymization enforced, no cloud, HIPAA-ready.",
            allowed_features=[
                "detection", "classification", "segmentation",
                "training", "inference",
                "video_processing", "export_onnx"
            ],
            disabled_features=[
                "cloud_training", "remote_inference", "model_sharing",
                "3d_reconstruction"  # Often not needed
            ],
            default_settings={
                "execution_mode": "local",
                "anonymization_enabled": True,
                "encryption_enabled": True
            },
            constraints=MandatoryConstraints(
                cloud_allowed=False,
                offload_allowed=False,
                encryption_required=True,
                anonymization_required=True,
                offline_only=True,
                audit_logging=True
            ),
            accent_color="#10b981",
            icon="heart-pulse"
        )
        
        # Aviation & Aerospace
        self._profiles["aviation"] = IndustryProfile(
            profile_id="aviation",
            name="Aviation & Aerospace",
            description="High-resolution inspection, OCR, defect detection.",
            allowed_features=self.ALL_FEATURES.copy(),
            disabled_features=[],
            default_settings={
                "execution_mode": "hybrid",
                "enable_ocr": True,
                "enable_high_res": True,
                "default_model": "yolov8l.pt"  # Large model for detail
            },
            constraints=MandatoryConstraints(
                cloud_allowed=True,
                offload_allowed=True,
                max_resolution=8192  # Support very high-res imagery
            ),
            accent_color="#3b82f6",
            icon="plane"
        )
        
        # Sports Analytics
        self._profiles["sports"] = IndustryProfile(
            profile_id="sports",
            name="Sports Analytics",
            description="Real-time tracking, video analysis, pose estimation.",
            allowed_features=self.ALL_FEATURES.copy(),
            disabled_features=[
                "3d_reconstruction",  # Usually not needed
                "mapping"
            ],
            default_settings={
                "execution_mode": "local",  # Real-time needs local
                "enable_tracking": True,
                "enable_pose": True,
                "default_model": "yolov8s.pt"  # Fast model for real-time
            },
            constraints=MandatoryConstraints(
                cloud_allowed=True,
                offload_allowed=False  # Real-time = local
            ),
            accent_color="#8b5cf6",
            icon="activity"
        )
    
    def _load_profiles_from_dir(self, config_dir: str):
        """Load custom profiles from JSON files."""
        for file_path in Path(config_dir).glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    profile = self._parse_profile(data)
                    self._profiles[profile.profile_id] = profile
            except Exception as e:
                print(f"Warning: Could not load profile from {file_path}: {e}")
    
    def _parse_profile(self, data: Dict) -> IndustryProfile:
        """Parse a profile from JSON data."""
        constraints_data = data.get("constraints", {})
        constraints = MandatoryConstraints(
            cloud_allowed=constraints_data.get("cloud_allowed", True),
            offload_allowed=constraints_data.get("offload_allowed", True),
            encryption_required=constraints_data.get("encryption_required", False),
            anonymization_required=constraints_data.get("anonymization_required", False),
            offline_only=constraints_data.get("offline_only", False),
            max_resolution=constraints_data.get("max_resolution"),
            audit_logging=constraints_data.get("audit_logging", False)
        )
        
        return IndustryProfile(
            profile_id=data["profile_id"],
            name=data.get("name", data["profile_id"]),
            description=data.get("description", ""),
            allowed_features=data.get("allowed_features", self.ALL_FEATURES),
            disabled_features=data.get("disabled_features", []),
            default_settings=data.get("default_settings", {}),
            constraints=constraints,
            accent_color=data.get("accent_color"),
            icon=data.get("icon")
        )
    
    def get_profile(self, profile_id: str) -> IndustryProfile:
        """Get a profile by ID. Returns 'general' if not found."""
        return self._profiles.get(profile_id, self._profiles["general"])
    
    def list_profiles(self) -> List[Dict]:
        """List all available profiles."""
        return [p.to_dict() for p in self._profiles.values()]
    
    def is_feature_allowed(self, profile_id: str, feature: str) -> bool:
        """Check if a feature is allowed for a profile."""
        profile = self.get_profile(profile_id)
        
        # If explicitly disabled, not allowed
        if feature in profile.disabled_features:
            return False
        
        # If allowed_features is not empty, feature must be in it
        if profile.allowed_features:
            return feature in profile.allowed_features
        
        return True
    
    def check_constraint(
        self, 
        profile_id: str, 
        action: str,
        context: Optional[Dict] = None
    ) -> tuple[bool, Optional[str]]:
        """
        Check if an action is allowed under profile constraints.
        
        Args:
            profile_id: Industry profile ID
            action: Action to check (e.g., "cloud_upload", "start_training")
            context: Optional context (e.g., {"encryption_enabled": False})
        
        Returns:
            (allowed, error_message)
        """
        profile = self.get_profile(profile_id)
        context = context or {}
        
        # Check cloud actions
        if action in ["cloud_upload", "cloud_training", "remote_inference"]:
            if not profile.constraints.cloud_allowed:
                return False, f"Cloud operations are not allowed for {profile.name} profile"
        
        # Check encryption requirement
        if profile.constraints.encryption_required:
            if not context.get("encryption_enabled", False):
                return False, f"Encryption is required for {profile.name} profile"
        
        # Check anonymization requirement
        if profile.constraints.anonymization_required:
            if not context.get("anonymization_enabled", False):
                return False, f"Data anonymization is required for {profile.name} profile"
        
        # Check offline-only
        if profile.constraints.offline_only:
            if action in ["cloud_upload", "cloud_training", "remote_inference", "fetch_remote_model"]:
                return False, f"{profile.name} profile requires offline-only operation"
        
        return True, None
    
    def apply_defaults(self, profile_id: str, config: Dict) -> Dict:
        """
        Apply profile default settings to a configuration.
        User settings override defaults.
        """
        profile = self.get_profile(profile_id)
        
        # Start with profile defaults
        result = profile.default_settings.copy()
        
        # User settings override
        result.update(config)
        
        return result
    
    def get_disabled_features(self, profile_id: str) -> List[str]:
        """Get list of disabled features for UI filtering."""
        profile = self.get_profile(profile_id)
        return profile.disabled_features.copy()
    
    def enforce_anonymization(self, profile_id: str, data: Any) -> Any:
        """
        Enforce anonymization if required by profile.
        This is a placeholder - actual implementation depends on data type.
        """
        profile = self.get_profile(profile_id)
        
        if not profile.constraints.anonymization_required:
            return data
        
        # TODO: Implement actual anonymization based on data type
        # - Images: Blur faces, remove EXIF
        # - Metadata: Remove PII
        # - Filenames: Anonymize if containing PII
        
        return data


# Singleton instance
profile_manager = IndustryProfileManager()
