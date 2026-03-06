"""
Dataset Version Manager for Singularity Vision
Tracks changes to datasets over time, enables snapshots and rollbacks.
"""

import os
import json
import shutil
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class DatasetVersion:
    """Represents a snapshot of the dataset at a point in time."""
    version_id: str
    name: str
    description: str
    created_at: str
    image_count: int
    annotation_count: int
    classes: List[str]
    file_hashes: Dict[str, str]  # filename -> hash
    parent_version: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class VersionDiff:
    """Represents changes between two versions."""
    added_images: List[str]
    removed_images: List[str]
    modified_images: List[str]
    added_annotations: int
    removed_annotations: int
    

class DatasetVersionManager:
    """
    Manages dataset versioning for a project.
    
    Features:
    - Create named snapshots of dataset state
    - Compare versions to see what changed
    - Restore dataset to a previous version
    - Track annotation changes over time
    """
    
    VERSION_DIR = ".versions"
    MANIFEST_FILE = "versions.json"
    
    def __init__(self, project_path: str):
        self.project_path = project_path
        self.dataset_path = os.path.join(project_path, "datasets")
        self.labels_path = os.path.join(project_path, "labels")
        self.versions_path = os.path.join(project_path, self.VERSION_DIR)
        self.manifest_path = os.path.join(self.versions_path, self.MANIFEST_FILE)
        
        # Ensure versions directory exists
        os.makedirs(self.versions_path, exist_ok=True)
        
        # Load or create manifest
        self.manifest = self._load_manifest()
        
    def _load_manifest(self) -> Dict:
        """Load the versions manifest file."""
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path, 'r') as f:
                return json.load(f)
        return {"versions": [], "current_head": None}
    
    def _save_manifest(self):
        """Save the versions manifest file."""
        with open(self.manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=2)
            
    def _compute_file_hash(self, file_path: str) -> str:
        """Compute SHA256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()[:16]  # Short hash
    
    def _get_dataset_state(self) -> Dict[str, str]:
        """Get current state of all dataset files as filename->hash mapping."""
        state = {}
        
        # Hash images
        if os.path.exists(self.dataset_path):
            for root, _, files in os.walk(self.dataset_path):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, self.dataset_path)
                        state[f"images/{rel_path}"] = self._compute_file_hash(full_path)
        
        # Hash labels
        if os.path.exists(self.labels_path):
            for root, _, files in os.walk(self.labels_path):
                for file in files:
                    if file.lower().endswith(('.txt', '.json')):
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, self.labels_path)
                        state[f"labels/{rel_path}"] = self._compute_file_hash(full_path)
                        
        return state
    
    def _count_annotations(self) -> int:
        """Count total annotations in the dataset."""
        count = 0
        if os.path.exists(self.labels_path):
            for root, _, files in os.walk(self.labels_path):
                for file in files:
                    if file.endswith('.txt'):
                        with open(os.path.join(root, file), 'r') as f:
                            count += len([l for l in f.readlines() if l.strip()])
        return count
    
    def _get_classes(self) -> List[str]:
        """Get list of classes from project config."""
        config_path = os.path.join(self.project_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                dataset_info = config.get("datasetInfo", {})
                if dataset_info:
                    return [c.get("name", "") for c in dataset_info.get("classes", [])]
        return []
    
    def create_version(
        self,
        name: str,
        description: str = ""
    ) -> DatasetVersion:
        """
        Create a new version snapshot of the current dataset state.
        
        Args:
            name: Human-readable name for the version
            description: Optional description of changes
            
        Returns:
            The created DatasetVersion
        """
        # Generate version ID
        timestamp = datetime.now()
        version_id = f"v_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        # Get current state
        file_hashes = self._get_dataset_state()
        image_count = len([k for k in file_hashes if k.startswith("images/")])
        annotation_count = self._count_annotations()
        classes = self._get_classes()
        
        # Create version object
        version = DatasetVersion(
            version_id=version_id,
            name=name,
            description=description,
            created_at=timestamp.isoformat(),
            image_count=image_count,
            annotation_count=annotation_count,
            classes=classes,
            file_hashes=file_hashes,
            parent_version=self.manifest.get("current_head")
        )
        
        # Save version data
        version_file = os.path.join(self.versions_path, f"{version_id}.json")
        with open(version_file, 'w') as f:
            json.dump(version.to_dict(), f, indent=2)
        
        # Update manifest
        self.manifest["versions"].append({
            "version_id": version_id,
            "name": name,
            "created_at": version.created_at,
            "image_count": image_count,
            "annotation_count": annotation_count
        })
        self.manifest["current_head"] = version_id
        self._save_manifest()
        
        return version
    
    def get_versions(self) -> List[Dict]:
        """Get list of all versions."""
        return self.manifest.get("versions", [])
    
    def get_version(self, version_id: str) -> Optional[DatasetVersion]:
        """Get a specific version by ID."""
        version_file = os.path.join(self.versions_path, f"{version_id}.json")
        if os.path.exists(version_file):
            with open(version_file, 'r') as f:
                data = json.load(f)
                return DatasetVersion(**data)
        return None
    
    def compare_versions(
        self,
        version_a_id: str,
        version_b_id: str
    ) -> Optional[VersionDiff]:
        """
        Compare two versions and return the differences.
        
        Args:
            version_a_id: Earlier version
            version_b_id: Later version
            
        Returns:
            VersionDiff showing changes between versions
        """
        version_a = self.get_version(version_a_id)
        version_b = self.get_version(version_b_id)
        
        if not version_a or not version_b:
            return None
        
        files_a = set(version_a.file_hashes.keys())
        files_b = set(version_b.file_hashes.keys())
        
        # Find image changes
        images_a = {f for f in files_a if f.startswith("images/")}
        images_b = {f for f in files_b if f.startswith("images/")}
        
        added = list(images_b - images_a)
        removed = list(images_a - images_b)
        
        # Find modified (same file, different hash)
        common = images_a & images_b
        modified = [
            f for f in common 
            if version_a.file_hashes[f] != version_b.file_hashes[f]
        ]
        
        return VersionDiff(
            added_images=added,
            removed_images=removed,
            modified_images=modified,
            added_annotations=max(0, version_b.annotation_count - version_a.annotation_count),
            removed_annotations=max(0, version_a.annotation_count - version_b.annotation_count)
        )
    
    def get_current_changes(self) -> Optional[VersionDiff]:
        """Compare current state to the last version."""
        if not self.manifest.get("current_head"):
            return None
            
        current_state = self._get_dataset_state()
        head = self.get_version(self.manifest["current_head"])
        
        if not head:
            return None
        
        files_head = set(head.file_hashes.keys())
        files_current = set(current_state.keys())
        
        images_head = {f for f in files_head if f.startswith("images/")}
        images_current = {f for f in files_current if f.startswith("images/")}
        
        added = list(images_current - images_head)
        removed = list(images_head - images_current)
        
        common = images_head & images_current
        modified = [
            f for f in common 
            if head.file_hashes[f] != current_state[f]
        ]
        
        current_annotations = self._count_annotations()
        
        return VersionDiff(
            added_images=added,
            removed_images=removed,
            modified_images=modified,
            added_annotations=max(0, current_annotations - head.annotation_count),
            removed_annotations=max(0, head.annotation_count - current_annotations)
        )
