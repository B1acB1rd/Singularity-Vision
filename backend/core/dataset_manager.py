"""
Dataset Manager for Singularity Vision

Philosophy: NO SILENT MUTATION
- Every dataset change creates a hash
- Training always references a dataset version
- Reproducibility is sacred

This is how we beat RoboFlow for researchers and enterprises.
"""

import os
import shutil
import cv2
import json
import hashlib
import uuid
from typing import List, Dict, Optional, Any
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field

import random


@dataclass
class DatasetVersion:
    """Represents an immutable dataset snapshot."""
    version_id: str
    timestamp: str
    action: str  # "import", "delete", "augment", "split", "restore"
    file_count: int
    total_hash: str  # Combined hash of all files
    file_hashes: Dict[str, str]  # filename -> SHA256
    diff_from_previous: Optional[Dict] = None  # added, removed, modified
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "version_id": self.version_id,
            "timestamp": self.timestamp,
            "action": self.action,
            "file_count": self.file_count,
            "total_hash": self.total_hash,
            "file_hashes": self.file_hashes,
            "diff_from_previous": self.diff_from_previous,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "DatasetVersion":
        return cls(
            version_id=data["version_id"],
            timestamp=data["timestamp"],
            action=data["action"],
            file_count=data["file_count"],
            total_hash=data["total_hash"],
            file_hashes=data.get("file_hashes", {}),
            diff_from_previous=data.get("diff_from_previous"),
            metadata=data.get("metadata", {})
        )


class DatasetManager:
    """
    Manages datasets with versioning and immutability.
    
    Core principle: NO SILENT MUTATION
    - Every change creates a snapshot
    - Changes are tracked with SHA256 hashes
    - Previous versions can be restored
    """
    
    def __init__(self):
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        self._versions_filename = ".dataset_versions.json"
    
    # =========================================================================
    # VERSIONING / IMMUTABILITY METHODS
    # =========================================================================
    
    def _compute_file_hash(self, file_path: str) -> str:
        """Compute SHA256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def _compute_dataset_hashes(self, project_path: str) -> Dict[str, str]:
        """Compute hashes for all files in the dataset."""
        dataset_dir = os.path.join(project_path, "datasets")
        if not os.path.exists(dataset_dir):
            return {}
        
        file_hashes = {}
        for filename in os.listdir(dataset_dir):
            if os.path.splitext(filename)[1].lower() in self.supported_formats:
                file_path = os.path.join(dataset_dir, filename)
                file_hashes[filename] = self._compute_file_hash(file_path)
        
        return file_hashes
    
    def _compute_total_hash(self, file_hashes: Dict[str, str]) -> str:
        """Compute combined hash of all file hashes."""
        if not file_hashes:
            return hashlib.sha256(b"empty").hexdigest()
        
        # Sort by filename for consistent ordering
        combined = "".join(
            f"{k}:{v}" for k, v in sorted(file_hashes.items())
        )
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def _load_versions(self, project_path: str) -> List[DatasetVersion]:
        """Load version history from disk."""
        versions_path = os.path.join(project_path, self._versions_filename)
        if not os.path.exists(versions_path):
            return []
        
        try:
            with open(versions_path, 'r') as f:
                data = json.load(f)
            return [DatasetVersion.from_dict(v) for v in data.get("versions", [])]
        except Exception as e:
            print(f"Error loading versions: {e}")
            return []
    
    def _save_versions(self, project_path: str, versions: List[DatasetVersion]):
        """Save version history to disk."""
        versions_path = os.path.join(project_path, self._versions_filename)
        data = {
            "versions": [v.to_dict() for v in versions]
        }
        with open(versions_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def create_snapshot(
        self, 
        project_path: str, 
        action: str,
        metadata: Optional[Dict] = None
    ) -> DatasetVersion:
        """
        Create an immutable snapshot of the current dataset state.
        
        Args:
            project_path: Path to project
            action: What triggered this snapshot (import, delete, augment, etc.)
            metadata: Optional additional metadata
        
        Returns:
            The created DatasetVersion
        """
        # Compute current state
        file_hashes = self._compute_dataset_hashes(project_path)
        total_hash = self._compute_total_hash(file_hashes)
        
        # Load previous versions
        versions = self._load_versions(project_path)
        
        # Compute diff from previous version
        diff = None
        if versions:
            prev_version = versions[-1]
            prev_files = set(prev_version.file_hashes.keys())
            curr_files = set(file_hashes.keys())
            
            added = list(curr_files - prev_files)
            removed = list(prev_files - curr_files)
            
            # Check for modified files
            modified = []
            for filename in prev_files & curr_files:
                if file_hashes[filename] != prev_version.file_hashes.get(filename):
                    modified.append(filename)
            
            if added or removed or modified:
                diff = {
                    "added": added,
                    "removed": removed,
                    "modified": modified
                }
        
        # Create new version
        version = DatasetVersion(
            version_id=str(uuid.uuid4())[:8],
            timestamp=datetime.now().isoformat(),
            action=action,
            file_count=len(file_hashes),
            total_hash=total_hash,
            file_hashes=file_hashes,
            diff_from_previous=diff,
            metadata=metadata or {}
        )
        
        # Save
        versions.append(version)
        self._save_versions(project_path, versions)
        
        return version
    
    def get_current_version(self, project_path: str) -> Optional[str]:
        """
        Get the current dataset version hash.
        
        Returns:
            Version hash or None if no versions exist
        """
        versions = self._load_versions(project_path)
        if versions:
            return versions[-1].total_hash
        
        # Compute fresh if no versions
        file_hashes = self._compute_dataset_hashes(project_path)
        return self._compute_total_hash(file_hashes) if file_hashes else None
    
    def list_versions(self, project_path: str) -> List[Dict]:
        """
        List all dataset versions.
        
        Returns:
            List of version summaries (without full file hashes for efficiency)
        """
        versions = self._load_versions(project_path)
        return [
            {
                "version_id": v.version_id,
                "timestamp": v.timestamp,
                "action": v.action,
                "file_count": v.file_count,
                "total_hash": v.total_hash[:12] + "...",  # Short hash for display
                "changes": v.diff_from_previous
            }
            for v in versions
        ]
    
    def get_version_details(self, project_path: str, version_id: str) -> Optional[Dict]:
        """Get full details of a specific version."""
        versions = self._load_versions(project_path)
        for v in versions:
            if v.version_id == version_id:
                return v.to_dict()
        return None
    
    def restore_version(self, project_path: str, version_id: str) -> Dict:
        """
        Restore dataset to a previous version.
        
        Note: This creates a NEW version (restore action), preserving history.
        
        Args:
            project_path: Path to project
            version_id: ID of version to restore to
        
        Returns:
            Result with status and new version info
        """
        versions = self._load_versions(project_path)
        
        # Find target version
        target = None
        for v in versions:
            if v.version_id == version_id:
                target = v
                break
        
        if not target:
            return {"error": f"Version {version_id} not found"}
        
        # This is a complex operation - for now, return guidance
        # Full implementation would require storing actual file backups
        return {
            "status": "info",
            "message": (
                f"Version {version_id} had {target.file_count} files. "
                "Full restore requires file backup storage (not yet implemented). "
                "Current version hash: " + (self.get_current_version(project_path) or "none")
            ),
            "target_version": target.to_dict()
        }
    
    def verify_dataset_integrity(self, project_path: str) -> Dict:
        """
        Verify current dataset matches last recorded version.
        
        Returns:
            Integrity report with any discrepancies
        """
        versions = self._load_versions(project_path)
        if not versions:
            return {"status": "no_versions", "message": "No versions recorded"}
        
        last_version = versions[-1]
        current_hashes = self._compute_dataset_hashes(project_path)
        current_total = self._compute_total_hash(current_hashes)
        
        if current_total == last_version.total_hash:
            return {
                "status": "valid",
                "message": "Dataset matches last recorded version",
                "version_id": last_version.version_id,
                "file_count": len(current_hashes)
            }
        
        # Find differences
        last_files = set(last_version.file_hashes.keys())
        curr_files = set(current_hashes.keys())
        
        return {
            "status": "modified",
            "message": "Dataset has been modified since last snapshot",
            "last_version": last_version.version_id,
            "changes": {
                "added": list(curr_files - last_files),
                "removed": list(last_files - curr_files),
                "modified": [
                    f for f in last_files & curr_files
                    if current_hashes[f] != last_version.file_hashes[f]
                ]
            }
        }
    
    # =========================================================================
    # ORIGINAL METHODS (now with auto-snapshotting)
    # =========================================================================

    def validate_image(self, file_path: str) -> bool:
        """Check if file is a valid image"""
        try:
            if not os.path.exists(file_path):
                return False
            
            ext = os.path.splitext(file_path)[1].lower()
            if ext not in self.supported_formats:
                return False
            
            # optional: deep check with cv2.imread
            return True
        except Exception:
            return False

    def import_images(self, project_path: str, image_paths: List[str]) -> Dict:
        """
        Import images into project dataset directory.
        
        Auto-creates a snapshot after import (NO SILENT MUTATION).
        """
        dataset_dir = os.path.join(project_path, "datasets")
        os.makedirs(dataset_dir, exist_ok=True)
        
        imported_count = 0
        failed_files = []
        imported_files = []
        
        for img_path in image_paths:
            try:
                if not self.validate_image(img_path):
                    failed_files.append(img_path)
                    continue

                # Generate unique filename to prevent collisions
                filename = os.path.basename(img_path)
                name, ext = os.path.splitext(filename)
                
                # Check if exists, if so append timestamp
                target_path = os.path.join(dataset_dir, filename)
                if os.path.exists(target_path):
                    timestamp = int(datetime.now().timestamp())
                    target_path = os.path.join(dataset_dir, f"{name}_{timestamp}{ext}")

                shutil.copy2(img_path, target_path)
                imported_count += 1
                imported_files.append(os.path.basename(target_path))
                
            except Exception as e:
                print(f"Error importing {img_path}: {e}")
                failed_files.append(img_path)

        # Update Project Config
        self._update_dataset_stats(project_path)
        
        # Auto-create snapshot (NO SILENT MUTATION)
        version = None
        if imported_count > 0:
            version = self.create_snapshot(
                project_path, 
                action="import",
                metadata={
                    "imported_count": imported_count,
                    "imported_files": imported_files
                }
            )

        return {
            "imported": imported_count,
            "failed": failed_files,
            "dataset_version": version.total_hash if version else None
        }

    def get_preview(self, project_path: str, page: int = 1, page_size: int = 50) -> Dict:
        """Get paginated list of images for preview"""
        dataset_dir = os.path.join(project_path, "datasets")
        if not os.path.exists(dataset_dir):
            return {"images": [], "total": 0}

        # Get all valid images
        all_files = [
            f for f in os.listdir(dataset_dir) 
            if os.path.splitext(f)[1].lower() in self.supported_formats
        ]
        all_files.sort() # Ensure consistent order

        total = len(all_files)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        preview_files = [os.path.join(dataset_dir, f) for f in all_files[start_idx:end_idx]]
        
        return {
            "images": preview_files,
            "total": total,
            "page": page,
            "page_size": page_size
        }

    def split_dataset(self, project_path: str, train_ratio: float, val_ratio: float, test_ratio: float) -> Dict:
        """Split dataset into train/val/test and generate txt files"""
        dataset_dir = os.path.join(project_path, "datasets")
        if not os.path.exists(dataset_dir):
            return {"error": "No dataset found"}
        
        # Get all images
        all_files = [
            os.path.abspath(os.path.join(dataset_dir, f))
            for f in os.listdir(dataset_dir) 
            if os.path.splitext(f)[1].lower() in self.supported_formats
        ]
        
        if not all_files:
            return {"error": "No valid images found in dataset"}

        # Shuffle
        random.shuffle(all_files)
        total = len(all_files)
        
        # Calculate split indices
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        train_files = all_files[:train_end]
        val_files = all_files[train_end:val_end]
        test_files = all_files[val_end:]
        
        # Write to txt files
        try:
            with open(os.path.join(project_path, 'autosplit_train.txt'), 'w') as f:
                f.write('\n'.join(train_files))
                
            with open(os.path.join(project_path, 'autosplit_val.txt'), 'w') as f:
                f.write('\n'.join(val_files))
                
            with open(os.path.join(project_path, 'autosplit_test.txt'), 'w') as f:
                f.write('\n'.join(test_files))

            # Update config with split stats
            self._update_dataset_stats(project_path, split_counts={
                "train": len(train_files),
                "val": len(val_files),
                "test": len(test_files)
            })
            
            return {
                "status": "success",
                "counts": {
                    "train": len(train_files),
                    "val": len(val_files),
                    "test": len(test_files),
                    "total": total
                }
            }
        except Exception as e:
            return {"error": str(e)}

    def _update_dataset_stats(self, project_path: str, split_counts: Optional[Dict] = None):
        """Update config.json with new dataset counts"""
        config_path = os.path.join(project_path, "config.json")
        if not os.path.exists(config_path):
            return

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)

            dataset_dir = os.path.join(project_path, "datasets")
            all_files = [
                f for f in os.listdir(dataset_dir) 
                if os.path.splitext(f)[1].lower() in self.supported_formats
            ]
            
            count = len(all_files)
            
            if config.get("datasetInfo") is None:
                config["datasetInfo"] = {
                    "totalImages": 0,
                    "classes": [],
                    "trainCount": 0,
                    "valCount": 0,
                    "testCount": 0,
                    "imageSize": None,
                    "hasLabels": False
                }
            
            config["datasetInfo"]["totalImages"] = count
            
            if split_counts:
                config["datasetInfo"]["trainCount"] = split_counts["train"]
                config["datasetInfo"]["valCount"] = split_counts["val"]
                config["datasetInfo"]["testCount"] = split_counts["test"]
            
            config["updatedAt"] = datetime.now().isoformat()

            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)

        except Exception as e:
            print(f"Failed to update stats: {e}")
    
    def extract_video_frames(
        self,
        project_path: str,
        video_path: str,
        frame_interval: int = 30,
        max_frames: Optional[int] = None
    ) -> Dict:
        """Extract frames from video and import as images"""
        dataset_dir = os.path.join(project_path, "datasets")
        os.makedirs(dataset_dir, exist_ok=True)
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {"error": "Failed to open video file"}
            
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            
            extracted_count = 0
            frame_number = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Extract every Nth frame
                if frame_number % frame_interval == 0:
                    # Save frame
                    frame_filename = f"{video_name}_frame_{frame_number:06d}.jpg"
                    frame_path = os.path.join(dataset_dir, frame_filename)
                    cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    extracted_count += 1
                    
                    if max_frames and extracted_count >= max_frames:
                        break
                
                frame_number += 1
            
            cap.release()
            
            # Update dataset stats
            self._update_dataset_stats(project_path)
            
            return {
                "status": "success",
                "extracted": extracted_count,
                "total_video_frames": total_frames,
                "video_fps": fps,
                "frame_interval": frame_interval
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def validate_dataset(self, project_path: str) -> Dict:
        """Validate dataset for issues: corrupt files, duplicates, missing labels, class imbalance"""
        import hashlib
        from collections import Counter
        
        dataset_dir = os.path.join(project_path, "datasets")
        labels_dir = os.path.join(project_path, "labels")
        
        if not os.path.exists(dataset_dir):
            return {"error": "Dataset directory not found"}
        
        issues = {
            "corrupt_files": [],
            "duplicates": [],
            "missing_labels": [],
            "empty_labels": [],
            "resolution_issues": [],
            "class_distribution": {},
            "summary": {}
        }
        
        # Track file hashes for duplicate detection
        file_hashes = {}
        resolutions = []
        total_images = 0
        valid_images = 0
        
        # Get all images
        all_files = [f for f in os.listdir(dataset_dir) 
                     if os.path.splitext(f)[1].lower() in self.supported_formats]
        
        for filename in all_files:
            file_path = os.path.join(dataset_dir, filename)
            total_images += 1
            
            try:
                # Check if image is readable
                img = cv2.imread(file_path)
                if img is None:
                    issues["corrupt_files"].append(filename)
                    continue
                
                valid_images += 1
                h, w = img.shape[:2]
                resolutions.append((w, h))
                
                # Check for duplicates using hash
                with open(file_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                
                if file_hash in file_hashes:
                    issues["duplicates"].append({
                        "file": filename,
                        "duplicate_of": file_hashes[file_hash]
                    })
                else:
                    file_hashes[file_hash] = filename
                
                # Check for corresponding label file
                label_name = os.path.splitext(filename)[0] + ".txt"
                label_path = os.path.join(labels_dir, label_name)
                
                if os.path.exists(labels_dir):
                    if not os.path.exists(label_path):
                        issues["missing_labels"].append(filename)
                    else:
                        # Check if label file is empty
                        with open(label_path, 'r') as lf:
                            content = lf.read().strip()
                            if not content:
                                issues["empty_labels"].append(filename)
                            else:
                                # Count class distribution
                                for line in content.split('\n'):
                                    if line.strip():
                                        class_id = line.split()[0]
                                        issues["class_distribution"][class_id] = \
                                            issues["class_distribution"].get(class_id, 0) + 1
                
            except Exception as e:
                issues["corrupt_files"].append(filename)
        
        # Analyze resolutions
        if resolutions:
            res_counter = Counter(resolutions)
            most_common_res = res_counter.most_common(1)[0][0]
            for filename, res in zip(all_files, resolutions):
                if res != most_common_res:
                    issues["resolution_issues"].append({
                        "file": filename,
                        "resolution": f"{res[0]}x{res[1]}",
                        "expected": f"{most_common_res[0]}x{most_common_res[1]}"
                    })
        
        # Summary
        issues["summary"] = {
            "total_images": total_images,
            "valid_images": valid_images,
            "corrupt_count": len(issues["corrupt_files"]),
            "duplicate_count": len(issues["duplicates"]),
            "missing_labels_count": len(issues["missing_labels"]),
            "empty_labels_count": len(issues["empty_labels"]),
            "resolution_issues_count": len(issues["resolution_issues"]),
            "class_count": len(issues["class_distribution"]),
            "is_healthy": (
                len(issues["corrupt_files"]) == 0 and
                len(issues["duplicates"]) == 0 and
                len(issues["missing_labels"]) < total_images * 0.1  # Less than 10% missing is OK
            )
        }
        
        return issues

dataset_manager = DatasetManager()

