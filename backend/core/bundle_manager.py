"""
Project Bundle Manager - Export and Import project bundles as ZIP archives
"""
import os
import json
import zipfile
import shutil
import tempfile
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
from core.security import (
    validate_path, validate_zip_member, validate_zip_safe,
    sanitize_filename, SecurityError, MAX_ZIP_SIZE_MB
)

logger = logging.getLogger("singularity.bundle")


class ProjectBundleManager:
    """
    Manages project export/import as portable ZIP bundles.
    
    Bundle structure:
    project_bundle.zip/
        config.json          # Project configuration
        manifest.json        # Bundle metadata (version, created, checksums)
        datasets/
            images/          # All images
            labels/          # All labels
        models/              # Trained model weights (optional)
        exports/             # Exported files (optional)
    """
    
    BUNDLE_VERSION = "1.0"
    
    def __init__(self):
        pass
    
    def export_bundle(
        self,
        project_path: str,
        output_path: str,
        include_models: bool = True,
        include_exports: bool = False
    ) -> Dict[str, Any]:
        """
        Export a project as a portable ZIP bundle.
        
        Args:
            project_path: Path to the project directory
            output_path: Path for the output ZIP file
            include_models: Include trained model weights
            include_exports: Include exported files (ONNX, etc.)
            
        Returns:
            Dict with export status and file info
        """
        try:
            # Validate project
            config_path = os.path.join(project_path, "config.json")
            if not os.path.exists(config_path):
                return {"error": "Project config.json not found"}
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Create manifest
            manifest = {
                "bundle_version": self.BUNDLE_VERSION,
                "project_name": config.get("name", "Unknown"),
                "project_id": config.get("id", ""),
                "created_at": datetime.now().isoformat(),
                "exported_by": "Singularity Vision",
                "includes": {
                    "datasets": True,
                    "models": include_models,
                    "exports": include_exports
                },
                "file_counts": {
                    "images": 0,
                    "labels": 0,
                    "models": 0
                }
            }
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                # Add config.json
                zf.write(config_path, "config.json")
                
                # Add datasets
                datasets_dir = os.path.join(project_path, "datasets")
                if os.path.exists(datasets_dir):
                    for root, dirs, files in os.walk(datasets_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arc_name = os.path.relpath(file_path, project_path)
                            zf.write(file_path, arc_name)
                            
                            # Count files
                            if "images" in arc_name:
                                manifest["file_counts"]["images"] += 1
                            elif "labels" in arc_name:
                                manifest["file_counts"]["labels"] += 1
                
                # Add models (optional)
                if include_models:
                    runs_dir = os.path.join(project_path, "runs")
                    if os.path.exists(runs_dir):
                        for root, dirs, files in os.walk(runs_dir):
                            for file in files:
                                if file.endswith('.pt'):
                                    file_path = os.path.join(root, file)
                                    arc_name = os.path.relpath(file_path, project_path)
                                    zf.write(file_path, arc_name)
                                    manifest["file_counts"]["models"] += 1
                
                # Add exports (optional)
                if include_exports:
                    exports_dir = os.path.join(project_path, "exports")
                    if os.path.exists(exports_dir):
                        for root, dirs, files in os.walk(exports_dir):
                            for file in files:
                                file_path = os.path.join(root, file)
                                arc_name = os.path.relpath(file_path, project_path)
                                zf.write(file_path, arc_name)
                
                # Add manifest
                manifest_json = json.dumps(manifest, indent=2)
                zf.writestr("manifest.json", manifest_json)
            
            # Get bundle size
            bundle_size = os.path.getsize(output_path)
            
            return {
                "status": "success",
                "bundle_path": output_path,
                "size_mb": round(bundle_size / (1024 * 1024), 2),
                "manifest": manifest
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def import_bundle(
        self,
        bundle_path: str,
        target_dir: str,
        overwrite: bool = False
    ) -> Dict[str, Any]:
        """
        Import a project from a ZIP bundle.
        
        Args:
            bundle_path: Path to the ZIP bundle
            target_dir: Directory to extract the project to
            overwrite: Overwrite if project already exists
            
        Returns:
            Dict with import status and project info
        """
        try:
            if not os.path.exists(bundle_path):
                return {"error": "Bundle file not found"}
            
            # Validate ZIP file for security
            try:
                validate_zip_safe(bundle_path)
            except SecurityError as e:
                logger.warning(f"ZIP security check failed: {e}")
                return {"error": str(e)}
            
            # Read manifest from bundle
            with zipfile.ZipFile(bundle_path, 'r') as zf:
                if "manifest.json" not in zf.namelist():
                    return {"error": "Invalid bundle: manifest.json not found"}
                
                manifest = json.loads(zf.read("manifest.json"))
                
                if "config.json" not in zf.namelist():
                    return {"error": "Invalid bundle: config.json not found"}
                
                config = json.loads(zf.read("config.json"))
            
            # Determine project directory
            project_name = sanitize_filename(config.get("name", "imported_project"))
            project_dir = os.path.join(target_dir, project_name)
            
            if os.path.exists(project_dir):
                if not overwrite:
                    return {"error": f"Project directory already exists: {project_dir}"}
                logger.warning(f"Overwriting existing project: {project_dir}")
                shutil.rmtree(project_dir)
            
            # Safe extraction with Zip Slip prevention
            os.makedirs(project_dir, exist_ok=True)
            
            with zipfile.ZipFile(bundle_path, 'r') as zf:
                for member in zf.infolist():
                    # Validate each member path
                    is_safe, target_path = validate_zip_member(
                        member.filename, project_dir
                    )
                    
                    if not is_safe:
                        logger.warning(f"Skipping dangerous ZIP member: {member.filename}")
                        continue
                    
                    # Create parent directories
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    
                    # Extract file
                    if not member.is_dir():
                        with zf.open(member) as source:
                            with open(target_path, 'wb') as target:
                                shutil.copyfileobj(source, target)
            
            logger.info(f"Bundle imported: {project_name}")
            
            return {
                "status": "success",
                "project_path": project_dir,
                "project_name": project_name,
                "manifest": manifest
            }
            
        except Exception as e:
            logger.error(f"Import failed: {e}")
            return {"error": str(e)}
    
    def get_bundle_info(self, bundle_path: str) -> Dict[str, Any]:
        """
        Get information about a bundle without extracting it.
        
        Args:
            bundle_path: Path to the ZIP bundle
            
        Returns:
            Dict with bundle metadata
        """
        try:
            if not os.path.exists(bundle_path):
                return {"error": "Bundle file not found"}
            
            with zipfile.ZipFile(bundle_path, 'r') as zf:
                if "manifest.json" not in zf.namelist():
                    return {"error": "Invalid bundle: manifest.json not found"}
                
                manifest = json.loads(zf.read("manifest.json"))
                
                # Get total size
                total_size = sum(info.file_size for info in zf.infolist())
                
                return {
                    "status": "success",
                    "manifest": manifest,
                    "compressed_size_mb": round(os.path.getsize(bundle_path) / (1024 * 1024), 2),
                    "uncompressed_size_mb": round(total_size / (1024 * 1024), 2),
                    "file_count": len(zf.namelist())
                }
                
        except Exception as e:
            return {"error": str(e)}


# Singleton instance
bundle_manager = ProjectBundleManager()
