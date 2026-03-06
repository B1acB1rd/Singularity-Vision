from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from core.dataset_manager import dataset_manager

router = APIRouter()

class ImportRequest(BaseModel):
    project_path: str
    image_paths: List[str]

@router.post("/import")
async def import_images(request: ImportRequest):
    try:
        result = dataset_manager.import_images(request.project_path, request.image_paths)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/preview")
async def get_preview(project_path: str, page: int = 1, page_size: int = 50):
    try:
        return dataset_manager.get_preview(project_path, page, page_size)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/image")
async def delete_image(project_path: str, image_path: str):
    """Delete an image from the dataset"""
    try:
        import os
        # Validate that image is within project
        abs_image = os.path.abspath(image_path)
        abs_project = os.path.abspath(project_path)
        
        if not abs_image.startswith(abs_project):
            raise HTTPException(status_code=403, detail="Cannot delete files outside project")
        
        if not os.path.exists(abs_image):
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Delete image
        os.remove(abs_image)
        
        # Also delete corresponding label if exists
        label_path = abs_image.replace('/images/', '/labels/').replace('\\images\\', '\\labels\\')
        for ext in ['.txt', '.json']:
            label_file = os.path.splitext(label_path)[0] + ext
            if os.path.exists(label_file):
                os.remove(label_file)
        
        return {"success": True, "deleted": abs_image}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate")
async def validate_dataset(project_path: str):
    # TODO: Implement validation
    return {"status": "not_implemented"}

class SplitRequest(BaseModel):
    project_path: str
    train_ratio: float
    val_ratio: float
    test_ratio: float

@router.post("/split")
async def split_dataset(request: SplitRequest):
    try:
        return dataset_manager.split_dataset(
            request.project_path, 
            request.train_ratio, 
            request.val_ratio, 
            request.test_ratio
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from typing import Optional

class ExtractFramesRequest(BaseModel):
    project_path: str
    video_path: str
    frame_interval: int = 30
    max_frames: Optional[int] = None

@router.post("/extract-frames")
async def extract_video_frames(request: ExtractFramesRequest):
    """Extract frames from a video file into the dataset"""
    try:
        result = dataset_manager.extract_video_frames(
            project_path=request.project_path,
            video_path=request.video_path,
            frame_interval=request.frame_interval,
            max_frames=request.max_frames
        )
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ValidateRequest(BaseModel):
    project_path: str


@router.post("/validate-full")
async def validate_dataset_full(request: ValidateRequest):
    """Run comprehensive dataset validation"""
    try:
        result = dataset_manager.validate_dataset(request.project_path)
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# DATASET VERSIONING ENDPOINTS (NO SILENT MUTATION)
# =============================================================================

class SnapshotRequest(BaseModel):
    project_path: str
    action: str = "manual"
    metadata: Optional[dict] = None


@router.post("/snapshot")
async def create_snapshot(request: SnapshotRequest):
    """
    Create a snapshot of the current dataset state.
    
    Philosophy: NO SILENT MUTATION - every change is tracked.
    """
    try:
        version = dataset_manager.create_snapshot(
            project_path=request.project_path,
            action=request.action,
            metadata=request.metadata
        )
        return {
            "status": "success",
            "version": version.to_dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/version")
async def get_current_version(project_path: str):
    """Get the current dataset version hash."""
    try:
        version_hash = dataset_manager.get_current_version(project_path)
        return {
            "project_path": project_path,
            "current_version": version_hash
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/versions")
async def list_versions(project_path: str):
    """List all dataset versions."""
    try:
        versions = dataset_manager.list_versions(project_path)
        return {
            "project_path": project_path,
            "version_count": len(versions),
            "versions": versions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/versions/{version_id}")
async def get_version_details(project_path: str, version_id: str):
    """Get full details of a specific version."""
    try:
        details = dataset_manager.get_version_details(project_path, version_id)
        if not details:
            raise HTTPException(status_code=404, detail=f"Version {version_id} not found")
        return details
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/verify-integrity")
async def verify_integrity(request: ValidateRequest):
    """
    Verify dataset integrity against last recorded version.
    
    Checks if files have been modified outside the platform.
    """
    try:
        result = dataset_manager.verify_dataset_integrity(request.project_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class RestoreVersionRequest(BaseModel):
    project_path: str
    version_id: str


@router.post("/restore")
async def restore_version(request: RestoreVersionRequest):
    """
    Restore dataset to a previous version.
    
    Note: Creates a new version record (restore action).
    """
    try:
        result = dataset_manager.restore_version(
            project_path=request.project_path,
            version_id=request.version_id
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# DATASET STATISTICS & ANALYTICS
# =============================================================================

@router.get("/stats")
async def get_dataset_stats(project_path: str):
    """
    Get comprehensive dataset statistics for analytics dashboard.
    
    Returns:
        - Total image count
        - Total annotation count
        - Class distribution
        - Average annotations per image
        - Dataset size in MB
    """
    import os
    import json
    
    try:
        datasets_dir = os.path.join(project_path, "datasets")
        annotations_dir = os.path.join(project_path, "annotations")
        
        # Count images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        total_images = 0
        total_size_bytes = 0
        
        if os.path.exists(datasets_dir):
            for root, _, files in os.walk(datasets_dir):
                for f in files:
                    ext = os.path.splitext(f.lower())[1]
                    if ext in image_extensions:
                        total_images += 1
                        file_path = os.path.join(root, f)
                        total_size_bytes += os.path.getsize(file_path)
        
        # Count annotations and class distribution
        total_annotations = 0
        class_distribution = {}
        
        if os.path.exists(annotations_dir):
            for f in os.listdir(annotations_dir):
                if f.endswith('.json'):
                    try:
                        with open(os.path.join(annotations_dir, f), 'r') as ann_file:
                            annotations = json.load(ann_file)
                            if isinstance(annotations, list):
                                total_annotations += len(annotations)
                                for ann in annotations:
                                    class_name = ann.get('class', ann.get('label', 'unknown'))
                                    class_distribution[class_name] = class_distribution.get(class_name, 0) + 1
                            elif isinstance(annotations, dict):
                                objects = annotations.get('objects', annotations.get('annotations', []))
                                total_annotations += len(objects)
                                for obj in objects:
                                    class_name = obj.get('class', obj.get('label', 'unknown'))
                                    class_distribution[class_name] = class_distribution.get(class_name, 0) + 1
                    except:
                        pass
        
        # Calculate averages
        avg_ann_per_image = total_annotations / max(total_images, 1)
        
        return {
            "total_images": total_images,
            "total_annotations": total_annotations,
            "class_distribution": class_distribution,
            "avg_annotations_per_image": round(avg_ann_per_image, 2),
            "dataset_size_mb": round(total_size_bytes / (1024 * 1024), 2),
            "class_count": len(class_distribution)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
