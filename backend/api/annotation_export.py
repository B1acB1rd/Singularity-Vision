"""
Export API Endpoints for Singularity Vision

Provides REST API for exporting annotations and projects:
- COCO format export
- Pascal VOC format export
- Full project bundle export
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import os

router = APIRouter(prefix="/export", tags=["export"])


class COCOExportRequest(BaseModel):
    project_path: str
    output_path: Optional[str] = None


class VOCExportRequest(BaseModel):
    project_path: str
    output_dir: Optional[str] = None


class BundleExportRequest(BaseModel):
    project_path: str
    output_path: str
    include_models: bool = False
    include_experiments: bool = True
    formats: List[str] = ["coco", "voc"]


@router.post("/coco")
async def export_coco(request: COCOExportRequest):
    """Export annotations in COCO format."""
    try:
        from core.annotation_exporter import coco_exporter
        
        output_path = request.output_path or os.path.join(
            request.project_path, "exports", "coco_annotations.json"
        )
        
        result_path = coco_exporter.export_from_project(
            request.project_path,
            output_path
        )
        
        return {
            "success": True,
            "format": "COCO",
            "output_path": result_path
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/voc")
async def export_voc(request: VOCExportRequest):
    """Export annotations in Pascal VOC format."""
    try:
        from core.annotation_exporter import voc_exporter, coco_exporter
        
        output_dir = request.output_dir or os.path.join(
            request.project_path, "exports", "voc"
        )
        
        # First load project data
        images, annotations, categories = coco_exporter._load_project_data(
            request.project_path
        )
        
        # Export to VOC format
        exported_files = voc_exporter.export_dataset(
            images, annotations, output_dir
        )
        
        return {
            "success": True,
            "format": "Pascal VOC",
            "output_dir": output_dir,
            "files_exported": len(exported_files)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bundle")
async def export_bundle(request: BundleExportRequest):
    """Export complete project as a portable bundle."""
    try:
        from core.annotation_exporter import bundle_exporter
        
        result_path = bundle_exporter.export(
            project_path=request.project_path,
            output_path=request.output_path,
            include_models=request.include_models,
            include_experiments=request.include_experiments,
            formats=request.formats
        )
        
        return {
            "success": True,
            "format": "Project Bundle",
            "output_path": result_path,
            "includes_models": request.include_models,
            "includes_experiments": request.include_experiments,
            "annotation_formats": request.formats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/formats")
async def list_formats():
    """List supported export formats."""
    return {
        "annotation_formats": [
            {
                "id": "coco",
                "name": "COCO",
                "description": "Common Objects in Context format - single JSON file",
                "file_extension": ".json"
            },
            {
                "id": "voc",
                "name": "Pascal VOC",
                "description": "Visual Object Classes format - one XML per image",
                "file_extension": ".xml"
            },
            {
                "id": "geojson",
                "name": "GeoJSON",
                "description": "Geographic JSON format for spatial data",
                "file_extension": ".geojson"
            }
        ],
        "bundle_options": [
            {
                "id": "include_models",
                "description": "Include trained model weights"
            },
            {
                "id": "include_experiments",
                "description": "Include experiment history and metrics"
            }
        ]
    }
