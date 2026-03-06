"""
Bundle API - Endpoints for project import/export bundles
"""
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import os
import tempfile
import logging
from core.bundle_manager import bundle_manager
from core.security import validate_path, sanitize_filename, PathValidationError, add_allowed_dir

router = APIRouter()
logger = logging.getLogger("singularity.api.bundle")


class ExportBundleRequest(BaseModel):
    project_path: str
    include_models: bool = True
    include_exports: bool = False


class ImportBundleRequest(BaseModel):
    bundle_path: str
    target_dir: str
    overwrite: bool = False


@router.post("/export")
async def export_bundle(request: ExportBundleRequest):
    """Export a project as a portable ZIP bundle"""
    try:
        # Validate project path
        try:
            validate_path(request.project_path, must_exist=True)
        except PathValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # Generate output path
        project_name = sanitize_filename(os.path.basename(request.project_path))
        output_dir = os.path.join(request.project_path, "exports", "bundles")
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, f"{project_name}_bundle.zip")
        
        result = bundle_manager.export_bundle(
            project_path=request.project_path,
            output_path=output_path,
            include_models=request.include_models,
            include_exports=request.include_exports
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        logger.info(f"Bundle exported: {output_path}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/import")
async def import_bundle(request: ImportBundleRequest):
    """Import a project from a ZIP bundle"""
    try:
        # Validate paths
        try:
            validate_path(request.bundle_path, must_exist=True, allowed_extensions=['.zip'])
            validate_path(request.target_dir, allow_creation=True)
        except PathValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        result = bundle_manager.import_bundle(
            bundle_path=request.bundle_path,
            target_dir=request.target_dir,
            overwrite=request.overwrite
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Import failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/info")
async def get_bundle_info(bundle_path: str):
    """Get information about a bundle without extracting it"""
    try:
        # Validate path
        try:
            validate_path(bundle_path, must_exist=True, allowed_extensions=['.zip'])
        except PathValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        result = bundle_manager.get_bundle_info(bundle_path)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download")
async def download_bundle(bundle_path: str):
    """Download a bundle file (restricted to exports directory)"""
    try:
        # Validate path - only allow downloads from exports directories
        validate_path(bundle_path, must_exist=True, allowed_extensions=['.zip'])
        
        # Additional check: only allow files in "exports" directories
        if "exports" not in bundle_path.replace("\\", "/"):
            logger.warning(f"Attempted download outside exports: {bundle_path}")
            raise HTTPException(
                status_code=403,
                detail="Downloads only allowed from exports directories"
            )
    except PathValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    return FileResponse(
        bundle_path,
        media_type="application/zip",
        filename=os.path.basename(bundle_path)
    )

