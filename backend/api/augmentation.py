"""
Data Augmentation API Endpoints

Provides augmentation preview and batch processing capabilities.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import cv2
import numpy as np
import json
import base64
import os
import logging

router = APIRouter()
logger = logging.getLogger("singularity.api.augmentation")


class AugmentationConfig(BaseModel):
    """Configuration for augmentation pipeline."""
    # Preprocessing
    resize: Optional[Dict[str, Any]] = {"enabled": True, "width": 640, "height": 640}
    normalize: bool = True
    grayscale: bool = False
    
    # Augmentations
    horizontal_flip: Optional[Dict[str, float]] = {"enabled": True, "prob": 0.5}
    vertical_flip: Optional[Dict[str, float]] = {"enabled": False, "prob": 0.5}
    rotation: Optional[Dict[str, Any]] = {"enabled": True, "prob": 0.3, "max_angle": 15}
    brightness: Optional[Dict[str, Any]] = {"enabled": True, "prob": 0.3, "range": 0.2}
    contrast: Optional[Dict[str, Any]] = {"enabled": True, "prob": 0.3, "range": 0.2}
    blur: Optional[Dict[str, Any]] = {"enabled": False, "prob": 0.1, "max_kernel": 5}
    noise: Optional[Dict[str, Any]] = {"enabled": False, "prob": 0.1, "intensity": 0.02}
    cutout: Optional[Dict[str, Any]] = {"enabled": False, "prob": 0.2, "size": 50}


class PreviewRequest(BaseModel):
    """Request for augmentation preview."""
    project_path: str
    image_path: str
    config: AugmentationConfig


class BatchAugmentRequest(BaseModel):
    """Request for batch augmentation."""
    project_path: str
    config: AugmentationConfig
    num_augmented_per_image: int = 3


def decode_image(file_bytes: bytes) -> np.ndarray:
    """Decode uploaded image bytes to numpy array."""
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def encode_image(img: np.ndarray, format: str = "png") -> bytes:
    """Encode numpy array to image bytes."""
    if format.lower() == "jpg" or format.lower() == "jpeg":
        _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    else:
        _, buffer = cv2.imencode('.png', img)
    return buffer.tobytes()


def apply_augmentation(img: np.ndarray, config: AugmentationConfig) -> np.ndarray:
    """Apply augmentation pipeline to an image."""
    result = img.copy()
    
    # Preprocessing
    if config.resize and config.resize.get("enabled"):
        w = config.resize.get("width", 640)
        h = config.resize.get("height", 640)
        result = cv2.resize(result, (w, h))
    
    if config.grayscale:
        result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)  # Keep 3 channels
    
    # Horizontal flip
    if config.horizontal_flip and config.horizontal_flip.get("enabled"):
        if np.random.random() < config.horizontal_flip.get("prob", 0.5):
            result = cv2.flip(result, 1)
    
    # Vertical flip
    if config.vertical_flip and config.vertical_flip.get("enabled"):
        if np.random.random() < config.vertical_flip.get("prob", 0.5):
            result = cv2.flip(result, 0)
    
    # Rotation
    if config.rotation and config.rotation.get("enabled"):
        if np.random.random() < config.rotation.get("prob", 0.3):
            max_angle = config.rotation.get("max_angle", 15)
            angle = np.random.uniform(-max_angle, max_angle)
            h, w = result.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            result = cv2.warpAffine(result, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    # Brightness
    if config.brightness and config.brightness.get("enabled"):
        if np.random.random() < config.brightness.get("prob", 0.3):
            range_val = config.brightness.get("range", 0.2)
            factor = np.random.uniform(1 - range_val, 1 + range_val)
            result = cv2.convertScaleAbs(result, alpha=factor, beta=0)
    
    # Contrast
    if config.contrast and config.contrast.get("enabled"):
        if np.random.random() < config.contrast.get("prob", 0.3):
            range_val = config.contrast.get("range", 0.2)
            factor = np.random.uniform(1 - range_val, 1 + range_val)
            mean = np.mean(result)
            result = cv2.convertScaleAbs(result, alpha=factor, beta=mean * (1 - factor))
    
    # Blur
    if config.blur and config.blur.get("enabled"):
        if np.random.random() < config.blur.get("prob", 0.1):
            max_kernel = config.blur.get("max_kernel", 5)
            kernel_size = np.random.choice([3, 5, 7])
            kernel_size = min(kernel_size, max_kernel)
            if kernel_size % 2 == 0:
                kernel_size += 1
            result = cv2.GaussianBlur(result, (kernel_size, kernel_size), 0)
    
    # Noise
    if config.noise and config.noise.get("enabled"):
        if np.random.random() < config.noise.get("prob", 0.1):
            intensity = config.noise.get("intensity", 0.02)
            noise = np.random.normal(0, intensity * 255, result.shape).astype(np.uint8)
            result = cv2.add(result, noise)
    
    # Cutout
    if config.cutout and config.cutout.get("enabled"):
        if np.random.random() < config.cutout.get("prob", 0.2):
            size = config.cutout.get("size", 50)
            h, w = result.shape[:2]
            x = np.random.randint(0, max(1, w - size))
            y = np.random.randint(0, max(1, h - size))
            result[y:y+size, x:x+size] = 0
    
    # Normalize (keep as uint8 for display)
    if config.normalize:
        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
    
    return result


@router.post("/preview")
async def preview_augmentation(
    image: UploadFile = File(...),
    config: str = Form(...)
):
    """
    Preview augmentation on a single image.
    Returns the augmented image as PNG.
    """
    try:
        # Parse config
        config_dict = json.loads(config)
        aug_config = AugmentationConfig(**config_dict)
        
        # Read and decode image
        file_bytes = await image.read()
        img = decode_image(file_bytes)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Failed to decode image")
        
        # Apply augmentation
        augmented = apply_augmentation(img, aug_config)
        
        # Encode and return
        result_bytes = encode_image(augmented, "png")
        return Response(content=result_bytes, media_type="image/png")
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid config JSON")
    except Exception as e:
        logger.error(f"Augmentation preview error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/preview-from-path")
async def preview_from_path(request: PreviewRequest):
    """
    Preview augmentation using image path from project.
    Returns base64 encoded image.
    """
    try:
        # Validate path
        if not os.path.exists(request.image_path):
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Read image
        img = cv2.imread(request.image_path)
        if img is None:
            raise HTTPException(status_code=400, detail="Failed to read image")
        
        # Apply augmentation
        augmented = apply_augmentation(img, request.config)
        
        # Encode to base64
        result_bytes = encode_image(augmented, "png")
        b64 = base64.b64encode(result_bytes).decode('utf-8')
        
        return {
            "success": True,
            "image": f"data:image/png;base64,{b64}",
            "width": augmented.shape[1],
            "height": augmented.shape[0]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Augmentation preview error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch")
async def batch_augment(request: BatchAugmentRequest):
    """
    Apply augmentation to all images in a project dataset.
    Creates augmented copies in a separate folder.
    """
    try:
        project_path = request.project_path
        datasets_dir = os.path.join(project_path, "datasets")
        augmented_dir = os.path.join(project_path, "datasets", "augmented")
        
        if not os.path.exists(datasets_dir):
            raise HTTPException(status_code=404, detail="Datasets directory not found")
        
        os.makedirs(augmented_dir, exist_ok=True)
        
        # Find all images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        original_images = []
        
        for f in os.listdir(datasets_dir):
            if os.path.splitext(f.lower())[1] in image_extensions:
                original_images.append(os.path.join(datasets_dir, f))
        
        if not original_images:
            raise HTTPException(status_code=400, detail="No images found in dataset")
        
        augmented_count = 0
        errors = []
        
        for img_path in original_images:
            try:
                img = cv2.imread(img_path)
                if img is None:
                    errors.append(f"Failed to read: {os.path.basename(img_path)}")
                    continue
                
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                
                for i in range(request.num_augmented_per_image):
                    augmented = apply_augmentation(img, request.config)
                    output_path = os.path.join(augmented_dir, f"{base_name}_aug_{i}.png")
                    cv2.imwrite(output_path, augmented)
                    augmented_count += 1
                    
            except Exception as e:
                errors.append(f"Error processing {os.path.basename(img_path)}: {str(e)}")
        
        return {
            "success": True,
            "original_count": len(original_images),
            "augmented_count": augmented_count,
            "output_dir": augmented_dir,
            "errors": errors if errors else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch augmentation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/save-config")
async def save_config(project_path: str, config: AugmentationConfig):
    """Save augmentation config to project."""
    try:
        config_path = os.path.join(project_path, "augmentation_config.json")
        
        with open(config_path, 'w') as f:
            json.dump(config.dict(), f, indent=2)
        
        return {"success": True, "path": config_path}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/load-config")
async def load_config(project_path: str):
    """Load augmentation config from project."""
    try:
        config_path = os.path.join(project_path, "augmentation_config.json")
        
        if not os.path.exists(config_path):
            return {"config": None, "message": "No saved config found"}
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        return {"config": config_dict}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
