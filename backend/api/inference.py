"""
Inference API - Endpoints for running model inference
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from core.inference_manager import inference_manager

router = APIRouter()

class InferenceRequest(BaseModel):
    project_path: str
    model_path: str
    image_path: str
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45

class BatchInferenceRequest(BaseModel):
    project_path: str
    model_path: str
    image_paths: List[str]
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45

class VideoInferenceRequest(BaseModel):
    model_path: str
    video_path: str
    output_path: Optional[str] = None
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45

@router.get("/models")
async def get_trained_models(project_path: str):
    """Get list of trained models available for inference"""
    try:
        models = inference_manager.get_trained_models(project_path)
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/run")
async def run_inference(request: InferenceRequest):
    """Run inference on a single image"""
    try:
        result = inference_manager.run_inference(
            model_path=request.model_path,
            image_path=request.image_path,
            conf_threshold=request.conf_threshold,
            iou_threshold=request.iou_threshold
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch")
async def run_batch_inference(request: BatchInferenceRequest):
    """Run inference on multiple images with progress tracking and summary stats"""
    try:
        result = inference_manager.run_batch_inference(
            model_path=request.model_path,
            image_paths=request.image_paths,
            conf_threshold=request.conf_threshold,
            iou_threshold=request.iou_threshold
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/video")
async def run_video_inference(request: VideoInferenceRequest):
    """Run inference on a video file"""
    try:
        result = inference_manager.run_video_inference(
            model_path=request.model_path,
            video_path=request.video_path,
            output_path=request.output_path,
            conf_threshold=request.conf_threshold,
            iou_threshold=request.iou_threshold
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/load")
async def load_model(model_path: str):
    """Pre-load a model for faster inference"""
    success = inference_manager.load_model(model_path)
    if success:
        return {"status": "loaded", "model_path": model_path}
    else:
        raise HTTPException(status_code=500, detail="Failed to load model")

from fastapi.responses import StreamingResponse
import base64

class WebcamRequest(BaseModel):
    model_path: str
    camera_id: int = 0
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45

@router.post("/webcam/start")
async def start_webcam_stream(request: WebcamRequest):
    """Start webcam inference stream - returns MJPEG stream"""
    def generate():
        for result in inference_manager.start_webcam_inference(
            model_path=request.model_path,
            camera_id=request.camera_id,
            conf_threshold=request.conf_threshold,
            iou_threshold=request.iou_threshold
        ):
            if "error" in result:
                yield f"data: {{'error': '{result['error']}'}}\n\n"
                break
            else:
                # Encode frame as base64 for SSE
                frame_b64 = base64.b64encode(result["frame"]).decode('utf-8')
                yield f"data: {{'frame': '{frame_b64}', 'detections': {result['detections']}, 'inference_time_ms': {result['inference_time_ms']}}}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )


# Inference History Endpoints
from core.inference_history import inference_history

@router.get("/history")
async def get_inference_history(limit: int = 50, input_type: Optional[str] = None):
    """Get inference history"""
    return {"history": inference_history.get_history(limit=limit, input_type=input_type)}


@router.get("/history/{result_id}")
async def get_inference_result(result_id: str):
    """Get specific inference result"""
    result = inference_history.get_result(result_id)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    return result


@router.delete("/history/{result_id}")
async def delete_inference_result(result_id: str):
    """Delete an inference result"""
    deleted = inference_history.delete_result(result_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Result not found")
    return {"status": "deleted"}


@router.get("/history/stats")
async def get_inference_stats():
    """Get inference statistics"""
    return inference_history.get_stats()
