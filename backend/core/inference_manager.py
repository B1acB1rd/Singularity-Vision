"""
Inference Manager - Handles YOLO model inference
"""
import os
from typing import List, Dict, Any, Optional
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import time

class InferenceManager:
    def __init__(self):
        self.loaded_models: Dict[str, YOLO] = {}
        self.default_model = None
    
    def load_model(self, model_path: str) -> bool:
        """Load a YOLO model for inference"""
        try:
            if model_path in self.loaded_models:
                return True
            
            model = YOLO(model_path)
            self.loaded_models[model_path] = model
            self.default_model = model
            return True
        except Exception as e:
            print(f"[InferenceManager] Failed to load model: {e}")
            return False
    
    def get_trained_models(self, project_path: str) -> List[Dict]:
        """List all trained models in a project"""
        models = []
        runs_dir = os.path.join(project_path, "runs", "detect")
        
        if not os.path.exists(runs_dir):
            return models
        
        for run_name in os.listdir(runs_dir):
            run_path = os.path.join(runs_dir, run_name)
            weights_dir = os.path.join(run_path, "weights")
            
            if os.path.exists(weights_dir):
                best_pt = os.path.join(weights_dir, "best.pt")
                last_pt = os.path.join(weights_dir, "last.pt")
                
                if os.path.exists(best_pt):
                    models.append({
                        "name": f"{run_name} (best)",
                        "path": best_pt,
                        "type": "trained"
                    })
                if os.path.exists(last_pt):
                    models.append({
                        "name": f"{run_name} (last)",
                        "path": last_pt,
                        "type": "trained"
                    })
        
        return models
    
    def run_inference(
        self,
        model_path: str,
        image_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ) -> Dict[str, Any]:
        """Run inference on a single image"""
        start_time = time.time()
        
        # Load model if not already loaded
        if model_path not in self.loaded_models:
            if not self.load_model(model_path):
                return {"error": "Failed to load model"}
        
        model = self.loaded_models[model_path]
        
        # Run inference
        try:
            results = model.predict(
                source=image_path,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False
            )
            
            inference_time = (time.time() - start_time) * 1000  # ms
            
            # Parse results
            detections = []
            if len(results) > 0:
                result = results[0]
                boxes = result.boxes
                
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = model.names[cls_id] if cls_id in model.names else f"class_{cls_id}"
                    
                    detections.append({
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "confidence": conf,
                        "class_id": cls_id,
                        "class_name": cls_name
                    })
            
            return {
                "status": "success",
                "detections": detections,
                "inference_time_ms": round(inference_time, 2),
                "image_path": image_path
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def run_batch_inference(
        self,
        model_path: str,
        image_paths: List[str],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Run inference on multiple images with progress tracking"""
        start_time = time.time()
        results = []
        total_detections = 0
        total_confidence = 0.0
        detection_count = 0
        errors = []
        
        total_images = len(image_paths)
        
        for idx, img_path in enumerate(image_paths):
            # Report progress
            if progress_callback:
                progress = ((idx) / total_images) * 100
                progress_callback(progress, idx, total_images, img_path)
            
            result = self.run_inference(model_path, img_path, conf_threshold, iou_threshold)
            
            if "error" in result:
                errors.append({"path": img_path, "error": result["error"]})
                results.append({
                    "image_path": img_path,
                    "status": "error",
                    "error": result["error"],
                    "detections": []
                })
            else:
                detections = result.get("detections", [])
                total_detections += len(detections)
                
                # Track confidence stats
                for det in detections:
                    total_confidence += det.get("confidence", 0)
                    detection_count += 1
                
                results.append({
                    "image_path": img_path,
                    "status": "success",
                    "detections": detections,
                    "inference_time_ms": result.get("inference_time_ms", 0)
                })
        
        # Final progress
        if progress_callback:
            progress_callback(100, total_images, total_images, "Complete")
        
        processing_time = time.time() - start_time
        avg_time_per_image = (processing_time / total_images) * 1000 if total_images > 0 else 0
        avg_confidence = (total_confidence / detection_count) if detection_count > 0 else 0
        
        return {
            "status": "success",
            "results": results,
            "summary": {
                "total_images": total_images,
                "successful": total_images - len(errors),
                "failed": len(errors),
                "total_detections": total_detections,
                "avg_confidence": round(avg_confidence, 3),
                "processing_time_s": round(processing_time, 2),
                "avg_time_per_image_ms": round(avg_time_per_image, 2),
                "images_per_second": round(total_images / processing_time, 2) if processing_time > 0 else 0
            },
            "errors": errors
        }
    
    def run_video_inference(
        self,
        model_path: str,
        video_path: str,
        output_path: Optional[str] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Run inference on a video file"""
        start_time = time.time()
        
        # Load model if not already loaded
        if model_path not in self.loaded_models:
            if not self.load_model(model_path):
                return {"error": "Failed to load model"}
        
        model = self.loaded_models[model_path]
        
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {"error": "Failed to open video file"}
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Setup output video writer if output path specified
            writer = None
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_count = 0
            total_detections = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Run inference on frame
                results = model.predict(
                    source=frame,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    verbose=False
                )
                
                # Draw detections
                if len(results) > 0:
                    result = results[0]
                    annotated_frame = result.plot()
                    total_detections += len(result.boxes)
                    
                    if writer:
                        writer.write(annotated_frame)
                else:
                    if writer:
                        writer.write(frame)
                
                # Progress callback
                if progress_callback and frame_count % 10 == 0:
                    progress = (frame_count / total_frames) * 100
                    progress_callback(progress, frame_count, total_frames)
            
            cap.release()
            if writer:
                writer.release()
            
            processing_time = time.time() - start_time
            avg_fps = frame_count / processing_time if processing_time > 0 else 0
            
            return {
                "status": "success",
                "frames_processed": frame_count,
                "total_detections": total_detections,
                "processing_time_s": round(processing_time, 2),
                "avg_fps": round(avg_fps, 1),
                "output_path": output_path
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def start_webcam_inference(
        self,
        model_path: str,
        camera_id: int = 0,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ):
        """Start webcam inference - yields frames with detections"""
        # Load model if not already loaded
        if model_path not in self.loaded_models:
            if not self.load_model(model_path):
                yield {"error": "Failed to load model"}
                return
        
        model = self.loaded_models[model_path]
        
        try:
            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                yield {"error": f"Failed to open camera {camera_id}"}
                return
            
            # Set camera properties for better performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                start_time = time.time()
                
                # Run inference
                results = model.predict(
                    source=frame,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    verbose=False
                )
                
                inference_time = (time.time() - start_time) * 1000
                
                # Get annotated frame
                if len(results) > 0:
                    annotated_frame = results[0].plot()
                    num_detections = len(results[0].boxes)
                else:
                    annotated_frame = frame
                    num_detections = 0
                
                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_bytes = buffer.tobytes()
                
                yield {
                    "frame": frame_bytes,
                    "detections": num_detections,
                    "inference_time_ms": round(inference_time, 2)
                }
            
            cap.release()
            
        except GeneratorExit:
            cap.release()
        except Exception as e:
            yield {"error": str(e)}

# Singleton instance
inference_manager = InferenceManager()


