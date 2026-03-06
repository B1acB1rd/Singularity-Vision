"""
Video Manager - Handles video processing, frame extraction, and object tracking
"""
import os
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import tempfile
import json


class VideoManager:
    """
    Manages video processing for tracking and analysis.
    
    Features:
    - Frame extraction
    - Object detection on video
    - Object tracking across frames
    - Annotated video export
    """
    
    def __init__(self):
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """
        Get metadata about a video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dict with video metadata
        """
        if not os.path.exists(video_path):
            return {"error": "Video file not found"}
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return {"error": "Could not open video file"}
            
            info = {
                "path": video_path,
                "filename": os.path.basename(video_path),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "duration_seconds": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS),
                "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
                "size_mb": round(os.path.getsize(video_path) / (1024 * 1024), 2)
            }
            
            cap.release()
            return info
            
        except Exception as e:
            return {"error": str(e)}
    
    def extract_frames(
        self,
        video_path: str,
        output_dir: str,
        interval: int = 1,
        max_frames: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Extract frames from a video at specified intervals.
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save extracted frames
            interval: Extract every Nth frame
            max_frames: Maximum number of frames to extract
            
        Returns:
            Dict with extraction results
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {"error": "Could not open video file"}
            
            frame_count = 0
            extracted = 0
            frame_paths = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % interval == 0:
                    frame_name = f"frame_{frame_count:06d}.jpg"
                    frame_path = os.path.join(output_dir, frame_name)
                    cv2.imwrite(frame_path, frame)
                    frame_paths.append(frame_path)
                    extracted += 1
                    
                    if max_frames and extracted >= max_frames:
                        break
                
                frame_count += 1
            
            cap.release()
            
            return {
                "status": "success",
                "total_frames": frame_count,
                "extracted_frames": extracted,
                "output_dir": output_dir,
                "frame_paths": frame_paths[:10]  # Return first 10 for preview
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def run_tracking(
        self,
        video_path: str,
        model_path: str,
        output_path: Optional[str] = None,
        conf_threshold: float = 0.5,
        tracker: str = "bytetrack",
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Run object tracking on a video using YOLO.
        
        Args:
            video_path: Path to input video
            model_path: Path to YOLO model weights
            output_path: Path for annotated output video
            conf_threshold: Detection confidence threshold
            tracker: Tracking algorithm (bytetrack, botsort)
            progress_callback: Callback for progress updates
            
        Returns:
            Dict with tracking results
        """
        try:
            from ultralytics import YOLO
            
            # Load model
            model = YOLO(model_path)
            
            # Get video info
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {"error": "Could not open video file"}
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            # Setup output video
            if output_path is None:
                base_name = os.path.splitext(video_path)[0]
                output_path = f"{base_name}_tracked.mp4"
            
            # Run tracking
            results = model.track(
                source=video_path,
                conf=conf_threshold,
                tracker=f"{tracker}.yaml",
                stream=True,
                persist=True
            )
            
            # Process results and write output
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            all_tracks = {}
            frame_idx = 0
            
            for result in results:
                frame = result.orig_img
                
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        if box.id is not None:
                            track_id = int(box.id[0])
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            xyxy = box.xyxy[0].cpu().numpy()
                            
                            # Store track data
                            if track_id not in all_tracks:
                                all_tracks[track_id] = {
                                    "class": model.names[cls],
                                    "class_id": cls,
                                    "frames": []
                                }
                            
                            all_tracks[track_id]["frames"].append({
                                "frame": frame_idx,
                                "bbox": xyxy.tolist(),
                                "confidence": conf
                            })
                            
                            # Draw on frame
                            x1, y1, x2, y2 = map(int, xyxy)
                            color = self._get_track_color(track_id)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            label = f"ID:{track_id} {model.names[cls]} {conf:.2f}"
                            cv2.putText(frame, label, (x1, y1 - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                out.write(frame)
                frame_idx += 1
                
                if progress_callback:
                    progress_callback(frame_idx, total_frames)
            
            out.release()
            
            return {
                "status": "success",
                "output_video": output_path,
                "total_frames": total_frames,
                "unique_tracks": len(all_tracks),
                "tracks": all_tracks
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _get_track_color(self, track_id: int) -> tuple:
        """Generate consistent color for a track ID"""
        np.random.seed(track_id)
        return tuple(int(c) for c in np.random.randint(0, 255, 3))


# Singleton instance
video_manager = VideoManager()
