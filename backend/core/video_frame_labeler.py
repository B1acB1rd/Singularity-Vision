"""
Video Frame Labeling Manager - Frame-by-frame annotation for videos
"""
import os
import cv2
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime


class VideoFrameLabeler:
    """
    Manages video frame-by-frame annotation.
    
    Features:
    - Extract frames for labeling
    - Track annotations across frames
    - Temporal interpolation
    - Export frame annotations
    """
    
    def __init__(self):
        pass
    
    def get_video_frames(
        self,
        video_path: str,
        start_frame: int = 0,
        frame_count: int = 10
    ) -> Dict[str, Any]:
        """
        Get frame information from a video.
        
        Args:
            video_path: Path to video file
            start_frame: Starting frame index
            frame_count: Number of frames to retrieve info for
            
        Returns:
            Dict with frame metadata
        """
        if not os.path.exists(video_path):
            return {"error": "Video file not found"}
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return {"error": "Could not open video"}
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            cap.release()
            
            return {
                "video_path": video_path,
                "total_frames": total_frames,
                "fps": fps,
                "width": width,
                "height": height,
                "duration_seconds": total_frames / fps if fps > 0 else 0,
                "start_frame": start_frame,
                "frame_count": min(frame_count, total_frames - start_frame)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def extract_frame(
        self,
        video_path: str,
        frame_index: int,
        output_dir: str
    ) -> Dict[str, Any]:
        """
        Extract a single frame from video.
        
        Args:
            video_path: Path to video
            frame_index: Frame index to extract
            output_dir: Directory to save frame
            
        Returns:
            Dict with frame path
        """
        try:
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                return {"error": f"Could not read frame {frame_index}"}
            
            os.makedirs(output_dir, exist_ok=True)
            
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            frame_path = os.path.join(output_dir, f"{video_name}_frame_{frame_index:06d}.jpg")
            
            cv2.imwrite(frame_path, frame)
            
            return {
                "status": "success",
                "frame_path": frame_path,
                "frame_index": frame_index
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def extract_keyframes(
        self,
        video_path: str,
        output_dir: str,
        interval: int = 30,
        max_frames: int = 100
    ) -> Dict[str, Any]:
        """
        Extract keyframes at regular intervals for labeling.
        
        Args:
            video_path: Path to video
            output_dir: Directory to save frames
            interval: Frame interval
            max_frames: Maximum frames to extract
            
        Returns:
            Dict with extracted frame paths
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return {"error": "Could not open video"}
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            os.makedirs(output_dir, exist_ok=True)
            
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            extracted = []
            
            frame_idx = 0
            while frame_idx < total_frames and len(extracted) < max_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    frame_path = os.path.join(output_dir, f"{video_name}_frame_{frame_idx:06d}.jpg")
                    cv2.imwrite(frame_path, frame)
                    extracted.append({
                        "frame_index": frame_idx,
                        "path": frame_path
                    })
                
                frame_idx += interval
            
            cap.release()
            
            return {
                "status": "success",
                "video_path": video_path,
                "extracted_count": len(extracted),
                "interval": interval,
                "frames": extracted
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def save_frame_annotation(
        self,
        project_path: str,
        video_name: str,
        frame_index: int,
        annotations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Save annotations for a specific video frame.
        
        Args:
            project_path: Project directory
            video_name: Video filename
            frame_index: Frame index
            annotations: List of annotations
            
        Returns:
            Dict with save status
        """
        labels_dir = os.path.join(project_path, "datasets", "video_labels", video_name)
        os.makedirs(labels_dir, exist_ok=True)
        
        label_path = os.path.join(labels_dir, f"frame_{frame_index:06d}.json")
        
        data = {
            "video_name": video_name,
            "frame_index": frame_index,
            "annotations": annotations,
            "saved_at": datetime.now().isoformat()
        }
        
        with open(label_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return {"status": "success", "path": label_path}
    
    def load_frame_annotation(
        self,
        project_path: str,
        video_name: str,
        frame_index: int
    ) -> Dict[str, Any]:
        """Load annotations for a specific video frame"""
        label_path = os.path.join(
            project_path, "datasets", "video_labels",
            video_name, f"frame_{frame_index:06d}.json"
        )
        
        if not os.path.exists(label_path):
            return {"annotations": [], "frame_index": frame_index}
        
        try:
            with open(label_path, 'r') as f:
                return json.load(f)
        except:
            return {"annotations": [], "frame_index": frame_index}
    
    def interpolate_annotations(
        self,
        start_annotation: Dict[str, Any],
        end_annotation: Dict[str, Any],
        start_frame: int,
        end_frame: int,
        target_frame: int
    ) -> Dict[str, Any]:
        """
        Linearly interpolate annotation between two keyframes.
        
        Args:
            start_annotation: Annotation at start frame
            end_annotation: Annotation at end frame
            start_frame: Start frame index
            end_frame: End frame index
            target_frame: Frame to interpolate for
            
        Returns:
            Interpolated annotation
        """
        if target_frame <= start_frame:
            return start_annotation
        if target_frame >= end_frame:
            return end_annotation
        
        # Calculate interpolation factor
        total_frames = end_frame - start_frame
        t = (target_frame - start_frame) / total_frames
        
        # Interpolate bounding box
        start_bbox = start_annotation.get("bbox", {})
        end_bbox = end_annotation.get("bbox", {})
        
        interpolated = {
            "class_id": start_annotation.get("class_id"),
            "class_name": start_annotation.get("class_name"),
            "track_id": start_annotation.get("track_id"),
            "interpolated": True,
            "bbox": {
                "x": start_bbox.get("x", 0) + t * (end_bbox.get("x", 0) - start_bbox.get("x", 0)),
                "y": start_bbox.get("y", 0) + t * (end_bbox.get("y", 0) - start_bbox.get("y", 0)),
                "width": start_bbox.get("width", 0) + t * (end_bbox.get("width", 0) - start_bbox.get("width", 0)),
                "height": start_bbox.get("height", 0) + t * (end_bbox.get("height", 0) - start_bbox.get("height", 0))
            }
        }
        
        return interpolated
    
    def export_video_annotations(
        self,
        project_path: str,
        video_name: str,
        output_format: str = "yolo"
    ) -> Dict[str, Any]:
        """
        Export all video frame annotations.
        
        Args:
            project_path: Project directory
            video_name: Video filename
            output_format: 'yolo', 'coco', or 'json'
            
        Returns:
            Dict with export status
        """
        labels_dir = os.path.join(project_path, "datasets", "video_labels", video_name)
        
        if not os.path.exists(labels_dir):
            return {"error": "No annotations found for this video"}
        
        export_dir = os.path.join(project_path, "exports", "video_annotations", video_name)
        os.makedirs(export_dir, exist_ok=True)
        
        all_annotations = []
        
        for filename in sorted(os.listdir(labels_dir)):
            if filename.endswith('.json'):
                with open(os.path.join(labels_dir, filename), 'r') as f:
                    data = json.load(f)
                all_annotations.append(data)
        
        if output_format == "json":
            output_path = os.path.join(export_dir, "annotations.json")
            with open(output_path, 'w') as f:
                json.dump(all_annotations, f, indent=2)
        
        elif output_format == "yolo":
            for ann_data in all_annotations:
                frame_idx = ann_data.get("frame_index", 0)
                txt_path = os.path.join(export_dir, f"frame_{frame_idx:06d}.txt")
                
                with open(txt_path, 'w') as f:
                    for ann in ann_data.get("annotations", []):
                        bbox = ann.get("bbox", {})
                        class_id = ann.get("class_id", 0)
                        
                        # Convert to YOLO format (normalized center x, y, w, h)
                        x_center = bbox.get("x", 0) + bbox.get("width", 0) / 2
                        y_center = bbox.get("y", 0) + bbox.get("height", 0) / 2
                        
                        f.write(f"{class_id} {x_center} {y_center} {bbox.get('width', 0)} {bbox.get('height', 0)}\n")
        
        return {
            "status": "success",
            "export_dir": export_dir,
            "format": output_format,
            "frame_count": len(all_annotations)
        }


# Singleton instance
video_frame_labeler = VideoFrameLabeler()
