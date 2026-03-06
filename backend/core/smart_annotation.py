"""
Smart Annotation Tools - Advanced annotation assistance features
"""
import os
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime


class SmartAnnotationTools:
    """
    Advanced annotation assistance features.
    
    Features:
    - Auto-snap to edges (edge detection)
    - Label validation rules
    - Annotation quality scoring
    - Auto-suggest annotations
    - Duplicate detection
    """
    
    def __init__(self):
        pass
    
    def snap_to_edges(
        self,
        image_path: str,
        bbox: Dict[str, float],
        snap_radius: int = 10
    ) -> Dict[str, Any]:
        """
        Snap bounding box edges to detected edges in image.
        
        Args:
            image_path: Path to image
            bbox: Initial bounding box {x, y, width, height}
            snap_radius: Pixel radius to search for edges
            
        Returns:
            Dict with snapped bounding box
        """
        if not os.path.exists(image_path):
            return {"error": "Image not found"}
        
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            # Detect edges
            edges = cv2.Canny(img, 50, 150)
            
            x = int(bbox.get("x", 0))
            y = int(bbox.get("y", 0))
            w = int(bbox.get("width", 0))
            h = int(bbox.get("height", 0))
            
            # Snap each edge
            new_x = self._find_nearest_edge(edges, x, 'vertical', snap_radius)
            new_y = self._find_nearest_edge(edges, y, 'horizontal', snap_radius)
            new_x2 = self._find_nearest_edge(edges, x + w, 'vertical', snap_radius)
            new_y2 = self._find_nearest_edge(edges, y + h, 'horizontal', snap_radius)
            
            snapped_bbox = {
                "x": new_x,
                "y": new_y,
                "width": new_x2 - new_x,
                "height": new_y2 - new_y
            }
            
            return {
                "original": bbox,
                "snapped": snapped_bbox,
                "adjustments": {
                    "x": new_x - x,
                    "y": new_y - y,
                    "width": (new_x2 - new_x) - w,
                    "height": (new_y2 - new_y) - h
                }
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _find_nearest_edge(
        self,
        edges: np.ndarray,
        position: int,
        direction: str,
        radius: int
    ) -> int:
        """Find nearest edge within radius"""
        h, w = edges.shape
        
        for offset in range(radius + 1):
            for sign in [0, 1, -1]:
                check_pos = position + sign * offset
                
                if direction == 'vertical':
                    if 0 <= check_pos < w:
                        col = edges[:, check_pos]
                        if np.any(col > 0):
                            return check_pos
                else:
                    if 0 <= check_pos < h:
                        row = edges[check_pos, :]
                        if np.any(row > 0):
                            return check_pos
        
        return position
    
    def validate_annotations(
        self,
        annotations: List[Dict[str, Any]],
        image_width: int,
        image_height: int,
        rules: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validate annotations against quality rules.
        
        Args:
            annotations: List of annotations
            image_width: Image width
            image_height: Image height
            rules: Validation rules
            
        Returns:
            Validation results
        """
        default_rules = {
            "min_bbox_size": 10,  # Minimum box size in pixels
            "max_bbox_size_ratio": 0.95,  # Max size as ratio of image
            "min_aspect_ratio": 0.1,
            "max_aspect_ratio": 10.0,
            "check_overlap": True,
            "max_overlap_ratio": 0.8
        }
        
        rules = {**default_rules, **(rules or {})}
        
        issues = []
        warnings = []
        
        for i, ann in enumerate(annotations):
            bbox = ann.get("bbox", {})
            x = bbox.get("x", 0)
            y = bbox.get("y", 0)
            w = bbox.get("width", 0)
            h = bbox.get("height", 0)
            
            # Check minimum size
            if w < rules["min_bbox_size"] or h < rules["min_bbox_size"]:
                issues.append({
                    "index": i,
                    "type": "too_small",
                    "message": f"Box {i} is too small ({w}x{h})"
                })
            
            # Check maximum size
            if w > image_width * rules["max_bbox_size_ratio"] or h > image_height * rules["max_bbox_size_ratio"]:
                warnings.append({
                    "index": i,
                    "type": "very_large",
                    "message": f"Box {i} covers most of the image"
                })
            
            # Check bounds
            if x < 0 or y < 0 or x + w > image_width or y + h > image_height:
                issues.append({
                    "index": i,
                    "type": "out_of_bounds",
                    "message": f"Box {i} extends outside image"
                })
            
            # Check aspect ratio
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio < rules["min_aspect_ratio"] or aspect_ratio > rules["max_aspect_ratio"]:
                warnings.append({
                    "index": i,
                    "type": "extreme_aspect_ratio",
                    "message": f"Box {i} has unusual aspect ratio ({aspect_ratio:.2f})"
                })
        
        # Check overlaps
        if rules["check_overlap"]:
            for i in range(len(annotations)):
                for j in range(i + 1, len(annotations)):
                    overlap = self._calculate_iou(
                        annotations[i].get("bbox", {}),
                        annotations[j].get("bbox", {})
                    )
                    if overlap > rules["max_overlap_ratio"]:
                        warnings.append({
                            "indices": [i, j],
                            "type": "high_overlap",
                            "message": f"Boxes {i} and {j} overlap significantly ({overlap:.2f})"
                        })
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "checked_count": len(annotations)
        }
    
    def _calculate_iou(self, bbox1: Dict, bbox2: Dict) -> float:
        """Calculate Intersection over Union of two boxes"""
        x1 = max(bbox1.get("x", 0), bbox2.get("x", 0))
        y1 = max(bbox1.get("y", 0), bbox2.get("y", 0))
        x2 = min(bbox1.get("x", 0) + bbox1.get("width", 0),
                 bbox2.get("x", 0) + bbox2.get("width", 0))
        y2 = min(bbox1.get("y", 0) + bbox1.get("height", 0),
                 bbox2.get("y", 0) + bbox2.get("height", 0))
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = bbox1.get("width", 0) * bbox1.get("height", 0)
        area2 = bbox2.get("width", 0) * bbox2.get("height", 0)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_quality_score(
        self,
        annotations: List[Dict[str, Any]],
        image_path: str
    ) -> Dict[str, Any]:
        """
        Calculate annotation quality score.
        
        Factors:
        - Bounding box precision (edge alignment)
        - Consistency of sizes
        - Coverage completeness
        
        Returns:
            Quality score and breakdown
        """
        if not annotations:
            return {"score": 0, "reason": "No annotations"}
        
        scores = {
            "count_score": min(len(annotations) / 5 * 100, 100),  # Expect ~5 objects
            "size_consistency": 0,
            "coverage": 0
        }
        
        # Size consistency
        sizes = [a.get("bbox", {}).get("width", 0) * a.get("bbox", {}).get("height", 0) 
                 for a in annotations]
        if len(sizes) > 1:
            mean_size = np.mean(sizes)
            std_size = np.std(sizes)
            cv = std_size / mean_size if mean_size > 0 else 1
            scores["size_consistency"] = max(0, 100 - cv * 50)
        else:
            scores["size_consistency"] = 100
        
        # Calculate coverage
        if os.path.exists(image_path):
            img = cv2.imread(image_path)
            if img is not None:
                h, w = img.shape[:2]
                total_area = w * h
                annotated_area = sum(
                    a.get("bbox", {}).get("width", 0) * a.get("bbox", {}).get("height", 0)
                    for a in annotations
                )
                coverage_ratio = annotated_area / total_area
                scores["coverage"] = min(coverage_ratio * 200, 100)  # Expect ~50% coverage
        
        overall = (scores["count_score"] + scores["size_consistency"] + scores["coverage"]) / 3
        
        return {
            "overall_score": round(overall, 1),
            "breakdown": scores,
            "annotation_count": len(annotations)
        }
    
    def find_duplicate_annotations(
        self,
        annotations: List[Dict[str, Any]],
        iou_threshold: float = 0.9
    ) -> List[Dict[str, Any]]:
        """
        Find potential duplicate annotations.
        
        Args:
            annotations: List of annotations
            iou_threshold: IOU threshold to consider as duplicate
            
        Returns:
            List of duplicate pairs
        """
        duplicates = []
        
        for i in range(len(annotations)):
            for j in range(i + 1, len(annotations)):
                # Same class?
                if annotations[i].get("class_id") == annotations[j].get("class_id"):
                    iou = self._calculate_iou(
                        annotations[i].get("bbox", {}),
                        annotations[j].get("bbox", {})
                    )
                    if iou >= iou_threshold:
                        duplicates.append({
                            "indices": [i, j],
                            "iou": round(iou, 3),
                            "class": annotations[i].get("class_name", "unknown")
                        })
        
        return duplicates
    
    def suggest_annotations(
        self,
        image_path: str,
        existing_annotations: List[Dict[str, Any]],
        model_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Suggest new annotations based on model inference.
        
        Args:
            image_path: Path to image
            existing_annotations: Already annotated boxes
            model_path: Optional model for inference
            
        Returns:
            Suggested annotations
        """
        # This would integrate with inference_manager for AI suggestions
        # For now, return a structure for future integration
        return {
            "status": "requires_model",
            "message": "Provide a model_path for AI-powered suggestions",
            "existing_count": len(existing_annotations)
        }


# Singleton instance
smart_annotation_tools = SmartAnnotationTools()
