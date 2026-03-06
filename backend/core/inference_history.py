"""
Inference History Manager - Store and retrieve inference results
"""
import os
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path


class InferenceHistoryManager:
    """
    Manages inference result history for analysis and review.
    
    Stores:
    - Inference results with timestamps
    - Detection counts and classes
    - Processing times
    - Model used
    """
    
    def __init__(self, history_dir: Optional[str] = None):
        if history_dir:
            self.history_dir = history_dir
        else:
            self.history_dir = os.path.join(
                os.path.expanduser("~"),
                ".singularity-vision",
                "inference_history"
            )
        os.makedirs(self.history_dir, exist_ok=True)
    
    def save_result(
        self,
        project_path: str,
        model_path: str,
        input_type: str,  # "image", "video", "batch", "webcam"
        input_path: str,
        detections: List[Dict[str, Any]],
        processing_time_ms: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save an inference result to history.
        
        Returns:
            result_id: Unique identifier for the result
        """
        import uuid
        
        result_id = str(uuid.uuid4())
        
        result = {
            "id": result_id,
            "timestamp": datetime.now().isoformat(),
            "project_path": project_path,
            "model_path": model_path,
            "model_name": os.path.basename(model_path),
            "input_type": input_type,
            "input_path": input_path,
            "input_name": os.path.basename(input_path),
            "detection_count": len(detections),
            "detections": detections,
            "processing_time_ms": processing_time_ms,
            "classes_detected": list(set(d.get("class", "") for d in detections)),
            "metadata": metadata or {}
        }
        
        # Save to file
        result_file = os.path.join(self.history_dir, f"{result_id}.json")
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Update index
        self._update_index(result_id, result)
        
        return result_id
    
    def _update_index(self, result_id: str, result: Dict[str, Any]):
        """Update the history index"""
        index_path = os.path.join(self.history_dir, "index.json")
        
        index = []
        if os.path.exists(index_path):
            try:
                with open(index_path, 'r') as f:
                    index = json.load(f)
            except:
                index = []
        
        # Add new entry (summary only)
        index.insert(0, {
            "id": result_id,
            "timestamp": result["timestamp"],
            "model_name": result["model_name"],
            "input_type": result["input_type"],
            "input_name": result["input_name"],
            "detection_count": result["detection_count"]
        })
        
        # Keep only last 1000
        index = index[:1000]
        
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)
    
    def get_history(
        self,
        limit: int = 50,
        input_type: Optional[str] = None,
        project_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get inference history index"""
        index_path = os.path.join(self.history_dir, "index.json")
        
        if not os.path.exists(index_path):
            return []
        
        try:
            with open(index_path, 'r') as f:
                index = json.load(f)
        except:
            return []
        
        # Filter
        if input_type:
            index = [r for r in index if r.get("input_type") == input_type]
        
        return index[:limit]
    
    def get_result(self, result_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific inference result"""
        result_file = os.path.join(self.history_dir, f"{result_id}.json")
        
        if not os.path.exists(result_file):
            return None
        
        try:
            with open(result_file, 'r') as f:
                return json.load(f)
        except:
            return None
    
    def delete_result(self, result_id: str) -> bool:
        """Delete an inference result"""
        result_file = os.path.join(self.history_dir, f"{result_id}.json")
        
        if os.path.exists(result_file):
            os.remove(result_file)
            
            # Update index
            index_path = os.path.join(self.history_dir, "index.json")
            if os.path.exists(index_path):
                with open(index_path, 'r') as f:
                    index = json.load(f)
                index = [r for r in index if r.get("id") != result_id]
                with open(index_path, 'w') as f:
                    json.dump(index, f, indent=2)
            
            return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics"""
        index_path = os.path.join(self.history_dir, "index.json")
        
        if not os.path.exists(index_path):
            return {"total": 0}
        
        try:
            with open(index_path, 'r') as f:
                index = json.load(f)
        except:
            return {"total": 0}
        
        stats = {
            "total": len(index),
            "by_type": {},
            "total_detections": sum(r.get("detection_count", 0) for r in index)
        }
        
        for r in index:
            t = r.get("input_type", "unknown")
            stats["by_type"][t] = stats["by_type"].get(t, 0) + 1
        
        return stats


# Singleton instance
inference_history = InferenceHistoryManager()
