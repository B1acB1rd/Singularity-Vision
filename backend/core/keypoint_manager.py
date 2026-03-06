"""
Keypoints Annotation Manager - Handle pose/keypoint annotations
"""
import os
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime


# Predefined skeleton configurations
SKELETON_PRESETS = {
    "coco_person": {
        "name": "COCO Person (17 keypoints)",
        "keypoints": [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ],
        "skeleton": [
            [0, 1], [0, 2], [1, 3], [2, 4],  # Head
            [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],  # Arms
            [5, 11], [6, 12], [11, 12],  # Torso
            [11, 13], [13, 15], [12, 14], [14, 16]  # Legs
        ],
        "colors": {
            "head": "#FF6B6B",
            "arms": "#4ECDC4",
            "torso": "#45B7D1",
            "legs": "#96CEB4"
        }
    },
    "hand_21": {
        "name": "Hand (21 keypoints)",
        "keypoints": [
            "wrist",
            "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
            "index_mcp", "index_pip", "index_dip", "index_tip",
            "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
            "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
            "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"
        ],
        "skeleton": [
            [0, 1], [1, 2], [2, 3], [3, 4],  # Thumb
            [0, 5], [5, 6], [6, 7], [7, 8],  # Index
            [0, 9], [9, 10], [10, 11], [11, 12],  # Middle
            [0, 13], [13, 14], [14, 15], [15, 16],  # Ring
            [0, 17], [17, 18], [18, 19], [19, 20]  # Pinky
        ],
        "colors": {
            "thumb": "#FF6B6B",
            "index": "#4ECDC4",
            "middle": "#45B7D1",
            "ring": "#96CEB4",
            "pinky": "#DDA0DD"
        }
    },
    "face_68": {
        "name": "Face (68 keypoints)",
        "keypoints": [f"point_{i}" for i in range(68)],
        "skeleton": [],  # Face typically uses contours, not skeleton
        "colors": {"face": "#FFB347"}
    },
    "custom": {
        "name": "Custom Skeleton",
        "keypoints": [],
        "skeleton": [],
        "colors": {}
    }
}


class KeypointAnnotation:
    """Represents a single keypoint annotation"""
    
    def __init__(
        self,
        image_id: str,
        skeleton_type: str,
        keypoints: List[Dict[str, Any]],
        class_name: str = "person",
        confidence: float = 1.0
    ):
        self.id = f"kp_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        self.image_id = image_id
        self.skeleton_type = skeleton_type
        self.keypoints = keypoints  # [{"x": float, "y": float, "visible": int}, ...]
        self.class_name = class_name
        self.confidence = confidence
        self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "image_id": self.image_id,
            "skeleton_type": self.skeleton_type,
            "class_name": self.class_name,
            "keypoints": self.keypoints,
            "confidence": self.confidence,
            "created_at": self.created_at
        }
    
    def to_coco_format(self) -> Dict[str, Any]:
        """Convert to COCO keypoint format"""
        kp_flat = []
        for kp in self.keypoints:
            kp_flat.extend([kp.get("x", 0), kp.get("y", 0), kp.get("visible", 2)])
        
        return {
            "keypoints": kp_flat,
            "num_keypoints": sum(1 for kp in self.keypoints if kp.get("visible", 0) > 0),
            "category_id": 1  # Default to person
        }


class KeypointManager:
    """
    Manages keypoint annotations for pose estimation.
    
    Supports:
    - Multiple skeleton presets (COCO, Hand, Face)
    - Custom skeleton definitions
    - COCO format export
    - Keypoint validation
    """
    
    def __init__(self):
        self.presets = SKELETON_PRESETS
    
    def get_presets(self) -> List[Dict[str, Any]]:
        """Get available skeleton presets"""
        return [
            {
                "id": key,
                "name": preset["name"],
                "keypoint_count": len(preset["keypoints"]),
                "has_skeleton": len(preset["skeleton"]) > 0
            }
            for key, preset in self.presets.items()
        ]
    
    def get_preset(self, preset_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific skeleton preset"""
        return self.presets.get(preset_id)
    
    def create_annotation(
        self,
        image_id: str,
        skeleton_type: str,
        keypoints: List[Dict[str, Any]],
        class_name: str = "person"
    ) -> KeypointAnnotation:
        """Create a new keypoint annotation"""
        return KeypointAnnotation(
            image_id=image_id,
            skeleton_type=skeleton_type,
            keypoints=keypoints,
            class_name=class_name
        )
    
    def validate_keypoints(
        self,
        keypoints: List[Dict[str, Any]],
        skeleton_type: str
    ) -> Dict[str, Any]:
        """Validate keypoint annotation"""
        preset = self.presets.get(skeleton_type)
        
        if not preset:
            return {"valid": False, "error": f"Unknown skeleton type: {skeleton_type}"}
        
        expected_count = len(preset["keypoints"])
        actual_count = len(keypoints)
        
        if actual_count != expected_count:
            return {
                "valid": False,
                "error": f"Expected {expected_count} keypoints, got {actual_count}"
            }
        
        # Check keypoint structure
        for i, kp in enumerate(keypoints):
            if "x" not in kp or "y" not in kp:
                return {
                    "valid": False,
                    "error": f"Keypoint {i} missing x or y coordinate"
                }
        
        return {"valid": True, "keypoint_count": actual_count}
    
    def save_annotations(
        self,
        project_path: str,
        image_id: str,
        annotations: List[KeypointAnnotation]
    ) -> Dict[str, Any]:
        """Save keypoint annotations for an image"""
        labels_dir = os.path.join(project_path, "datasets", "keypoints")
        os.makedirs(labels_dir, exist_ok=True)
        
        # Use image name without extension
        base_name = os.path.splitext(image_id)[0]
        label_path = os.path.join(labels_dir, f"{base_name}.json")
        
        data = {
            "image_id": image_id,
            "annotations": [ann.to_dict() for ann in annotations],
            "saved_at": datetime.now().isoformat()
        }
        
        with open(label_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return {"status": "success", "path": label_path, "count": len(annotations)}
    
    def load_annotations(
        self,
        project_path: str,
        image_id: str
    ) -> List[Dict[str, Any]]:
        """Load keypoint annotations for an image"""
        base_name = os.path.splitext(image_id)[0]
        label_path = os.path.join(project_path, "datasets", "keypoints", f"{base_name}.json")
        
        if not os.path.exists(label_path):
            return []
        
        try:
            with open(label_path, 'r') as f:
                data = json.load(f)
            return data.get("annotations", [])
        except:
            return []
    
    def export_coco_keypoints(
        self,
        project_path: str,
        output_path: str
    ) -> Dict[str, Any]:
        """Export all keypoints in COCO format"""
        keypoints_dir = os.path.join(project_path, "datasets", "keypoints")
        
        if not os.path.exists(keypoints_dir):
            return {"error": "No keypoint annotations found"}
        
        coco_output = {
            "info": {
                "description": "Keypoint annotations from Singularity Vision",
                "version": "1.0",
                "date_created": datetime.now().isoformat()
            },
            "categories": [{
                "id": 1,
                "name": "person",
                "keypoints": SKELETON_PRESETS["coco_person"]["keypoints"],
                "skeleton": SKELETON_PRESETS["coco_person"]["skeleton"]
            }],
            "images": [],
            "annotations": []
        }
        
        ann_id = 1
        img_id = 1
        
        for filename in os.listdir(keypoints_dir):
            if filename.endswith('.json'):
                with open(os.path.join(keypoints_dir, filename), 'r') as f:
                    data = json.load(f)
                
                image_id = data.get("image_id", filename[:-5])
                
                coco_output["images"].append({
                    "id": img_id,
                    "file_name": image_id
                })
                
                for ann in data.get("annotations", []):
                    kp_flat = []
                    for kp in ann.get("keypoints", []):
                        kp_flat.extend([
                            kp.get("x", 0),
                            kp.get("y", 0),
                            kp.get("visible", 2)
                        ])
                    
                    coco_output["annotations"].append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": 1,
                        "keypoints": kp_flat,
                        "num_keypoints": sum(1 for kp in ann.get("keypoints", []) if kp.get("visible", 0) > 0)
                    })
                    ann_id += 1
                
                img_id += 1
        
        with open(output_path, 'w') as f:
            json.dump(coco_output, f, indent=2)
        
        return {
            "status": "success",
            "path": output_path,
            "images": img_id - 1,
            "annotations": ann_id - 1
        }


# Singleton instance
keypoint_manager = KeypointManager()
