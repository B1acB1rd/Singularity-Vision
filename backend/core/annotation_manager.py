import os
import json
from typing import List, Dict, Any

class AnnotationManager:
    def __init__(self):
        pass

    def get_classes(self, project_path: str) -> List[str]:
        """Get list of classes from project config"""
        config_path = os.path.join(project_path, "config.json")
        if not os.path.exists(config_path):
            return []
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config.get("datasetInfo", {}).get("classes", [])
        except Exception:
            return []

    def add_class(self, project_path: str, class_name: str) -> List[str]:
        """Add a new class to the project"""
        config_path = os.path.join(project_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError("Project config not found")
            
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        if "datasetInfo" not in config:
            config["datasetInfo"] = {}
        if "classes" not in config["datasetInfo"]:
            config["datasetInfo"]["classes"] = []
            
        classes = config["datasetInfo"]["classes"]
        if class_name not in classes:
            classes.append(class_name)
            
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        return classes

    def remove_class(self, project_path: str, class_name: str) -> List[str]:
        """Remove a class from the project"""
        config_path = os.path.join(project_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError("Project config not found")
            
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        classes = config.get("datasetInfo", {}).get("classes", [])
        if class_name in classes:
            classes.remove(class_name)
            
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        return classes

    def save_annotation(self, project_path: str, image_name: str, annotations: List[Dict[str, Any]], image_size: Dict[str, int]):
        """
        Save annotations in YOLO format
        annotations: List of dicts with {cls_idx, x, y, w, h} (normalized)
        """
        labels_dir = os.path.join(project_path, "datasets", "labels")
        os.makedirs(labels_dir, exist_ok=True)
        
        # YOLO format expects .txt file with same name as image
        base_name = os.path.splitext(image_name)[0]
        label_path = os.path.join(labels_dir, f"{base_name}.txt")
        
        with open(label_path, 'w') as f:
            for ann in annotations:
                # YOLO format: <class> <x_center> <y_center> <width> <height>
                row = f"{ann['cls_idx']} {ann['x']} {ann['y']} {ann['w']} {ann['h']}\n"
                f.write(row)

    def get_annotation(self, project_path: str, image_name: str) -> List[Dict[str, Any]]:
        """Load annotations for an image"""
        base_name = os.path.splitext(image_name)[0]
        # Check both datasets/labels and side-by-side (if we decide to support that)
        # For now, strict structure: datasets/labels/
        label_path = os.path.join(project_path, "datasets", "labels", f"{base_name}.txt")
        
        if not os.path.exists(label_path):
            return []
            
        annotations = []
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    annotations.append({
                        "cls_idx": int(parts[0]),
                        "x": float(parts[1]),
                        "y": float(parts[2]),
                        "w": float(parts[3]),
                        "h": float(parts[4])
                    })
        except Exception as e:
            print(f"Error loading annotation {label_path}: {e}")
            
        return annotations

    def export_to_voc(
        self,
        project_path: str,
        image_name: str,
        annotations: List[Dict[str, Any]],
        image_size: Dict[str, int],
        classes: List[str]
    ) -> str:
        """
        Export annotations to Pascal VOC XML format.
        
        Args:
            project_path: Path to project
            image_name: Image filename
            annotations: List of normalized YOLO format annotations
            image_size: Dict with 'width' and 'height'
            classes: List of class names
            
        Returns:
            Path to created XML file
        """
        from xml.etree.ElementTree import Element, SubElement, ElementTree
        
        # Create output directory
        voc_dir = os.path.join(project_path, "exports", "voc")
        os.makedirs(voc_dir, exist_ok=True)
        
        # Create root element
        annotation_elem = Element("annotation")
        
        # Add folder and filename
        folder = SubElement(annotation_elem, "folder")
        folder.text = "images"
        
        filename = SubElement(annotation_elem, "filename")
        filename.text = image_name
        
        # Add size
        size = SubElement(annotation_elem, "size")
        width_elem = SubElement(size, "width")
        width_elem.text = str(image_size.get("width", 0))
        height_elem = SubElement(size, "height")
        height_elem.text = str(image_size.get("height", 0))
        depth_elem = SubElement(size, "depth")
        depth_elem.text = "3"
        
        # Add segmented
        segmented = SubElement(annotation_elem, "segmented")
        segmented.text = "0"
        
        # Convert normalized YOLO to absolute VOC coordinates
        img_width = image_size.get("width", 1)
        img_height = image_size.get("height", 1)
        
        for ann in annotations:
            obj = SubElement(annotation_elem, "object")
            
            # Class name
            name = SubElement(obj, "name")
            cls_idx = ann.get("cls_idx", 0)
            name.text = classes[cls_idx] if cls_idx < len(classes) else f"class_{cls_idx}"
            
            # Pose and flags
            pose = SubElement(obj, "pose")
            pose.text = "Unspecified"
            truncated = SubElement(obj, "truncated")
            truncated.text = "0"
            difficult = SubElement(obj, "difficult")
            difficult.text = "0"
            
            # Bounding box (convert from center to corner)
            x_center = ann.get("x", 0) * img_width
            y_center = ann.get("y", 0) * img_height
            width = ann.get("w", 0) * img_width
            height = ann.get("h", 0) * img_height
            
            xmin = int(x_center - width / 2)
            ymin = int(y_center - height / 2)
            xmax = int(x_center + width / 2)
            ymax = int(y_center + height / 2)
            
            bndbox = SubElement(obj, "bndbox")
            SubElement(bndbox, "xmin").text = str(max(0, xmin))
            SubElement(bndbox, "ymin").text = str(max(0, ymin))
            SubElement(bndbox, "xmax").text = str(min(img_width, xmax))
            SubElement(bndbox, "ymax").text = str(min(img_height, ymax))
        
        # Write XML file
        base_name = os.path.splitext(image_name)[0]
        xml_path = os.path.join(voc_dir, f"{base_name}.xml")
        
        tree = ElementTree(annotation_elem)
        tree.write(xml_path, encoding="unicode", xml_declaration=True)
        
        return xml_path

    def import_from_voc(
        self,
        voc_xml_path: str,
        classes: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Import annotations from Pascal VOC XML format.
        
        Args:
            voc_xml_path: Path to VOC XML file
            classes: List of class names
            
        Returns:
            List of normalized YOLO format annotations
        """
        import xml.etree.ElementTree as ET
        
        if not os.path.exists(voc_xml_path):
            return []
        
        try:
            tree = ET.parse(voc_xml_path)
            root = tree.getroot()
            
            # Get image size
            size = root.find("size")
            if size is None:
                return []
            
            width = int(size.findtext("width", "1"))
            height = int(size.findtext("height", "1"))
            
            annotations = []
            
            for obj in root.findall("object"):
                name = obj.findtext("name", "")
                
                # Find class index
                if name in classes:
                    cls_idx = classes.index(name)
                else:
                    # Add new class
                    classes.append(name)
                    cls_idx = len(classes) - 1
                
                bndbox = obj.find("bndbox")
                if bndbox is None:
                    continue
                
                xmin = int(float(bndbox.findtext("xmin", "0")))
                ymin = int(float(bndbox.findtext("ymin", "0")))
                xmax = int(float(bndbox.findtext("xmax", "0")))
                ymax = int(float(bndbox.findtext("ymax", "0")))
                
                # Convert to YOLO normalized format
                box_width = xmax - xmin
                box_height = ymax - ymin
                x_center = xmin + box_width / 2
                y_center = ymin + box_height / 2
                
                annotations.append({
                    "cls_idx": cls_idx,
                    "x": x_center / width,
                    "y": y_center / height,
                    "w": box_width / width,
                    "h": box_height / height
                })
            
            return annotations
            
        except Exception as e:
            print(f"Error parsing VOC XML {voc_xml_path}: {e}")
            return []

annotation_manager = AnnotationManager()
